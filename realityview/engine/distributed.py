"""Process setup for single-process and multi-GPU (DistributedDataParallel) training.

Started through torchrun, every process finds RANK, WORLD_SIZE and LOCAL_RANK in its environment and joins
the process group. Started plainly, there is one process and no process group. The distributed code path is
also taken with a single process when it comes from torchrun, so it can be exercised on one GPU.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as torch_dist


@dataclass
class ProcessInfo:
    rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup(device: str = "auto", backend: str = "auto") -> ProcessInfo:
    """`device`: "auto" (GPU when there is one), "cuda" or "cpu".
    `backend`: "auto" (nccl on GPU, gloo on CPU), "nccl" or "gloo". Several processes sharing one GPU, which
    only makes sense for a test, need "gloo"."""
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto, cuda or cpu, got {device}")
    use_cuda = torch.cuda.is_available() if device == "auto" else device == "cuda"
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        return ProcessInfo(0, 1, torch.device("cuda" if use_cuda else "cpu"), False)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if use_cuda:
        device_obj = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device_obj)
    else:
        device_obj = torch.device("cpu")
    if not torch_dist.is_initialized():
        name = ("nccl" if use_cuda else "gloo") if backend == "auto" else backend
        # The main process evaluates and writes checkpoints while the others wait in a collective call:
        # the default limit of 10 minutes for such a call is too short for a long evaluation.
        timeout = datetime.timedelta(hours=2)
        try:
            torch_dist.init_process_group(name, timeout=timeout, device_id=device_obj if use_cuda else None)
        except TypeError:           # PyTorch versions without the device_id argument
            torch_dist.init_process_group(name, timeout=timeout)
    return ProcessInfo(rank, world, device_obj, True)


def wrap(model: torch.nn.Module, info: ProcessInfo, find_unused_parameters: bool = False) -> torch.nn.Module:
    """DistributedDataParallel around the model when distributed, the model itself otherwise."""
    if not info.distributed:
        return model
    device_ids = [info.device.index] if info.device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids,
                                                     find_unused_parameters=find_unused_parameters)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def mean_across_processes(value: torch.Tensor, info: ProcessInfo) -> float:
    if not info.distributed:
        return float(value)
    total = value.detach().clone().float()
    torch_dist.all_reduce(total, op=torch_dist.ReduceOp.SUM)
    return float(total) / info.world_size


def any_process(flag: bool, info: ProcessInfo) -> bool:
    """True on every process if `flag` is true on at least one. Keeps the processes stopping at the same step."""
    if not info.distributed:
        return flag
    value = torch.tensor([1.0 if flag else 0.0], device=info.device)
    torch_dist.all_reduce(value, op=torch_dist.ReduceOp.MAX)
    return bool(value.item() > 0)


def cleanup(info: ProcessInfo) -> None:
    if info.distributed and torch_dist.is_initialized():
        torch_dist.barrier()
        torch_dist.destroy_process_group()
