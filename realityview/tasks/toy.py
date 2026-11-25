"""A tiny regression task. It exists to test the training loop without data or pretrained weights.

The model has dropout, so a step uses random numbers: resuming is exact only if the loop re-seeds them.
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    """y = sin(3 x0) + x1 * x2 with a fixed generator; sample i depends on i only."""

    def __init__(self, length: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(length, 3, generator=g) * 2 - 1
        self.y = torch.sin(3 * self.x[:, 0]) + self.x[:, 1] * self.x[:, 2]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i], "index": i}


class ToyTask:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        toy = cfg.get("toy", {})
        self.train_size = toy.get("train_size", 256)
        self.eval_size = toy.get("eval_size", 64)
        self.width = toy.get("width", 32)
        self.lr = toy.get("lr", 1.0e-2)
        self.dropout = toy.get("dropout", 0.1)
        self.stop_file_at_step = toy.get("stop_file_at_step")      # for tests: ask the run to stop at this step
        self.is_main = int(os.environ.get("RANK", "0")) == 0

    def build_model(self):
        return torch.nn.Sequential(torch.nn.Linear(3, self.width), torch.nn.GELU(), torch.nn.Dropout(self.dropout),
                                   torch.nn.Linear(self.width, self.width), torch.nn.GELU(),
                                   torch.nn.Linear(self.width, 1))

    def build_train_dataset(self):
        return ToyDataset(self.train_size, seed=1)

    def build_eval_dataset(self):
        return ToyDataset(self.eval_size, seed=2)

    def param_groups(self, model):
        return [{"name": "first", "params": list(model[0].parameters()), "lr": self.lr},
                {"name": "rest", "params": [p for m in list(model)[1:] for p in m.parameters()], "lr": self.lr * 0.5}]

    def training_step(self, model, batch, step):
        if self.is_main and step == self.stop_file_at_step:
            open(os.path.join(self.cfg["run_dir"], "STOP"), "w").close()
        loss = torch.nn.functional.mse_loss(model(batch["x"]).squeeze(-1), batch["y"])
        return loss, {"mse": float(loss.detach())}

    def evaluate(self, model, loader, device):
        total, count = 0.0, 0
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            total += float(((model(x).squeeze(-1) - y) ** 2).sum())
            count += len(y)
        return {"mse": total / max(count, 1)}
