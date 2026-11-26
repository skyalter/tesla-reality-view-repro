#!/usr/bin/env python3
"""Encode the fixed text prompt of the stage-2 diffusion model once, with Wan2.1's text encoder (umT5-XXL).

The diffusion backbone keeps its text branch, frozen, and always gets the same prompt, so the 11 GB text
encoder is needed only here. The result is a small file; training and sampling read it and never load the
text encoder.

    python scripts/encode_prompt.py --weights <dir>/wan2.1-text-encoder --out assets/stage2_prompt.safetensors

``--weights`` holds ``models_t5_umt5-xxl-enc-bf16.pth`` and ``google/umt5-xxl/tokenizer.json``
(``scripts/download_weights.py --set stage2_text``); it can be deleted afterwards. Needs the ``tokenizers``
package. ``--prompt`` can be given several times; the first one is stored as ``prompt``, the others as
``extra_<n>`` (for checks).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

DEFAULT_PROMPT = "realistic dashcam footage of a car driving on a city street"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    prompts = args.prompt or [DEFAULT_PROMPT]

    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    from realityview.models.diffusion.wan.t5 import umt5_xxl

    tok = Tokenizer.from_file(os.path.join(args.weights, "google", "umt5-xxl", "tokenizer.json"))
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = umt5_xxl(encoder_only=True, dtype=dtype, device=args.device).eval().requires_grad_(False)
    state = torch.load(os.path.join(args.weights, "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    del state
    out, meta = {}, {}
    for n, text in enumerate(prompts):
        text = " ".join(text.split())
        ids = torch.tensor([tok.encode(text, add_special_tokens=True).ids], device=args.device)
        with torch.no_grad():
            emb = model(ids, torch.ones_like(ids))[0]
        key = "prompt" if n == 0 else f"extra_{n}"
        out[key] = emb.float().cpu().half()
        meta[key] = text
        print(f"{key}: {ids.shape[1]} tokens, embedding {tuple(emb.shape)}, mean |x| {float(emb.float().abs().mean()):.4f}  "
              f"{text!r}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out, metadata=meta)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
