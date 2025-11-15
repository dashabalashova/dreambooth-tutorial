#!/usr/bin/env python3
"""
python inference.py \
  --model "stabilityai/stable-diffusion-3.5-medium" \
  --prompts "dreambooth-tutorial/prompts/mc_cat.txt" \
  --out "dreambooth-tutorial/outputs/inference-prior" \
  --lora "dreambooth-tutorial/outputs/trained-sd3-lora-prior"
"""

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def read_prompts(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def try_load_lora(pipe, lora_dir):
    if not lora_dir:
        return False
    p = Path(lora_dir)
    if not p.exists():
        print(f"[LORA] directory not found: {lora_dir}")
        return False
    # diffusers provides load_attn_procs for many LoRA artifacts — try it simply
    try:
        if hasattr(pipe.unet, "load_attn_procs"):
            pipe.unet.load_attn_procs(str(p))
            print(f"[LORA] Applied attn_procs from: {lora_dir}")
            return True
        else:
            print("[LORA] pipe.unet has no method load_attn_procs — skipping.")
            return False
    except Exception as e:
        print("[LORA] Failed to load attn_procs:", e)
        return False

def main(args):
    prompts = read_prompts(args.prompts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = (args.device.startswith("cuda") and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print("Loading pipeline:", args.model)
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Try apply LoRA (very simple)
    if args.lora:
        try_load_lora(pipe, args.lora)

    gen = torch.Generator(device=device)
    if args.seed is not None:
        gen.manual_seed(args.seed)

    for i, prompt in enumerate(prompts):
        if i >= args.max:
            break
        print(f"[{i+1}/{min(len(prompts), args.max)}] Prompt: {prompt}")
        if device.type == "cuda":
            # use amp for faster/fp16 inference on GPU
            with torch.cuda.amp.autocast():
                out = pipe(prompt,
                           num_inference_steps=args.steps,
                           guidance_scale=args.guidance,
                           generator=gen,
                           height=args.height,
                           width=args.width)
        else:
            out = pipe(prompt,
                       num_inference_steps=args.steps,
                       guidance_scale=args.guidance,
                       generator=gen,
                       height=args.height,
                       width=args.width)

        img = out.images[0]
        fname = out_dir / f"gen_{i:03d}.png"
        img.save(fname)
        print("Saved ->", fname)

    print("All done. Output folder:", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--prompts", type=str, default="dreambooth-tutorial/prompts/mc_cat.txt")
    p.add_argument("--out", type=str, default="dreambooth-tutorial/outputs/inference_simple")
    p.add_argument("--lora", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max", type=int, default=20)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    args = p.parse_args()
    main(args)
