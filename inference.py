#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py
Inference for SD3 + LoRA using float32 components (no fp16).
Mirrors the final-inference pattern from train_dreambooth_lora_sd3.py:
 - loads tokenizers + three text encoders, VAE, transformer explicitly
 - constructs StableDiffusion3Pipeline with those components (torch_dtype=torch.float32)
 - loads LoRA via pipeline.load_lora_weights(dir) or (dir, weight_name=file)
"""

import argparse
from pathlib import Path
import sys
import torch
from PIL import Image
import os

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from diffusers import (
    AutoencoderKL,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)

# -------------------------
# Helpers (copied / adapted from train script)
# -------------------------
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str = None, subfolder: str = "text_encoder"):
    """
    Determine which class to use for a text encoder (CLIPTextModelWithProjection or T5EncoderModel),
    by reading the pretrained config (same logic as training script).
    """
    cfg = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = cfg.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported as text encoder class")


def load_text_encoders(class_one, class_two, class_three, pretrained_model_name_or_path, revision=None):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision
    )
    text_encoder_three = class_three.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_3", revision=revision
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# -------------------------
# LoRA load helper
# -------------------------
def load_lora_into_pipeline(pipe, lora_path: str):
    """
    Accept either:
      - local directory produced by save_lora_weights
      - local .safetensors file (pytorch_lora_weights.safetensors) -> call load_lora_weights(parent, weight_name=filename)
      - Hugging Face repo id (string) -> pass that string to load_lora_weights
    """
    if lora_path is None:
        print("[info] No --lora provided; skipping LoRA load.")
        return

    p = Path(lora_path)
    if p.exists():
        if p.is_file():
            parent = str(p.parent)
            weight_name = p.name
            print(f"[info] Loading LoRA from file: {p} -> using parent={parent}, weight_name={weight_name}")
            pipe.load_lora_weights(parent, weight_name=weight_name)
        else:
            print(f"[info] Loading LoRA from directory: {p}")
            pipe.load_lora_weights(str(p))
    else:
        # assume it's a repo id like username/repo
        print(f"[info] Loading LoRA from HF repo id: {lora_path}")
        pipe.load_lora_weights(lora_path)


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True, help="e.g. stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--revision", default=None)
    p.add_argument("--variant", default=None, help="model variant (if used during training)")
    p.add_argument("--lora", default=None, help="path to directory or .safetensors file or HF repo id")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--outdir", type=str, default="outputs/inference_no_fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    if args.prompt is None and args.prompts_file is None:
        print("Specify --prompt or --prompts_file")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load tokenizers
    print("[info] Loading tokenizers...")
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)
    tokenizer_three = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision)

    # Choose classes for text encoders and load them
    print("[info] Determining text encoder classes and loading them...")
    cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder")
    cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
    cls_three = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3")

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        cls_one, cls_two, cls_three, args.pretrained_model_name_or_path, revision=args.revision
    )

    # Load VAE and transformer explicitly (as in train script) and force float32
    print("[info] Loading VAE and transformer (float32)...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    transformer = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant)

    # set requires_grad False and cast to float32 (we avoid fp16 here)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    device = args.device
    print(f"[info] Moving models to device={device} (dtype=float32)")
    vae.to(device, dtype=torch.float32)
    transformer.to(device, dtype=torch.float32)
    text_encoder_one.to(device, dtype=torch.float32)
    text_encoder_two.to(device, dtype=torch.float32)
    text_encoder_three.to(device, dtype=torch.float32)

    # Build pipeline using loaded components (torch_dtype=float32)
    print("[info] Constructing StableDiffusion3Pipeline (float32)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        text_encoder_3=text_encoder_three,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32,
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # Load LoRA (dir / file / repo)
    if args.lora:
        load_lora_into_pipeline(pipe, args.lora)

    # Read prompts
    prompts = []
    if args.prompts_file:
        pf = Path(args.prompts_file)
        if not pf.exists():
            print(f"[error] Prompts file not found: {pf}")
            sys.exit(2)
        with open(pf, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [args.prompt.strip()]

    # generate
    for idx, prompt in enumerate(prompts):
        seed = args.seed + idx
        generator = torch.Generator(device=device).manual_seed(seed) if device.startswith("cuda") else torch.Generator().manual_seed(seed)
        prompt_safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:80].strip()
        prompt_out = outdir / f"{idx:03d}_{prompt_safe}"
        prompt_out.mkdir(parents=True, exist_ok=True)
        print(f"[info] Generating prompt {idx}: '{prompt}' -> {prompt_out}")

        output = pipe(
            prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance,
            num_images_per_prompt=args.num_images,
            generator=generator,
            height=args.height,
            width=args.width,
        )

        saved = []
        for i, im in enumerate(output.images):
            path = prompt_out / f"img_{i:02d}.png"
            im.save(path)
            saved.append(path)
        print("[ok] Saved:", saved)


if __name__ == "__main__":
    main()
