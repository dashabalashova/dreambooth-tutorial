# DreamBooth tutorial – Stable Diffusion + LoRA

Large text-to-image models can generate high-quality, diverse images from text, but they typically cannot faithfully reproduce and recontextualize a specific subject from just a few reference photos. DreamBooth fine-tunes a pretrained text-to-image model to bind a unique identifier to your subject, enabling novel photorealistic renditions of that subject in different poses, scenes, and artistic styles. This tutorial shows how to train a LoRA-style DreamBooth on Stable Diffusion 3.5-medium and run inference for recontextualization, art rendition, and property modification.

<begin> <p align="center"> <img src="data/cat_gmc/gmc_003.jpeg" width="24%" /> <img src="data/cat_gmc/gmc_004.jpeg" width="24%" /> <img src="data/cat_gmc/gmc_005.jpeg" width="24%" /> <img src="data/cat_gmc/gmc_006.jpeg" width="24%" /> <br/> </p> <end>
<begin> <p align="center"> <img src="data/cat_gmc/gmc_007.jpeg" width="24%" /> <img src="data/cat_gmc/gmc_008.jpeg" width="24%" /> <img src="data/cat_gmc/gmc_011.jpeg" width="24%" /> <img src="data/cat_gmc/gmc_010.jpeg" width="24%" /> <br/> </p> <end>

---

## Overview

This guide walks you through:

1. Creating a VM and connecting via SSH
2. Installing the required software (Hugging Face `diffusers`, PEFT, etc.)
3. Running the training
4. Running the inference

---

## Repository layout (example)

The tutorial expects the repository layout like this (adjust paths as needed):

* `dreambooth-tutorial/`
* `data/cat_gmc/` – instance images (your example subject images)
* `data/prompts/recontextualization.txt` – prompts for inference
* `scripts/sd3_lora.sh` – helper script to launch training
* `inference.py` – inference helper for SD3 + LoRA
* `outputs/sd3-lora/` – where LoRA weights and generated images go

---

## 1. Create VM & connect 

The workflow requires a VM equipped with a GPU that has at least 75 GB of VRAM. The example runs were performed on an NVIDIA H200. You can create a Compute virtual machine (VM) through the [web console](https://console.nebius.com), the CLI, or the Terraform provider, see the [Nebius AI Cloud documentation](https://docs.nebius.com/compute/virtual-machines/manage) for instructions. After creating VM:
```
ssh <user>@<IP>
```

---

## 2. Install environment & dependencies

Create a Python virtual environment, install `diffusers` and requirements for the SD3 DreamBooth example.

```
# clone the repo for diffusers and create a venv

git clone https://github.com/huggingface/diffusers
python3 -m venv ~/.venvs/dreambooth
source ~/.venvs/dreambooth/bin/activate
pip install --upgrade pip

# install diffusers in editable mode and example requirements

cd diffusers
pip install -e .
cd examples/dreambooth
pip install -r requirements_sd3.txt

# install PEFT and wandb

python -m pip install "peft>=0.17.0"
pip install wandb

# login to Hugging Face

hf auth login --token hf_<your_token>
hf auth whoami
```

---

## 3. Training

For fine-tuning, 24 instance images were used: two – gmc_001.jpeg and gmc_002.jpeg – were sourced from [Freepik (author: fxquadro, free license)](https://www.freepik.com/author/fxquadro), and the remaining 22 images in `data/cat_gmc/` were generated with [Nano Banana](https://nanobanana.ai/). Helper script `scripts/sd3_lora.sh` launches `train_dreambooth_lora_sd3.py` via `accelerate` with example arguments:
```
# make script executable (one time) and run it

chmod +x dreambooth-tutorial/scripts/sd3_lora.sh
dreambooth-tutorial/scripts/sd3_lora.sh
```

The script encapsulates these arguments:
* `--pretrained_model_name_or_path` – base SD3 model (`stabilityai/stable-diffusion-3.5-medium`)
* `--instance_data_dir` – your subject images folder (`data/cat_gmc`)
* `--class_data_dir` – class images used for prior preservation
* `--instance_prompt` / `--class_prompt` – textual prompts used during training
* `--with_prior_preservation` and `--prior_loss_weight` – enable prior preservation (recommended for DreamBooth)
* `--resolution` – training size (`512`)
* `--train_batch_size`, `--gradient_accumulation_steps` – effective batch size control
* `--max_train_steps` – number of optimization steps

---

## 4. Inference

Use the included `inference.py` helper to load LoRA weights and run SD3 inference. 

Flags:
* `--lora` – path to a LoRA directory, a `.safetensors` file, or an HF repo id containing LoRA weights.
* `--prompts_file` or `--prompt` – provide prompts (one per line when using a file).
* `--num_images` – images per prompt
* `--num_steps` – number of diffusion steps (28 in example for balance of speed/quality)
* `--guidance` – classifier-free guidance scale (commonly 7.5–8.5)
* `--height` / `--width` – output resolution

---

Place your inference prompts (one per line) in `data/prompts/<prompts>.txt` or call `--prompt "your prompt here"`. Output images are saved under the `--outdir` path. Each prompt will create a directory `NNN_<prompt-safe-name>/img_XX.png`.

Recontextualization inference:
```
python dreambooth-tutorial/inference.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --lora dreambooth-tutorial/outputs/sd3-lora/pytorch_lora_weights.safetensors \
  --prompts_file dreambooth-tutorial/data/prompts/recontextualization.txt \
  --outdir dreambooth-tutorial/outputs/sd3-lora/images \
  --num_images 4 --num_steps 28 --guidance 7.5 --height 1024 --width 1024
```
<begin> <p align="center"> <img src="outputs/sd3-lora/images/002_a photo of gmc cat playfully hiding in a cardboard box_ candid_ natural light/img_00.png" width="24%" /> <img src="outputs/sd3-lora/images/005_a photo of gmc cat yawning with whiskers in sharp focus_ macro lens_ ultra high/img_02.png" width="24%" /> <img src="outputs/sd3-lora/images/011_a photo of gmc cat napping in a vintage suitcase_ cozy editorial style_ warm fil/img_00.png" width="24%" /> <img src="outputs/sd3-lora/images/017_a photo of gmc cat walking on a mossy forest path_ cinematic lighting/img_00.png" width="24%" /> <br/> </p> <end>

Art rendition inference:
```
python dreambooth-tutorial/inference.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --lora dreambooth-tutorial/outputs/sd3-lora/pytorch_lora_weights.safetensors \
  --prompts_file dreambooth-tutorial/data/prompts/art.txt \
  --outdir dreambooth-tutorial/outputs/sd3-lora/art_images \
  --num_images 4 --num_steps 28 --guidance 7.5 --height 1024 --width 1024
```
<begin> <p align="center"> <img src="outputs/sd3-lora/art_images/002_a cubist portrait of gmc cat in the style of Pablo Picasso/img_03.png" width="24%" /> <img src="outputs/sd3-lora/art_images/003_a surreal tableau of gmc cat in the style of Salvador Dalí/img_00.png" width="24%" /> <img src="outputs/sd3-lora/art_images/005_a ukiyo-e print of gmc cat in the style of Katsushika Hokusai/img_00.png" width="24%" /> <img src="outputs/sd3-lora/art_images/011_a stained glass window of gmc cat in the style of Gothic cathedral art/img_02.png" width="24%" /> <br/> </p> <end>

Property modification inference:
```
python dreambooth-tutorial/inference.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
  --lora dreambooth-tutorial/outputs/sd3-lora/pytorch_lora_weights.safetensors \
  --prompts_file dreambooth-tutorial/data/prompts/property.txt \
  --outdir dreambooth-tutorial/outputs/sd3-lora/property_images \
  --num_images 4 --num_steps 28 --guidance 7.5 --height 1024 --width 1024
```
<begin> <p align="center"> <img src="outputs/sd3-lora/property_images/002_a cross of a gmc cat and a rabbit_ with elongated ears and feline face/img_02.png" width="24%" /> <img src="outputs/sd3-lora/property_images/010_a cyborg gmc cat with delicate mechanical implants_ still recognizably gmc/img_02.png" width="24%" /> <img src="outputs/sd3-lora/property_images/013_an origami gmc cat folded from patterned paper_ clear gmc silhouette/img_02.png" width="24%" /> <img src="outputs/sd3-lora/property_images/015_a stitched plush gmc cat toy made of fabric with visible seams and embroidery/img_00.png" width="24%" /> <br/> </p> <end>

## 5. Clean up (optional)
Open [Nebius AI web console](https://console.nebius.com). Find the VM you want to delete. Click the three vertical dots (⋮) on the instance row and choose Delete — in the confirmation dialog, check Delete boot disk if you also want the instance disk removed.