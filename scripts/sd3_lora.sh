#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
export INSTANCE_DIR="dreambooth-tutorial/data/cat_mc"
export CLASS_DIR="dreambooth-tutorial/data/class_cat"
export OUTPUT_DIR="dreambooth-tutorial/outputs/sd3-lora"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of gmc cat" \
  --class_prompt="a photo of a cat, full body, natural light" \
  --validation_prompt="a photo of gmc cat walking in the forest" \
  --with_prior_preservation \
  --prior_loss_weight 0.05 \
  --num_class_images=24 \
  --learning_rate=5e-04 \
  --text_encoder_lr=5e-05 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="AdamW" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --max_train_steps=456 \
  --report_to="wandb" \
  --validation_epochs=25 \
  --seed=0
