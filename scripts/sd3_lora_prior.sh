# at top of scripts/sd3_lora_prior.sh
#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
export INSTANCE_DIR="data/cat_mc"
export CLASS_DIR="data/cat_class"
export OUTPUT_DIR="outputs/trained-sd3-lora-prior"

accelerate launch train_dreambooth_lora_sd3.py \
  --with_prior_preservation \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of <m-c> cat" \
  --class_prompt="a photo of a cat" \
  --validation_prompt="a photo of <m-c> cat walking in the forest" \
  --prior_loss_weight=0.05 \
  --learning_rate=1e-4 \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="AdamW" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=2000 \
  --report_to="wandb" \
  --validation_epochs=5 \
  --seed="0"