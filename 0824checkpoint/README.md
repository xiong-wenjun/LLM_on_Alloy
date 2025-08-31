---
library_name: peft
license: other
base_model: /mnt-nfsdata/MaterialCode/base-model/DeepSeek-R1-0528-Qwen3-8B
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: 0824checkpoint
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 0824checkpoint

This model is a fine-tuned version of [/mnt-nfsdata/MaterialCode/base-model/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co//mnt-nfsdata/MaterialCode/base-model/DeepSeek-R1-0528-Qwen3-8B) on the 0723_sft_dataset_4k_clean_reformat_unit_new_native_think_ready_alpaca dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- total_eval_batch_size: 64
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 10.0

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.6.0+cu124
- Datasets 3.6.0
- Tokenizers 0.21.1