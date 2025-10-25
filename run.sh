LOG_NAME=Test.log
export HF_ENDPOINT=https://hf-mirror.com
export ACCELERATE_LOG_LEVEL=info
export VLLM_DISABLE_COMPILE_CACHE=1
# export CUDA_VISIBLE_DEVICES=0,1

rm -f $LOG_NAME

accelerate launch \
    --config_file /gemini/code/OpenR1/open_r1/accelerate_configs/zero3.yaml \
    --num_processes 4 \
    -- \
    /gemini/code/OpenR1/grpo.py \
    --config /gemini/code/OpenR1/config_demo.yaml \
    --model_name_or_path /gemini/code/merged_model_ckpt861 \
    --max_completion_length 8096 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.1 \
    --vllm_mode colocate \
    --output_dir /gemini/code/OpenR1/model_ckpt/test_grpo 2>&1 | tee $LOG_NAME
    