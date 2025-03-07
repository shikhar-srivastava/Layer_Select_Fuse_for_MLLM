#!/bin/bash



export HF_ENDPOINT=https://hf-mirror.com
export XDG_CACHE_HOME=/code/hf_cache
export HF_HOME=/code/hf_cache
export TRANSFORMERS_CACHE=/code/hf_cache
export TRANSFORMERS_OFFLINE=1
conda install openjdk=8 -y

# Define the model paths
models=(
    "/path/to/model"
)


for model_path in "${models[@]}"; do
    echo "Running evaluation for model: $model_path"

    model_name=$(basename $model_path)

    echo "Starting first evaluation for $model_name..."
    CUDA_VISIBLE_DEVICES=0 nohup accelerate launch --num_processes=1 -m lmms_eval \
        --model llava --model_args pretrained="$model_path" \
        --tasks textvqa --batch_size 1 --log_samples \
        --log_samples_suffix $model_name --output_path ./logs/ \
        > ${model_name}.log 2>&1 &

    wait

    echo "Evaluation completed for model: $model_name"
done

echo "All evaluations completed."
