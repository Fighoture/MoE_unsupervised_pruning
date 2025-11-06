#!/bin/bash
model_path="/home/sagemaker-user/user-default-efs/moe_prune/model/Qwen/Qwen1___5-MoE-A2___7B"
# model_path="/home/sagemaker-user/user-default-efs/moe_prune/model/deepseek-ai/DeepSeek-V2-Lite"
dataset_dir="dataset"
# dataset_name_list="c4"
# dataset_name_list="MathInstruct.json,code_alpaca_20k.json,finance_alpaca.json"
dataset_name_list="MedInstruct-52k.json"
batch_size=32
sample_number=1000
by_domain=1 # only available when using uns method
random_state=42

use_layerwise_pruning=1
#layerwise_pruning_method="kmeans_prune"
layerwise_pruning_method="hierarchical_prune"
#layerwise_pruning_method="hierarchical_prune_with_entropy"
# layerwise_pruning_method="seer_prune"
#layerwise_pruning_method="hsic_prune"
layerwise_cluster_number=12 # only available when using uns method
layerwise_prune_rate=0.1

use_global_pruning=1
#global_pruning_method="kmeans_prune"
global_pruning_method="hierarchical_prune"
global_cluster_number=8 # only available when using uns method
global_prune_rate=0.1

args=(
    --model_path "$model_path"
    --dataset_dir "$dataset_dir"
    --dataset_name_list "$dataset_name_list"
    --batch_size "$batch_size"
    --sample_number "$sample_number"
    --by_domain "$by_domain"
    --random_state "$random_state"
)

if [ "$use_layerwise_pruning" -eq 1 ]; then
    args+=(
        --layerwise_pruning_method "$layerwise_pruning_method"
        --layerwise_cluster_number "$layerwise_cluster_number"
        --layerwise_prune_rate "$layerwise_prune_rate"
    )
fi

if [ "$use_global_pruning" -eq 1 ]; then
    args+=(
        --global_pruning_method "$global_pruning_method"
        --global_cluster_number "$global_cluster_number"
        --global_prune_rate "$global_prune_rate"
    )
fi

CUDA_VISIBLE_DEVICES="0,1" python pruning_mask.py "${args[@]}"
