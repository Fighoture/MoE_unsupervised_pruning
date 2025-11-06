import torch
import os
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def compute_similarity(embedding):
    distance_matrix = squareform(pdist(embedding.to(torch.float16).detach().numpy()))
    similarity_matrix = 1 - distance_matrix / np.max(distance_matrix)
    return similarity_matrix


if __name__ == "__main__":
    model = "deepseek"
    dataset_name = "MedInstruct-52k"
    math_expert_output_dir = f"pruned_result/DeepSeek-V2-Lite/sample_1000/{dataset_name}_expert_output_hidden"
    save_dir = f"visual/deepseek_{dataset_name}_expert_similarity"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    layer_output = []
    for layer_idx in range(1, 27):
        file_name = f"layer_{layer_idx}.pth"
        file_path = f"{math_expert_output_dir}/{file_name}"
        data = torch.load(file_path)
        data = data.mean(dim=1)
        # 1 7 13 19 25
        if layer_idx % 6 == 1:
            similarity_matrix = compute_similarity(data)
            sns.heatmap(similarity_matrix, fmt='d', cmap='YlGnBu')
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            save_path = f"{save_dir}/layerwise_{file_name}.png"
            plt.savefig(save_path)
            plt.close()
        layer_output.append(data)

    layer_output = torch.stack(layer_output, dim=0)
    global_level_output = layer_output.mean(dim=1)
    global_similarity_matrix = compute_similarity(global_level_output)
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(global_similarity_matrix, fmt='d', cmap='YlGnBu')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    save_path = f"{save_dir}/global_similarity_matrix.png"
    plt.savefig(save_path)
    plt.close()
