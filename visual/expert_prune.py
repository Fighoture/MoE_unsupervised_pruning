import numpy as np
import json
import os


def map_to_matrix(prune_map):
    result = []
    sorted_info = sorted(prune_map.items(), key=lambda x: int(x[0]))
    for (layer_id, prune_list) in sorted_info:
        mask = np.ones(expert_num, dtype=int)
        mask[prune_list] = 0
        result.append(mask)
    return np.array(result)


if __name__ == "__main__":
    model = "deepseek"
    # model = "qwen"
    if model == "deepseek":
        expert_num = 64
    elif model == "qwen":
        expert_num = 60
    else:
        raise ValueError("Model must be either 'deepseek' or 'qwen'")
    subject_list = ["MathInstruct", "code_alpaca_20k", "finance_alpaca", "MedInstruct-52k"]
    load_dir = f"pruned_result/{model}_sota_prune"
    save_dir = "visual/prune_matrix"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for subject in subject_list:
        file_path = f"{load_dir}/{subject}_prune.json"
        with open(file_path, 'r') as f:
            prune_map = json.load(f)
        prune_matrix = map_to_matrix(prune_map)
        save_path = f"{save_dir}/{model}_{subject}_prune.npy"
        np.save(save_path, prune_matrix)

