import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model_list = ["deepseek", "qwen"]
    dataset_list = ["MathInstruct", "code_alpaca_20k", "finance_alpaca", "MedInstruct-52k"]
    for model in model_list:
        for dataset in dataset_list:
            A = np.load(f"./visual/prune_matrix/{model}_{dataset}_prune.npy") 
            
            plt.figure(figsize=(8, 4))
            plt.imshow(
                A, 
                cmap="binary",
                interpolation="nearest",
                aspect="auto", 
                origin="upper",
            )
            
            plt.title("DeepSeek - MedInstruct", fontsize=12)
            plt.xlabel("Expert ID")
            plt.ylabel("Layer ID")
            
            plt.xticks([0, 12, 24, 36, 48, 63], [0, 12, 24, 36, 48, 63])
            plt.yticks([0, 5, 10, 15, 20, 25], [0, 5, 10, 15, 20, 25])
            
            save_path = f"visual/prune_plot/{model}_{dataset}.png"
            plt.savefig(save_path)
            plt.close()
