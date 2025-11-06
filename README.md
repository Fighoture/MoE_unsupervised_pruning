# Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

## Set Up
### Requirements
1. torch+cu121 - 2.3.0
2. transformers - 4.43.3
3. flash_attn - 2.6.3
4. scikit-learn - 1.5.1
5. peft - 0.10.0

### Dataset
- [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- [finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)
- [MedInstruct](https://huggingface.co/datasets/xz97/MedInstruct)

### Action
- Files in **transformers_py** should be copied to the path of **transformers** package of your running environment. (eg. /path/to/your/envs/{#env_name}/lib/python3.10/site-packages/transformers)
- Files in **qwen2moe_py** should be copied to the path of **qwen2_moe** model from your transformers package. (eg. /path/to/your/envs/{#env_name}/lib/python3.10/site-packages/transformers/models/qwen2_moe)
- Files in **deepseek_model_py** should be copied to the path of **deepseek** model downloaded from HuggingFace.

### Running scripts
To prune the DeepseekV2Lite model with the suggested parameters, you can run the script below:
```
./scripts/pruning_mash.sh >/path/to/your/log
```

## Citation
```
  @article{guo2025arxiv,
  title={Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models},
  author={Hongcheng Guo, Juntao Yao, Boyang Wang, Junjia Du, Shaosheng Cao, Donglin Di, Shun Zhang, Zhoujun Li},
  journal={arXiv preprint arXiv:2504.07807},
  year={2025},
  url={https://arxiv.org/abs/2504.07807}
  }
```
