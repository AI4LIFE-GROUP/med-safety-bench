# MedSafetyBench

This repository contains the ```MedSafetyBench``` benchmark dataset and code from the paper [MedSafetyBench: Evaluating and Improving the
Medical Safety of Large Language Models](https://arxiv.org/abs/2403.03744) [NeurIPS 2024, Datasets and Benchmarks]. 


## Dataset

The ```MedSafetyBench``` benchmark dataset is designed to evaluate and improve the medical safety of large language models.

**Note**: *This dataset contains content that may be used for harmful purposes. It should be used for research only. By using this dataset, you agree to use it for research only.*

### Dataset structure

The dataset consists of:

- 1,800 medical safety demonstrations, where each safety demonstration consists of a harmful medical request and a corresponding safe response. For the harmful medical requests, 900 are developed using GPT4 (prompting) and 900 are developed using Llama2-7b-chat (jailbreaking). The safe responses are generated using GPT4. The 1,800 medical safety demonstrations are randomly split into a training set and a test set.
  - Training set (n=900): ```datasets/train``` (referred to as ```MedSafety-Improve``` in the paper)
  - Test set (n=900): ```datasets/test``` (referred to as ```MedSafety-Eval``` in the paper)
 
- 74,374 harmful medical requests generated using Llama-3-8B-Instruct (forcing responses to begin with "Sure"). These were generated after completion of the paper. We include these harmful medical requests as an additional resource.
  - ```datasets/med_harm_llama3```
 

### Updates
- V1 [June 2024]. This version of the dataset consisted of a training set with 900 medical safety demonstrations and a test set with 900 harmful medical requests.
- V2 [October 2024]. Current version described above. Added corresponding safe responses to the test set. Also added harmful medical requests generated using Llama-3-8B-Instruct.


## Code 

### Dataset generation
- Generate harmful medical requests using Llama2-7b-chat, jailbreaking (```exps/adv_attack/gcg.py```)
- Generate harmful medical requests using Llama-3-8B-Instruct, forcing responses to begin with "Sure" (```exps/adv_attack/generate_prompts.py```)
- Generate safe responses to harmful medical requests (```exps/exp03_generate_safe_responses.py```)

### Experiments
- Prompt LLMs using harmful requests (general and medical) (```exps/exp01_prompt_models.py```)
- Evaluate responses of LLMs to harmful requests (```exps/exp02_eval_responses.py```)
- Fine-tune medical LLMs using demonstrations of medical safety, general safety, or both (```training/finetuning.py```)
  - Medical safety: 900 demonstrations from MedSafetyBench training set (```datasets/train-splits-used-for-ft/med_safety/ft_safety_med_n900.json```)
  - General safety: 900 demonstrations from [Bianchi et al. (2023)](https://arxiv.org/abs/2309.07875) (```datasets/train-splits-used-for-ft/gen_safety/ft_safety_gen_n900.json```)
  - Both safety: The 900x2 demonstrations pooled together (```datasets/train-splits-used-for-ft/both_safety/ft_safety_both_n1800.json```)
  - Data splits used for fine-tuning: ```datasets/train-splits-used-for-ft```
- Prompt fine-tuned medical LLMs using harmful requests (general and medical) (```exps/exp01_prompt_models.py```)
- Evaluate responses of fine-tuned medical LLMs to harmful requests (```exps/exp02_eval_responses.py```)
- Evaluate the medical performance of medical LLMs before and after fine-tuning (```meditron/evaluation/inference_pipeline.sh```)


## Citation

```
@article{han2024towards,
  title={MedSafetyBench: Evaluating and Improving the Medical Safety of Large Language Models},
  author={Han, Tessa and Kumar, Aounon and Agarwal, Chirag and Lakkaraju, Himabindu},
  journal={NeurIPS},
  year={2024}
}
```
