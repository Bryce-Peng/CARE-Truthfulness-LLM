# CARE: Context-Aligned Representation Editing and Tuning with Procedural Priors for Truthfulness Improvement in Large Language Models

This repository contains the official implementation of the paper:

> **Context-Aligned Representation Editing and Tuning with Procedural Priors for Truthfulness Improvement in Large Language Models**  
> Xianghui Peng, Guozheng Rao, Jiahuan Zhang, Li Zhang  
> *Expert Systems With Applications*, 2026  
> DOI: [10.1016/j.eswa.2026.133433](https://doi.org/10.1016/j.eswa.2026.133433)

If you find this work useful, please cite:

```bibtex
@article{peng2026care,
  title={Context-Aligned Representation Editing and Tuning with Procedural Priors for Truthfulness Improvement in Large Language Models},
  author={Peng, Xianghui and Rao, Guozheng and Zhang, Jiahuan and Zhang, Li},
  journal={Expert Systems With Applications},
  year={2026},
  doi={10.1016/j.eswa.2026.133433}
}
```

## Quick-Start Guide

Below are the minimal commands to reproduce the CARE environment and run the official Llama-2 7B Chat experiments.  

------------------------------------------------
### 1. Environment Preparation

```bash
# 1.1 Create a clean Python 3.12 conda env
conda create -n care python=3.12 -y

# 1.2 Activate it
conda activate care

# 1.3 Install project dependencies
bash env.sh
```

------------------------------------------------
### 2. Run CARE Evaluation

We provide two scripts that reproduce Table-1 in the paper.

```bash
# 3.1 Enter the scripts directory
cd scripts

# 3.2 CARE-TF (Token-level Fine-grained)
bash llama2_chat_7B_CARETF.sh

# 3.3 CARE-PO (Position-level)
bash llama2_chat_7B_CAREPO.sh
```

For any other bug please open an issue and attach the full error traceback.