# DeepFail

Failure Analysis of Deepfake Image Detection In-the-Wild

## ⚙️ Usage

### Installation

**Clone the Repo**  
Clone the repo using git or download and extract the zip file.
```bash
git clone https://github.com/kodamaChameleon/DeepFail.git
cd DeepFail
```

**Environment**  
Conda is the preferred environment for this project.
```bash
conda create -n DeepFail python=3.13
conda activate DeepFail
pip install -r requirements.txt
```

Copy .env_example to .env and update required variables.
```bash
cp .env_example .env
```

### Data Processing
**Download**  
Download the required datasets using.
```
python3 run.py --download
```
> [!CAUTION]
> Make sure you have **plenty** of disk space before running this command. These are **big** datasets.  
> - FFHQ (95.77GB)
> - SFHQ-T2I (24.5GB)
> - TPDNE (4.77GB)

## ✨ Acknowledgements

This project was made possible by:

**Universal Fake Detect**
```bibtex
@inproceedings{ojha2023fakedetect,
      title={Towards Universal Fake Image Detectors that Generalize Across Generative Models}, 
      author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
      booktitle={CVPR},
      year={2023},
}
```

**Simple Preserved and Augmented FEatures (SAFE)**
```bibtex
@inproceedings{li2025improving,
  title={Improving synthetic image detection towards generalization: An image transformation perspective},
  author={Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Feng, Fuli},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  pages={2405--2414},
  year={2025}
}
```

**Coding assistance from ChatGPT by OpenAI**
```bibtex
@software{openai2026chatgpt,
  author={OpenAI},
  title={ChatGPT (GPT-5.2)},
  year={2026},
  url={https://chat.openai.com},
  note={Large language model used for code assistance}
}
```
