# [CVPR 2026 Findings Workshop]Adapting Large VLMs with Iterative and Manual Instructions for Generative Low-light Enhancement

<p align="center">
Xiaoran Sun<sup>1,†</sup>, Liyan Wang<sup>1,†</sup>, Cong Wang<sup>2</sup>, Yeying Jin<sup>3</sup>, Kin-man Lam<sup>4</sup>, Zhixuan Su<sup>1,*</sup>, Yang Yang<sup>5</sup>, Jinshan Pan<sup>1</sup>
</p>

<p align="center">
<sup>1</sup>Dalian University of Technology &nbsp;&nbsp; <sup>2</sup>University of California, San Francisco &nbsp;&nbsp; <sup>3</sup>National University of Singapore &nbsp;&nbsp; <sup>4</sup>The Hong Kong Polytechnic University &nbsp;&nbsp; <sup>5</sup>Nanjing University of Science and Technology
</p>

## Dependencies and Installation

- Python 3.10
- PyTorch 2.0.1

### 1. Create Conda Environment
```
conda create --n vlm-imi python=3.10
conda activate vlm-imi
```
### 2. Install PyTorch
```
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
### 3. Install Dependencies
```
cd VLM-IMI
pip install -r requirements.txt
```
## Data Preparation
You can refer to the following links to download the datasets.
- LOL: [LOL](https://daooshee.github.io/BMVC2018website/)
- LSRW: [LSRW](https://github.com/JianghaiSCU/R2RNet)
- RAISE：[RAISE](https://loki.disi.unitn.it/RAISE/index.php)
- LOLI-Street：[LOLI-Street](https://github.com/tanvirnwu/TriFuse_ACCV_2024)

Then, you can use [LLaVA](https://github.com/haotian-liu/LLaVA) to generate instructions, the models can be downloaded from [Google Dirve](https://drive.google.com/file/d/11uH6y7jBKzj2s2fo7L8oJX-88xLiGHYb/view?usp=sharing).

```
cd datasets
python make_dataset.py
```

Then, put them in the following folder:

```
├── data
      ├── lowlight
           ├── low
           ├── high
           └── text
```

Finally, run:

```
python preprocess.py
```

## Training
You can download the required models from [T5](https://drive.google.com/file/d/1UI0f-riwlINe4-ZZpbZVbuFwX5hurXyG/view?usp=sharing) and [StableDiffusion](https://drive.google.com/file/d/1UgOzTYnvafgrxeCrzYoMZnfe-7aDubFK/view?usp=sharing).
Then, run:

```
CUDA_VISIBLE_DEVICES="0,1" accelerate launch train.py --enable_xformers_memory_efficient_attention --gradient_checkpointing
```

## Testing
```
python test.py
```
