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
pip install -r requirements.txt
```
