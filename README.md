# DeepFail

Failure Analysis of Deepfake Image Detection In-the-Wild

# Setup

## Installation

**Clone the Repo**  
Clone the repo using git or download and extract the zip file.
```
git clone https://github.com/kodamaChameleon/DeepFail.git
cd DeepFail
```

**Environment**  
Conda is the preferred environment for this project.
```
conda create -n DeepFail python=3.13
conda activate DeepFail
pip install -r requirements.txt
```

Copy .env_example to .env and update required variables.
```
cp .env_example .env
```

## Data Processing
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