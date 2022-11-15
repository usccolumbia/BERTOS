# BERTOS
BERTOS: transformer for oxidation state prediction

## Table of Contents
- [Installations](##installations)

- [Datasets](##datasets)

- [Usage](##usage)

- [Pretrained Models](##pretrained-models)

- [Acknowledgement](##acknowledgement)

## Installations
1. PyTorch 
```
conda create -n bertos
conda activate bertos
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
2. Other packagess
```
pip install -r requirements.txt`
```

## Datasets
Our training process is carried out on our [BERTOS_datasets](https://figshare.com/account/projects/153468/articles/21554817). After downloading and extracting the data under `dataset` folder, you will find the following three folders `ICSD`, 'ICSD_CN', `ICSD_CN_oxide`, and `ICSD_oxide`.

## Usage
### A Quick Run
`bash train_BERTOS.sh`

### Training
An example is to train a BERTOS model on the ICSD dataset.
`python train_BERTOS.py  --config_name ./random_config/  --dataset_name materials_icsd.py   --max_length 100  --per_device_train_batch_size 256  --learning_rate 1e-3  --num_train_epochs 500    --output_dir ./icsd`

 If you want to change the dataset, you can change different dataset file, like `materials_icsdcn.py`, `materials_icsdcno.py`, and `materials_icsdo.py`. And you can also follow the intructions of [huggingface]() to build you own customized datase.
## Pretrained Models
Our trained models can be downloaded from [BERTOS_models](https://figshare.com/account/projects/153468/articles/21554823), and you can use it as a test or predict model.
