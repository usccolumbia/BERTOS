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
pip install -r requirements.txt
```  
## Datasets  
Our training process is carried out on our [BERTOS_datasets](https://figshare.com/account/projects/153468/articles/21554817). After downloading and extracting the data under `dataset` folder, you will find the following three folders `ICSD`, `ICSD_CN`, `ICSD_CN_oxide`, and `ICSD_oxide`.

## Usage
### A Quick Run
Quickly run the script to train a BERTOS.
```
bash train_BERTOS.sh
```  
### Training
An example is to train a BERTOS model on the ICSD dataset.  
```
python train_BERTOS.py  --config_name $CONFIG NAME$  --dataset_name $DATASET FILE$   --max_length $MAX LENGTH$  --per_device_train_batch_size $BATCH SIZE$  --learning_rate $LEARNING RATE$  --num_train_epochs $EPOCHS$    --output_dir $OUTPUT DIRECTORY$
```
 If you want to change the dataset, you can use different dataset file, like `materials_icsd.py`, `materials_icsdcn.py`, `materials_icsdcno.py`, and `materials_icsdo.py`. And you can also follow the intructions of [huggingface]() to build you own custom dataset.

### Predict
Run `getOS.py` file to get predicted oxidation states for input formulas or input csv file containing multiple formula.
```
python getOS.py --i $FORMULAS$
python getOS.py --f $FORMULAS CSV FILE$
```

## Pretrained Models
Our trained models can be downloaded from [BERTOS_models](https://figshare.com/account/projects/153468/articles/21554823), and you can use it as a test or predict model.



## Acknowledgement
```
@article{wolf2019huggingface,  
  title={Huggingface's transformers: State-of-the-art natural language processing},  
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and Funtowicz, Morgan and others},  
  journal={arXiv preprint arXiv:1910.03771},  
  year={2019}  
}
```
