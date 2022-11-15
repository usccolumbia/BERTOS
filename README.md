# BERTOS
BERTOS: transformer language model for oxidation state prediction

Machine Learning and Evolution Laboratory <br>
Department of computer science and Engineering <br>
University of South Carolina

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
if you only has CPU on your computer, try this:
```
pip install transformers[torch]
```

2. Other packagess
```
pip install -r requirements.txt
```  

## Datasets  
Our training process is carried out on our BERTOS datasets. After extracting the data under `datasets` folder, you will get the following four folders `ICSD`, `ICSD_CN`, `ICSD_CN_oxide`, and `ICSD_oxide`.

## Usage
### A Quick Run
Quickly run the script to train a BERTOS using the OS-ICSD-CN training set and save the model into the `./output_icsdcn` folder.
```
bash train_BERTOS.sh
```  
### Training
The command to to train a BERTOS model.  
```
python train_BERTOS.py  --config_name $CONFIG NAME$  --dataset_name $DATASET LOADER$   --max_length $MAX LENGTH$  --per_device_train_batch_size $BATCH SIZE$  --learning_rate $LEARNING RATE$  --num_train_epochs $EPOCHS$    --output_dir $OUTPUT DIRECTORY$
```
We use `ICSD_CN` dataset as an example:
```
python train_BERTOS.py  --config_name ./random_config   --dataset_name materials_icsd_cn.py   --max_length 100  --per_device_train_batch_size 256  --learning_rate 1e-3  --num_train_epochs 500    --output_dir ./output_icsdcn
```
 If you want to change the dataset, you can use different dataset file to replace `$DATASET LOADER$`, like `materials_icsd.py`, `materials_icsdcn.py`, `materials_icsdcno.py`, and `materials_icsdo.py`. And you can also follow the intructions of [huggingface]() to build you own custom dataset.

### Predict
Run `getOS.py` file to get predicted oxidation states for a input formula or input formulas.csv file containing multiple formulas.
```
python getOS.py --i Your_formula(e.g. SrTiO3)
python getOS.py --f formulas.csv --model_name_or_path ./trained_models/ICSD_CN
```

## Pretrained Models
Our trained models can be downloaded from figshare [BERTOS models](https://figshare.com/articles/online_resource/BERTOS_model/21554823), and you can use it as a test or prediction model.

## Acknowledgement
We use the transformer model as implmented in Huggingface.
```
@article{wolf2019huggingface,  
  title={Huggingface's transformers: State-of-the-art natural language processing},  
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and Funtowicz, Morgan and others},  
  journal={arXiv preprint arXiv:1910.03771},  
  year={2019}  
}
```

## Cite our work
```
Nihang Fu,†,§ Jeffrey Hu,†,§ Ying Feng,‡ Hanno zur Loye,¶ and Jianjun Hu, Composition based oxidation state prediction of
materials using deep learning. 2022.

```
