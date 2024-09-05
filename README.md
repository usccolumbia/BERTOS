# BERTOS
BERTOS: transformer language model for oxidation state prediction

Citation: Fu, Nihang, Jeffrey Hu, Ying Feng, Gregory Morrison, Hans‐Conrad zur Loye, and Jianjun Hu. "Composition Based Oxidation State Prediction of Materials Using Deep Learning Language Models." Advanced Science (2023): 2301011. [Link](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202301011)


Nihang Fu, Jeffrey Hu, Ying Feng, Jianjun Hu* <br>

Machine Learning and Evolution Laboratory <br>
Department of computer science and Engineering <br>
University of South Carolina

[Online Toolbox]([https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202301011](http://www.materialsatlas.org/bertos)

## Table of Contents
- [Installations](#Installations)

- [Datasets](#Datasets)

- [Usage](#Usage)

- [Pretrained Models](#Pretrained-models)

- [Performance](#Performance)

- [Acknowledgement](#Acknowledgement)

## Installations

0. Set up a virtual environment
```
conda create -n bertos
conda activate bertos
```

1. PyTorch and transformers for computers with Nvidia GPU.
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge transformers
```
If you only have CPU on your computer, try this:
```
pip install transformers[torch]
```
If you are using Mac M1 chip computer, following [this tutorial](https://jamescalam.medium.com/hugging-face-and-sentence-transformers-on-m1-macs-4b12e40c21ce) or [this one](https://towardsdatascience.com/hugging-face-transformers-on-apple-m1-26f0705874d7) to install pytorch and transformers.

2. Other packagess
```
pip install -r requirements.txt
```  

## Datasets  
Our training process is carried out on our BERTOS datasets. After extracting the data under `datasets` folder, you will get the following four folders `ICSD`, `ICSD_CN`, `ICSD_CN_oxide`, and `ICSD_oxide`.

## Usage
### A Quick Run
Quickly run the script to train a BERTOS using the OS-ICSD-CN training set and save the model into the `./model_icsdcn` folder.
```
bash train_BERTOS.sh
```  
### Training
The command is to train a BERTOS model.  
```
python train_BERTOS.py  --config_name $CONFIG_NAME$  --dataset_name $DATASET_LOADER$   --max_length $MAX_LENGTH$  --per_device_train_batch_size $BATCH_ SIZE$  --learning_rate $LEARNING_RATE$  --num_train_epochs $EPOCHS$    --output_dir $MODEL_OUTPUT_DIRECTORY$
```
We use `ICSD_CN` dataset as an example:
```
python train_BERTOS.py  --config_name ./random_config   --dataset_name materials_icsd_cn.py   --max_length 100  --per_device_train_batch_size 256  --learning_rate 1e-3  --num_train_epochs 500    --output_dir ./model_icsdcn
```
 If you want to change the dataset, you can use a different dataset file to replace `$DATASET_LOADER$`, like `materials_icsd.py`, `materials_icsdcn.py`, `materials_icsdcno.py`, and `materials_icsdo.py`. And you can also follow the intructions of [huggingface]() to build your own custom dataset.

### Predict
Run `getOS.py` file to get predicted oxidation states for an input formula or input formulas.csv file containing multiple formulas. <br>
Using your model:
```
python getOS.py --i SrTiO3 --model_name_or_path ./model_icsdcn
python getOS.py --f formulas.csv --model_name_or_path ./model_icsdcn

```
Using pretrained model:
```
python getOS.py --i SrTiO3 --model_name_or_path ./trained_models/ICSD_CN
python getOS.py --f formulas.csv --model_name_or_path ./trained_models/ICSD_CN
```

## Pretrained Models
Our trained models can be downloaded from figshare [BERTOS models](https://figshare.com/articles/online_resource/BERTOS_model/21554823), and you can use it as a test or prediction model.


## Performance

![Performance](performances.png)
Removing `OS`, the datasets under `datasets` folder correspond to the datasets in the figure.

## Acknowledgement
We use the transformer model as implemented in Huggingface.
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
Fu, Nihang, Jeffrey Hu, Ying Feng, Gregory Morrison, Hans‐Conrad zur Loye, and Jianjun Hu. "Composition Based Oxidation State Prediction of Materials Using Deep Learning Language Models." Advanced Science (2023): 2301011. [PDF](https://arxiv.org/pdf/2211.15895)

```

# Contact
If you have any problem using BERTOS, feel free to contact via [funihang@gmail.com](mailto:funihang@gmail.com).
