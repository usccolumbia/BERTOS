# python getOS.py  --i SO2
# python getOS.py  --f formulas.csv

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import ClassLabel, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from transformers import BertTokenizerFast

import numpy as np


from pymatgen.io.cif import CifParser
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

import torch.nn.functional as F

import pandas as pd


#import pymatgen

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test trained model."
    )
    parser.add_argument(
        "--i",
        type=str,
        default=None,
        help="Input formula",
    )
    
    parser.add_argument(
        "--f",
        type=str,
        default=None,
        help="Input file",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='./trained_model/icsdcn/',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
  
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='./tokenizer',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer_name_or_path = args.tokenizer_name
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name_or_path, do_lower_case=False)
    
    padding = "max_length" if args.pad_to_max_length else False
    
    # Load model config
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=14)
   
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
        
    if args.i is not None:
        print("Input formula -------> ", args.i)
        comp = Composition(args.i)        
        comp_dict = comp.to_reduced_dict
        #print(comp_dict)
        
        input_seq = ""
        for ele in comp_dict.keys():
            for count in range(int(comp_dict[ele])):
                input_seq = input_seq + ele + " "
                
        #print(input_seq)
        
    
        tokenized_inputs = torch.tensor(tokenizer.encode(input_seq, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        #print('input: ', tokenized_inputs)    
        
       
        outputs = model(tokenized_inputs)
        predictions = outputs.logits.argmax(dim=-1)
        probs = torch.max(F.softmax(outputs[0], dim=-1), dim=-1)
        #print(probs[0][0][1])
        
    
        true_pred = predictions[0][1:-1]
        true_probs = probs[0][0][1:-1]
        #print(true_pred)
        
        tmp = input_seq.split()
        outstr = ''
        for i, ele in enumerate(tmp):
            outstr += ele
            true_os = true_pred[i].item() - 5
            prob = true_probs[i].item()
            outstr = outstr + '(' + str(true_os) + '  ' + str(prob) + ') '
         
        print("Get Oxidation State: ", outstr)
    
    if args.f is not None:
        print("Input file ------->", args.f)
        df = pd.read_csv(args.f, header=None)
        formulas = df[0]
        
        all_outs = []
        for item in formulas:       
            comp = Composition(item)        
            comp_dict = comp.to_reduced_dict
            #print(comp_dict)
            
            input_seq = ""
            for ele in comp_dict.keys():
                for count in range(int(comp_dict[ele])):
                    input_seq = input_seq + ele + " "
                    
            #print(input_seq)
            tokenized_inputs = torch.tensor(tokenizer.encode(input_seq, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            #print('input: ', tokenized_inputs)    
            
            outputs = model(tokenized_inputs)
            predictions = outputs.logits.argmax(dim=-1)
            probs = torch.max(F.softmax(outputs[0], dim=-1), dim=-1)
            #print(probs[0][0][1])
            
        
            true_pred = predictions[0][1:-1]
            true_probs = probs[0][0][1:-1]
            #print(true_pred)
            
            tmp = input_seq.split()
            outstr = ''
            for i, ele in enumerate(tmp):
                outstr += ele
                true_os = true_pred[i].item() - 5
                prob = true_probs[i].item()
                outstr = outstr + '(' + str(true_os) + '  ' + str(prob) + ') '
             
            #print("Get Oxidation State: ", outstr)
            
            all_outs.append(outstr)
            
            out_df = pd.DataFrame(all_outs)
            out_df.to_csv('predictedOS.csv', header=None, index=None)
    

    
        
if __name__ == "__main__":
    main()