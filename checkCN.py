# for a formula: python getOS.py  --i SO2
# for a csv file conatining multiple formulas: python getOS.py  --f formulas.csv

import argparse
import json
import logging
import os
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
)
from transformers import BertTokenizerFast

import numpy as np

from pymatgen.io.cif import CifParser
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

import torch.nn.functional as F

import pandas as pd

def merge_os(osstr):
    #Sr(+2:1.00) Ti(+4:1.00) O(-2:1.00) O(-2:1.00) O(-2:1.00)
    items = osstr.split(" ")
    elementos={}
    for x in items:
        if x in elementos:
            elementos[x]+=1
        else:
            elementos[x]=1
    out=''
    for x in elementos:
        if elementos[x]==1:
            out+=x+" "
        else:
            e=x.split('(')[0]
            out+=f'{e}{elementos[x]}({"".join(x.split("(")[1:])} '
    return out.strip()

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
        default='./trained_models/ICSD_CN/',
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
        "--ignore_mismatched_sizes",
        action="store_true",
        default=True,
        help="ignore_mismatched_sizes set to True by default.",
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
    model.eval()
    if (args.i is not None) and (args.f is not None):    
        print("Please input a formula (using --i) or give the csv file with some formulas (using --f)")
        return 
    
    if args.i is not None:
        print("Input formula -------> ", args.i)
        comp = Composition(args.i)        
        comp_dict = comp.to_reduced_dict

        input_seq = ""
        for ele in comp_dict.keys():
            for count in range(int(comp_dict[ele])):
                input_seq = input_seq + ele + " "
                
    
        tokenized_inputs = torch.tensor(tokenizer.encode(input_seq, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        
        outputs = model(tokenized_inputs)
        predictions = outputs.logits.argmax(dim=-1)
        probs = torch.max(F.softmax(outputs[0], dim=-1), dim=-1)
        
    
        true_pred = predictions[0][1:-1]
        true_probs = probs[0][0][1:-1]
        
        
        tmp = input_seq.split()
        outstr = ''
        count_cn = 0
        for i, ele in enumerate(tmp):
            outstr += ele
            true_os = true_pred[i].item() - 5
            count_cn += true_os
            if true_os>0:
                true_os='+'+str(true_os)
            prob = true_probs[i].item()
            
            outstr = outstr +f'({true_os}:{prob:.2f}) '
        outstr=merge_os(outstr) 

        print("Predicted Oxidation States:\n ", outstr)

        if count_cn == 0:
            print("Charge Neutral? Yes")
        else:
            print("Charge Neutral? No")
    
    if args.f is not None:
        print("Input file ------->", args.f)
        df = pd.read_csv(args.f, header=None)
        formulas = df[0]
        
        all_outs = []
        for item in formulas:       
            comp = Composition(item)        
            comp_dict = comp.to_reduced_dict
            
            input_seq = ""
            for ele in comp_dict.keys():
                for count in range(int(comp_dict[ele])):
                    input_seq = input_seq + ele + " "
                    
            tokenized_inputs = torch.tensor(tokenizer.encode(input_seq, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
 
            
            outputs = model(tokenized_inputs)
            predictions = outputs.logits.argmax(dim=-1)
            probs = torch.max(F.softmax(outputs[0], dim=-1), dim=-1)
            
        
            true_pred = predictions[0][1:-1]
            true_probs = probs[0][0][1:-1]
            
            tmp = input_seq.split()

            cn_count = 0
            outstr = ''
            for i, ele in enumerate(tmp):
                outstr += ele
                true_os = true_pred[i].item() - 5

                cn_count += true_os

                if true_os>0:
                    true_os='+'+str(true_os)
                prob = true_probs[i].item()
                
                outstr = outstr +f'({true_os}:{prob:.2f}) '
            outstr=merge_os(outstr)  
            
            if cn_count == 0:
                all_outs.append([item, outstr, "True"])
            else:
                all_outs.append([item, outstr, "False"])

        out_df = pd.DataFrame(all_outs)
        out_df.columns = ["formula", "predicted OS", "charge neutrality"]

        #add _OS to the input filename as output file
        outfile='.'.join(args.f.split(".")[0:-1])+"_OS_CN."+args.f.split(".")[-1]

        out_df.to_csv(outfile, index=None)
        print("Output file ------>",f"{outfile} <-- check for the predicted oxidation states")

        
if __name__ == "__main__":
    main()
