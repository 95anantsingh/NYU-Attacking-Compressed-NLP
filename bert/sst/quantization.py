import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset,load_metric
import pandas as pd
import argparse
# from pytorch_quantization import quant_modules
# quant_modules.initialize()
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-q", help = "value: 8")

args = parser.parse_args()
device = 'cpu'

MAX_LENGTH = 128

tokenizer = BertTokenizer.from_pretrained('./bert_main_sst_model')
model = BertForSequenceClassification.from_pretrained('./bert_main_sst_model', return_dict=True)
model.eval().to(device)
print(model)
val_dataset = load_dataset('glue', 'sst2', split='validation')
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.remove_columns(['label'])
val_dataset = val_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

def main():
    if args.q == '8':
        dt = torch.qint8
    elif args.q == '16':
        dt = torch.float16
    
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=dt
    )
    model_size = print_size_of_model(model,"fp32")
    if "8" in args.q:
        quantized_model_size = print_size_of_model(quantized_model,"int8")
    else:
        quantized_model_size = print_size_of_model(quantized_model,"float16")
        
    print("{0:.2f} times smaller".format(model_size/quantized_model_size))
    
    input_ids =  torch.tensor(val_dataset['input_ids'][:1])
    token_type_ids = torch.tensor(val_dataset['token_type_ids'][:1])
    attention_mask = torch.tensor(val_dataset['attention_mask'][:1])
    dummy_input = [input_ids, attention_mask, token_type_ids]
    
    traced_model = torch.jit.trace(quantized_model, dummy_input, strict=False)
    model_path = 'bert_quantized'+str(args.q)+'_sst_model/'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    torch.jit.save(traced_model, model_path+"model.pt")
    
if __name__ == '__main__':
    main()