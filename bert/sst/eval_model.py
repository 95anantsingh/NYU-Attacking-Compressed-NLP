import os
import torch
device='cuda' if torch.cuda.is_available() else 'cpu' 
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric
import pandas as pd
import utils_uat_sst as utils
import argparse
import pickle
import numpy as np
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("-t", help = "0/1")
parser.add_argument("-v", help = "value: main/quantized/distilled")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

val_dataset = load_dataset('glue', 'sst2', split='validation')
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.remove_columns(['label'])

tokenizer = BertTokenizer.from_pretrained('./bert_main_sst_model')

if 'quantized' not in args.v:
    model = BertForSequenceClassification.from_pretrained('./bert_'+args.v+'_sst_model', return_dict=True)
else:
    model = torch.jit.load("bert_"+args.v+"_sst_model/model.pt")
    
model.eval().to(device)

utils.add_hooks(model)
embedding_weight = utils.get_embedding_weight(model) # also save the word embedding matrix

MAX_LENGTH = 128

def get_accuracy(orig, preds):
    new_preds= (np.argmax(preds, axis=1))
    return accuracy_score(orig, new_preds)

def validate_trigger_acc(sents, trigger, label):
    if trigger == None:
        sent = deepcopy(sents)
    else:    
        sent = [trigger +" "+x for x in sents]
    pred = []
    for inds in range(0,len(sent)-32,32):
        encoded_pair = tokenizer(sent[inds:inds+32], return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LENGTH).to(device)
        output_labels = model(**encoded_pair)
        pred.extend(output_labels['logits'].cpu().detach().numpy())
    acc = get_accuracy([label]*len(pred), pred)
    return acc

def main():
    dataset_label_filter = int(args.t)
    df_targeted = val_dataset.filter(lambda x: x['labels']==dataset_label_filter)
    sents = df_targeted['sentence']
    trigger = ''
    final_acc= validate_trigger_acc(sents, trigger, dataset_label_filter)
    print(final_acc)
if __name__ == '__main__':
    main()