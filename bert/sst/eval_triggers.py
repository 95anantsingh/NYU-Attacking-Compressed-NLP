import os
import torch
device='cuda' if torch.cuda.is_available() else 'cpu' 
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric
import pandas as pd
import utils_uat_sst as utils
import argparse
import pickle
import numpy as np
from copy import deepcopy
import torch.nn as nn
import copy
parser = argparse.ArgumentParser()
parser.add_argument("-t", help = "0/1")
parser.add_argument("-attacker", help = "value: main/quantized/distilled")
parser.add_argument("-attacked", help = "value: main/quantized/distilled")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

val_dataset = load_dataset('glue', 'sst2', split='validation')
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.remove_columns(['label'])

tokenizer = BertTokenizer.from_pretrained('./bert_main_sst_model')

def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, len(num_layers_to_keep)):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList

    return copyOfModel

if args.attacked=='main':
    tokenizer = BertTokenizer.from_pretrained('./bert_'+args.attacked+'_sst_model')
    model = BertForSequenceClassification.from_pretrained('./bert_'+args.attacked+'_sst_model', return_dict=True)
elif args.attacked == 'distil':
    tokenizer = DistilBertTokenizer.from_pretrained('./bert_'+args.attacked+'_sst_model')
    model = DistilBertForSequenceClassification.from_pretrained('./bert_'+args.attacked+'_sst_model', return_dict=True)
elif args.attacked == 'distilled':
    tokenizer = BertTokenizer.from_pretrained('./bert_'+args.attacked+'_sst_model')
    model = BertForSequenceClassification.from_pretrained('./bert_main_sst_model', return_dict=True)    
    student =  deleteEncodingLayers(model, [1,5,8,12])
    m = torch.load('./bert_'+args.attacked+'_sst_model/pytorch_model.bin')
    student.load_state_dict(m)
elif args.attacked == 'pruned':
    tokenizer = BertTokenizer.from_pretrained('./bert_main_sst_model')
    model = BertForSequenceClassification.from_pretrained('./bert_'+args.attacked+'_sst_model', return_dict=True)

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
    f = open('uat_'+str(args.attacker)+'_'+str(args.t)+'.txt', 'r')
    triggs = f.read()
    f.close()
    trigg_list = triggs.split('|')
    trigg_list = [' '.join(x.split(":")[0].split(',')) for x in trigg_list]
    print(trigg_list)
    final_accs = []
    for k in trigg_list:
        final_acc= validate_trigger_acc(sents, k, dataset_label_filter)
        final_accs.append(final_acc)
    print(final_accs)
    acc_without_trigg = final_accs.pop()
    
    avg_trigger_acc = np.array(final_accs).mean()
    
    f = open('eval_uat_'+str(args.attacker)+'_'+str(args.attacked)+'_'+str(args.t)+'.txt', 'w')
    
    f.write(str(acc_without_trigg)+","+ str(avg_trigger_acc))
    f.close()
    
if __name__ == '__main__':
    main()