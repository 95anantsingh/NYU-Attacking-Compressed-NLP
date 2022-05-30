import os
import torch
device='cuda' if torch.cuda.is_available() else 'cpu' 
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric
import pandas as pd
import utils_uat_sst as utils
import argparse
import pickle
import numpy as np
import random
from copy import deepcopy
import  copy
import torch.nn as nn
import heapq
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument("-t", help = "0/1")
parser.add_argument("-v", help = "value: main/quantized/distilled")

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# train_dataset = load_dataset('glue', 'sst2', split='train')
val_dataset = load_dataset('glue', 'sst2', split='validation')

# train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

val_dataset = val_dataset.remove_columns(['label'])
# train_dataset = train_dataset.remove_columns(['label'])

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

if args.v=='main':
    tokenizer = BertTokenizer.from_pretrained('./bert_main_sst_model')
    model = BertForSequenceClassification.from_pretrained('./bert_'+args.v+'_sst_model', return_dict=True)
elif args.v == 'distil':
    tokenizer = DistilBertTokenizer.from_pretrained('./bert_'+args.v+'_sst_model')
    model = DistilBertForSequenceClassification.from_pretrained('./bert_'+args.v+'_sst_model', return_dict=True)
elif args.v == 'distilled':
    tokenizer = BertTokenizer.from_pretrained('./bert_main_sst_model')
    stud = BertForSequenceClassification.from_pretrained('./bert_main_sst_model', return_dict=True)    
    model =  deleteEncodingLayers(stud, [1,5,8,12])
    m = torch.load('./bert_'+args.v+'_sst_model/pytorch_model.bin')
    model.load_state_dict(m)
elif args.v == 'pruned':
    tokenizer = BertTokenizer.from_pretrained('./bert_main_sst_model')
    from nn_pruning.patch_coordinator import SparseTrainingArguments
    from nn_pruning.patch_coordinator import ModelPatchingCoordinator

    sparse_args = SparseTrainingArguments(
        dense_pruning_method="topK:1d_alt", 
        attention_pruning_method= "topK", 
        initial_threshold= 1.0, 
        final_threshold= 0.5, 
        initial_warmup= 1,
        final_warmup= 3,
        attention_block_rows=32,
        attention_block_cols=32,
        attention_output_with_dense= 0
    )
    mpc = ModelPatchingCoordinator(
        sparse_args=sparse_args, 
        device=device, 
        cache_dir="checkpoints", 
        logit_names="logits", 
        teacher_constructor=None)
    model = BertForSequenceClassification.from_pretrained('./bert_main_sst_model', return_dict=True) 
    mpc.patch_model(model)

    m = torch.load('./bert_'+args.v+'_sst_model/pytorch_model.bin')
    model.load_state_dict(m)
    mpc.compile_model(model)
    
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
    all_trigs = {}
    ws = ['a','an','the','to','in','with','are','or','so']
    for k in range(10):
        trigger = ' '.join(random.sample(ws, 3))
        print("before:", trigger, validate_trigger_acc(sents, trigger, dataset_label_filter))
        b =32
        for ind in range(len(sents)//b -1):
            data = {'sentence1':sents[ind*b: ind*b+b], 'label':[dataset_label_filter]*b}
            averaged_grad_sent = utils.get_average_grad(model, data, trigger, len(trigger.split()), tokenizer)

            cand_trigger_token_ids_sent = utils.hotflip_attack(averaged_grad_sent,
                                                            embedding_weight,
                                                            num_candidates=40,increase_loss = True)
            trigger_token_ids_sent, loss_sent = utils.get_best_candidates(model,
                                                                  data, trigger.split(),
                                                                  cand_trigger_token_ids_sent, tokenizer,
                                                                  beam_size = 1, increase_loss = True)
            trigger = ' '.join(trigger_token_ids_sent)

            final_acc = validate_trigger_acc(sents, trigger, dataset_label_filter)
            all_trigs[trigger]=final_acc
    
    res = sorted(all_trigs.items(), key=itemgetter(1), reverse=False)[:10]
    f = open('uat_'+str(args.v)+'_'+str(args.t)+'.txt', 'w')
    res_str = ''
    for k,v in res:
        res_str+=','.join(k.split()) + ':'+ str(v) + '|'
    f.write(res_str)
    f.close()
if __name__ == '__main__':
    main()