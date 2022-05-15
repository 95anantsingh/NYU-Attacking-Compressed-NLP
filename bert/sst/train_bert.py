import os
import torch
device='cuda' if torch.cuda.is_available() else 'cpu' 
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric

train_dataset = load_dataset('glue', 'sst2', split='train')
val_dataset = load_dataset('glue', 'sst2', split='validation')
test_dataset = load_dataset('glue', 'sst2', split='test')

train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

val_dataset = val_dataset.remove_columns(['label'])
test_dataset = test_dataset.remove_columns(['label'])
train_dataset = train_dataset.remove_columns(['label'])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LENGTH = 128
train_dataset = train_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
val_dataset = val_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',      
    overwrite_output_dir = 'True', # output directory
    learning_rate=1e-5,
    num_train_epochs=2,
    warmup_steps = 600,
    weight_decay=0.01,
    per_device_train_batch_size=32,               # batch size per device during training
    per_device_eval_batch_size=32,                # batch size for evaluation
    logging_dir='./logs',            
    logging_steps=100,
    do_train=True,
    do_eval=True,
    no_cuda=False,
    load_best_model_at_end=False,
    save_total_limit = 1,
    save_strategy = "no",
    seed=42,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics
)





train_out = trainer.train()
model.save_pretrained('./bert_main_sst_model/')
tokenizer.save_pretrained('./bert_main_sst_model/')