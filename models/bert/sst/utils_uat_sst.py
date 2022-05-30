from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import torch.optim as optim
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score
import random
import pandas as pd
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
extracted_grads = []

def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30522: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()

# add hooks for embeddings
def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30522: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_full_backward_hook(extract_grad_hook)

def hotflip_attack(averaged_grad, embedding_matrix,
                   increase_loss=False, num_candidates=1):
    
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))        
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()

def evaluate_batch(model, data, trig, tokenizer, MAX_LENGTH = 128):
    sent1 = [trig +" "+x for x in data['sentence1']]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoded_pair = tokenizer(sent1, return_tensors='pt', truncation=True, padding='max_length',max_length=MAX_LENGTH).to(device) 
    label = data['label']
    output_label = model(**encoded_pair)
    pred = output_label['logits']
    
    logit_label = torch.zeros(len(sent1), 2).to(device)
    logit_label[:, label]=1
    
    loss = torch.nn.BCEWithLogitsLoss()(pred, logit_label)
    output_dict={'logits': pred, 'loss': loss}
    return output_dict

def get_average_grad(model, batch, trigger_token_ids, trig_len, tokenizer):
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss = evaluate_batch(model, batch, trigger_token_ids, tokenizer)['loss']
    loss.backward()
    grads = extracted_grads[0].cpu()
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad_sent = averaged_grad[1:1+trig_len]
    return averaged_grad_sent

def get_best_candidates(model, x_batch, trigger_ids, candidates, tokenizer, beam_size=1,\
                        increase_loss=False):
    if increase_loss:
        beamer = heapq.nlargest
    else:
        beamer = heapq.nsmallest
    
    loss_per_candidate = get_loss_per_candidate(0, model, x_batch, trigger_ids, candidates, tokenizer)
    rand_ind = random.randint(0,beam_size-1)
    top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    for idx in range(1, len(trigger_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates: 
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, x_batch, cand, candidates, tokenizer))
        top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    if increase_loss:
        output = max(top_candidates, key=itemgetter(1))
    else:
        output = min(top_candidates, key=itemgetter(1))
    return output[0], output[1] 

def get_loss_per_candidate(index, model, batch, trigger, cand_trigger_token_ids, tokenizer):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    loss_per_candidate = []
    curr_loss = evaluate_batch(model, batch, ' '.join(trigger), tokenizer)['loss'].cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger), curr_loss))
    
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger)
#         print(tokenizer.decode(int(cand_trigger_token_ids[index][cand_id])))
        
        trigger_token_ids_one_replaced[index] = tokenizer.convert_ids_to_tokens(int(cand_trigger_token_ids[index][cand_id]))
        loss = evaluate_batch(model, batch, ' '.join(trigger_token_ids_one_replaced), tokenizer)['loss'].cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate