import sys
import os.path
import torch
import torch.optim as optim
import torch.nn as nn
import random
random.seed(42)
from reader_new import StanfordSentimentTreeBankDatasetReader_NEW
# from allennlp.data.iterators import BucketIterator, BasicIterator
# from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.data_loaders import SimpleDataLoader
from operator import itemgetter
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
import utils_sst_uat2 as utils
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-t", help = "0/1")
parser.add_argument("-v", help = "value: main/quantized/distilled")
args = parser.parse_args()

class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

def main():
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer

    reader = StanfordSentimentTreeBankDatasetReader_NEW(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
    dev_data = reader.read('./data/dev.txt')
    
    vocab_path = "./lstm_main_sst_model/w2v_vocab"
    vocab = Vocabulary.from_files(vocab_path)

    embedding_path = "./data/crawl-300d-2M.vec.zip"
    weight = _read_pretrained_embeddings_file(embedding_path,
                                              embedding_dim=300,
                                              vocab=vocab,
                                              namespace="tokens")
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=300,
                                weight=weight,
                                trainable=False)
    word_embedding_dim = 300

    # Initialize model, cuda(), and optimizer
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    if 'quantized' not in args.v:
        model_path = "./lstm_"+args.v+"_sst_model/w2v_" + "model.th"
        model = LstmClassifier(word_embeddings, encoder, vocab)

        with open(model_path, 'rb') as f:
            model.cuda()
            model.load_state_dict(torch.load(f))
        device = 'cuda'
    else:
        model_path = "./lstm_main_sst_model/w2v_model.th"
        model = LstmClassifier(word_embeddings, encoder, vocab)
        
        with open(model_path, 'rb') as f:
            model.cuda()
            model.load_state_dict(torch.load(f))
            
        model.cpu()
        model = torch.quantization.quantize_dynamic(
            model,{nn.LSTM}, dtype=torch.qint8
        )
        device = 'cpu'
        print("Quantized!")
        
    model.train().to(device)
    print(model)
    
    # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
    # We use the gradient later in the attack.
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # also save the word embedding matrix

    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    dataset_label_filter = args.t
    targeted_dev_data = []
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)
            
    dl = SimpleDataLoader(targeted_dev_data, batch_size=128, shuffle=True)
    dl.index_with(vocab)
    # get accuracy before adding triggers
    utils.get_accuracy(model, dl, vocab, trigger_token_ids=None, device = device)
    model.train() # rnn cannot do backwards in train mode

    # initialize triggers which are concatenated to the input
    num_trigger_tokens = 3
    
    n_eps = 10
    # sample batches, update the triggers, and repeat
    all_trigs = {}
    ws = ['a','an','the','to','in','with','are','or','so']
    for k in range(n_eps):
        trigger = random.sample(ws, num_trigger_tokens)
        trigger_token_ids = [vocab.get_token_index(x) for x in trigger]
    
        for batch in dl:
            # get accuracy with current triggers
            utils.get_accuracy(model, dl, vocab, trigger_token_ids, device = device)
            model.train() # rnn cannot do backwards in train mode

            # get gradient w.r.t. trigger embeddings for current batch
            averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids)

            # pass the gradients to a particular attack to generate token candidates for each token.
            cand_trigger_token_ids = utils.hotflip_attack(averaged_grad,
                                                            embedding_weight,
                                                            trigger_token_ids,
                                                            num_candidates=10,
                                                            increase_loss=True)

            # Tries all of the candidates and returns the trigger sequence with highest loss.
            trigger_token_ids = utils.get_best_candidates(model,
                                                          batch,
                                                          trigger_token_ids,
                                                          cand_trigger_token_ids)
            
            final_acc = utils.get_accuracy(model, dl, vocab, trigger_token_ids, device = device)
            trigger = ','.join([vocab.get_token_from_index(x) for x in trigger_token_ids])
            all_trigs[trigger]=final_acc

    # print accuracy after adding triggers
    res = sorted(all_trigs.items(), key=itemgetter(1), reverse=False)[:10]
    f = open('uat_'+str(args.v)+'_'+str(args.t)+'.txt', 'w')
    res_str = ''
    for k,v in res:
        res_str+=','.join(k.split()) + ':'+ str(v) + '|'
    f.write(res_str)
    f.close()
    
if __name__ == '__main__':
    main()