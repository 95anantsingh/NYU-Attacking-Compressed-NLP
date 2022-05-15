import sys
import os.path
import torch
import torch.optim as optim
from reader_new import StanfordSentimentTreeBankDatasetReader_NEW
# from allennlp.data.iterators import BucketIterator, BasicIterator
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
from allennlp.data.data_loaders import SimpleDataLoader
import torch.nn as nn
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-t", help = "0/1")
parser.add_argument("-attacker", help = "value: main/quantized/distilled")
parser.add_argument("-attacked", help = "value: main/quantized/distilled")
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

EMBEDDING_TYPE = "w2v" # what type of word embeddings to use

def main():
    # load the binary SST dataset.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader_NEW(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
    dev_data = reader.read('./data/dev.txt')
    
    vocab_path = "./lstm_main_sst_model/" + EMBEDDING_TYPE + "_" + "vocab"

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

    
    model_path = "./lstm_"+args.attacked+"_sst_model/w2v_model.th"

    model = LstmClassifier(word_embeddings, encoder, vocab)

    with open(model_path, 'rb') as f:
        model.cuda()
        model.load_state_dict(torch.load(f))

    model.to(device) # rnn cannot do backwards in train mode
    
    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    dataset_label_filter = args.t
    targeted_dev_data = []
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)
    
    dl = SimpleDataLoader(targeted_dev_data, batch_size=128, shuffle=True)
    dl.index_with(vocab)
    
    f = open('uat_'+str(args.attacker)+'_'+str(args.t)+'.txt', 'r')
    triggs = f.read()
    f.close()
    trigg_list = triggs.split('|')
    trigg_list = [x.split(":")[0].split(',') for x in trigg_list][:-1]
    print(trigg_list)
    final_accs = []
    for k in trigg_list:
        trig = [vocab.get_token_index(x) for x in k]
        final_acc= utils.get_accuracy(model, dl, vocab, trigger_token_ids=trig, device=device)
        final_accs.append(final_acc)
    print(final_accs)
    acc_without_trigg = utils.get_accuracy(model, dl, vocab, trigger_token_ids=None, device=device)
    
    avg_trigger_acc = np.array(final_accs).mean()
    
    f = open('eval_uat_'+str(args.attacker)+'_'+str(args.attacked)+'_'+str(args.t)+'.txt', 'w')
    
    f.write(str(acc_without_trigg)+","+ str(avg_trigger_acc))
    f.close()
    
if __name__ == '__main__':
    main()