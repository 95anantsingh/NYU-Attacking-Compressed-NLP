import sys
import os.path
import torch
import torch.optim as optim
import torch.quantization
import torch.nn as nn

from reader_new import StanfordSentimentTreeBankDatasetReader_NEW
# from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.data_loaders import SimpleDataLoader

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
parser.add_argument("-q", help = "value: 8/16")
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

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

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
    
    model_path = "./lstm_main_sst_model/w2v_model.th"
    model = LstmClassifier(word_embeddings, encoder, vocab)
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    if args.q == '8':
        dt = torch.qint8
    elif args.q == '16':
        dt = torch.float16
        
    model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model,{nn.Linear,nn.LSTM}, dtype=dt
    )
    
    model_size = print_size_of_model(model,"fp32")
    if "8" in args.q:
        quantized_model_size = print_size_of_model(quantized_model,"int8")
    else:
        quantized_model_size = print_size_of_model(quantized_model,"float16")

    print("{0:.2f} times smaller".format(model_size/quantized_model_size))
    
    dl = SimpleDataLoader(list(dev_data), batch_size=128, shuffle=True)
    dl.index_with(vocab)
    
    for x in dl:
        inp = (x)
        break
    
    dummy_input = [inp['tokens'], inp['label']]
    traced_model = torch.jit.trace(quantized_model, dummy_input, strict=False)
    
    model_path = 'lstm_quantized'+str(args.q)+'_sst_model/'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    torch.jit.save(traced_model, model_path+"model.pt")
    
if __name__ == '__main__':
    main()