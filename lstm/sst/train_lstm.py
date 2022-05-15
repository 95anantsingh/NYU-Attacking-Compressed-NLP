import os.path
import torch
import torch.optim as optim
# from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
#     StanfordSentimentTreeBankDatasetReader
from reader_new import StanfordSentimentTreeBankDatasetReader_NEW
from allennlp.data.iterators import BucketIterator, BasicIterator
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
import pandas as pd





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
    # load the binary SST dataset.
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
#     # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader_NEW(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer},
                                                    use_subtrees=True)
    train_data = reader.read('./data/train.txt')
#     print(train_data)
    reader = StanfordSentimentTreeBankDatasetReader_NEW(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
    dev_data = reader.read('./data/dev.txt')

    vocab = Vocabulary.from_instances(train_data)
    print(vocab)

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
    model = LstmClassifier(word_embeddings, encoder, vocab)
    model.cuda()




    # where to save the model
    model_path = "./saved_lstm_sst_model/" + EMBEDDING_TYPE + "_" + "model.th"
    vocab_path = "./saved_lstm_sst_model/" + EMBEDDING_TYPE + "_" + "vocab"
    
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    
    iterator.index_with(vocab)
    
    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_data,
                      validation_dataset=dev_data,
                      num_epochs=4,
                      patience=1,
                      cuda_device=0)
                      
    trainer.train()




    
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(vocab_path)
    
if __name__ == '__main__':
    main()