import argparse
import os
from functools import partial

import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torchtext.data import Field

from enet import consts
from enet.corpus.Data import ACE2005Dataset, WordField
from enet.models.ee import EDModel
from enet.testing import EDTester
from enet.training import train
from enet.util import log
from pytorch_pretrained_bert import BertTokenizer
from enet.corpus.glove_loader import WordLoader


class EERunner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        parser.add_argument("--test", help="validation set", default="test.json")
        parser.add_argument("--train", help="training set", default="test.json", required=False)
        parser.add_argument("--dev", help="development set", required=False, default="dev.json")
        parser.add_argument("--bert", default="bert")

        parser.add_argument("--batch", help="batch size", default=16, type=int)
        parser.add_argument("--epochs", help="n of epochs", default=99999, type=int)

        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        parser.add_argument("--optimizer", default="adadelta")
        parser.add_argument("--lr", default=0.5, type=float)
        parser.add_argument("--l2decay", default=1e-5, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)
        parser.add_argument("--keep_event", default=0, type=int)

        parser.add_argument("--out", help="output model path", default="out")
        parser.add_argument("--finetune", help="pretrained model path")
        parser.add_argument("--earlystop", default=10, type=int)
        parser.add_argument("--restart", default=999999, type=int)

        parser.add_argument("--device", default="cuda:1")
        parser.add_argument("--hps", help="model hyperparams", required=False,
                            default="{'loss_alpha': 5, 'bert_layers': 4}")
        parser.add_argument("--dropout", help="dropout rate", default=0.3)
        parser.add_argument("--contiguous", help="include contiguous trigger", default=True)
        parser.add_argument("--seq_lambda", default=1, type=float)

        self.a = parser.parse_args()

    def set_device(self, device="cpu"):
        self.device = torch.device(device)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def load_model(self, fine_tune: str):
        """
        Load model from checkpoint if finetune is not None
        :param fine_tune: checkpoint path
        :return: instance of EDModel
        """
        mymodel = EDModel(self.a.hps, self.get_device(), bert_path=self.a.bert, dropout_rate=float(self.a.dropout))
        if fine_tune is not None:
            mymodel.load_model(fine_tune)
        mymodel.to(self.get_device())

        return mymodel

    def get_tester(self, voc_i2s):
        return EDTester(voc_i2s)

    def get_token_id(self, path):
        token_id = {}
        with open(path, 'r', encoding="utf-8") as reader:
            lines = reader.readlines()
            for line in lines:
                token = ' '.join(line.split()[:-1])
                id = int(line.split()[-1])
                if token in token_id:
                    print(token)
                token_id[token] = id
        return token_id

    def get_class_label_list(self, path, token_id):
        class_label_list = [[]] * len(token_id)
        with open(path, 'r', encoding="utf-8") as reader:
            lines = reader.readlines()
            for line in lines:
                token = ' '.join(line.split()[:-1])
                id = token_id[token]
                label = line.split()[-1].split(',')[:-1]
                class_label_list[id] = list(map(float, label))
        return class_label_list

    def get_class_count_list(self, path, token_id):
        class_count_list = [[]] * len(token_id)
        with open(path, 'r', encoding="utf-8") as reader:
            lines = reader.readlines()
            for line in lines:
                token = ' '.join(line.split()[:-1])
                id = token_id[token]
                label = line.split()[-1].split(',')[:-1]
                class_count_list[id] = list(map(float, label))
        return class_count_list

    def get_embedding(self, embedding_list):
        embedding_numpy = np.array(embedding_list)
        emb = nn.Embedding(embedding_numpy.shape[0], embedding_numpy.shape[1])
        emb.weight.data.copy_(torch.from_numpy(embedding_numpy))
        emb.weight.requires_grad = False
        emb.to(self.get_device())
        return emb

    def run(self):
        """
        Run the project.
        """
        print("Running on", self.a.device)
        print("Load BERT:", self.a.bert)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)

        # create training set
        if self.a.train:
            log('loading corpus from %s' % self.a.train)

        def tokenize(text):
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            return ids

        tokenizer = BertTokenizer.from_pretrained(self.a.bert,
                                                  never_split=["[CLS]", "[SEP]"])
        TokensField = Field(lower=False, batch_first=True, include_lengths=True,
                            tokenize=(lambda s: tokenize(s)), pad_token=0, unk_token=None, use_vocab=False)
        WordsField = WordField(lower=False, batch_first=True, use_vocab=False, tokenize=(lambda s: tokenize(s)))
        TriggerLabelField = Field(lower=False, batch_first=True, pad_token=consts.PADDING_LABEL, unk_token=None)
        TokenLabelField = Field(lower=False, batch_first=True, pad_token=consts.PADDING_LABEL, unk_token=None)
        WordIdField = Field(lower=False, batch_first=True, pad_token=0, unk_token=None, use_vocab=False,
                            include_lengths=True)
        WordLabelIdField = Field(lower=False, batch_first=True, pad_token=0, unk_token=0, use_vocab=False)

        data_dir = self.a.test.split('/')[0]
        complete_filename = lambda x: os.path.join(data_dir, x)
        consts.complete_filename = complete_filename

        word_id = self.get_token_id(complete_filename("word_class_id.txt"))
        print('read...')
        all_word_id = self.get_token_id(complete_filename("all_word_class_id.txt"))
        print('list...')
        word_label_list = self.get_class_label_list(complete_filename("word_class_label.txt"), word_id)

        word_count_list = self.get_class_count_list(complete_filename("word_class_count.txt"), word_id)
        word_num_list = self.get_class_count_list(complete_filename("word_class_num.txt"), word_id)
        print('embedding...')
        word_label_emb = self.get_embedding(word_label_list)
        word_count_emb = self.get_embedding(word_count_list)
        word_num_emb = self.get_embedding(word_num_list)

        print('embedding was built...')
        self.a.word_label_emb = word_label_emb
        self.a.word_count_emb = word_count_emb
        self.a.word_num_emb = word_num_emb
        # 这里的 fields 会自动映射 json 文件里的结果
        train_set = ACE2005Dataset(path=self.a.train, min_len=None,
                                   fields={"words": ("WORDS", TokensField),
                                           "word-list": ("WLIST", WordsField),
                                           "golden-event-mentions": ("LABEL", TriggerLabelField),
                                           "golden-token-labels": ("TOKENLABEL", TokenLabelField),
                                           "word-ids": ("WIDS", WordIdField),
                                           "word-label-ids": ("WLABELIDS", WordLabelIdField)},
                                   keep_events=self.a.keep_event,
                                   tokenizer=tokenizer,
                                   all_word_id=all_word_id,
                                   word_id=word_id)
        self.a.contiguous = eval(self.a.contiguous)
        dev_set = ACE2005Dataset(path=self.a.dev,
                                 fields={"words": ("WORDS", TokensField),
                                         "word-list": ("WLIST", WordsField),
                                         "golden-event-mentions": ("LABEL", TriggerLabelField),
                                         "golden-token-labels": ("TOKENLABEL", TokenLabelField),
                                         "word-ids": ("WIDS", WordIdField),
                                         "word-label-ids": ("WLABELIDS", WordLabelIdField)},
                                 keep_events=0,
                                 include_contigous=self.a.contiguous,
                                 tokenizer=tokenizer,
                                 all_word_id=all_word_id,
                                 word_id=word_id)

        TriggerLabelField.build_vocab(train_set.LABEL)
        TokenLabelField.build_vocab(
            train_set.LABEL)  # must be LABEL instead of TOKENLABEL, or the vocab.stoi is not same

        consts.O_LABEL = TriggerLabelField.vocab.stoi["O"]
        print(TriggerLabelField.vocab.stoi)
        test_set = ACE2005Dataset(path=self.a.test,
                                  fields={"words": ("WORDS", TokensField),
                                          "word-list": ("WLIST", WordsField),
                                          "golden-event-mentions": ("LABEL", TriggerLabelField),
                                          "golden-token-labels": ("TOKENLABEL", TokenLabelField),
                                          "word-ids": ("WIDS", WordIdField),
                                          "word-label-ids": ("WLABELIDS", WordLabelIdField)},
                                  keep_events=0,
                                  include_contigous=self.a.contiguous,
                                  tokenizer=tokenizer,
                                  all_word_id=all_word_id,
                                  word_id=word_id)

        self.a.hps = eval(self.a.hps)

        # 词向量大小
        # 事件预测空间
        if "oc" not in self.a.hps:
            self.a.hps["oc"] = len(TriggerLabelField.vocab.itos)

        # 评测
        tester = self.get_tester(TriggerLabelField.vocab.itos)

        if self.a.finetune:
            log('init model from ' + self.a.finetune)
            model = self.load_model(self.a.finetune)
            log('model loaded, there are %i sets of params' % len(model.parameters_requires_grads()))
        else:
            model = self.load_model(None)

        if self.a.optimizer == "adadelta":
            optimizer_constructor = partial(torch.optim.Adadelta, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay)
        elif self.a.optimizer == "adam":
            optimizer_constructor = partial(torch.optim.Adam, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay)
        else:
            optimizer_constructor = partial(torch.optim.SGD, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay,
                                            momentum=0.9)

        log('optimizer in use: %s' % str(self.a.optimizer))

        if not os.path.exists(self.a.out):
            os.mkdir(self.a.out)

        log('init complete\n')

        self.a.label_i2s = TriggerLabelField.vocab.itos
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer

        train(
            model=model,
            train_set=train_set,
            dev_set=dev_set,
            test_set=test_set,
            optimizer_constructor=optimizer_constructor,
            epochs=self.a.epochs,
            tester=tester,
            parser=self.a
        )
        log('Done!')


if __name__ == "__main__":
    EERunner().run()
