import copy

import numpy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm

from enet.models.model import Model

from enet.util import BottledXavierLinear
from enet.models.EEBert import EEBertModel
from enet import consts


class EDModel(Model):
    def __init__(self, hyps, device=torch.device("cpu"), bert_path=None, dropout_rate=0.3):
        if bert_path is None:
            raise AttributeError("Bert 路径未设置")

        super(EDModel, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.device = device
        self.c = 4

        # bert
        # self.bert = BertModel.from_pretrained(bert_path)
        self.bert = EEBertModel.from_pretrained(bert_path, sentence_layers=3)

        self.word_embedding = nn.Embedding(20085, 300)
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.read_embed()))
        self.word_embedding.to(device)
        self.word_embedding_linear = nn.Linear(300, 350, bias=False).to(device)
        nn.init.eye_(self.word_embedding_linear.weight)

        print("dropout rate:", dropout_rate)
        # Dropout
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        # Output Linear
        # self.ol = BottledXavierLinear(in_features=hyps["bert_layers"] * self.bert.config.hidden_size, out_features=hyps["oc"]).to(device=device)
        self.ol = BottledXavierLinear(in_features=self.bert.config.hidden_size+350+self.hyperparams["oc"] - 1, out_features=self.hyperparams["oc"] - 1).to(device=device)
        self.word_linear = nn.Linear(350+self.hyperparams["oc"] - 1, 350+self.hyperparams["oc"] - 1).to(device)
        nn.init.eye_(self.word_linear.weight)
        self.word_class_ol = BottledXavierLinear(in_features=350, out_features=self.hyperparams["oc"] - 1).to(device)
        self.ol_norm = LayerNorm(self.ol.linear.in_features, elementwise_affine=False)
        # Move to right device
        self.to(self.device)

    def dynamic_multipooling(self, inp):
        multipooling = torch.ones((inp.shape[0], inp.shape[1], 2, inp.shape[2])).to(device=self.device) # batch * seq * 2 * dim; 2 means one for trigger left words and one for trigger right words
        for i in range(inp.shape[1]):
            '''
            # context without trigger
            if i <= 1:
                multipooling[:, i, 0, :] = torch.zeros_like(inp[:,0,:])
            else:
                multipooling[:, i, 0, :] = torch.max(inp[:, 1:i, :], dim=1)[0]
            if i == 0 or i == inp.shape[1]-1:
                multipooling[:, i, 1, :] = torch.zeros_like(inp[:,0,:])
            else:
                multipooling[:, i, 1, :] = torch.max(inp[:, i+1:, :], dim=1)[0]
            '''
            # context with trigger
            if i <= 0:
                multipooling[:, i, 0, :] = torch.zeros_like(inp[:,0,:])
            else:
                multipooling[:, i, 0, :] = torch.max(inp[:, 1:i+1, :], dim=1)[0]
            if i == 0:
                multipooling[:, i, 1, :] = torch.zeros_like(inp[:,0,:])
            else:
                multipooling[:, i, 1, :] = torch.max(inp[:, i:, :], dim=1)[0]
        # y = torch.cat((multipooling[:,:,0,:], inp[:, :, :] ,multipooling[:,:,1,:]), dim=2).to(self.device)
        y = torch.sum(multipooling, dim=2, keepdim=False).to(self.device)
        return y

    def forward(self, tokens, x_len, word, word_count, prob_mask, word_num):
        '''
            extracting event triggers
        '''
        # mask
        mask = numpy.zeros(shape=tokens.size(), dtype=numpy.uint8)
        segment = numpy.zeros(shape=tokens.size(), dtype=numpy.uint8)
        for i, l in enumerate(x_len):
            mask[i, 0:l] = numpy.ones(shape=(l), dtype=numpy.uint8)
        mask = torch.LongTensor(mask).to(self.device)
        segment = torch.LongTensor(segment).to(self.device)

        embedding_output = self.bert.embeddings(tokens, segment)

        word_embedding = self.word_embedding(word)
        # word_embedding = word_embedding.detach()
        word_embedding = self.word_embedding_linear(word_embedding)
        # word_embedding = word_embedding.detach()
        word_class_output = self.word_class_ol(word_embedding)
        # word_bias = word_bias.unsqueeze(2)
        # word_embedding = torch.cat((word_embedding, word_count, word_bias), dim=2)
        word_embedding = torch.cat((word_embedding, word_count), dim=2)

        word_vec = self.word_linear(word_embedding)
        word_vec = word_vec + word_embedding
        alpha = torch.min(torch.ones_like(word_num), word_num / self.c)
        alpha = alpha.unsqueeze(-1)
        # alpha = alpha.view(*alpha.shape, -1)
        word_vec = word_vec * alpha

        word_vec = prob_mask(word_vec, "word")

        embedding_output = self.dropout(embedding_output)

        # feed into bert
        bert_encode_layers = None
        if self.training:
            self.bert.train()
            bert_encode_layers, _ = self.bert(tokens, segment, mask, embedding_output=embedding_output)
        else:
            with torch.no_grad():
                bert_encode_layers, _ = self.bert(tokens, segment, mask, embedding_output=embedding_output)

        h = bert_encode_layers[-self.hyperparams["bert_layers"]]  # (x_len, 4*768)
        h = self.dynamic_multipooling(h)
        h = h[:, 1:]

        if not self.training:
            bert_trust_mask = torch.ones_like(h)
            statistical_trust_mask = word_num > 0
            statistical_trust_mask = statistical_trust_mask.unsqueeze(-1).float()
            statistical_trust_mask = prob_mask(statistical_trust_mask, "word")
            # print(bert_trust_mask.shape)
            # print(statistical_trust_mask.shape)
            # print(word_vec.shape)
            statistical_trust_mask = statistical_trust_mask * torch.ones_like(word_vec)
            trust_mask = torch.cat((bert_trust_mask, statistical_trust_mask), dim=-1)
            trust_mask = trust_mask.detach()
        # h = torch.cat((h, exclude_embedding_output), dim=-1)
        # h = torch.cat((h[:,1:], token_vec), dim=-1)
        # print(h.shape)
        # print(word_vec.shape)
        h = torch.cat((h, word_vec), dim=-1)
        h = self.ol_norm(h)
        if not self.training:
            h = h * trust_mask
        trigger_logits = self.ol(h)
        return trigger_logits, word_class_output

    def calculate_loss_ed(self, label, pred, word_class_output, word_class_labels,
                          word_class_num):
        '''
        Calculate loss for a batched output of ed

        :param pred: FloatTensor, (batch_size, seq_len, output_class)
        :param mask: ByteTensor, mask of padded batched input sequence, (batch_size, seq_len)
        :param label: LongTensor, golden label of paadded sequences, (batch_size, seq_len)
        :param word_class_labels: the label for FSMLC
        :return: ed loss, trigger probability, token loss(not used), token class label(not used), word loss
        '''
        y = label.clone()
        pred_log_softmax = F.log_softmax(pred, dim=2)
        y[y > 0] -= 1
        y = y.view(-1,1)
        pred_prob = pred_log_softmax
        pred_log_softmax = pred_log_softmax.view(-1, pred_log_softmax.shape[2])
        y_ = torch.zeros_like(pred_log_softmax)
        y_.scatter_(1, y, 1.)
        y = label
        y[y>0] = 1
        y = y.view(-1,)
        y = y.float()

        pred_loss = torch.sum(pred_log_softmax*y_, dim=1)
        loss = -torch.sum(pred_loss*y) / y.sum().float()

        word_log_softmax = F.log_softmax(word_class_output, dim=2)
        word_log_softmax = word_log_softmax.view(-1, word_log_softmax.shape[2])

        word_class_labels = word_class_labels.view(-1, word_class_labels.shape[2])
        statistical_mask = word_class_num > 0
        word_class_num_reciprocal = torch.reciprocal(word_class_num + 1e-8)
        word_class_num_reciprocal = word_class_num_reciprocal * statistical_mask.float()
        word_class_num_reciprocal = word_class_num_reciprocal.view(-1, 1)
        word_class_num_reciprocal = word_class_num_reciprocal.detach()
        word_loss = torch.sum(word_class_num_reciprocal*word_class_labels*word_log_softmax, dim=1)
        y = torch.sum(word_class_labels, dim=1).sum()
        word_loss = -torch.sum(word_loss) / y.float()
        return loss, pred_prob, word_loss

    def read_embed(self):
        emb_list = [[]]*20085
        with open(consts.complete_filename('word_embedding.txt'), 'r') as reader:
            lines = reader.readlines()
            for i, line in enumerate(lines):
                vec = line.split()
                # print(line.split()[-2:])
                # return
                emb_list[i] = list(map(float, vec))
        emb_np = numpy.array(emb_list)
        return emb_np