import json
import math
import sys

import torch
import torch.nn as nn
from torch.nn import init

from enet.corpus.Mask import Mask


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        #        print("Bottle forward...")
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0] * size[1]))
        return out.view(-1, size[0], size[1])


class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OrthogonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.orthogonal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class BottledLinear(Bottle, nn.Linear):
    pass


class BottledXavierLinear(Bottle, XavierLinear):
    pass


class BottledOrthogonalLinear(Bottle, OrthogonalLinear):
    pass


def log(*args, **kwargs):
    print(file=sys.stdout, flush=True, *args, **kwargs)


def logerr(*args, **kwargs):
    print(file=sys.stderr, flush=True, *args, **kwargs)


def logonfile(fp, *args, **kwargs):
    fp.write(*args, **kwargs)


def progressbar(cur, total, other_information):
    percent = '{:.2%}'.format(cur / total)
    if type(other_information) is str:
        log("\r[%-50s] %s %s" % ('=' * int(math.floor(cur * 50 / total)), percent, other_information))
    else:
        log("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent))


def save_hyps(hyps, fp):
    json.dump(hyps, fp)


def load_hyps(fp):
    hyps = json.load(fp)
    return hyps


def generate_mask(words, max_word_len, max_token_len, device, method='first'):
    word_list = []  # (batch, max_word_len)
    token_list = []  # (batch, max_token_len)
    for sentence in words:
        a_word_list = []
        a_token_list = []
        wlist = sentence['_LIST_']
        for i, w in enumerate(wlist):
            a_word_list.append(i)
            a_token_list.extend([i] * len(sentence[w]))
        while len(a_word_list) < max_word_len:
            a_word_list.append(0)
        while len(a_token_list) < max_token_len:
            a_token_list.append(-1)
        word_list.append(a_word_list)
        token_list.append(a_token_list)

    mask = Mask(method=method, word_list=word_list, token_list=token_list, device=device)
    return mask


def run_over_data(model, optimizer, data_iter, need_backward, tester, device, label_i2s,
                  maxnorm, save_output, seq_lambda = 0.9, word_count_emb=None, word_num_emb=None):
    """
    Run an epoch on data iter
    :param data_iter: data iterator
    :param need_backward: whether or not backward
    :param device: gpu device
    :param label_i2s: label int2str
    :param seq_lambda: beta  of FSMLC
    :param word_count_emb: generate the word-event co-occurrence frequencies
    :param word_num_emb: not used in SSJDN
    :return: loss, F1
    """
    if need_backward:
        model.test_mode_off()
    else:
        model.test_mode_on()
    log("Seq train")
    running_loss = 0.0
    sequence_loss = 0.0
    classes_loss = 0.0

    # 事件标签
    e_y = []
    e_y_ = []
    # argument label
    ae_y = []
    ae_y_ = []

    cnt = 0
    for batch in data_iter:
        optimizer.zero_grad()
        cnt += 1

        tokens, x_len = batch.WORDS
        words = batch.WLIST
        labels = batch.LABEL
        token_labels = batch.TOKENLABEL

        word_ids, w_len = batch.WIDS
        word_label_id = batch.WLABELIDS
        word_class_count = word_count_emb(word_label_id)
        word_class_num = word_num_emb(word_label_id)
        word_num_sum = torch.sum(word_class_num, dim=-1)

        mask = generate_mask(words, labels.size(1), torch.max(x_len.cpu()) - 1, device, method='average')
        old_labels = labels.clone()
        # forward
        trigger_logits, word_class_output = model.forward(tokens, x_len, word_ids, word_class_count, mask,
                                                                        word_num_sum)
        ae = []
        ae_ = []

        labels = token_labels
        # calculate loss
        seq_loss, trigger_prob, w_loss = model.calculate_loss_ed(labels, trigger_logits, word_class_output,
                                                                 word_class_count, word_num_sum)

        loss = seq_loss + seq_lambda * w_loss
        # loss = seq_loss

        if need_backward:
            loss.backward()
            if 1e-6 < maxnorm and model.parameters_requires_grad_clipping() is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters_requires_grad_clipping(), maxnorm)

            optimizer.step()
            optimizer.zero_grad()
        trigger_logits = mask(trigger_prob, type='prob')
        trigger_logits = trigger_logits.reshape(-1, trigger_logits.shape[2])
        # metrics
        trigger_ = torch.argmax(trigger_logits.cpu(), 1).tolist()
        old_labels[old_labels>0] -= 1
        old_labels = old_labels.cpu().view(-1, ).tolist()
        e_y.extend(old_labels)
        e_y_.extend(trigger_)
        ae_y.extend(ae)
        ae_y_.extend(ae_)


        running_loss += loss.item()
        sequence_loss += seq_loss.item()
        classes_loss += w_loss.item()

    if save_output:
        with open(save_output, "w", encoding="utf-8") as f:
            f.write("y,y_\n")
            for y, y_ in zip(e_y, e_y_):
                f.write("%d,%d\n" % (y, y_))

    running_loss = running_loss / cnt
    sequence_loss = sequence_loss / cnt
    classes_loss = classes_loss / cnt
    all_summary = tester.summary_report(e_y, e_y_, ae_y, ae_y, label_i2s, role_i2s=None)
    log("Epoch finish")

    def display(report):
        d = lambda s: print(s.center(30, "-"))
        d("")
        d(" loss : {:.6f} ".format(running_loss))
        d(" Trigger Identification ")
        d(" P: {:.6f} R: {:.6f} F1: {:.6f}".format(*report["t-i"]))
        d(" Trigger Classification ")
        d(" P: {:.6f} R: {:.6f} F1: {:.6f}".format(*report["t-c"]))
        d(" seq loss : {:.6f} ".format(sequence_loss))
        d(" class loss : {:.6f} ".format(classes_loss))

    display(all_summary)

    return running_loss, all_summary["t-c"][-1], all_summary["a-c"][-1]
