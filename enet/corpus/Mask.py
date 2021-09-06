#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
import torch
import numpy as np

class Mask:
    def __init__(self, method, word_list, token_list, device):
        self.mask = None
        self.word_list = word_list
        self.token_list = token_list
        self.prob_mask = np.zeros((len(word_list), len(token_list[0]), len(word_list[0])))
        self.word_mask = np.zeros((len(word_list), len(word_list[0]), len(token_list[0])))
        self.__set_mask(method, device)

    def __set_mask(self, method, device):

        def set_prob_mask():
            prob_mask = torch.zeros((len(token_list), len(word_list)))
            word_mask = torch.zeros((len(word_list), len(token_list)))
            if method == 'first':
                occur_list = [False] * len(word_list)
                for i, j in enumerate(token_list):
 #                   print(j)
                    if occur_list[j]:
                        continue
                    else:
                        prob_mask[i, j] = 1
                        occur_list[j] = True
            elif method == 'last':
                start = 0
                for i, j in enumerate(token_list):
                    if j != start:
                        prob_mask[i-1, j-1] = 1
                        start = j
                prob_mask[-1, -1] = 1 # the last one must be 1
            elif method == 'average':
                start = 0
                count = 0
                finish_word = []
                for i, j in enumerate(token_list):
                    if j != start:
                        prob_mask[i-count:i, start] = 1.0 / count
                        word_mask[start, i-count:i] = 1.0
                        count = 1
                        finish_word.append(True)
                        if j == -1:
                            break
                        else:
                            start = j
                    else:
                        count += 1
                if count == len(token_list):
                    prob_mask[:count, 0] = 1.0 / count
                    word_mask[0, :count] = 1.0
                elif len(finish_word) < len(word_list) and start == token_list[-1]:
                    prob_mask[-count:, len(finish_word)] = 1.0 / count
                    word_mask[start, -count:] = 1.0
                else:
                    pass
            else:
                raise NotImplementedError
            return prob_mask.numpy(), word_mask.numpy()

        i = 0
        for word_list, token_list in zip(self.word_list, self.token_list):
            self.prob_mask[i], self.word_mask[i] = set_prob_mask()
            i += 1
        self.prob_mask = torch.from_numpy(self.prob_mask).float().to(device=device)
        self.word_mask = torch.from_numpy(self.word_mask).float().to(device)

    def __call__(self, x, type = None):
        '''
        process masking
        :param args:
        :param kwargs:
        :return:
        '''
        if type == 'prob':
            x = torch.transpose(x, 1, 2)
            y = torch.matmul(x, self.prob_mask)
            y = torch.transpose(y, 1, 2)
            return y
        elif type == 'word':
            x = torch.transpose(x, 1, 2)
            y = torch.matmul(x, self.word_mask)
            y = torch.transpose(y, 1, 2)
            return y
        else:
            raise ValueError('Wrong mask type')
