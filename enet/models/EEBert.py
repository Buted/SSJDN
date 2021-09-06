#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertEncoder

class EEBertEncoder(BertEncoder):
    def __init__(self, config, sentence_layers):
        """
        :param sentence_layers: int, the last sentence layer
        """
        super(EEBertEncoder, self).__init__(config)
        self.frozen_layers = len(self.layer) - sentence_layers

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            if i == self.frozen_layers:
                hidden_states = hidden_states.detach()
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class EEBertModel(BertModel):
    def __init__(self, config, sentence_layers=3):
        super(EEBertModel, self).__init__(config)
        self.encoder = EEBertEncoder(config, sentence_layers)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                embedding_output=None):
        """
        Override forward function of BertModel, but add `embedding_output` parameter.
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param output_all_encoded_layers:
        :param embedding_output:
        :return:
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if embedding_output is None:
            embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

