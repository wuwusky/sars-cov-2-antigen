import pandas as pd
import numpy as np
# from codes.token_tools import Tokenizer

from collections import OrderedDict


class Tokenizer(object):
    padding_token = '<pad>'
    mask_token = '<mask>'
    start_token = class_token = '<cls>'
    unknown_token = '<unk>'
    end_token = seperate_token = '<sep>'

    padding_token_id = 0
    mask_token_id = 1
    start_token_id = class_token_id = 2
    unknown_token_id = 3
    end_token_id = seperate_token_id = 4

    special_token_ids = [padding_token_id, mask_token_id, start_token_id, unknown_token_id]

    vocab = OrderedDict([
        (padding_token, 0),
        (mask_token, 1),
        (class_token, 2),
        (unknown_token, 3),
        (seperate_token, 4),
        ('A', 5),
        ('B', 6),
        ('C', 7),
        ('D', 8),
        ('E', 9),
        ('F', 10),
        ('G', 11),
        ('H', 12),
        ('I', 13),
        ('K', 14),
        ('L', 15),
        ('M', 16),
        ('N', 17),
        ('O', 18),
        ('P', 19),
        ('Q', 20),
        ('R', 21),
        ('S', 22),
        ('T', 23),
        ('U', 24),
        ('V', 25),
        ('W', 26),
        ('X', 27),
        ('Y', 28),
        ('Z', 29)
    ])

    def tokenize(self, sequence):
        if len(sequence) <= 3:
            return ['unknown_token']
        tokens = []
        for x in sequence:
            tokens.append(x)
        return tokens
    
    def convert_token_to_id(self, token):
        if token not in self.vocab:
            return Tokenizer.unknown_token_id
        else:
            return Tokenizer.vocab[token]
    
    def convert_tokens_to_ids(self, tokens):
        return [self.convert_token_to_id(token) for token in tokens]

    def gen_token_ids(self, sequence):
        tokens = []
        tokens += self.tokenize(sequence)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids





## extract sequence information dataset
def pad_to_max_seq(seq_ids, max_len, pad_id=0):
    seq_len = len(seq_ids)
    if seq_len < max_len:
        padded_tokens = [pad_id for _ in range(max_len-seq_len)]
        seq_ids_new = seq_ids + padded_tokens
    elif seq_len > max_len:
        seq_ids_new = seq_ids[:max_len]
    else:
        seq_ids_new = seq_ids
    seq_ids_new = np.array(seq_ids_new)
    return seq_ids_new
