# import pandas as pd
import numpy as np
# from codes.token_tools import Tokenizer

from collections import OrderedDict, Counter


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


    '''''
    亲水性 1
    疏水性 2
    酸性   3
    碱性   4
    '''''
    voc_sp = OrderedDict([
        ('G', 1),
        ('A', 2),
        ('V', 2),
        ('L', 2),
        ('I', 2),
        ('F', 2),
        ('W', 2),
        ('Y', 1),
        ('D', 3),
        ('N', 1),
        ('E', 3),
        ('K', 4),
        ('Q', 1),
        ('M', 2),
        ('S', 1),
        ('T', 1),
        ('C', 1),
        ('P', 2),
        ('H', 4),
        ('R', 4),
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
            return 0
        else:
            return Tokenizer.vocab[token]
    
    def convert_tokens_to_ids(self, tokens):
        return [self.convert_token_to_id(token) for token in tokens]

    def gen_token_ids(self, sequence):
        tokens = []
        tokens += self.tokenize(sequence)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids
    



    
    def convert_token_to_id_sp(self, token):
        if token not in self.vocab:
            return Tokenizer.unknown_token_id
        else:
            return Tokenizer.voc_sp[token]
    
    def convert_tokens_to_ids_sp(self, tokens):
        return [self.convert_token_to_id_sp(token) for token in tokens]

    def gen_token_ids_sp(self, sequence):
        tokens = []
        tokens += self.tokenize(sequence)
        token_ids = self.convert_tokens_to_ids_sp(tokens)
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

import random

def random_pad_to_max_seq(seq_ids, max_len, pad_id=0):
    seq_len = len(seq_ids)
    if seq_len < max_len:
        if  random.random() < 0.5:
            pad_tokens = [pad_id for _ in range(max_len-seq_len)]
            seq_ids_new = seq_ids + pad_tokens
        else:
            len_pad = max_len-seq_len
            len_pad_pre = random.randint(1, len_pad-2)
            len_pad_post = len_pad - len_pad_pre

            pad_tokens_pre = [pad_id for _ in range(len_pad_pre)]
            pad_tokens_post = [pad_id for _ in range(len_pad_post)]

            seq_ids_new = pad_tokens_pre + seq_ids + pad_tokens_post
    elif seq_len > max_len:
        seq_ids_new = seq_ids[:max_len]
    else:
        seq_ids_new = seq_ids
    
    seq_ids_new = np.array(seq_ids_new)
    return seq_ids_new


def random_seq_mask(data_seq, ratio=0.1):
    ## len * ch
    # data[:,0:1] vh
    # data[:,1:2] vl
    # data[:,2:3] ch
    # data[:,3:4] cl
    # data[:,4:5] vh sp
    # data[:,5:6] vl sp
    # data[:,6:7] ch sp
    # data[:,7:8] cl sp
    for i in range(len(data_seq)):
        ## mask vh and vhsp
        if random.random() < ratio:
            data_seq[i,0:1] = -1
            data_seq[i,4:5] = -1
        ## mask vl and vl sp
        if random.random() < ratio:
            data_seq[i,1:2] = -1
            data_seq[i,5:6] = -1
    return data_seq
    



def resample_dataset(data_seq, data_lbl):
    data_seq_re = []
    data_lbl_re = []
    temp_count = Counter(data_lbl)
    temp_max = max(temp_count.values())
    print(temp_count)

    list_index_all = []
    for i in range(1, 6, 1):
        list_idnex_temp = []
        while True:
            for mm in range(len(data_lbl)):
                if data_lbl[mm] == i:
                    list_idnex_temp.append(mm)
                if len(list_idnex_temp) >= temp_max:
                    break
            if len(list_idnex_temp) >= temp_max:
                break
        list_index_all += list_idnex_temp
    



    for temp_index in list_index_all:
        data_seq_re.append(data_seq[temp_index])
        data_lbl_re.append(data_lbl[temp_index])
    



    print(Counter(data_lbl_re))

    return data_seq_re, data_lbl_re


