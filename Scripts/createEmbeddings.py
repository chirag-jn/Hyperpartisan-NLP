from defaults import *
import argparse
import os
import json
import math
import numpy as np
from optparse import OptionParser
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

batch_size = 50
maxTokens = 200
maxSentences = 200
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')   

def setParser():
    parser = OptionParser()
    parser.add_option("--bsize", help="Batch Size", type=int, default=50)
    parser.add_option("--maxtokens", help="Max tokens in a sentence", type=int, default=200)
    parser.add_option("--maxsentences", help="Max sentences in an article", type=int, default=200)
    options, _ = parser.parse_args()
    return options

def openTextConvert():
    with open(train_text_tsv, 'r') as tsv:
        for ts in tsv:
            arr = ts.split('\t')
            sent = arr[4]
            arr_sent = sent.split('<splt>')
            for i in range(len(arr_sent)):
                arr_sent[i] = "[CLS] " + arr_sent[i] + " [SEP]"
            # print(arr_sent[0])
            arrTokenize(arr_sent)
            # TODO: Remove this break
            break

def get_tensors(indexed_tokens,segments_ids):
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor, segments_tensors

def arrTokenize(arr):
    for i in arr:
        tokens,indexed_tokens = sentTokenize(i)
        # TODO: Remove break
        break

def sentTokenize(sent):
    global bert_tokenizer
    tokens = bert_tokenizer.tokenize(sent) 
    #index of tokens in BERT
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokens)

    for tup in zip(tokens, indexed_tokens):
        print (tup)    
    print(tokens)
    return tokens, indexed_tokens

if __name__=='__main__':
    options = setParser()
    batch_size = options.bsize
    maxTokens = options.maxtokens
    maxSentences = options.maxsentences
    openTextConvert()

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()    
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')