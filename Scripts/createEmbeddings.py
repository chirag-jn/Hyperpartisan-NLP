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

bert_model_name = 'bert-base-cased'
bert_tokenizer = None
bert_model = None

# model = None

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
            # print(arr_sent)   
            finalSent = '[CLS] '
            for i in range(len(arr_sent)):
                finalSent += arr_sent[i].strip() + ' [SEP] '
                # arr_sent[i] = "[CLS] " + arr_sent[i] + " [SEP]"
            # print(arr_sent[0])

            tokenized, indexed_tokens, segment_ids = bertSentTokenize(finalSent)

            indexed_tokens_tensor, segment_ids_tensor = get_tensors(indexed_tokens, segment_ids)
            encoded_layers = getEncodingLayers(indexed_tokens_tensor, segment_ids_tensor)

            # TODO: Remove this break
            break

def getEncodingLayers(tokens_tensor, segments_tensors):
    # Predict hidden state features for each layer
    global bert_model
    with torch.no_grad():
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)      
    return encoded_layers

def extractEmbeddings():
    pass

def get_segments_ids(tokens):
    segments_ids = [1] * len(tokens)
    print('seg id len', len(segments_ids))
    return segments_ids

def get_tensors(indexed_tokens,segments_ids):
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    return tokens_tensor, segments_tensors

def bertSentTokenize(sent):
    global bert_tokenizer

    # TODO: Change length here
    tokenized = bert_tokenizer.tokenize(sent)[:512]
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized)
    segment_ids = get_segments_ids(tokenized)
    
    return tokenized, indexed_tokens, segment_ids
    # print(segment_ids)
    # print(tokenized)
    # for i in arr:
    #     tokens,indexed_tokens = sentTokenize(i)
    #     segments_ids = get_segments_ids(tokens)
    #     tokens_tensor,segments_tensors = get_tensors(indexed_tokens,segments_ids)
    #     # TODO: Remove break
    #     break

def get_vectors(model,tokens_tensor,segments_tensors):
    # Convert the hidden state embeddings into single token vectors

    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)   
    token_embeddings = [] 
    # For the 5th token in our sentence, select its feature values from layer 5.

    # For each token in the sentence...
    for token_i in range(len(tokenized_text)):
    
    # Holds 12 layers of hidden states for each token 
        hidden_layers = [] 
    
    # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):
            
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]
            
            hidden_layers.append(vec)
            
        token_embeddings.append(hidden_layers)

    # Sanity check the dimensions:
    print ("Number of tokens in sequence:", len(token_embeddings))
    print ("Number of layers per token:", len(token_embeddings[0])) 
    sentence_embedding = torch.mean(encoded_layers[11], 1)

# def sentTokenize(sent):
#     global bert_tokenizer
#     tokens = bert_tokenizer.tokenize(sent) 
#     #index of tokens in BERT
#     indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokens)

#     for tup in zip(tokens, indexed_tokens):
#         print (tup)    
#     print(tokens)
#     return tokens, indexed_tokens

def loadBertModel():
    global bert_tokenizer, bert_model
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=False)   

    bert_model = BertModel.from_pretrained(bert_model_name)
    
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    bert_model.eval()   
    print('Bert loaded')
    return None

if __name__=='__main__':
    # options = setParser()
    # batch_size = options.bsize
    # maxTokens = options.maxtokens
    # maxSentences = options.maxsentences

    loadBertModel()
    openTextConvert()