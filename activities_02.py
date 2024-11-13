import os
import json
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
import spacy
import matplotlib.pyplot as plt
import regex as re


def tokenize_json(file_path):
    with open (file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        text = data['text']
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('portuguese'))
        tokens = [token for token in tokens if token not in stop_words]
        print(tokens)
    return tokens
file = 'dois/342.json'

def tokenize_multiples_json(directory):
    all_tokens = []
    vocab = defaultdict(lambda: len(vocab))
    for file in os.listdir(directory):
        if file.endswith('.json'):
            file_path = os.path.join(directory, file)
            tokens = tokenize_json(file_path)
            tokens_ids = [vocab[token] for token in tokens]
            all_tokens.append(tokens_ids)
            #print(tokens_ids)
            #print('tamaño:', len(tokens_ids))
    return all_tokens

directory = "dois"
tokenize_multiples_json(directory)
all_tokens = tokenize_multiples_json(directory)   
print('length of all tokens:',len(all_tokens))     


def get_stats(tokens):
    counts = {}
    for i in range(len(tokens)- 1):
        pair = (tokens[i], tokens[i+1])
        counts[pair] =  counts.get(pair, 0) + 1
        return counts
stats = get_stats(all_tokens[0])
print (stats)

top_pair = max(stats, key=stats.get)
print('Top pair:', top_pair)



def merge(ids, pair, idx):
    newids = []
    i = 0
    while i< len(ids):
        if i <len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids
    
tokens2 = merge(all_tokens, top_pair, 256) 
print('Tokens2:', tokens2)
print('length of tokens2:', len(tokens2))   

vocab_size = 276
num_merges = vocab_size - 256
ids = list(all_tokens[0])

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + len(merges)
print(f"merging {pair} into a new token {idx}")
ids = merge(ids, pair, idx)
merges [pair] = idx    


#print("tokens length:", len(tokens))
#print("ids length:", len(ids))
#print(f"compression ratio: {len(tokens) / len(ids):2f}X")

#decoding
vocab = {idx: bytes ([idx] )for idx in range(256) }
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab [p0] + vocab [p1]

def decode (ids): 
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
print(decode([73]))

#encoding
def encode(text):
    tokens =  list(text.encode("utf-8"))
    while True:
        stats =  get_stats(tokens)
        pair = min(stats, key= lambda p: merges.get(p, float ("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens =  merge(tokens, pair, idx)
    return tokens
print(encode("O tempo é muito poderoso!"))


#Conta com que frequencia ocurrem as combinações
b = {}
for words in all_tokens[:1]:
    ch = ['<S>'] + list(words) + ['<E>']
    for ch1, ch2 in zip(words, words[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) +1
        #print(ch1, ch2)
        
a = sorted(b.items(), key= lambda kv: -kv[1])
#print('print sorted:', a)


sequences = all_tokens
max_len = max(len(seq) for seq in sequences)
padded_sequences = [seq + [0] *(max_len - len(seq))for seq in sequences]
matriz = torch.tensor((padded_sequences), dtype= torch.int32)

forma = matriz.shape
print ('a matriz torcs:', forma)

submatriz = matriz [:10, :10]
matriz_float = np.array(submatriz, dtype=np.float32)
np.set_printoptions(suppress=True)
print('Submatriz:', matriz_float)

p = matriz_float
p = p/p.sum()
print ('Calculo:', p)

def def_perplexity (vector_probability):
    vector_probability = np.where(vector_probability ==0, 1e-10, vector_probability)
    entropia = -np.sum(vector_probability * np.log2(vector_probability))
    perplexity = np.exp2(entropia)
    return perplexity

vector_prob = np.array(p)
perplexity = def_perplexity(vector_prob)
print('La perplexidad es:', perplexity)

