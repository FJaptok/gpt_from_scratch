import re
import tqdm
from collections import Counter
from byte_pair_encoding import load,create_corpus,get_vocab,merge_corpus,byte_pair_encoding

train = load('Shakespeare_clean_train.txt')

test = load('Shakespeare_clean_test.txt')

validation = load('Shakespeare_clean_valid.txt')

def unigram(vocab, corpus):
    counts = {}
    for token in vocab:
        counts[token] = 0
    
    for token in corpus:
        counts[token] += 1
    
    return counts

vocab, corpus = byte_pair_encoding(train, 50)

corpus2 = merge_corpus(vocab,test)
counts = unigram(vocab,corpus2)

print(counts)


