import re
import tqdm
from collections import Counter
from byte_pair_encoding import load,create_corpus,get_vocab,merge_corpus,byte_pair_encoding

def count_occurences(corpus):
    """ Counts the occurences of all tokens inside a given corpus as a dictionary """
    # get all tokens inside a corpus
    vocab = get_vocab(corpus)
    counts = {}
    for token in vocab:
        counts[token] = 0
    # count the occurences of each token
    for token in corpus:
        counts[token] += 1
    
    return counts

def probability(corpus):
    """ Returns a hashmap containing all possible token combinations 
    with their probabilites according to their occurences in a given corpus """
    hash_map = {}
    vocab = get_vocab(corpus)
    counts = count_occurences(corpus)

    # create entries in the hash_map for all possible combinations
    for a in vocab:
        for b in vocab:
            hash_map[(a,b)] = 0

    # and count the occurences of each combination in the corpus
    for i in range(len(corpus)):
        try:
            hash_map[(corpus[i-1],corpus[i])] += 1
        except:
            pass
    
    # create entries for the probabilites for each combination
    probabilties = {}
    for a in vocab:
        for b in vocab:
            probabilties[(a,b)] = 0.0

    # and compute the probability with the occurences of the combination and the first token itself
    for pair in hash_map:
        probabilties[pair] = hash_map[pair] / counts[pair[0]]

    return probabilties


train = load('Shakespeare_clean_train.txt')

test = load('Shakespeare_clean_test.txt')

validation = load('Shakespeare_clean_valid.txt')

vocab, corpus = byte_pair_encoding(train, 50)

corpus2 = merge_corpus(vocab,test)
probs = probability(corpus2)

print(probs)


