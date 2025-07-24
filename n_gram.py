import re
import tqdm
import time
from itertools import product
from collections import Counter
from byte_pair_encoding import load,create_corpus,get_vocab,merge_corpus,byte_pair_encoding

# def count_words(corpus):
#     """ Counts the occurences of all tokens inside a given corpus as a dictionary """
#     # get all tokens inside a corpus
#     vocab = get_vocab(corpus)
#     counts = {}
#     for token in vocab:
#         counts[token] = 0
#     # count the occurences of each token
#     for token in corpus:
#         counts[token] += 1
    
#     return counts

def count_combinations(corpus, n):
    """ Counts the occurences of all tokens inside a given corpus as a dictionary """
    counts = {}
    
    for i in range(len(corpus)):
        for j in range(1, n+1):
            try:
                if tuple(corpus[i-n+1:i+1]) in counts:
                    counts[tuple(corpus[i-j+1:i+1])] += 1
                else:
                    counts[tuple(corpus[i-j+1:i+1])] = 1
            except:
                pass
    return counts

class n_gram:
    """ A clss to create an N-gram for a given n """
    def __init__(self, n):
        self.n = n
    
    def get_probabilities(self, corpus):
        """ This function takes a corpus and returns sets the probabilities for all possible combinations accordingly """
        # get the counts for all occuring combinations
        counts = count_combinations(corpus, self.n)
        probabilities = {}

        # we consider all occuring combinations given by "counts" 
        for tokens in counts:
            # add an exception for the uni-gram probabilities
            if len(tokens) > 1:
                if counts[tokens[:-1]] == 0:
                    probabilities[tokens] = 0.0
                else:
                    # and calculate the probabilities accordingly with the full conditional probability and 
                    # the conditional probability for the combined token - the last token
                    probabilities[tokens] = counts[tokens] / counts[tokens[:-1]]
            else:
                # uni-gram probabilities are calculated with the number of occurences of the token over the whole corpus
                probabilities[tokens] = counts[tokens] / len(corpus)
        
        self.probabilities = probabilities

if __name__ == '__main__':

    train = load('Shakespeare_clean_train.txt')
    test = load('Shakespeare_clean_test.txt')
    val = load('Shakespeare_clean_valid.txt')

    corpus = create_corpus(train)

    uni_gram = n_gram(1)
    uni_gram.get_probabilities(corpus)

    bi_gram = n_gram(2)
    bi_gram.get_probabilities(corpus)

    tri_gram = n_gram(3)
    tri_gram.get_probabilities(corpus)

    four_gram = n_gram(4)
    four_gram.get_probabilities(corpus)


