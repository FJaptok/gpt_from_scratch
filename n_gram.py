import re
import tqdm
from collections import Counter
from byte_pair_encoding import load,create_corpus,get_vocab,merge_corpus,byte_pair_encoding
import pickle
import os 


def get_n_gram(corpus, n):
    n_gram_counts = {}
    counts = {}
    for i in range(len(corpus)):
        current_token = corpus[i]
        if current_token in counts:
            counts[current_token] += 1
        else:
            counts[current_token] = 1
        if n == 1:
            return cal_probs_from_counts(counts)
        
    bigram_counts = {}
    for i in range(len(corpus)):
        try:
            current_bigram = (corpus[i-1], corpus[i])
            if current_bigram in bigram_counts:
                bigram_counts[current_bigram] += 1
            else:
                bigram_counts[current_bigram] = 1
        except:
            pass
    if n == 2:
        return cal_probs_from_counts(counts, bigram_counts)
    
    trigram_counts = {}
    for i in range(len(corpus)):
        try:
            current_trigram = (corpus[i-2], corpus[i-1], corpus[i]) # This looks like a typo, should be corpus[i] for the last element
            if current_trigram in trigram_counts:
                trigram_counts[current_trigram] += 1
            else:
                trigram_counts[current_trigram] = 1
        except: 
            pass
    if n == 3:
        return cal_probs_from_counts(counts, bigram_counts, trigram_counts)
    
    fourgram_counts = {}
    for i in range(len(corpus)):
        try:
            current_fourgram = (corpus[i-3], corpus[i-2], corpus[i-1], corpus[i]) # These indices look incorrect for a fourgram
            if current_fourgram in fourgram_counts:
                fourgram_counts[current_fourgram] += 1
            else:
                fourgram_counts[current_fourgram] = 1
        except:
            pass
    return cal_probs_from_counts(counts, bigram_counts, trigram_counts, fourgram_counts)

            


def cal_probs_from_counts(unigram_counts, bigram_counts=None, trigram_counts=None, fourgram_counts=None):
    uni_probs = {}
    total_unigram_count = sum(unigram_counts.values()) # Calculate total once
    for unigram, count in unigram_counts.items(): # Iterate over items for clarity
        prob = count / total_unigram_count
        uni_probs[unigram] = prob
    
    if bigram_counts is None: # Use 'is None' for checking None
        return uni_probs
    
    bi_probs = {}
    for bigram, count in bigram_counts.items():
        # Ensure the unigram count for the first element of the bigram exists to avoid KeyError
        if bigram[0] in unigram_counts and unigram_counts[bigram[0]] > 0:
            prob = count / unigram_counts[bigram[0]]
        else:
            prob = 0.0 # Assign 0 or handle smoothing for unseen unigrams
        bi_probs[bigram] = prob         
    
    if trigram_counts is None:
        return uni_probs, bi_probs
    
    tri_probs = {}
    for trigram, count in trigram_counts.items():
        # Ensure the bigram count for the first two elements of the trigram exists and is not zero
        prefix_bigram = (trigram[0], trigram[1])
        if prefix_bigram in bigram_counts and bigram_counts[prefix_bigram] > 0:
            prob = count / bigram_counts[prefix_bigram]
        else:
            prob = 0.0 # Assign 0 or handle smoothing for unseen bigrams
        tri_probs[trigram] = prob
    
    if fourgram_counts is None:
        return uni_probs, bi_probs, tri_probs
    
    four_probs = {}
    for fourgram, count in fourgram_counts.items():
        # Ensure the trigram count for the first three elements of the fourgram exists and is not zero
        prefix_trigram = (fourgram[0], fourgram[1], fourgram[2])
        if prefix_trigram in trigram_counts and trigram_counts[prefix_trigram] > 0:
            prob = count / trigram_counts[prefix_trigram]
        else:
            prob = 0.0 # Assign 0 or handle smoothing for unseen trigrams
        four_probs[fourgram] = prob
    
    return uni_probs, bi_probs, tri_probs, four_probs


def simple_interpolate(key, 
                       lambdas,
                       uni_probs, 
                       bi_probs = None, 
                       tri_probs = None, 
                       four_probs = None):
    
    
    if bi_probs == None:
        print("interpolate needs at least two args")
        return 0

    if sum(lambdas) != 1.0:
        print("lambdas need to sum to 1.0")
        return 0
    
    value = 0

    # uni gram
    value += lambdas[0] * uni_probs[key[0]]
    
    # bi gram
    value += lambdas[1] * bi_probs[(key[0], key[1])]

    # tri gram
    if tri_probs:
        value += lambdas[2] * tri_probs[(key[0], key[1], key[2])]

    # four gram
    if four_probs:
        value += lambdas[3] * four_probs[(key[0], key[1], key[2], key[3])]
    

    return value


def split_corpus_into_n(text, n):

    splits = []
    for i in range(len(text) - (n-1)):
        splits.append(text[i:i+n])
    
    return splits



train = load('Shakespeare_clean_train.txt')
test = load('Shakespeare_clean_test.txt')
validation = load('Shakespeare_clean_valid.txt')

vocab_file = 'vocab_k-800.pkl'
if os.path.exists(vocab_file):
    print(f"Loading vocabulary from {vocab_file}...")
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    corpus = None 
else:
    print("Vocabulary file not found. Generating vocabulary...")
    vocab, corpus = byte_pair_encoding(train, 800)
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_file}")

corpus2 = merge_corpus(vocab,test)

four_gram = get_n_gram(corpus2, 4)
#print(four_gram)

snippet = corpus2[:10]
print("test corpus : ", snippet, "\n")

test_corpus = split_corpus_into_n(snippet, 3)

val = 0
for text in test_corpus:
    tmp = simple_interpolate(key=text, 
                   lambdas=[0.2, 0.2, 0.6],
                   uni_probs=four_gram[0],
                   bi_probs=four_gram[1],
                   tri_probs=four_gram[2]
                   )
    
    print(f"prob for : {text} : ", tmp)

    val += tmp

print("final probs : ", val)
