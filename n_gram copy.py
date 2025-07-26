import re
import tqdm
from collections import Counter
from byte_pair_encoding import load,create_corpus,get_vocab,merge_corpus,byte_pair_encoding
import pickle
import os 

train = load('Shakespeare_clean_train.txt')
test = load('Shakespeare_clean_test.txt')
#print(type(test))
validation = load('Shakespeare_clean_valid.txt')

#vocab_file = 'vocab_k-800.pkl'
vocab_file = 'vocab.pkl'
if os.path.exists(vocab_file):
    print(f"Loading vocabulary from {vocab_file}...")
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    corpus = None 
else:
    print("Vocabulary file not found. Generating vocabulary...")
    vocab, corpus = byte_pair_encoding(train, 50)
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_file}")

corpus2 = merge_corpus(vocab,train)
#corpus_test = merge_corpus(vocab,test)

#print(corpus2)


def get_n_gram(corpus, n):
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


def calc_prob(counts,  n_gram, v, laplace=True):
    prob = 1 
    n_gram = tuple(n_gram)
    print(n_gram)
    #print(counts)
    if len(n_gram) == 1:
        if laplace:
            prob = counts[0].get(n_gram) + 1 / sum(counts[0].values()) + v
        else:
            try:
                prob = counts[0].get(n_gram) / sum(counts[0].values())
            except:
                prob = 1
    elif laplace:
        prob = counts[-1].get(n_gram) + 1 / counts[-2].get(n_gram[0:-1]) + v
    else:
        try:
            prob = counts[-1].get(n_gram) / counts[-1].get(n_gram[0:-1])
        except:
            prob = 1 
    return prob


def calc_perplexity(sequence, counts, interpolate=False, lambdas = None):
    #sequence = create_corpus(sequence)
    sequence = merge_corpus(vocab, sequence)
    v = len(vocab)
    prob = 1
    if len(counts) == 1:
        for i in range(len(sequence)):
            if interpolate and lambdas is not None:
                pass
            else:
                prob *= calc_prob(counts, sequence[i], v) 
    if len(counts) == 2:
        prob = calc_prob(counts, sequence[0], v, laplace=True)
        for i in range(1, len(sequence)):
            if interpolate and lambdas is not None:
                pass
            else:
                prob *= calc_prob(counts, sequence[i-1:i], v)
    if len(counts) == 3:
        prob = calc_prob(counts, sequence[0], v) * calc_prob(counts, sequence[0:1], v)
        for i in range(2, len(sequence)):
            if interpolate and lambdas is not None:
                pass
            else:
                prob *= calc_prob(counts, sequence[i-2:i], v)
    if len(counts) == 4:
        prob = calc_prob(counts, sequence[0], v) * calc_prob(counts, sequence[0:1], v) * calc_prob(counts, sequence[0:2], v)
        for i in range(3, len(sequence)):
            if interpolate and lambdas is not None:
                pass
            else:
                prob *= calc_prob(counts, sequence[i-2:i], v)
    return prob**(-1/len(sequence))


def simple_interpolate(key, 
                       lambdas,
                       n_grams):
    
    
    if n_grams[1] == None:
        print("interpolate needs at least two args")
        return 0

    if sum(lambdas) != 1.0:
        print("lambdas need to sum to 1.0")
        return 0
    
    value = 0

    # uni gram
    value += lambdas[0] * n_grams[0][key[0]]
    
    # bi gram
    value += lambdas[1] * n_grams[1][(key[0], key[1])]

    # tri gram
    try:
        value += lambdas[2] * n_grams[2][(key[0], key[1], key[2])]
    except:
        pass

    # four gram
    try:
        value += lambdas[3] * n_grams[3][(key[0], key[1], key[2], key[3])]
    except:
        pass

    return value


def split_corpus_into_n(text, n):

    splits = []
    for i in range(len(text) - (n-1)):
        splits.append(text[i:i+n])
    
    return splits

def call_interpolation_on_string(text, models, lambdas):
    
    value = 0
    i = 0
    while i < len(text):
        split_text = text[i:i+len(models)]
        value = simple_interpolate(split_text, lambdas=lambdas, n_grams=models)
        print(f"probs for {split_text} : ", value)

        if i == 10:
            break

        i += 1


uni_gram, bi_gram, tri_gram = get_n_gram(corpus2, 3)

models = [uni_gram, bi_gram, tri_gram]


lambdas = [0.2, 0.2, 0.6]
call_interpolation_on_string(corpus2, models, lambdas)


perplexity = calc_perplexity("".join(corpus2[:10]), models)

print(perplexity)

