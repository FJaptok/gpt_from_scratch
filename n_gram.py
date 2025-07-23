from byte_pair_encoding import create_corpus, merge_corpus, load




def load_pickle(file_path):
    """ Loads a vocabulary from a pickle file """
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def uni_gram(corpus, vocab, norm="laplace"):
    """ Merges unigrams in a given corpus according to a given vocabulary """
    counter = {}
    if norm=="laplace":
        increment = 1
    else:
        increment = 0
    
    for token in corpus:
        if token in vocab:
            counter[token] = counter.get(token, increment) + 1
    return counter

def bi_gram(corpus, vocab, norm="laplace"):
    """ Merges bigrams in a given corpus according to a given vocabulary """
    counter = {}
    if norm == "laplace":
        increment = 1
    else:
        increment = 0
    
    for token_1 in vocab:
        for token_2 in vocab:
            token = (token_1, token_2)
            counter[token] = increment
    
    
    for i in range(len(corpus) - 1):
        token = (corpus[i] , corpus[i + 1])
        if token in counter:
            counter[token] += 1
    return counter

path_2_pickle = 'vocab.pickle'
path_2_train = "Shakespeare_clean_train.txt"
# load pickle vocab
created_vocab = load_pickle(path_2_pickle)

#train = load(path_2_train)

#corpus2 = create_corpus(train)

#corpus3 = merge_corpus(created_vocab,corpus2)
#import pickle
#with open('corpus3.pickle', 'wb') as f:
#    pickle.dump(corpus3, f)

corpus3 = load_pickle('corpus3.pickle')

#print(corpus3[:1000]) 

counter_uni = uni_gram(corpus3, created_vocab)
#print(counter_uni)

counter_bi = bi_gram(corpus3, created_vocab, "laplace")
#print(counter_bi)

joined_probability = []

i = 0
for pair in counter_bi:
    
    token_1, token_2 = pair
    
    uni_prob = counter_uni[token_1]
    #print("uni prob :", uni_prob)
    
    bi_prob = counter_bi[pair]
    #print("bi prob : ", bi_prob)

    print(f"joined prob {pair} :", bi_prob / uni_prob)

    if i == 100:
        break
    i += 1

