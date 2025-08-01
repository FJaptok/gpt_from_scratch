from byte_pair_encoding import load,create_corpus,get_vocab,segment,byte_pair_encoding
import math
import time

class n_gram:
    """ A clss to create an N-gram for a given n """
    def __init__(self, n):
        self.n                      = n
        self.probabilities          = {}
        self.inter_probabilities    = {}
        self.vocab                  = []
    
    def get_probabilities(self, corpus):
        """ This function takes a corpus and sets the probabilities for all possible combinations accordingly """
        # get the counts for all occuring combinations
        counts = self.count_combinations(corpus)
        self.vocab = get_vocab(corpus)

        # we consider all occuring combinations given by "counts" 
        for tokens in counts:
            # add an exception for the uni-gram probabilities since they are calculated differently
            if len(tokens) > 1:
                # since we excluded count combinations from "counts" this condition should never be true but it is still a save
                if counts[tokens[:-1]] == 0:
                    self.probabilities[tokens] = 0.0
                else:
                    # and calculate the probabilities accordingly with the full conditional probability and 
                    # the conditional probability for the combined token - the last token
                    # added + 1 and + len(self.vocab) as laplace smoothing
                    self.probabilities[tokens] = (counts[tokens] + 1) / (counts[tokens[:-1]] + len(self.vocab))
            else:
                # uni-gram probabilities are calculated with the number of occurences of the token over the whole corpus
                self.probabilities[tokens] = (counts[tokens] + 1) / (len(corpus) + len(self.vocab))

    
    def interpolate(self, corpus, parameters):
        """ This function computes the interpolated probabilities. 
        parameters[0] gives the lambda value for the uni-gram probabilities, parameters[1] for the bi-gram probabilities etc """

        # o if not enough lambdas were given
        assert len(parameters) == self.n, f"Number of lambdas given: {len(parameters)}, does not match the model's N: {self.n}"
        # raise error if the given lambdas do not sum up to 1.0
        assert sum(parameters) == 1.0, f"The lambda values must sum up to 1.0"

        # calculate the probabilities for the given corpus 
        self.get_probabilities(corpus)
        self.inter_probabilities = {}

        # calculate the interpolated probabilities
        for token in self.probabilities:
            new_prob = 0.0
            # but only for the conditional token probabilities of the correct length
            if len(token) == self.n:
                for i in range(len(token)):
                    # calculate the new probabilities and use the given labdas from "parameters" for the interpolated probabilities
                    new_prob += parameters[i]*self.probabilities[token[-(i+1):]]
                self.inter_probabilities[token] = new_prob
            else:
                pass
    
    def count_combinations(self, corpus):
        """ Counts the occurences of all tokens inside a given corpus as a dictionary """
        counts = {}
        
        for i in range(len(corpus)):
            for j in range(self.n):
                try:
                    if len(corpus[i-j:i+1]) != 0:
                        if tuple(corpus[i-j:i+1]) in counts:
                            counts[tuple(corpus[i-j:i+1])] += 1
                        else:
                            counts[tuple(corpus[i-j:i+1])] = 1
                except:
                    pass
        return counts
    
    def generate(self, start_token, max_iter, use_inter = False):
        """ Generates and returns a string with the interpolated or normal probabilities starting with a single given token and for a given token length. """
        
        generation = [start_token]

        # use interpolated probabilities for generation if the flag is True
        if use_inter:
            for _ in range(max_iter):
                possible = {}
                # gather the context for the model's N
                context = tuple(generation[-self.n+1:])

                for token in self.inter_probabilities:
                    # check all probabilities and collect the ones starting with the tokens inside the context window
                    # the second condition exists because we only consider probabilities where something follows the given context
                    if token[:len(context)] == context and len(token) > len(context):
                        possible[token] = self.inter_probabilities[token]

                # get the most probable following token/s
                sorted_dict = dict(sorted(possible.items(), key=lambda item: item[1]))
                most_common = next(iter(sorted_dict))

                # small for loop if we want to append multiple new tokens at once
                for subword in most_common[len(context):]:
                    generation.append(subword)
        
        else:
            for _ in range(max_iter):
                possible = {}
                # gather the context for the model's N
                context = tuple(generation[-self.n+1:])

                for token in self.probabilities:
                    # check all probabilities and collect the ones starting with the tokens inside the context window
                    # the second condition exists because we only consider probabilities where something follows the given context
                    if token[:len(context)] == context and len(token) > len(context):
                        possible[token] = self.probabilities[token]

                # get the most probable following token/s
                sorted_dict = dict(sorted(possible.items(), key=lambda item: item[1]))
                most_common = next(iter(sorted_dict))

                # small for loop if we want to append multiple new tokens at once
                for subword in most_common[len(context):]:
                    generation.append(subword)

        return ''.join(generation)


def perplexity(corpus, n_gram, use_inter = False):
    """ This function computes the perplexity for a given n_gram and corpus and can use the standard probabilities or the interpolated ones """
    if use_inter:
        for i in range(len(corpus)):
            if i == n_gram.n-1:
                if tuple(corpus[i+1-n_gram.n:i+1]) in n_gram.inter_probabilities:
                    prblty = math.log(n_gram.inter_probabilities[tuple(corpus[i+1-n_gram.n:i+1])])
                else:
                    n_gram.inter_probabilities[tuple(corpus[i+1-n_gram.n:i+1])] = 1 / len(n_gram.vocab)
                    prblty = math.log(n_gram.inter_probabilities[tuple(corpus[i+1-n_gram.n:i+1])])
            if i > n_gram.n-1:
                if tuple(corpus[i+1-n_gram.n:i+1]) in n_gram.inter_probabilities:
                    prblty += math.log(n_gram.inter_probabilities[tuple(corpus[i+1-n_gram.n:i+1])])
                else:
                    n_gram.inter_probabilities[tuple(corpus[i+1-n_gram.n:i+1])] = 1 / len(n_gram.vocab)
                    prblty += math.log(n_gram.inter_probabilities[tuple(corpus[i+1-n_gram.n:i+1])])
        perplexity = math.exp((-1/len(corpus))*prblty)

    else:
        for i in range(len(corpus)):
            if i == n_gram.n-1:
                if tuple(corpus[i+1-n_gram.n:i+1]) in n_gram.probabilities:
                    prblty = math.log(n_gram.probabilities[tuple(corpus[i+1-n_gram.n:i+1])])
                else:
                    n_gram.probabilities[tuple(corpus[i+1-n_gram.n:i+1])] = 1 / len(n_gram.vocab)
                    prblty = math.log(n_gram.probabilities[tuple(corpus[i+1-n_gram.n:i+1])])
            if i > n_gram.n-1:
                if tuple(corpus[i+1-n_gram.n:i+1]) in n_gram.probabilities:
                    prblty += math.log(n_gram.probabilities[tuple(corpus[i+1-n_gram.n:i+1])])
                else:
                    n_gram.probabilities[tuple(corpus[i+1-n_gram.n:i+1])] = 1 / len(n_gram.vocab)
                    prblty += math.log(n_gram.probabilities[tuple(corpus[i+1-n_gram.n:i+1])])
        perplexity = math.exp((-1/len(corpus))*prblty)
    
    return perplexity


if __name__ == '__main__':
    train = load('Shakespeare_clean_train.txt')
    test = load('Shakespeare_clean_test.txt')
    val = load('Shakespeare_clean_valid.txt')

    train_corpus = create_corpus(train)
    test_corpus = create_corpus(test)
    val_corpus = create_corpus(val)

    vocab, segmented_train = byte_pair_encoding(train, 100)
    segmented_test = segment(vocab,test)
    segmented_val = segment(vocab,val)


    uni_gram = n_gram(1)
    uni_gram.get_probabilities(segmented_train)

    perp = perplexity(segmented_test, uni_gram, False)
    print("uni-gram: ", perp)
    perp = perplexity(segmented_test, uni_gram, True)
    print("uni-gram inter: ", perp)


    bi_gram = n_gram(2)
    #bi_gram.get_probabilities(segmented_train)
    bi_gram.interpolate(segmented_train, [0.3,0.7])
    generated = bi_gram.generate('the_', 20)

    perp = perplexity(segmented_test, bi_gram, False)
    print("Bi-gram: ", perp)
    perp = perplexity(segmented_test, bi_gram, True)
    print("Bi-gram inter: ", perp)
    print("Bi_gram generated: ", generated)


    tri_gram = n_gram(3)
    #tri_gram.get_probabilities(segmented_train)
    tri_gram.interpolate(segmented_train, [0.2, 0.6, 0.2])
    generated = tri_gram.generate('the_', 20)

    perp = perplexity(segmented_test, tri_gram, False)
    print("tri-gram: ", perp)
    perp = perplexity(segmented_test, tri_gram, True)
    print("tri-gram inter: ", perp)
    print("tri_gram generated: ", generated)


    four_gram = n_gram(4)
    #four_gram.get_probabilities(segmented_train)
    four_gram.interpolate(segmented_train, [0.1, 0.4, 0.2, 0.3])
    generated = four_gram.generate('the_', 20)

    perp = perplexity(segmented_test, four_gram, False)
    print("four-gram: ", perp)
    perp = perplexity(segmented_test, four_gram, True)
    print("four-gram inter: ", perp)
    print("four_gram generated: ", generated)


    five_gram = n_gram(5)
    #four_gram.get_probabilities(segmented_train)
    five_gram.interpolate(segmented_train, [0.1, 0.4, 0.1, 0.1, 0.3])
    generated = five_gram.generate('the_', 20)

    perp = perplexity(segmented_test, five_gram, False)
    print("five-gram: ", perp)
    perp = perplexity(segmented_test, five_gram, True)
    print("five-gram inter: ", perp)
    print("five_gram generated: ", generated)


    six_gram = n_gram(6)
    #four_gram.get_probabilities(segmented_train)
    six_gram.interpolate(segmented_train, [0.1, 0.3, 0.1, 0.1, 0.2, 0.2])
    generated = six_gram.generate('the_', 20)

    perp = perplexity(segmented_test, six_gram, False)
    print("six-gram: ", perp)
    perp = perplexity(segmented_test, six_gram, True)
    print("six-gram inter: ", perp)
    print("six_gram generated: ", generated)