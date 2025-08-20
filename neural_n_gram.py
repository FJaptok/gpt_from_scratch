from byte_pair_encoding import load,create_corpus,get_vocab,segment,byte_pair_encoding
import numpy as np
import torch


def gen_mapping(sequence):
    int_to_ch = {}
    ch_to_int = {}
    chars = []

    for i in sequence:
        chars.append(i)
    
    chars = list(set(chars))

    for i,char in enumerate(chars):
        ch_to_int[char] = i
        int_to_ch[i] = char
    
    return ch_to_int, int_to_ch

def encode(sequence):
    ch_to_int, int_to_ch = gen_mapping(sequence)

    new_corpus = []

    for token in sequence:
        new_corpus.append(ch_to_int[token])
    
    return new_corpus, int_to_ch

def decode(sequence, dict):

    decoded = ""

    for token in sequence:
        decoded += dict[token]
    
    return decoded

def create_inputs(data, block_size, batch_size):
    data = torch.tensor(data, dtype=torch.long)
    ix = torch.randint(len(data)-block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y



class BigramLanguageModel:
    def __init__(self, vocab_size):
        self.look_up = np.random.rand(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.look_up(idx)
        


if __name__ == '__main__':
    train = load('Shakespeare_clean_train.txt')
    test = load('Shakespeare_clean_test.txt')
    val = load('Shakespeare_clean_valid.txt')

    train_corpus = create_corpus(train)
    test_corpus = create_corpus(test)
    val_corpus = create_corpus(val)

    encoding, int_to_ch = encode(train)
    print(encoding[:100])

    x,y = create_inputs(encoding, 5, 64)
    print("x: ",x)
    print("x.shape: ", x.shape)
    print("y: ",y)
    print("y.shape: ", y.shape)

    # decoding = decode(encoding, int_to_ch)
    # print(decoding[:100])