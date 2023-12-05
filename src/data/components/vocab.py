import numpy as np
import torch
from torch.nn import Embedding
import re
import time
#### tokenizer
import pythainlp




############################## utils ###############################

# coding=utf-8
# Copyright (c) 2021 VinAI Research

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }

def replace_all(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text

############################## vocab ###############################
class Vocab:
    def __init__(self,
                 ckpt_path:str = None,
                 vocab_path:str = None,
                 weights_path:str = None,
                 stride: int = 0,
                 init_special_symbol: bool = True,
                 tokenizer:str = "vi",
                 device: str = "cpu"):
        self.weights = []
        self.stride = stride
        self.embedding = None
        self.vocab = dict()
        self.vocab_size = 0
        self.device = device
        if not ckpt_path and (not vocab_path or not weights_path):
            raise AssertionError("What the heck do u want me to do ?")
        if not vocab_path or not weights_path: 
            self.stride = 3
            self.weights = [0, 0, 0]
            self.load(ckpt_path, self.stride)
            if init_special_symbol:
                self.vocab['<pad>'] = 0
                self.weights[0] = np.full(self.dim, 0)
                self.vocab['<sos>'] = 1
                self.weights[1] = np.full(self.dim,2.5)
                self.vocab['<eos>'] = 2
                self.weights[2] = np.full(self.dim, 5)
                self.weights = np.vstack(self.weights)
            self.weights = torch.from_numpy(self.weights.astype(np.float32))
            self.embedder = Embedding.from_pretrained(self.weights, freeze=False, padding_idx=0)
        else:
            self.load_weights(vocab_path, weights_path, device)
        
        if tokenizer == 'vi':
            self.tokenizer = pyvi.ViTokenizer()
        elif tokenizer == 'lao':
            print("Init lao tokenizer")
            print(len(self.vocab.values()))
            self.tokenizer = pythainlp.tokenize.Tokenizer(self.vocab.keys(), engine='newmm')
        
        
        
        
    def load_weights(self, vocab_path, weights_path, device):
        file = open(vocab_path, "r")
        self.vocab = dict()
        header = file.readline()
        vocab_size, dim, stride = header.strip().split()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.stride = int(stride)
        print("Loading {} words with {} features with {} addition keys".format(self.vocab_size, self.dim, self.stride)) 
        for line in file:
            parts = line.split("\t")
            id = int(parts[-1])
            word = " ".join(parts[:-1])
            self.vocab[word] = id
        self.weights = torch.load(weights_path).to(device)
        print("Weight dimension:", self.weights.size())
        self.embedder = Embedding.from_pretrained(self.weights, freeze=False, padding_idx=0, )
        print(self.vocab[' '])
        pass
        
        
    def load(self, path, stride):
        file = open(path, "r")
        if file is None:
            raise FileNotFound("Not exist")
        header = file.readline()
        self.vocab_size, self.dim = header.strip().split()
        self.dim = int(self.dim)
        self.vocab_size = int(self.vocab_size)
        i = int(stride)
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < self.dim + 1:
                word = " "
            else:
                word = " ".join(parts[:len(parts) - self.dim])
                
            self.vocab[word] = i
            weight = np.array(parts[-self.dim:]).astype(np.float32)
            if weight.shape[0] != self.dim:
                print(weight.shape[0], self.dim)
                print(word)
                raise AssertionError("Embedding dimension not match")
            self.weights.append(weight)
            i += 1
            # stop point
            if i == self.vocab_size + stride:
                print("Done", i, "words")
                break
    
    def number_decomposition(self, seq):
        
            
    def tokenize(self, seq):
        seq = replace_all(seq, dict_map)
        # seq = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", seq)
        print(seq)
        tokens = []
        if isinstance(self.tokenizer, pythainlp.tokenize.Tokenizer):
            print("Lao tokenizing")
            tokens = self.tokenizer.word_tokenize(seq)
        else:
            tokens = self.tokenizer.tokenize(seq)
        tokens = [token.strip(u'\u200b') for token in tokens if token != '\u200b']
        print(tokens)

        return tokens
        
    
    def get_weights(self):
        return self.weights
    
    def get_vocab(self):
        return self.vocab
        
    def to_index(self, tokens):
        ids = []
        skipped = []
        for token in tokens:
            try:
                ids.append(self.vocab[token])
            except:
                skipped.append(token)
        return torch.LongTensor(ids), skipped
    
    def embed(self, ids):
        if not torch.is_tensor(ids):
            ids = torch.LongTensor(ids.copy()).to(self.device)
        elif ids.device != self.device:
            ids.to(self.device)
        return self.embedder(ids)
    
    def __len__(self):
        return self.vocab_size
    
    def get_emb(self):
        return self.embedder
    
    def to(device):
        self.device = device
        self.embedder.to(device)

def load_dict(path):
    file = open(path, "r")
    header = file.readline()
    vocab = []
    for line in file:
        parts = line.strip().split("\t")
        vocab.append(parts[0])
    
    return vocab

if __name__ == "__main__":
    # tokens = sample.strip().split(" ")
    
    vocab = Vocab(vocab_path="/work/hpc/potato/laos_vi/data/embedding/laos_fix.txt", 
                  weights_path="/work/hpc/potato/laos_vi/data/embedding/laos_v100d.pt",
                  stride=0, 
                  tokenizer='lao',
                  init_special_symbol=False)
    # vocab = Vocab(ckpt_path="/work/hpc/potato/laos_vi/data/embedding/word2vec_vi_words_100dims.txt",
    #               stride=0,
    #               init_special_symbol=True)
    # command this if u saved weights
    # dictionary = list(vocab.get_vocab().items())
    # f = open("data/embedding/vi_dictionary.txt", "w")
    # f.write(" ".join((str(vocab.vocab_size), str(vocab.dim), str(vocab.stride))))
    # for items in dictionary:
    #     f.write("\n" + " ".join([str(item) for item in items]))
    # f.close()
    # weights = vocab.get_weights()
    # torch.save(weights, "data/embedding/vi_emb.pt")
    file = open("/work/hpc/potato/laos_vi/data/label/train_clean.dat", "r")
    laos = []
    unknowns = []
    for line in file:
        parts = line.strip().split("\t")
        tokens = vocab.tokenize(parts[0])
        print(tokens)
        indexes, unks = vocab.to_index(tokens)
        emb = vocab.embed(indexes)
        print(indexes)
        unknowns += unks
        # time.sleep(0.1)
        laos.append(parts[0])
    
    file.close()
    output= open("/work/hpc/potato/laos_vi/data/embedding/train_laos_unknowns.txt", "w")
    for unk in unknowns:
        output.write(unk + "\n")
    
    output.close()
    # tokens = vocab.tokenize(sample)
    # indexes = vocab.to_index(tokens)
    # print(indexes)
    # emb = vocab.embed(indexes)
    # print(emb.size(), emb)
    