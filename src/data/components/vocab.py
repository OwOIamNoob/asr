import numpy as np
import torch
from torch.nn import Embedding
import pyvi.ViTokenizer
import re
import time
import os
#### tokenizer
import pythainlp
from laonlp.corpus import lao_words



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
    "ຆ": "ฆ",  # PALI GHA
    "ຉ": "ฉ",  # PALI CHA
    "ຌ": "ฌ",  # PALI JHA
    "ຎ": "ญ",  # PALI NYA
    "ຏ": "ฏ",  # PALI TTA
    "ຐ": "ฐ",  # PALI TTHA
    "ຑ": "ฑ",  # PALI DDA
    "ຒ": "ฒ",  # PALI DDHA
    "ຓ": "ณ",  # PALI NNA
    "ຘ": "ธ",  # PALI DHA
    "ຠ": "ภ",  # PALI BHA
    "ຨ": "ศ",  # SANSKRIT SHA
    "ຩ": "ษ",  # SANSKRIT SSA
    "ຬ": "ฬ",
    }

def replace_all(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text



############################## vocab ###############################
class Vocab:
    def __init__(self,
                 ckpt_path:str = "",
                 vocab_path:str = "",
                 weights_path:str = "",
                 stride: int = 0,
                 init_special_symbol: bool = True,
                 tokenizer:str = "vi",
                 device: str = "cpu"):
        self.weights = []
        self.stride = stride
        self.embedder = None
        self.vocab = dict()
        self.idx_to_text = dict()
        self.vocab_size = 0
        self.device = device
        
        self.export = {0:1000000, 1:1000000, 2:1000000}
        self.library = dict()
        
        
        if not os.path.isfile(ckpt_path) and not os.path.isfile(vocab_path) and not os.path.isfile(weights_path):
            raise AssertionError("What the heck do u want me to do ?")
        if not os.path.isfile(vocab_path) and not os.path.isfile(weights_path): 
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
        elif not os.path.isfile(weights_path):
            self.load_dict(vocab_path)
        else:
            self.load_weights(vocab_path, weights_path, device)
        
        self.idx_to_text = dict(zip(self.vocab.values(), self.vocab.keys()))
        
        if tokenizer == 'lao':
            for word in lao_words():
                if self.vocab.keys().__contains__(word):
                    self.export[self.vocab[word]] = 1
            print(len(self.export))
            self.tokenizer = pythainlp.tokenize.Tokenizer(lao_words() + list(self.vocab.keys()), engine='longest')
        else:
            # file = open("/work/hpc/potato/laos_vi/data/embedding/pyvi_dict.txt", "r")
            # for line in file:
            #     parts = line.strip().split()
            #     words = ('_'.join(parts), ''.join(parts), ' '.join(parts))
            #     for word in words:
            #         if self.vocab.keys().__contains__(word) is True:
            #             id = self.vocab[word]
            #             self.idx_to_text[id] = words[0]
            #             self.vocab[words[0]] = id
            #             self.export[id] = 1
            # print(len(self.export))
            # file.close()
            self.tokenizer = pythainlp.tokenize.Tokenizer(list(self.vocab.keys()))
        
        # self.idx_to_text = dict(zip(self.vocab.values(), self.vocab.keys()))
        
        
    def load_dict(self, vocab_path):
        file = open(vocab_path, "r")
        self.vocab = dict()
        header = file.readline()
        vocab_size, dim, stride = header.strip().split()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.stride = int(stride)
        print("Loading {} words with {} features with {} addition keys".format(self.vocab_size, self.dim, self.stride)) 
        for line in file:
            parts = line.strip().split()
            id = int(parts[-1])
            word = " ".join(parts[:-1])
            if word not in self.vocab:
                self.vocab[word] = id
        
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
            parts = line.strip().split()
            id = int(parts[-1])
            word = " ".join(parts[:-1])
            if word not in self.vocab.keys():
                self.vocab[word] = id
        self.weights = torch.load(weights_path).to(device)
        print("Weight dimension:", self.weights.size())
        self.embedder = Embedding.from_pretrained(self.weights, freeze=True, padding_idx=0, )
        pass
        
        
    def view(self, tokens):
        de_stash = [re.sub(r"_", " ", token) if len(token) > 2 else token for token in tokens ]
        return " ".join(de_stash)
    
    
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
        number = []
        checkpoint = 0
        same = False
        for idx in range(len(seq)):
            if idx == len(seq) - 1:
                number.append(seq[checkpoint:idx + 1])
            elif seq[idx + 1].isnumeric() != same or seq[idx + 1].isnumeric() is True:
                number.append(seq[checkpoint:idx + 1])
                checkpoint = idx + 1
                same = seq[idx + 1].isnumeric()
        # print(number)
        return number  
    
    def pyvi_join(self, tokens):
        checkpoint = 0
        final = []
        for i in range(len(tokens)):
            if i == len(tokens) - 1:
                string = ''.join(tokens[checkpoint:])
                # checkpoint = i + 1
                final.append(string)
            elif tokens[i] == ' ':
                string = ''.join(tokens[checkpoint:i])
                checkpoint = i + 1
                final.append(string)
            
        return final
        
        
    def tokenize(self, seq):
        seq = seq.lower()
        seq = replace_all(seq, dict_map)
        # seq = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", seq)
        # print(seq)
        tokens = []
        if isinstance(self.tokenizer, pythainlp.tokenize.Tokenizer):
            # print("Lao tokenizing")
            tokens = self.tokenizer.word_tokenize(seq)
        else:
            tokens = self.tokenizer.tokenize(seq)
            tokens = self.pyvi_join(tokens)
            
        final = []
        for token in tokens:
            token = re.sub(r" ", "", token)
            if token == ' ' or token == "":
                continue
            if token != '\u200b':
                token = token.strip(u'\u200b')
            if token in self.vocab:
                index = self.vocab[token]
                final += [token]
            else:
                final  += self.number_decomposition(token)
        # print(final)
        return final
        
    def get_weights(self):
        return self.weights
    
    def get_vocab(self):
        return self.vocab
        
    def to_index(self, tokens):
        ids = []
        skipped = []
        for token in tokens:
            if token != ' ':
                try:
                    id = int(self.vocab[token])
                    ids.append(id)
                    if self.export.keys().__contains__(id) is False:
                        self.export[id] = 1
                    else:
                        self.export[id] += 1
                except:
                    ids.append(self.vocab['<pad>'])
                    self.export[2] += 1
                    skipped.append(token)

        return torch.LongTensor(ids), skipped
    
    def embed(self, ids, target_device):
        # if not embedder:
        #     raise NotImplementError
        if not torch.is_tensor(ids):
            ids = torch.LongTensor(ids.copy()).to(self.device)
        elif ids.device != self.device:
            ids = ids.to(self.device)
        return self.embedder(ids).to(target_device)
    
    def get_used_vocab(self):
        return self.export
    
    def __len__(self):
        return self.vocab_size
    
    def get_emb(self):
        return self.embedder
    
    def to(self, device):
        self.device = device
        self.embedder.to(device)

    def decode(self, ids):
        return [self.idx_to_text[id] for id in ids]
    
    def get_topk(self, top_k: int):
        ids = sorted(list(self.export.items()), key= lambda a: a[1], reverse=True)
        words = [self.idx_to_text[id] for id, freq in ids]
        return np.array(ids), words
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
    
    vocab = Vocab(vocab_path="/work/hpc/potato/laos_vi/data/embedding/laos_glove_dict.txt", 
                  weights_path="/work/hpc/potato/laos_vi/data/embedding/laos_glove_v100d.pt",
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
    # file = open("/work/hpc/potato/laos_vi/data/label/train_clean.dat", "r")
    # laos = []
    # unknowns = []
    # for line in file:
    #     parts = line.strip().split("\t")
    #     tokens = vocab.tokenize(parts[0])
    #     print(tokens)
    #     indexes, unks = vocab.to_index(tokens)
    #     emb = vocab.embed(indexes)
    #     print(indexes)
    #     unknowns += unks
    #     # time.sleep(0.1)
    #     laos.append(parts[0])
    
    # file.close()
    # output= open("/work/hpc/potato/laos_vi/data/embedding/t_train_laos_unknowns.txt", "w")
    # for unk in unknowns:
    #     output.write(unk + "\n")
    
    # output.close()
    sample = "ຖ້າ ເຈົ້າ ຮູ້ສຶກ ຢ້ານ ທີ່ ຈະ ປະກາດ ຂໍ ໃຫ້ ຊ້ອມ ເວົ້າ ໃນ ແບບ ທີ່ ຍັງ ບໍ່ ມີ ເປົ້າ ຫມາຍ ປະກາດ I like you"
    tokens = vocab.tokenize(sample)
    print(tokens)
    indexes, unks = vocab.to_index(tokens)
    print(indexes, unks)
    emb = vocab.embed(indexes)
    print(emb.size(), emb)
    