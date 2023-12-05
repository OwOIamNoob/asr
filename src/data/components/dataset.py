from torch.utils.data import Dataset
import os
import laonlp
import pyvi
import pythainlp
from tokenizer.LaoNLP.laonlp.corpus.lao_words import lao_words
from tokenizer.LaoNLP.laonlp.word_vector.word2vec import Word2Vec
from vocab import Vocab
laos_embedding = laonlp.word_vector.Word2Vec(model="skip-gram"),
laos_tokenizer = pythainlp.tokenize.Tokenizer(lao_words(), engine="mm"),
vi_tokenizer = pyvi.ViTokenizer()

class LaosDataset(Dataset):
    def __init__(self,
                 data_dir:str,
                 file_type:str,
                 suffix:list,
                 concat:bool,
                 input_tokenizer:pythainlp.tokenize.Tokenizer(lao_words(), engine="mm"),
                 target_tokenizer:pyvi.ViTokenizer,
                 input_vocab: Word2Vec,
                 target_vocab: Vocab):
        self.data_dir = data_dir
        self.type = file_type
        self.suffix = suffix
        self.concat = concat
        self.data = self.prepare()
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        
    def prepare(self):
        plugins = []
        for name in self.suffix:
            path = os.path.join(self.data_dir, self.type + name)
            with open(path, "r") as file:
                for line in file:
                    x, y = line.strip().split("\t")
                    plugins.append(self.encode(x, y))
        
        return plugins
    
    def encode(self, inp, tgt):
        inp_ids = self.input_vocab.to_index(self.input_tokenizer.word_tokenize(inp))
        tgt_ids = self.target_vocab.to_index(self.target_tokenizer.tokenize(tgt))
        return {'input': inp_ids, 'target': tgt_ids}

            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        return self.data[index]

class Collator(Object):
    def __init__(self, masked_language_model=False, pad_val=0):
        self.masked_language_model = masked_language_model
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.pad_val = pad_val
        
    def __call__(self, batch):
        inputs = []
        targets = []
        target_weights = []
        max_label_len = max(sample["target"].len() for sample in batch)
        max_input_len = max(sample["input"].len() for sample in batch)
        target_len = max(max_label_len, max_input_len)
        
        if pad_val is None:
            pad_val = 0
            
        for sample in batch:
            # padding and append
            inp = np.pad(inp, 
                         ((0, target_len - inp.len())), 
                         'constant',
                         constant_values=(pad_val,))
            inputs.append(inp)            
            
            tgt= np.pad(tgt,
                        ((0, target_len - sample["target"])),
                        'constant',
                        constant_values=(pad_val,))
            targets.append(tgt)

            
        tgt = np.array(targets, dtype=np.float32)
        inp = np.array(inputs, dtype=np.float32)
        
            
            
            
        