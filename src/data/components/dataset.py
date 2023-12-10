from torch.utils.data import Dataset
import os
from src.data.components.vocab import Vocab
from torch.nn.functional import pad
import torch 
import numpy as np

class LaosDataset(Dataset):
    def __init__(self,
                 data_dir:str,
                 file_type:str,
                 suffix:list,
                 input_vocab: Vocab,
                 target_vocab: Vocab):
        self.data_dir = data_dir
        self.type = file_type
        self.suffix = suffix
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.data = self.prepare()

        print(self.input_vocab, self.target_vocab)
        
        
    def prepare(self):
        print("Loading dataset")
        plugins = []
        for name in self.suffix:
            path = os.path.join(self.data_dir, self.type + "_" + name + ".dat")
            with open(path, "r") as file:
                for line in file:
                    try:
                        x, y = line.strip().split("\t")
                    except:
                        print(line)
                        raise ValueError("Cannot parse")
                    plugins.append(self.encode(x, y))
        
        return plugins
    
    def encode(self, inp, tgt):
        inp_ids = self.input_vocab.to_index(self.input_vocab.tokenize(inp))[0]
        tgt_ids = self.target_vocab.to_index(self.target_vocab.tokenize(tgt))[0]
        return {'input': inp_ids, 'target': tgt_ids}

            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        return self.data[index]

class Collator:
    def __init__(self, masked_language_model=False, pad_val=0):
        self.masked_language_model = masked_language_model
        self.pad_val = pad_val
        
    def __call__(self, batch):
        inputs = []
        targets = []
        input_lengths = []
        target_lengths = []
        max_label_len = max(sample["target"].size(0) for sample in batch)
        max_input_len = max(sample["input"].size(0) for sample in batch)
        target_len = max(max_label_len, max_input_len)
        
        if self.pad_val is None:
            self.pad_val = 0
            
        for sample in batch:
            # padding and append
            inp_length = sample['input'].size(0)
            inp = pad(sample['input'], 
                         (0, target_len - inp_length), 
                         'constant',
                         value=self.pad_val)
            input_lengths.append(inp_length)
            inputs.append(inp)            
            
            
            tgt_length = sample['target'].size(0)
            tgt= pad(sample['target'],
                        (0, target_len - tgt_length),
                        'constant',
                        value=self.pad_val)
            targets.append(tgt)
            target_lengths.append(tgt_length)
            
        tgt = torch.stack(targets)
        inp = torch.stack(inputs)
        return {"inputs": torch.stack(targets),
                "targets": torch.stack(inputs),
                "input_lengths": torch.IntTensor(input_lengths),
                "target_lengths": torch.IntTensor(target_lengths)}
        
if __name__ == "__main__":
    lao_vocab = Vocab(vocab_path="/work/hpc/potato/laos_vi/data/embedding/laos_glove_dict.txt",
                      weights_path="/work/hpc/potato/laos_vi/data/embedding/laos_glove_v100d.pt",
                      stride=0,
                      tokenizer='lao',
                      init_special_symbol=False)
    vi_vocab = Vocab(vocab_path="/work/hpc/potato/laos_vi/data/embedding/vi_dict.txt",
                     stride=0)
    dataset = LaosDataset(  data_dir="/work/hpc/potato/laos_vi/data/label/",
                            file_type="train",
                            suffix=["clean"],
                            input_vocab=lao_vocab,
                            target_vocab=vi_vocab
                            )
    
    data = dataset.__getitem__(5)
    print(data)
    
    