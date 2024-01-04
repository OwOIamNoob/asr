from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
from src.data.components.vocab import Vocab
from torch.nn.functional import pad
import torch 
import numpy as np
from collections import defaultdict
import random

class LaosDataset(Dataset):
    def __init__(self,
                 data_dir:str,
                 file_type:str,
                 suffix:list,
                 input_vocab: Vocab,
                 target_vocab: Vocab):
        super().__init__()
        self.data_dir = data_dir
        self.type = file_type
        self.suffix = suffix
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.cluster_indicies = defaultdict(list)
        self.data = self.prepare()

        print(self.input_vocab, self.target_vocab)
        
        
    def prepare(self):
        print("Loading dataset")
        i = 0
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
                    plugin = self.encode(x, y)
                    cluster_id = max(plugin['input'].size(0), plugin['target'].size(0))
                    self.cluster_indicies[cluster_id].append(i)
                    i += 1
                    plugins.append(plugin)
        
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
    def __init__(self, masked_language_model=False, sos_id=0, eos_id=1, pad_id=[2], target_vocab_size=1, max_length=256):
        self.masked_language_model = masked_language_model
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.target_vocab_size=target_vocab_size
        self.max_length = max_length
        
    def __call__(self, batch):
        inputs = []
        targets = []
        input_lengths = []
        target_lengths = []
        max_label_len = max(sample["target"].size(0) for sample in batch)
        max_input_len = max(sample["input"].size(0) for sample in batch)
        
        if max_label_len > self.max_length or max_input_len > self.max_length - 2:
            raise ValueError("String too big gawk gawk")
        
        target_len = max(max_label_len, max_input_len)

        for sample in batch:
            # padding and append
            inp = pad(sample['input'],
                      (0, 1),
                      'constant',
                      value=self.eos_id)
            inp_length = inp.size(0)
            
            input_lengths.append(inp.size(0))
            inp = pad(inp, 
                         (0, self.max_length - inp_length), 
                         'constant',
                         value=self.pad_id)
            
            inputs.append(inp)            
            
            tgt = pad(  sample['target'],
                        (0, 1),
                        'constant',
                        value=self.eos_id)
            tgt_length = tgt.size(0)
            target_lengths.append(tgt.size(0))
            tgt = pad(   tgt,
                        (0, self.max_length - tgt_length),
                        'constant',
                        value=self.pad_id)
            
            targets.append(tgt)
            
            
        # tgt = torch.stack(targets)
        # inp = torch.stack(inputs)
        return {"targets":  torch.stack(targets).type(torch.int64),
                "inputs": torch.stack(inputs).type(torch.int64),
                "input_lengths": torch.LongTensor(input_lengths),
                "target_lengths": torch.LongTensor(target_lengths)}

class ClusterSampler(Sampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_added_sample = 0
    
    def __iter__(self):
        batch_list = []
        for seq_length, indicies in self.dataset.cluster_indicies.items():
            if self.shuffle:
                random.shuffle(indicies)

            for i in range(0, len(indicies), self.batch_size):
                batch_list += self.fill(indicies[i:i + self.batch_size])
        return iter(batch_list)
    
    def __len__(self) -> int:
        return len(self.dataset) + self.total_added_sample
    
    def fill(self, batch):
        if len(batch) == self.batch_size:
            return batch
        
        addition = self.batch_size - len(batch)
        self.total_added_sample += addition
        
        return batch + random.choices(batch, k=addition)
    
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
    
    