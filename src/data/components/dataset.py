from torch.utils.data import Dataset
import os
import laonlp
import pyvi
import pythainlp
from tokenizer.LaoNLP.laonlp.corpus.lao_words import lao_words

laos_embedding = laonlp.word_vector.Word2Vec(model="skip-gram"),
laos_tokenizer = pythainlp.tokenize.Tokenizer(lao_words(), engine="mm"),
vi_tokenizer = pyvi.ViTokenizer()

class LaosDataset(Dataset):
    def __init__(self,
                 data_dir:str,
                 file_type:str,
                 suffix:list,
                 concat:bool):
        self.data_dir = data_dir
        self.type = file_type
        self.suffix = suffix
        self.concat = concat
        self.data = self.prepare()
        
    def prepare(self):
        plugins = []
        for name in self.suffix:
            path = os.path.join(self.data_dir, self.type + name)
            with open(path, "r") as file:
                for line in file:
                    x, y = line.strip().split("\t")
                    plugins.append({'x':x, 'y':y})
        
        return plugins
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        return self.data[index]
    
class LaosTransformedDataset(Dataset):
    def __init__(self, 
                 dataset: Dataset,
                 task: str,
                 src_embedding = laos_embedding,
                 src_tokenizer = laos_tokenizer,
                 target_tokenizer = vi_tokenizer):
        self.dataset = dataset
        self.task = task 
        self.src_embedding = src_embedding
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.target_embedding  = 
        