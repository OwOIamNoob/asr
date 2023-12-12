from laonlp.word_vector.word2vec import Word2Vec
import numpy as np
import torch
# import gensim.downloader

dict1 = open("/work/hpc/potato/laos_vi/data/embedding/laos_fix.txt", "r")
vocab1_size, dim1, stride1 = dict1.readline().strip().split()
dim1 = int(dim1)
vocab1_size = int(vocab1_size)
stride1 = int(stride1)
dict2 = open("/work/hpc/potato/laos_vi/data/embedding/glove_dictionary.txt", "r")
vocab2_size, dim2, stride2 = dict2.readline().strip().split()
dim2 = int(dim2)
vocab2_size = int(vocab2_size)
stride2 = int(stride2)
vocab = dict()

spacing = vocab1_size
stride = 2
out_file = open("/work/hpc/potato/laos_vi/data/embedding/laos_glove_dict.txt", "w")
out_file.write(str(vocab1_size + vocab2_size - stride2 + stride) + "\t" + str(dim2) + "\t" + str(stride1 + stride))
for line in dict1:
    parts = line.strip().split("\t")
    id = int(parts[-1]) + stride
    word = " ".join(parts[:-1])
    out_file.write("\n" + word + "\t" + str(id))
    
for line in dict2:
    parts = line.strip().split("\t")
    id = int(parts[-1]) + spacing + stride1 + stride
    word = " ".join(parts[:-1])
    out_file.write("\n" + word + "\t" + str(id))

out_file.close()
dict1.close()
dict2.close()



ts1 = torch.load("/work/hpc/potato/laos_vi/data/embedding/laos_v100d_test.pt")
ts2 = torch.load("/work/hpc/potato/laos_vi/data/embedding/glove_v100d.pt")
print(ts1.size(), ts2.size())
sos = torch.full((1, 100), -1, dtype=torch.float32)
eos = torch.full((1, 100), 1, dtype=torch.float32)
ts = torch.cat((sos, eos, ts1, ts2))
print(ts.size())
torch.save(ts, "/work/hpc/potato/laos_vi/data/embedding/laos_glove_v100d.pt")
    
    
    