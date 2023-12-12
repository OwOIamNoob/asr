import torch
import numpy as np
import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.data.components.vocab import Vocab


file = open("/work/hpc/potato/laos_vi/data/embedding/pyvi_dict.txt", "r")
pyvi_vocab = set()
for line in file:
    parts = line.strip().split()
    pyvi_vocab.add(('_'.join(parts), ''.join(parts), ' '.join(parts)))

print(len(pyvi_vocab))
# print(pyvi_vocab)
idx = np.loadtxt("/work/hpc/potato/laos_vi/data/embedding/vi_reduced.txt").astype(int)
# indexes = set(idx)
vocab = Vocab(vocab_path="/work/hpc/potato/laos_vi/data/embedding/vi_dict.txt",
              weights_path="/work/hpc/potato/laos_vi/data/embedding/vi_dict.pt",
              stride=3,
              init_special_symbol=False,
              tokenizer='vi')
print(idx.max(), len(vocab.idx_to_text))
dataset_vocab = vocab.decode(idx)
vi_vocab = dict()
for words in list(pyvi_vocab):
    index, unk = vocab.to_index(words)
    for token in index:
        if token != 2 and token not in vi_vocab:
            vi_vocab[token] = words[0]
            break

for word, idx in zip(dataset_vocab, idx):
    if idx not in vi_vocab.keys():
        vi_vocab[idx] = word
# vi_vocab[0] = '<sos>'
print(len(vi_vocab))
# if len(unk) > 0:
#     print(unk)
#     raise ValueError("Unexpected token")
output = open("/work/hpc/potato/laos_vi/data/embedding/vi_reduce_remap.txt", "w")
prev_idx = -1
indexes = []
for index, word in sorted(zip(vi_vocab.keys(), vi_vocab.values()), key=lambda a: a[0]):
    if index == prev_idx:
        continue
    prev_idx = index
    output.write("\n" + word + "\t" + str(len(indexes)))
    indexes.append(index)

indexes = torch.LongTensor(indexes)
embed = vocab.embed(indexes)
torch.save(embed, "/work/hpc/potato/laos_vi/data/embedding/vi_reduce_remap.pt")
output.close()
file.close()
     

