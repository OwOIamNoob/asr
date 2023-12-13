import torch
import numpy as np
import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.data.components.vocab import Vocab
from laonlp.corpus.lao_words import lao_words
import math
# file = open("/work/hpc/potato/laos_vi/data/embedding/pyvi_dict.txt", "r")
# pyvi_vocab = set()
# for line in file:
#     parts = line.strip().split()
#     pyvi_vocab.add(('_'.join(parts), ''.join(parts), ' '.join(parts)))

# print(len(pyvi_vocab))
# print(pyvi_vocab)
laos_vocab = ['<sos>', "<eos>", "<pad>"] + lao_words() 
dataset_idx = sorted(np.loadtxt("/work/hpc/potato/laos_vi/data/embedding/laos_reduced.txt").astype(int).tolist())
# indexes = set(idx)
vocab = Vocab(vocab_path="/work/hpc/potato/laos_vi/data/embedding/laos_glove_v100d.txt",
              weights_path="/work/hpc/potato/laos_vi/data/embedding/laos_glove_v100d.pt",
              stride=3,
              init_special_symbol=False,
              tokenizer='lao')
print(vocab.idx_to_text[0], vocab.idx_to_text[1], vocab.idx_to_text[2])
print(vocab.vocab["<eos>"], vocab.vocab["<sos>"], vocab.vocab["<pad>"])
dataset_vocab = vocab.decode(dataset_idx)
print(dataset_vocab[:3], dataset_idx[:3])
laos_vocab += dataset_vocab
output_vocab = dict()
for word in laos_vocab:
    index, unk = vocab.to_index([word])
    if len(index) > 0 and index[0] not in output_vocab:
        output_vocab[index[0]] = word
    else:
        print("Skip")

for word, idx in zip(dataset_vocab, dataset_idx):
    if output_vocab.__contains__(idx) is False:
        output_vocab[idx] = word
# vi_vocab[0] = '<sos>'
print("Total size:", len(output_vocab))
# if len(unk) > 0:
#     print(unk)
#     raise ValueError("Unexpected token")
output = open("/work/hpc/potato/laos_vi/data/embedding/laos_reduce_remap.txt", "w")
prev_idx = -1
indexes = []
for index, word in sorted(zip(output_vocab.keys(), output_vocab.values()), key=lambda a: a[0]):
    if index == prev_idx:
        continue
    prev_idx = index
    output.write("\n" + word + "\t" + str(len(indexes)))
    indexes.append(index)

indexes = torch.LongTensor(indexes)
embed = vocab.embed(indexes, "cpu")
torch.save(embed, "/work/hpc/potato/laos_vi/data/embedding/laos_reduce_remap.pt")
output.close()
# file.close()
     

