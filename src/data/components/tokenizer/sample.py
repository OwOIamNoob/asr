from laonlp.word_vector.word2vec import Word2Vec
import numpy as np
import torch

model = Word2Vec("skip-gram").get_model()
vocab = model.index_to_key
index = range(len(vocab))
weights = model.__getitem__(index)
nan_pos = np.array(np.where(np.isnan(weights))).T
for pos in nan_pos:
    print(pos)
    weights[pos[0], pos[1]] = -2
weights =  torch.from_numpy(weights.astype(np.float32)).reshape((-1,300))
dim = weights.shape[1]
weights = np.pad(weights, ((1, 0), (0, 0)), 'constant', constant_values=(0., ))
if len(weights.shape) < 2:
    print(weights.shape)
    weights = weights.reshape((-1, 300))
torch.save(weights, "/work/hpc/potato/laos_vi/data/embedding/laos_v300d.pt")
stride = 3
file = open("/work/hpc/potato/laos_vi/data/embedding/lao_dictionary.txt", "w")
file.write(str(int(len(vocab) + stride)) + "\t" + str(dim) + "\t" + str(stride))
for idx, word in enumerate(vocab):
    if not word:
        print(idx)
        continue
    index = idx + stride 
    file.write("\n" + word + "\t" + str(int(index)))
file.close()
    