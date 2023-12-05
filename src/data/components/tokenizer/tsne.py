# from sklearn.manifold import TSNE
import torch
import numpy as np
from laonlp.word_vector.word2vec import Word2Vec

model = Word2Vec("skip-gram").get_model()
vocab = model.index_to_key
file = open("/work/hpc/potato/laos_vi/data/embedding/laos_fix_v1.txt", "w")
skip_key = []
key = []
for idx, word in enumerate(vocab):
    if not word:
        skip_key.append(idx)
        continue
    
    key.append(idx)
weights = model.__getitem__(key)
print([vocab[k] for k in skip_key])
weights[np.isnan(weights)] = -1.
weights[np.isinf(weights)] = 1.
weights = np.pad(weights.reshape((-1, 300)),
                 ((1, 0), (0, 0)),
                 'constant',
                 constant_values=(0, )).astype(np.float32)
stride = 1
print(weights.shape)
x = input("Continue?")
if x == 'n':
    exit 0
file.write(str(len(key) + stride) + "\t" + str(weights.shape[1]) + "\t" + str(stride))
file.write("\n<pad>" + "\t" + "0")
for k in key:
    file_id = k + stride
    word = vocab[k]
    file.write("\n" + word + "\t" + str(file_id))
file.close()

tensor = torch.FloatTensor(weights)
# tensor = torch.load("/work/hpc/potato/laos_vi/data/embedding/laos_v300d.pt")
if not torch.is_tensor(tensor):
    tensor = torch.tensor(tensor).to("cuda:2")
# if np.isnan(tensor).any() or np.isinf(tensor).any():
#     raise ValueError("What the fuck")
u, s, v = torch.pca_lowrank(tensor)
print(u.shape, s.shape, v.shape)
reduced = tensor @ v[:, :100]

# model = TSNE(n_components=100, perplexity=30, method='exact', verbose=1)
# reduced = model.fit_transform(tensor)
# np.savetxt("/work/hpc/potato/laos_vi/data/embedding/laos_v100d_numpy.txt", reduced)
# reduced = torch.tensor(reduced)
torch.save(reduced, "/work/hpc/potato/laos_vi/data/embedding/laos_v100d_test.pt")