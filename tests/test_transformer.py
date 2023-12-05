import torch
from torch import Tensor

import rootutils

rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)


from src.models.transformer_module import TransformerLitModule

from src.models.transformer.model.encoder import Encoder
from src.models.transformer.model.decoder import Decoder


input_dim = 300
vocab_size = 69
d_ff = 8
d_model = 8
num_layers = 2

encoder = Encoder(
    vocab_size=vocab_size,
    input_dim=input_dim,
    d_ff=d_ff,
    d_model=d_model,
    num_layers=num_layers,
    # All else go with default
)

decoder = Decoder(
    vocab_size=vocab_size,
    d_ff=d_ff,
    d_model=d_model,
    num_layers=num_layers,
    max_length=100,
    # All go with default
)

model = TransformerLitModule(
    encoder=encoder,
    decoder=decoder,
    pad_id=0,
    sos_id=1,
    eos_id=2,
    teacher_forcing_ratio=0.5,
)

torch.manual_seed(0)

batch_size = 5
max_seq_len = 16

def get_inputs():
    inputs = torch.zeros((batch_size, max_seq_len, input_dim), dtype=torch.float)
    input_lengths = torch.zeros((batch_size), dtype=torch.long)

    for batch in range(batch_size):
        input_lengths[batch] = torch.randint(
            low=1,
            high=max_seq_len+1,
            size=(1,),
            dtype=torch.long
        ).item()

        for i in range(int(input_lengths[batch])):
            inputs[batch,i] = torch.rand(
                size=(input_dim,),
            )
    return inputs, input_lengths

def get_targets():
    targets = torch.zeros((max_seq_len, vocab_size), dtype=torch.long)
    target_lengths = torch.randint(
                        low=1,
                        high=max_seq_len+1,
                        size=(1,),
                        dtype=torch.long
                    ).item()

    for i in range(int(target_lengths)):
        one = torch.randint(
            low=5,
            high=vocab_size+1,
            size=(1,),
            dtype=torch.long
        ).item()
        targets[i,one] = torch.ones((1,),dtype=torch.long).item()

    return targets, target_lengths

inputs, input_lengths = get_inputs()
targets, target_lengths = get_targets()

print(inputs.size())
print(input_lengths)

# output = model.forward(inputs,input_lengths)
# print(output.keys())

loss = model.training_step(batch=(inputs, targets, input_lengths, target_lengths))
print(loss)
