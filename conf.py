"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_worker = 4
# visible_gpus = "0,1,2,3"
visible_gpus = "4,5,6,7"

# model parameter setting
batch_size = 36
max_len = 1024
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1
# method = "diff"
# model_path =  "/raid2/bazaarz/lgbt/transformer/saved/model-0.007121720826521768-diff.pt"
method = None
model_path = "/raid2/bazaarz/lgbt/transformer/saved/model-3.0858828595609986-None.pt"


# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 200
warmup = 100
epoch = 10
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
seed = 42
stride = 64

# tinystrory len 999

