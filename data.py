"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import os
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    
seed_everything(seed)

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenizer = tokenizer.encoder,
                    detokenizer= tokenizer.decoder,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()
# loader.build_vocab(train_data=train, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                    batch_size=batch_size,
                                                    max_len= max_len)

src_pad_idx = 100264
trg_pad_idx = 100264
trg_sos_idx = 100265

enc_voc_size = tokenizer.encoder.max_token_value
dec_voc_size = tokenizer.encoder.max_token_value
