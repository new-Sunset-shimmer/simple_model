"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.blocks.decoder_layer_diff import DecoderLayer_diff
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device, method):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)
        if method == None:
            print("None")
            self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                    ffn_hidden=ffn_hidden,
                                                    n_head=n_head,
                                                    drop_prob=drop_prob
                                                    )
                                            for _ in range(n_layers)])
        else:
            print("diff")
            self.layers = nn.ModuleList([DecoderLayer_diff(d_model=d_model,
                                                    ffn_hidden=ffn_hidden,
                                                    n_head=n_head,
                                                    drop_prob=drop_prob,
                                                    n_layers = n_layers
                                                    )
                                        for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    # def forward(self, trg, enc_src, trg_mask, src_mask):
    #     trg = self.emb(trg)

    #     for layer in self.layers:
    #         trg = layer(trg, enc_src, trg_mask, src_mask)

    #     # pass to LM head
    #     output = self.linear(trg)
    #     return output
    def forward(self, trg, trg_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, trg_mask)

        # pass to LM head
        output = self.linear(trg)
        return output