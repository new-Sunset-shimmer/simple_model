"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention_diff import MultiheadDiffAttn as MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer_diff(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, n_layers):
        super(DecoderLayer_diff, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, depth = n_layers)
        self.norm1 = LayerNorm(d_model=d_model)
        # self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        # self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, trg_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(dec, rel_pos = None, attn_mask=trg_mask)
        
        # 2. add and norm
        # x = self.dropout1(x)
        x = self.norm1(x + _x)

        # if enc is not None:
        #     # 3. compute encoder - decoder attention
        #     _x = x
        #     x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
        #     # 4. add and norm
        #     x = self.dropout2(x)
        #     x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        # x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
