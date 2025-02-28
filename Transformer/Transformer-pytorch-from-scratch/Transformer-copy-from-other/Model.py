import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-product Attention 
    q: (batch_size, num_heads, seq_len_q, d_k)
    k: (batch_size, num_heads, seq_len_k, d_k)
    v: (batch_size, num_heads, seq_len_v, d_v)

    attn: (batch_size, num_heads, seq_len_q, seq_len_k)
    output: (batch_size, num_heads, seq_len_q, d_v)
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        计算 attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) 时
        k.transpose(2, 3) 改变 k 的大小为 (batch_size, num_heads, d_k, seq_len_k)
        结果 attn 的大小为 (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2,3))

        if mask is not None:
            attn = attn.masked_fill(mask==0,-1e9)

        attn = self.dropout(F.softmax(attn,dim=-1))
        """
        v: (batch_size, num_heads, seq_len_v, d_v)
       attn(batch_size, num_heads, seq_len_q, seq_len_k)
        """
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    """定义这个多头注意力机制的类"""
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        n_head  :注意力头的数量
        d_model :输入和输出的维度,通常是嵌入向量的维度
        d_k     :每个注意力头的查询和键的维度, 通常是 d_model 除以 n_head 的值
        d_v     :每个注意力头的值的维度。
        droput  :
        """
        super().__init__()
        self.n_head = n_head
        self.d_k = d_K
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head*d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False) # 结合模型理解

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) # 
    
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        if mask is not None:    
            mask = mask.unsqueeze(1) # For head axis broadcasting.
        q, attn = self.attention(q,k,v,mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1,2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q,attn

"""不熟悉"""
class PositionwiseFeedForward(nn.Module):
    """一个标准的FNN模型"""
    """FNN的模型结构是什么"""
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x+= residual
        x = self.layer_norm(x)
        return x

"""不熟悉"""
class EncoderLayer(nn.Module):
    """MultiHeadAttention+FNN"""
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attention(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output = self.feed_forward(enc_output)
        return enc_output, enc_self_attn

"""更不熟悉了"""
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    
    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask) # 因为当前的decode是掩盖的，所以使用enc_output作为value
        dec_output = self.feed_forward(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

"""构筑模型"""

def get_pad_mask(seq, pad_idx):
    """非填充值的位置为True，填充值的位置为False。这个掩码用于在注意力机制中忽略填充值"""
    return (seq != pad_idx).unsqueeze(-2)

"""可视化它"""
def get_subsequent_mask(seq):
    """屏蔽序列中当前时间步之后的所有时间步的信息"""
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_in, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
    
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        position = torch.arange(n_position).unsqueeze(1)  # [n_position, 1]
        div_term = torch.pow(10000, torch.arange(0, d_hid, 2) / d_hid)
        angle_rads = position / div_term # [n_position, d_hid // 2]

        # 偶数维度用 sin，奇数维度用 cos
        sinusoid_table = torch.zeros((n_position, d_hid))  # 初始化表格
        sinusoid_table[:, 0::2] = torch.sin(angle_rads)  # 偶数维度
        sinusoid_table[:, 1::2] = torch.cos(angle_rads)  # 奇数维度
        
        return sinusoid_table.unsqueeze(0)  # [1, n_position, d_hid]

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx) # ? pad_idx
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = dropout
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attn=False):
        enc_slf_attn_list = []

        enc_output = self.slf_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5 # ??
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output) # ?? 结构是这样子的吗
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, self_attn_mask=src_mask)
            enc_self_attn_list += [enc_slf_attn] if return_attn else []
        
        if return_attn:
            return enc_output, enc_self_attn_list
        else:
            return enc_output


class Decoder(nn.Module):
    def __init__(self, n_tgt_vocab, d_word_vec, n_layers, d_k, d_v, d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):
        super().__init__()
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = dropout
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm  = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model 
    
    def forward(self, tgt_seq, tgt_mask, enc_output, src_mask, return_attn=False):
        dec_slf_attn_list = []
        dec_enc_attn_list = []

        dec_output =  self.tgt_word_emb(tgt_seq)
        if self.scale_emd:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, slf_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attn else []
            dec_enc_attn_list += [dec_enc_attn] if return_attn else []
        
        if return_attn:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        else:
            return dec_output
    
class Transformer(nn.Module):
    def __init__(
        self, 
        n_src_vocab, 
        n_tgt_vocab, 
        src_pad_idx, 
        tgt_pad_idx, 
        d_word_vec=512, 
        d_model=512, 
        d_inner=2048, 
        n_layers=6, 
        n_head=8, 
        d_k=64, 
        d_v=64, 
        dropout=0.1, 
        n_position=200, 
        tgt_emb_prj_weight_sharing=True, 
        emb_src_tgt_weight_sharing=True, 
        scale_emb_or_prj='prj'):
        
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if tgt_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            pad_idx=src_pad_idx,
            dropout=dropout,
            scale_emb=scale_emb
        )

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            pad_idx=tgt_pad_idx,
            dropout=dropout,
            scale_emb=scale_emb
        )

        # 这是一个线性层，将解码器的输出映射为目标语言词汇表的 logits
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)

        # 使用 Xavier 初始化方法对所有权重矩阵进行初始化
        for p in self.parameters():
            if p.dim > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        # 如果启用了 tgt_emb_prj_weight_sharing，则该层的权重会与目标语言嵌入矩阵共享
        if tgt_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight

        # 如果启用 emb_src_tgt_weight_sharing，则源语言和目标语言的嵌入矩阵权重会被共享。
        if emb_src_tgt_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight
    
    def forward(self, src_seq, tgt_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        tgt_mask = get_pad_mask(tgt_seq, self.tgt_pad_idx) & get_subsequent_mask(tgt_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(tgt_seq, tgt_mask, enc_output, src_mask)

        # 将解码器输出映射为目标语言词汇表的 logits
        seq_logit = self.tgt_word_prj(dec_output)

        # 如果启用了投影缩放，则对 logits 进行缩放
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        # 将 logits 展平为 [batch_size * seq_len, vocab_size] 的形状，便于计算损失函数
        return seq_logit.view(-1, seq_logit.size(2))
