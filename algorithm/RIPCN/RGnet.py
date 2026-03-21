# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import math
import torch.nn.init as init
import torch.nn.functional as F
import einops

from .graph_algo import *

"""
Implementation of UGnet
Tcnblock: extract time feature
SpatialBlock: extract the spatial feature
"""

def TimeEmbedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x: [b, c_in, time, n_nodes]
        # Lk: [3, n_nodes, n_nodes]
        if len(Lk.shape) == 2: # if supports_len == 1:
            Lk=Lk.unsqueeze(0)
        x_c = torch.einsum("bktnm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta,
                            x_c) + self.b  # [b, c_out, time, n_nodes]
        return torch.relu(x_gc + x)

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, : -self.chomp_size]


class TcnBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation_size=1, droupout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(3, self.kernel_size), padding=(1, self.padding), dilation=(1, self.dilation_size))

        self.chomp = Chomp(self.padding)
        self.drop =  nn.Dropout(droupout)

        self.net = nn.Sequential(self.conv, self.chomp, self.drop)

        self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None


    def forward(self, x):
        # x: (B, C_in, V, T) -> (B, C_out, V, T)
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)

        return out + x_skip

class ResidualBlock(nn.Module):  # 所有调用这个类的改kernel_size参数 1,3,5,7
    def __init__(self, c_in, c_out, config, kernel_size=3):
        """
        :param c_in: in channels
        :param c_out: out channels
        :param kernel_size:
        TCN convolution
            input: (B, c_in, V, T)
            output:(B, c_out, V, T)
        """
        super().__init__()
        self.tcn1 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.tcn2 = TcnBlock(c_out, c_out, kernel_size=kernel_size)
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1,1))
        self.t_conv = nn.Conv2d(config.d_h, c_out, (1,1))
        self.spatial = SpatialBlock(config.supports_len, c_out, c_out)

        self.norm = nn.LayerNorm([config.V, c_out])
    def forward(self, x, t, A_hat):
        # x: (B, c_in, V, T), return (B, c_out, V, T)

        h = self.tcn1(x)

        if t != None:
            h += self.t_conv(t[:, :, None, None])

        h = self.tcn2(h)

        h = self.norm(h.transpose(1,3)).transpose(1,3) # (B, c_out, V, T)

        h = h.transpose(2,3) #(B, c_out, V, T)
        h = self.spatial(h, A_hat).transpose(2,3) # (B, c_out, V, T)
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        """
        :param c_in: in channels, out channels
        :param c_out:
        """
        super().__init__()
        self.res = ResidualBlock(c_in, c_out, config, kernel_size=3)

    def forward(self, x, t, supports):
        # x: (B, c_in, V, T), return (B, c_out, V, T)

        return self.res(x, t, supports)

class Downsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in,  kernel_size= (1,3), stride=(1,2), padding=(0,1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, supports):
        _ = t
        _ = supports
        return self.conv(x)


class  UpBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        super().__init__()
        self.res = ResidualBlock(c_in*2, c_out, config, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res(x, t, supports)

class Upsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_in, (1, 4), (1, 2), (0, 1))

    def forward(self, x, t, supports):
        _ = t
        _ = supports
        return  self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, c_in, config):
        super().__init__()
        self.res1 = ResidualBlock(c_in, c_in, config, kernel_size=3)
        self.res2 = ResidualBlock(c_in, c_in, config, kernel_size=3)

    def forward(self, x, t, supports):
        x = self.res1(x, t, supports)

        x = self.res2(x, t, supports)

        return x


class attention3(nn.Module): 
    def __init__(self):
        super(attention3, self).__init__()
    def forward(self, query, key, value, dropout=None):
        d = key.size(-1)
        alpha = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d)
        alpha = F.softmax(alpha, dim=-1)
        out = torch.matmul(alpha, value)
        return out, alpha
        
class MultiheadAttention(nn.Module):
    def __init__(self, head, embedding_size, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        # assert embedding_size % head == 0 # 得整分
        self.head = head
        self.W_K = nn.Linear(embedding_size, head*embedding_size)
        self.W_Q = nn.Linear(embedding_size, head*embedding_size)
        self.W_V = nn.Linear(embedding_size, head*embedding_size)
        self.fc = nn.Linear(head*embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.d_k = embedding_size
        self.attention = attention3()
    def forward(self, x):
        V = x.shape[1]
        batch_size = x.size(0)
        # x = x.to(torch.float32)
        # 转换成多头，一次矩阵乘法即可完成
        query = self.W_Q(x).view(batch_size, V, self.head, -1, self.d_k)
        key = self.W_K(x).view(batch_size, V, self.head, -1, self.d_k)
        value = self.W_V(x).view(batch_size, V, self.head, -1, self.d_k)
        out, alpha = self.attention(query, key, value, self.dropout)
        out = out.view(batch_size, V, -1, self.d_k * self.head)
        out = self.fc(out)
        return out, alpha


class RGnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.for_NPPC = config.for_NPPC
        self.d_h = config.d_h
        self.T_p = config.T_p
        self.T_h = config.T_h
        T = config.T
        F = self.F = config.F * 2 if config.for_NPPC else config.F  # 主成分网络的输入历史的x和预测的x
        self.T_mid = T if config.for_NPPC else T * 2  # 主成分值预测未来的即可
        self.final_channel = config.final_channel

        self.n_blocks = 2
        
        # number of resolutions
        dim_hid = [self.d_h] + [self.d_h * mult for mult in config.channel_multipliers]
        resolutions = [[dim_hid[i], dim_hid[i + 1]] for i in range(len(dim_hid) - 1)]
        n_resolutions = len(resolutions)

        self.cap = torch.from_numpy(config.cap).to(self.config.device).to(torch.float32)[None,None,:,None]
        self.max_speed = torch.from_numpy(config.max_speed).to(self.config.device).to(torch.float32)[None,None,:,None]
        self.ttu = torch.from_numpy(config.ttu).to(self.config.device).to(torch.float32)[None,None,:,None]
        
        # for gcn
        self.a1 = asym_adj(config.A)
        self.a2 = asym_adj(np.transpose(config.A))
        self.a1 = torch.from_numpy(self.a1).to(self.config.device)
        self.a2 = torch.from_numpy(self.a2).to(self.config.device)
        config.supports_len = 4
        self.V = self.a1.shape[0]
        
        # first half of U-Net = decreasing resolution
        down = []
        # number of channels
        for i in range(n_resolutions):
            in_channels = resolutions[i][0]
            out_channels = resolutions[i][1]
            for _ in range(self.n_blocks):
                down.append(DownBlock(in_channels, out_channels, config))
                in_channels = out_channels

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, config)

        # #### Second half of U-Net - increasing resolution
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            in_channels = resolutions[i][1]
            out_channels = resolutions[i][1]
            for b in range(self.n_blocks):
                up.append(UpBlock(in_channels, out_channels, config))
                if b == self.n_blocks - 2: out_channels = resolutions[i][0]

        self.up = nn.ModuleList(up)
        self.x_proj = nn.Conv2d(self.F, self.d_h, (1,1))
        self.f_proj = nn.Sequential(nn.Linear(1, self.d_h),
                                    nn.ELU(),
                                    nn.Linear(self.d_h, self.d_h))
        self.r_att = MultiheadAttention(8, self.d_h)  # 8头注意力 4,8,12,16
        self.r_out = nn.Sequential(nn.Linear(self.d_h, self.d_h), 
                                   nn.ELU(),
                                   nn.Linear(self.d_h, 1))
        self.out = nn.Sequential(
                                 nn.Conv2d(self.d_h, self.d_h, kernel_size=(3, 5), padding=(1, 0), dilation=(1, 1)),
                                 nn.ELU(),
                                 nn.Conv2d(self.d_h, self.d_h, kernel_size=(3, 5), padding=(1, 0), dilation=(1, 1)),
                                 nn.ELU(),
                                 nn.Conv2d(self.d_h, self.d_h, kernel_size=(3, 5), padding=(1, 0), dilation=(1, 1)),
                                 nn.ELU(),
                                 nn.Conv2d(self.d_h, self.final_channel, (1,1)),
                                )
        


    def forward(self, x: torch.Tensor, r_h, t: torch.Tensor, c):
        """
        :param x: x_t of current diffusion step, (B, F, V, T)
        :param t: diffsusion step
        :param c: condition information
            used information in c:
                x_masked: (B, F, V, T)
        :return:
        """
        # 先计算阻抗，再映射
        batch_num = x.shape[0]
        r_h = self.f_proj(r_h.permute(0,2,1,3)) #[:, :, :, None, :]  # [32, 170, 12, 32]
        r_p, _ = self.r_att(r_h)  # (B,V,T,1d)  [32, 170, 12, 32]
        r = torch.cat([r_h, r_p], 2)
        r = self.r_out(r)  # (B,V,T,1)  [32, 170, 24, 1]
        r = r.permute(0, 3, 1, 2)  # [32, 1, 170, 24]
        
        
        a1 = self.a1.repeat(batch_num*(self.T_h + self.T_p), 1, 1)
        a2 = self.a2.repeat(batch_num*(self.T_h + self.T_p), 1, 1)
        a1, a2 = a1.reshape(batch_num, 1, self.T_h+self.T_p, self.V, self.V), a2.reshape(batch_num, 1, self.T_h+self.T_p, self.V, self.V)
        
        row_indices, col_indices = torch.nonzero(self.a1.detach().cpu() > 0, as_tuple=True)
        r_diff_1 = torch.zeros_like(a1)
        r_diff_1[:,:,:,row_indices, col_indices] = (r[:,:,col_indices,:] - r[:,:,row_indices,:]).transpose(-1,-2)
        
        row_indices, col_indices = torch.nonzero(self.a2.detach().cpu() > 0, as_tuple=True)
        r_diff_2 = torch.zeros_like(a2)
        r_diff_2[:,:,:,row_indices, col_indices] = (r[:,:,col_indices,:] - r[:,:,row_indices,:]).transpose(-1,-2)
       
        supports = torch.cat([a1, a2, r_diff_1, r_diff_2], dim=1)  # [32, 4, 24, 170, 170]
        

        x_masked, pos_w, pos_d = c  # x_masked: (B, F, V, T), pos_w: (B,T,1,1), pos_d: (B,T,1,1)
        x = torch.cat((x, x_masked), dim=3) if x_masked != None else x  # (B, F, V, 2 * T)

        x = self.x_proj(x)

        t = TimeEmbedding(t, self.d_h) if t != None else None

        h = [x]

        for m in self.down:
            x = m(x, t, supports)
            h.append(x)
            
        x = self.middle(x, t, supports)

        for m in self.up:
            s = h.pop()
            x = torch.cat((x, s), dim=1)
            x = m(x, t, supports)
        e = self.out(x)
        return r[...,-self.T_p:], e