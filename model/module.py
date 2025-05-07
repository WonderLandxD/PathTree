import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce
from math import ceil
from torch_geometric.nn import GATConv


def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim_head, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values


        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(q, k_landmarks, einops_eq)
        sim2 = einsum(q_landmarks, k_landmarks, einops_eq)
        sim3 = einsum(q_landmarks, k, einops_eq)
        # sim1 = einsum(einops_eq, q, k_landmarks)
        # sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        # sim3 = einsum(einops_eq, q_landmarks, k)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads
        out = F.adaptive_max_pool2d(out, (1, self.dim_head))
        out = out.squeeze(2)

        # out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        # out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


class Self_Attention_light(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=11):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//heads,  # dim_head = 64,  
            heads = heads,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = False,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            # dropout=0.1
        )
        # self.to_out = nn.AdaptiveAvgPool2d((1, dim))

    def forward(self, x):
        out = self.attn(self.norm(x))   # ([1, 13, 256, 39])
        
        return out


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 12):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x, attention_only=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # chunk * N * n_classes
        raw_A = A
        if attention_only:
            return raw_A
        A = torch.transpose(A, 2, 1)
        A = F.softmax(A, dim=2)
        feat = torch.matmul(A, x)

        return feat
    

class BiTreeGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(BiTreeGNN, self).__init__()
        self.conv1 = GATConv(num_node_features, num_node_features // 2, heads=8, dropout=0.)
        self.conv2 = GATConv(num_node_features * 4, num_node_features, heads=4, concat=False, dropout=0.)
        self.W_1 = nn.Linear(num_node_features, num_node_features)
        self.W_2 = nn.Linear(num_node_features, num_node_features)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index_1, edge_index_2):
        H_1 = self.activation(self.conv1(x, edge_index_1))
        H_1 = self.conv2(H_1, edge_index_1)

        H_2 = self.activation(self.conv1(x, edge_index_2))
        H_2 = self.conv2(H_2, edge_index_2)

        sum_embedding = self.activation(self.W_1(H_1 + H_2))
        bi_embedding = self.activation(self.W_2(H_1 * H_2))
        # cat_embedding = self.activation(self.linear3(torch.cat([e_h, e_Nh], dim=2)))
        embedding = sum_embedding + bi_embedding

        return embedding
    

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = (anchor - positive).pow(2).sum(1)
        
        negative_distance = (anchor - negative).pow(2).sum(1)
        
        losses = torch.relu(positive_distance - negative_distance + self.margin)
        
        return losses.mean()