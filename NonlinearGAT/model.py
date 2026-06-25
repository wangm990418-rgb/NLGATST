"""Nonlinear graph attention layers used by NLGATST."""

from __future__ import annotations

import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch_scatter import scatter_add


class NonlinearGAT_S(nn.Module):
    """Softmax-based nonlinear graph attention encoder with an MLP decoder."""

    def __init__(self, nfeat, nhid, dropout, alpha, nheads, s_f=1.0):
        super().__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.s_f = s_f
        self.attentions = [
            GATConv_S(nfeat, nhid, negative_slope=alpha, attn_drop=dropout)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module(f"attention_{i}", attention)
        self.out_att = GATConv_S(
            nhid * nheads,
            nhid,
            negative_slope=alpha,
            attn_drop=dropout,
            concat=False,
        )
        self.decoder = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, 2000),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, graph, x, edge, p):
        x = F.dropout(x, self.dropout, training=self.training)
        support = []
        for i in range(self.nheads):
            support.append(self.attentions[i](graph, x, edge, p[0, i], self.s_f))
        x = torch.cat(support, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(graph, x, edge, p=p[0, i], s_f=self.s_f))
        reconstructed = self.decoder(F.dropout(x, self.dropout, training=self.training))
        return x, reconstructed


class GATConv_S(nn.Module):
    """Graph attention layer with nonlinear neighbor reweighting."""

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        concat=False,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.concat = concat
        self.activation = activation

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge, p, s_f):
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError("Zero in-degree nodes produce invalid outputs.")

            feat_src, feat_dst, h_dst, dst_prefix_shape = self._project_features(feat)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})

            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            att = self.attn_drop(edge_softmax(graph, self.leaky_relu(graph.edata.pop("e"))))
            graph.edata["a"] = att

            p = torch.sigmoid(p) * s_f
            feat_flat = feat_src.squeeze()
            scale = p * feat_flat
            softmax = torch.exp(scale - scale.max())
            denom = scatter_add(
                softmax[edge[1], :] * att.squeeze(1),
                edge[0],
                dim=0,
                dim_size=softmax.size(0),
            )[edge[0], :]
            weight = softmax[edge[1], :] / (denom + 1e-6)
            weight = F.dropout(weight, 0.5, training=self.training)
            output = scatter_add(
                weight * feat_flat[edge[1], :] * att.squeeze(1),
                edge[0],
                dim=0,
                dim_size=softmax.size(0),
            )
            rst = output.unsqueeze(1)

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            rst = torch.flatten(rst, start_dim=1)
            return F.elu(rst) if self.concat else rst

    def _project_features(self, feat):
        if isinstance(feat, tuple):
            src_prefix_shape = feat[0].shape[:-1]
            dst_prefix_shape = feat[1].shape[:-1]
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            if hasattr(self, "fc_src"):
                feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            return feat_src, feat_dst, h_dst, dst_prefix_shape

        prefix_shape = feat.shape[:-1]
        h_src = h_dst = self.feat_drop(feat)
        projected = self.fc(h_src).view(*prefix_shape, self._num_heads, self._out_feats)
        return projected, projected, h_dst, prefix_shape


NonlinearGAT_G = NonlinearGAT_S
NonlinearGAT_P = NonlinearGAT_S


class GATConv(nn.Module):
    """Standard graph attention layer retained for ablation studies."""

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        concat=False,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.concat = concat
        self.activation = activation

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError("Zero in-degree nodes produce invalid outputs.")

            feat_src, feat_dst, h_dst, dst_prefix_shape = self._project_features(feat)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, self.leaky_relu(graph.edata.pop("e"))))
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            rst = torch.flatten(rst, start_dim=1)
            return F.elu(rst) if self.concat else rst

    def _project_features(self, feat):
        if isinstance(feat, tuple):
            src_prefix_shape = feat[0].shape[:-1]
            dst_prefix_shape = feat[1].shape[:-1]
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            if hasattr(self, "fc_src"):
                feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            return feat_src, feat_dst, h_dst, dst_prefix_shape

        prefix_shape = feat.shape[:-1]
        h_src = h_dst = self.feat_drop(feat)
        projected = self.fc(h_src).view(*prefix_shape, self._num_heads, self._out_feats)
        return projected, projected, h_dst, prefix_shape


class EstimateP(nn.Module):
    """Learnable nonlinear aggregation parameter."""

    def __init__(self, n):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor(1, n))
        stdv = 1.0 / math.sqrt(self.p.size(1))
        torch.nn.init.uniform_(self.p.data, -stdv, stdv)

    def forward(self):
        return self.p
