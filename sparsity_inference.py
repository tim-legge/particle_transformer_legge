
import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import weaver
import fastjet


import mplhep as hep

hep.style.use(hep.style.ROOT)
plt.style.use(hep.style.ROOT)

import argparse

parser = argparse.ArgumentParser(description='select model to run inference on')
parser.add_argument('-m', '--model', type=str, required=True, help='model to run inference on (10_pct, seed_0, or seed_1)')

model = parser.parse_args().model

def _clip(a, min_value, max_value):
    assert isinstance(a, ak.Array), "expected awkward array"
    main_list = []
    for i in range(len(a)):
        sublist = ak.to_list(a[i])
        sublist = np.clip(sublist, min_value, max_value)
        main_list.append(sublist)
    return ak.from_iter(main_list)

def _pad(a, maxlen=128, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

def build_features_and_labels(tree, transform_features=True):

    # load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])

    # compute new features
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])
    a['part_d0'] = np.tanh(a['part_d0val'])
    a['part_dz'] = np.tanh(a['part_dzval'])

    # apply standardization
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0
        a['part_d0err'] = _clip(a['part_d0err'], 0, 1)
        a['part_dzerr'] = _clip(a['part_dzerr'], 0, 1)

    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'], # not used in ParT
        'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_charge',
            'part_isChargedHadron',
            'part_isNeutralHadron',
            'part_isPhoton',
            'part_isElectron',
            'part_isMuon',
            'part_d0',
            'part_d0err',
            'part_dz',
            'part_dzerr',
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask']
    }

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)

    label_list = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)

    return out


def build_features_and_labels_qg(tree, transform_features=True):
    """Build features for QuarkGluon dataset based on qg_kinpid.yaml"""
    # load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label'])

    # compute new features
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])

    # apply standardization based on qg_kinpid.yaml
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0

    # Feature list for QuarkGluon (kinematic + particle ID)
    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'],
        'pf_features': [
            'part_pt_log',
            'part_e_log', 
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_charge',
            'part_isChargedHadron',
            'part_isNeutralHadron',
            'part_isPhoton',
            'part_isElectron',
            'part_isMuon',
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask']
    }

    def _pad(a, maxlen=128, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)

    # Labels for QuarkGluon (binary classification)
    out['label'] = a['label'].to_numpy().astype('int')
    
    return out

def build_features_and_labels_tl(tree, transform_features=True):
    """Build features for TopLandscape dataset based on top_kin.yaml"""
    # load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label'])

    # compute new features (same as QG)
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])

    # apply standardization based on top_kin.yaml (same as QG)
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0

    # Feature list for TopLandscape (same kinematic features as QG)
    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'],
        'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel', 
            'part_logerel',
            'part_deltaR',
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask']
    }

    def _pad(a, maxlen=128, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)

    # Labels for TopLandscape (binary classification) 
    out['label'] = a['label'].to_numpy().astype('int')

    return out


from typing import List, Optional
import timeit
import awkward as ak
import torch
import torch.nn as nn
from torch.nn import Parameter 
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
from typing import Optional, Tuple
_is_fastpath_enabled: bool = True
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)
linear = torch._C._nn.linear
import math
import random
import warnings
import copy
from torch._C import _add_docstr, _infer_size

from functools import partial
from weaver.utils.logger import _logger
import os
import uproot
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information from different representation subspaces.

    This MultiheadAttention layer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O

    where :math:`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``nn.MultiheadAttention`` will use the optimized implementations of
    ``scaled_dot_product_attention()`` when possible.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        return_attns: bool = False,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.return_attns = return_attns
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        return_attns: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Compute attention outputs using query, key, and value embeddings.

            Supports optional parameters for padding, masks and attention weights.

        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and float masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
                and achieve the best performance for MHA.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
            is_causal: If specified, applies a causal mask as attention mask.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``attn_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """  # noqa: B950
        why_not_fast_path = ""
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = (
                "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
            )
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(
                _arg_requires_grad(x) for x in tensor_args
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            if self.return_attns:
                attn_output, attn_output_weights, pre_softmax_attention, pre_softmax_interaction = multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v,
                self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights, is_causal=is_causal,
                return_attns=self.return_attns,
            )
                pre_softmax_attention.cpu().detach()
                pre_softmax_interaction.cpu().detach()
            else:
                attn_output, attn_output_weights = multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v,
                self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights, is_causal=is_causal,
                return_attns=self.return_attns,
            )
        else:
            if self.return_attns:
                attn_output, attn_output_weights, pre_softmax_attention, pre_softmax_interaction = multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal,
                    return_attns=self.return_attns,
                )
                pre_softmax_attention.cpu().detach()
                pre_softmax_interaction.cpu().detach()
            else:
                attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                return_attns=self.return_attns,
            )
        if self.batch_first and is_batched:
            if self.return_attns:
                pre_softmax_attention.cpu().detach()
                pre_softmax_interaction.cpu().detach()
                return attn_output.transpose(1, 0), attn_output_weights, pre_softmax_attention, pre_softmax_interaction
            else:
                return attn_output.transpose(1, 0), attn_output_weights
        else:
            if self.return_attns:
                    pre_softmax_attention.cpu().detach()
                    pre_softmax_interaction.cpu().detach()
                    return attn_output, attn_output_weights, pre_softmax_attention, pre_softmax_interaction
            else:
                return attn_output, attn_output_weights

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len
                ).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type

def _canonical_mask(
    mask: Optional[Tensor],
    mask_name: str,
    other_type: Optional[torch.dtype],
    other_name: str,
    target_type: torch.dtype,
    check_other: bool = True,
) -> Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                pass
                #warnings.warn(
                #    f"Support for mismatched {mask_name} and {other_name} "
                #    "is deprecated. Use same type for both instead."
                #)
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(
                mask, float("-inf")
            )
    return mask
def _none_or_dtype(input: Optional[Tensor]) -> Optional[torch.dtype]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")
def get_fastpath_enabled() -> bool:
    """Returns whether fast path for TransformerEncoder and MultiHeadAttention
    is enabled, or ``True`` if jit is scripting.

    ..note:
        The fastpath might not be run even if ``get_fastpath_enabled`` returns
        ``True`` unless all conditions on inputs are met.
    """
    if not torch.jit.is_scripting():
        return _is_fastpath_enabled
    return True

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""Perform the in-projection step of the attention operation, using packed weights.

    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = (
                proj.unflatten(-1, (3, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = (
                kv_proj.unflatten(-1, (2, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def _mha_shape_check(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    num_heads: int,
):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, (
            "For batched (3-D) `query`, expected `key` and `value` to be 3-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, (
            "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, (
                "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert (
                    attn_mask.shape == expected_shape
                ), f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}"
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor"
        )

    return is_batched

def set_fastpath_enabled(value: bool) -> None:
    """Sets whether fast path is enabled"""
    global _is_fastpath_enabled
    _is_fastpath_enabled = value


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    return_attns: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Forward method for MultiHeadAttention.

    See :class:`torch.nn.MultiheadAttention` for details.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
            return_attns=return_attns,
        )

    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, (
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    )
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, (
        f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    )
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], (
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        )
    else:
        assert key.shape == value.shape, (
            f"key shape {key.shape} does not match value shape {value.shape}"
        )

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, (
            "use_separate_proj_weight is False but in_proj_weight is None"
        )
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, (
            "use_separate_proj_weight is True but q_proj_weight is None"
        )
        assert k_proj_weight is not None, (
            "use_separate_proj_weight is True but k_proj_weight is None"
        )
        assert v_proj_weight is not None, (
            "use_separate_proj_weight is True but v_proj_weight is None"
        )
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, (
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        )
        assert static_k.size(2) == head_dim, (
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        )
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, (
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        )
        assert static_v.size(2) == head_dim, (
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        )
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        _B, _Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))

        assert not (is_causal and attn_mask is None), (
            "FIXME: is_causal not implemented for need_weights"
        )

        if attn_mask is not None:
            pre_softmax_interaction = attn_mask
            pre_softmax_interaction.detach()
            pre_softmax_attention = torch.bmm(q_scaled, k.transpose(-2, -1))
            pre_softmax_attention.detach()
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k.transpose(-2, -1)
            )
            print(attn_output_weights.shape)
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        
        attn_saved_weights = attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        #if dropout_p > 0.0:
        #    attn_output_weights = torch.dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        if return_attns:
            return attn_output, attn_saved_weights, pre_softmax_attention, pre_softmax_interaction
        else:
            return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None



# Moe ParT modified to store router output, accessed via MoeRouterHook

from typing import List, Optional
import timeit
import awkward as ak
import torch
import torch.nn as nn
from torch.nn import Parameter 
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
from typing import Optional, Tuple
_is_fastpath_enabled: bool = True
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)
linear = torch._C._nn.linear
import math
import random
import warnings
import copy
from torch._C import _add_docstr, _infer_size

from functools import partial
from weaver.utils.logger import _logger
import os
import uproot
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes
import mplhep as hep
import matplotlib.pyplot as plt
plt.style.use(hep.style.ROOT)


import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import logging
import numpy as np

_logger = logging.getLogger(__name__)

@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True,
                 moe_num_experts=4, moe_top_k=1,
                 moe_capacity_factor=1.25, moe_aux_loss_coef=0.01, moe_router_jitter=0.0,
                 return_attns=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio
        self.return_attns = return_attns

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
            return_attns=return_attns,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None

        if not (moe_num_experts and moe_num_experts > 0):
            raise ValueError('moe_num_experts must be >= 1 for MoE-only Block')
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_aux_loss_coef = moe_aux_loss_coef
        self.moe_router_jitter = moe_router_jitter
        self.router = nn.Linear(embed_dim, moe_num_experts)
        experts = []
        for _ in range(moe_num_experts):
            experts.append(nn.Sequential(
                nn.Linear(embed_dim, self.ffn_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(activation_dropout),
                nn.LayerNorm(self.ffn_dim) if scale_fc else nn.Identity(),
                nn.Linear(self.ffn_dim, embed_dim),
            ))
        self.experts = nn.ModuleList(experts)
        self.register_buffer('_last_aux_loss', torch.tensor(0.0), persistent=False)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            if not self.return_attns:
                x = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)
            if self.return_attns:
                x, attention, pre_softmax_attention, pre_softmax_interaction = self.attn(x, x, x, key_padding_mask=padding_mask,
                    attn_mask=attn_mask, average_attn_weights=False, return_attns=True)
                #pre_softmax_attention = self.attn(x, x, x, key_padding_mask=padding_mask, 
                #    attn_mask=attn_mask, average_attn_weights=False)[2]
                #pre_softmax_interaction = self.attn(x, x, x, key_padding_mask=padding_mask,
                #    attn_mask=attn_mask, average_attn_weights=False)[3]
                attention.cpu().detach()
                pre_softmax_attention.cpu().detach()
                pre_softmax_interaction.cpu().detach()
                self.attention = attention
                self.pre_softmax_attention = pre_softmax_attention
                self.pre_softmax_interaction = pre_softmax_interaction
            
        if padding_mask is not None:
            self.padding_mask = padding_mask

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        #x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        # reshape to tokens-first for routing
        seq_len, batch_size, embed_dim = x.shape
        tokens = x.reshape(seq_len * batch_size, embed_dim)
        with torch.no_grad():
            router_logits = self.router(tokens)
            if self.training and self.moe_router_jitter > 0:
                noise = torch.empty_like(router_logits).uniform_(0, 1)
                noise = -torch.log(-torch.log(noise.clamp(min=1e-9)))
                router_logits = router_logits + self.moe_router_jitter * noise
            gates = torch.softmax(router_logits, dim=-1)
            # Load-balancing aux: generalized for top-k routing
            mean_prob_per_expert = gates.mean(dim=0)
            if self.moe_top_k == 1:
                token_choice = gates.argmax(dim=-1)
                frac_per_expert = torch.stack([
                    (token_choice == e).float().mean() for e in range(self.moe_num_experts)
                ])
            else:
                topk_idx_aux = gates.topk(k=min(self.moe_top_k, self.moe_num_experts), dim=-1).indices
                assigned = torch.zeros_like(gates)
                assigned.scatter_(1, topk_idx_aux, 1.0)
                # Normalize so sum_e frac_per_expert ~= 1 like top-1 case
                frac_per_expert = assigned.mean(dim=0) / float(self.moe_top_k)
            aux = (mean_prob_per_expert * frac_per_expert).sum() * (self.moe_num_experts ** 2)
            self._last_aux_loss = aux.detach()

        output_tokens = torch.zeros_like(tokens)
        if self.moe_top_k == 1:
            # top-1 routing (Switch-style)
            top1_idx = gates.argmax(dim=-1)
            top1_w = gates.gather(1, top1_idx.unsqueeze(1)).squeeze(1)
            capacity = int(self.moe_capacity_factor * math.ceil(tokens.size(0) / max(1, self.moe_num_experts)))
            for e in range(self.moe_num_experts):
                mask = (top1_idx == e)
                if mask.any():
                    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    if idx.numel() > capacity:
                        sel = top1_w[idx].topk(capacity, sorted=False).indices
                        idx = idx[sel]
                    expert_in = tokens.index_select(0, idx)
                    expert_out = self.experts[e](expert_in)
                    expert_w = top1_w.index_select(0, idx).unsqueeze(1)
                    expert_out = expert_out * expert_w
                    output_tokens.index_copy_(0, idx, expert_out)
        else:
            # top-k routing: send tokens to k experts and sum weighted outputs
            k = min(self.moe_top_k, self.moe_num_experts)
            topk_vals, topk_idx = gates.topk(k=k, dim=-1)
            # save topk_idx for hooking, analysis
            self.topk_idx = topk_idx
            # Renormalize selected gates to sum to 1 per token
            denom = topk_vals.sum(dim=1, keepdim=True).clamp(min=1e-9)
            topk_w = topk_vals / denom
            # save topk_w for hooking, analysis
            self.topk_w = topk_w
            # Capacity per expert accounts for k assignments per token
            capacity = int(self.moe_capacity_factor * math.ceil((tokens.size(0) * k) / max(1, self.moe_num_experts)))
            for e in range(self.moe_num_experts):
                mask_e = (topk_idx == e)
                if mask_e.any():
                    rows, cols = torch.nonzero(mask_e, as_tuple=True)
                    if rows.numel() > capacity:
                        weights_e = topk_w[rows, cols]
                        sel = weights_e.topk(capacity, sorted=False).indices
                        rows = rows.index_select(0, sel)
                        cols = cols.index_select(0, sel)
                        weights_e = weights_e.index_select(0, sel)
                    else:
                        weights_e = topk_w[rows, cols]
                    expert_in = tokens.index_select(0, rows)
                    expert_out = self.experts[e](expert_in)
                    expert_out = expert_out * weights_e.unsqueeze(1)
                    # Multiple experts can contribute to the same token; accumulate
                    output_tokens.index_add_(0, rows, expert_out)
        x = output_tokens.view(seq_len, batch_size, embed_dim)
        #x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x
        


class MoeParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 #interpretability of pre-softmax
                 return_attns=False,
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 moe_num_experts=4,
                 moe_top_k=1,
                 moe_layers=None,
                 moe_capacity_factor=1.25,
                 moe_aux_loss_coef=0.01,
                 moe_router_jitter=0.0,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.return_attns = return_attns

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        self.default_cfg = default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        self.cfg_block = cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_layers = set(moe_layers) if moe_layers is not None else set()
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_aux_loss_coef = moe_aux_loss_coef
        self.moe_router_jitter = moe_router_jitter
        if not (self.moe_num_experts and self.moe_num_experts > 0):
            raise ValueError('moe_num_experts must be >= 1 for MoE-only MoeParticleTransformer')
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        blocks = []
        for _ in range(num_layers):
            blocks.append(Block(
                **cfg_block,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_capacity_factor=self.moe_capacity_factor,
                moe_aux_loss_coef=self.moe_aux_loss_coef,
                moe_router_jitter=self.moe_router_jitter,
                return_attns=self.return_attns
            ))
        self.blocks = nn.ModuleList(blocks)
        self.cls_blocks = nn.ModuleList([
            Block(
                **cfg_cls_block,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_capacity_factor=self.moe_capacity_factor,
                moe_aux_loss_coef=self.moe_aux_loss_coef,
                moe_router_jitter=self.moe_router_jitter,
            ) for _ in range(num_cls_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            self._moe_aux_loss = torch.tensor(0.0, device=x.device)
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
                aux = getattr(block, '_last_aux_loss', None)
                if aux is not None:
                    self._moe_aux_loss = self._moe_aux_loss + self.moe_aux_loss_coef * aux

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output



import torch

class MoeParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.mod = MoeParticleTransformer(**kwargs)
        self.attention_matrix = None
        self.interactionMatrix = None
        self.pre_mask_attention_matrices = []

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        output = self.mod(features, v=lorentz_vectors, mask=mask)
        #self.attention_matrix = self.mod.getAttention()
        #self.interactionMatrix = self.mod.getInteraction()
        #self.pre_mask_attention_matrices = self.get_pre_mask_attention_matrices()
        return output

    def get_attention_matrix(self):
        return self.attention_matrix

    def get_interactionMatrix(self):
        return self.interactionMatrix

class Pre_Softmax_Hook:
    def __init__(self, model):
        self.model = model
        self.kwargs = self.model.kwargs

        if isinstance(self.model, MoeParticleTransformerWrapper):
            self.model = self.model.mod

        for module in self.model.blocks:
            #if layer_name in name:
                # register forward hook functions
            handle_postsoft_attn = module.register_forward_hook(lambda *args, **kwargs: Pre_Softmax_Hook.get_attention(self, *args, **kwargs))
            #handle_attn = module.register_forward_hook(lambda *args, **kwargs: Pre_Softmax_Hook.get_pre_softmax_attention(self, *args, **kwargs))
            #handle_inter = module.register_forward_hook(lambda *args, **kwargs: Pre_Softmax_Hook.get_pre_softmax_interaction(self, *args, **kwargs))

            print(f"Registered hook onto module")
    
        self.handle_postsoft_attn = handle_postsoft_attn
        #self.handle_attn = handle_attn
        #self.handle_inter = handle_inter

        self.embed_dim = self.model.default_cfg['embed_dim']
        self.num_heads = self.model.default_cfg['num_heads']
        self.seq_len = self.embed_dim

        self.attentions = torch.empty((0, self.num_heads, self.seq_len, self.seq_len), dtype=torch.float32)
        self.pre_softmax_attentions = torch.empty((0, self.num_heads, self.seq_len, self.seq_len), dtype=torch.float32)
        self.pre_softmax_interactions = torch.empty((0, self.num_heads, self.seq_len, self.seq_len), dtype=torch.float32)

    # hooks will grab from the outputs of Block
    # meaning we don't need to modify forward methods for anything else
    # 3 hours later -- lol this was wrong

    def get_attention(self, module, input, output):
        print('Getting attention...')

        # handle batching - we will divide 1st dimension by the number such that it will comport with num_heads
        print(f'Got shape:{output[1].shape}')
        output_hooked = module.attention
        output_split = output_hooked.view(output_hooked.shape[0]//self.num_heads, self.num_heads, output_hooked.shape[1], output_hooked.shape[2])
        output_unsqueezed = output_split.unsqueeze(dim=0)
    
        if self.attentions.shape[0] == 0:
            self.attentions = torch.empty((0, output_hooked.shape[0]//self.num_heads, self.num_heads, output_unsqueezed.shape[3], output_unsqueezed.shape[4]), dtype=torch.float32)

        self.attentions = torch.cat((self.attentions, output_unsqueezed), dim=0) # shape 

    def get_pre_softmax_attention(self, module, input, output):
        print('Getting pre_softmax attention...')

        output_hooked = module.pre_softmax_attention
        output_split = output_hooked.view(output_hooked.shape[0]//self.num_heads, self.num_heads, output_hooked.shape[1], output_hooked.shape[2])
        output_unsqueezed = output_split.unsqueeze(dim=0)
    
        if self.pre_softmax_attentions.shape[0] == 0:
            self.pre_softmax_attentions = torch.empty((0, output_hooked.shape[0]//self.num_heads, self.num_heads, output_unsqueezed.shape[3], output_unsqueezed.shape[4]), dtype=torch.float32)

        self.pre_softmax_attentions = torch.cat((self.pre_softmax_attentions, output_unsqueezed), dim=0)
        
        #self.sorted_attentions = self.sort(self.pre_softmax_attentions)

    def get_pre_softmax_interaction(self, module, input, output):
        print('Getting pre-softmax interaction...')

        output_hooked = module.pre_softmax_interaction
        output_split = output_hooked.view(output_hooked.shape[0]//self.num_heads, self.num_heads, output_hooked.shape[1], output_hooked.shape[2])
        output_unsqueezed = output_split.unsqueeze(dim=0)
    
        #print(f'{output_split[0].shape}')

        if self.pre_softmax_interactions.shape[0] == 0:
            self.pre_softmax_interactions = torch.empty((0, output_hooked.shape[0]//self.num_heads, self.num_heads, output_unsqueezed.shape[3], output_unsqueezed.shape[4]), dtype=torch.float32)

        self.pre_softmax_interactions = torch.cat((self.pre_softmax_interactions, output_unsqueezed), dim=0)

    def get_padding(self, tensor, mask):
        '''
        Gives a list of jet lengths for the tensors stored in self.pre_softmax_attentions and self.pre_softmax_interactions.
        Useful for plotting attention-related 

        Args:
        - tensor: The input tensor to process.
        - config: Model configuration dict. If 'num_layers' is not present, defaults to 8 as in original ParT.

        Outputs:
        - padding_limits: list of jet lengths.
        '''
        config = self.kwargs

        if 'num_layers' in config:
            num_layers = config['num_layers']
        else:
            num_layers = 8

        num_jets = tensor.shape[0] // num_layers
        
        #tensor_viewed = tensor.view(num_layers, num_jets, self.num_heads, self.seq_len, self.seq_len)
        tensor_as_np = tensor.numpy() # should be (layers, jets, heads, jet_length, jet_length)
        #tensor_as_np = np.split(tensor_as_np, indices_or_sections=num_jets, axis=0) # now listed by layer -> (jet_num, head, jet_length, jet_length)
        tensor_as_ak = ak.from_numpy(tensor_as_np)

        padding_limits = []

        for jet_idx in range(tensor_as_np.shape[1]):
            padding_limit = np.sum(mask[jet_idx]).astype(int)
            padding_limits.append(padding_limit)
            #tensor_as_ak[:,jet_idx,:,:,:] = tensor_as_ak[:,jet_idx,:,:padding_limit, :padding_limit]
        
        print(f'Padding Limit: {padding_limits}')

        return padding_limits

    def clear(self):
        self.pre_softmax_attentions = torch.empty((0, self.num_heads, self.seq_len, self.seq_len), dtype=torch.float32)
        self.pre_softmax_interactions = torch.empty((0, self.num_heads, self.seq_len, self.seq_len), dtype=torch.float32)
        self.handle_attn.remove()
        self.handle_inter.remove()
        #self.cls_handle_attn.remove()


def get_model(data_type=None, **kwargs):

    if data_type == 'jc_full' or None:
        cfg = dict(
            input_dim=17,
            num_classes=10,
            # network configurations
            pair_input_dim=4,
            use_pre_activation_pair=False,
            embed_dims=[128, 512, 128],
            pair_embed_dims=[64, 64, 64],
            num_heads=8,
            num_layers=8,
            num_cls_layers=2,
            block_params=None,
            cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
            fc_params=[],
            activation='gelu',
            # misc
            trim=True,
            for_inference=False,
            # interpretability of pre-softmax
            return_attns=True,
            # MoE settings
            moe_num_experts=8,
            moe_top_k=2,
            moe_capacity_factor=1.25,
            moe_aux_loss_coef=0.01,
            moe_router_jitter=0.0,
        )
    cfg.update(**kwargs)

    model = MoeParticleTransformerWrapper(**cfg)

    model_info = {
        }
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()


MoE_model = get_model('jc_full')[0]

if model == '10_pct':
    modelpath = f'net_best_epoch_state.pt'
elif model == 'seed_0':
    modelpath = f'models/jc_100_seed_0_net_epoch_state.pt'
elif model == 'seed_1':
    modelpath = f'models/jc_100_seed_1_net_epoch_state.pt'

MoE_statedict = torch.load(modelpath, map_location=torch.device('cpu'))
MoE_model.load_state_dict(MoE_statedict)
#router_hook = Router_Hook(model=MoE_model)
pre_softmax_hook = Pre_Softmax_Hook(model=MoE_model)

with open(f'/moe-interpretability-pv/moe_sparsity_hists_{model}/counter.txt', 'r') as f:
    start_idx = int(f.read().strip())

howmanyjets = 1000

while start_idx < 10:

    features = np.load('/moe-interpretability-pv/datasets/jc_full_pf_features.npy', allow_pickle=True)[start_idx::100000//howmanyjets]
    masks = np.load('/moe-interpretability-pv/datasets/jc_full_pf_mask.npy', allow_pickle=True)[start_idx::100000//howmanyjets]
    labels = np.load('/moe-interpretability-pv/datasets/jc_full_labels.npy', allow_pickle=True)[start_idx::100000//howmanyjets]
    vectors = np.load('/moe-interpretability-pv/datasets/jc_full_pf_vectors.npy', allow_pickle=True)[start_idx::100000//howmanyjets]
    points = np.load('/moe-interpretability-pv/datasets/jc_full_pf_points.npy', allow_pickle=True)[start_idx::100000//howmanyjets]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    idx_to_label = {
        0: 'Higgs BB',
        1: 'Higgs CC',
        2: 'Higgs GG',
        3: 'Higgs QQL',
        4: 'Higgs 4Q',
        5: 'Top BL',
        6: 'Top BQQ',
        7: 'W QQ',
        8: 'QCD',
        9: 'Z QQ'
    }

    MoE_model.eval()
    with torch.no_grad():
        y_pred= MoE_model(torch.from_numpy(points),torch.from_numpy(features),
                                    torch.from_numpy(vectors),torch.from_numpy(masks))

    attentions = pre_softmax_hook.attentions
    pad_limits = pre_softmax_hook.get_padding(pre_softmax_hook.attentions, masks)
    print(attentions.shape)

    unpadded_attentions = []
    for jet_idx in range(attentions.shape[1]):
        unpadded_attentions.append(attentions[:,jet_idx,:, :pad_limits[jet_idx], :pad_limits[jet_idx]].cpu().numpy().flatten())
    import numpy as np
    import matplotlib.pyplot as plt

    # Assuming attention is already a list or array
    # Flattening the attention array
    flattened_attention = np.empty(0)
    for item in unpadded_attentions:
        flattened_attention = np.concatenate(flattened_attention, np.array(item))

    # Define number of bins for the probability distribution
    num_bins = 20

    # Define the bin edges between 0 and 1, using 20 evenly spaced bins
    bin_edges = np.linspace(0, 1, num_bins + 1)

    # Function to process data in chunks and compute histogram
    def process_in_chunks(attention_iterator, chunk_size=100000, bin_edges=bin_edges):
        hist_counts = np.zeros(len(bin_edges) - 1)  # Initialize histogram counts for bins

        # Process each chunk of attention data
        for chunk in attention_iterator:
            # Flatten the chunk to ensure it's 1D and processable by np.histogram
            chunk = np.array(chunk).flatten()

            # Calculate histogram for this chunk
            hist, _ = np.histogram(chunk, bins=bin_edges)

            # Accumulate the counts
            hist_counts += hist
        
        np.save(f'sparsity_hist_counts_start_idx_{start_idx}.npy', hist_counts)

    # Simulate loading a large dataset in chunks (e.g., from a file or other source)
    def attention_generator(attention, chunk_size):
        """Simulate chunked data loader for large dataset."""
        for i in range(0, len(attention), chunk_size):
            yield attention[i:i + chunk_size]
    
    process_in_chunks(attention_generator(flattened_attention, chunk_size=100000))

    print(f'Completed start_idx: {start_idx}')
    start_idx += 1
    with open(f'/moe-interpretability-pv/moe_sparsity_hists_{model}/counter.txt', 'w') as f:
        f.write(str(start_idx))

for file in os.listdir(f'/moe-interpretability-pv/moe_sparsity_hists_{model}/'):
    if file.startswith('sparsity_hist_counts_start_idx_') and file.endswith('.npy'):
        hist_counts = np.load(os.path.join(f'/moe-interpretability-pv/moe_sparsity_hists_{model}/', file))
        if 'total_hist_counts.npy' in os.listdir(f'/moe-interpretability-pv/moe_sparsity_hists_{model}/'):
            total_hist_counts = np.load(os.path.join(f'/moe-interpretability-pv/moe_sparsity_hists_{model}/', 'total_hist_counts.npy'))
            total_hist_counts += hist_counts
        else:
            total_hist_counts = hist_counts
        np.save(os.path.join(f'/moe-interpretability-pv/moe_sparsity_hists_{model}/', 'total_hist_counts.npy'), total_hist_counts)

total_hist_counts = np.load(os.path.join(f'/moe-interpretability-pv/moe_sparsity_hists_{model}/', 'total_hist_counts.npy'))
total_hist_sum = np.sum(total_hist_counts)
probabilities = total_hist_counts / total_hist_sum  # Normalize to get probabilities

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, probabilities, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title(f'Attention Distribution {model}', fontsize=16)
plt.savefig(f'Attention Distribution {model}.pdf', bbox_inches="tight")

# Show the plot
plt.show()

self_attending_counts = np.zeros((attentions.shape[0], attentions.shape[2]))  # layers x heads
for layer_idx in range(attentions.shape[0]):
    for head_idx in range(attentions.shape[2]):
        # track how often the diagonal elements are largest in each row
        for jet_idx in range(attentions.shape[1]):
            attention_matrix = attentions[layer_idx][jet_idx][head_idx, 0:pad_limits[jet_idx], 0:pad_limits[jet_idx]].cpu().numpy()
            count = 0
            for i in range(attention_matrix.shape[0]):
                row = attention_matrix[i]
                if np.argmax(row) == i:
                    count += 1
            self_attending_counts[layer_idx][head_idx] = count  # fraction of self-attending rows
self_attending_fractions = self_attending_counts / np.sum(pad_limits)
        #print(f'Layer {layer_idx}, Head {head_idx}, Self-attending fraction: {self_attending_fraction:.4f}')
        
print('Self-attending fractions per layer and head:')
print(self_attending_fractions)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.lines import Line2D

# Flatten the data for sorting
data = []
num_layers, num_heads = self_attending_fractions.shape
for l in range(num_layers):
    for h in range(num_heads):
        data.append({'fraction': self_attending_fractions[l, h], 'layer': l, 'head': h})

# Sort by fraction in descending order
data.sort(key=lambda x: x['fraction'], reverse=True)

# Extract sorted data
fractions = [d['fraction'] for d in data]
labels = [f"L{d['layer']} H{d['head']}" for d in data]
layers = [d['layer'] for d in data]

# Setup colors
# Use a colormap that gives distinct colors for layers
cmap = plt.get_cmap('viridis', num_layers)
colors = [cmap(l) for l in layers]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
bars = ax.bar(range(len(fractions)), fractions, color=colors)

# Labels and title
ax.set_ylabel('Self-Attending Fraction')
ax.set_title('Self-Attending Heads')
ax.set_xticks(range(len(fractions)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_xlim(-0.5, len(fractions) - 0.5)

# Add legend for layers
legend_elements = [Line2D([0], [0], color=cmap(i), lw=4, label=f'Layer {i}') for i in range(num_layers)]
ax.legend(handles=legend_elements, title="Layer")

plt.tight_layout()
plt.savefig(f'self_attending_heads_{model}.png')

def get_subjets(px, py, pz, e, N_SUBJETS=3, JET_ALGO="kt", jet_radius=0.8):
    """
    Declusters a jet into exactly N_SUBJETS using the JET_ALGO and jet_radius provided.

    Args:
        px [np.ndarray]: NumPy array of shape ``[num_particles]`` containing the px of each particle inside the jet
        py [np.ndarray]: NumPy array of shape ``[num_particles]`` containing the py of each particle inside the jet
        pz [np.ndarray]: NumPy array of shape ``[num_particles]`` containing the pz of each particle inside the jet
        e [np.ndarray]: NumPy array of shape ``[num_particles]`` containing the e of each particle inside the jet
        N_SUBJETS [int]: Number of subjets to decluster the jet into
            (default is 3)
        JET_ALGO [str]: The jet declustering algorithm to use. Choices are ["CA", "kt", "antikt"]
            (default is "CA")
        jet_radius [float]: The jet radius to use when declustering
            (default is 0.8)

    Returns:
        subjet_idx [np.array]: NumPy array of shape ``[num_particles]`` with elements
                                representing which subjet the particle belongs to
        subjet_vectors [list]: includes bjet information (e.g. px, py, pz)

    """
    import awkward as ak
    import fastjet
    import vector

    if JET_ALGO == "kt":
        JET_ALGO = fastjet.kt_algorithm
    elif JET_ALGO == "antikt":
        JET_ALGO = fastjet.antikt_algorithm
    elif JET_ALGO == "CA":
        JET_ALGO = fastjet.cambridge_algorithm

    jetdef = fastjet.JetDefinition(JET_ALGO, jet_radius)

    # define jet directly not an array of jets
    jet = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": e,
        },
        with_name="MomentumArray4D",
    )

    pseudojet = [
        fastjet.PseudoJet(particle.px.item(), particle.py.item(), particle.pz.item(), particle.E.item()) for particle in jet
    ]

    cluster = fastjet.ClusterSequence(pseudojet, jetdef)

    # cluster jets
    jets = cluster.inclusive_jets()
    print(len(jets))
    #assert len(jets) == 1

    # get the 3 exclusive jets
    subjets = cluster.exclusive_subjets(jets[0], N_SUBJETS)
    assert len(subjets) == N_SUBJETS

    # sort by pt
    subjets = sorted(subjets, key=lambda x: x.pt(), reverse=True)

    # define a subjet_idx placeholder
    subjet_idx = ak.zeros_like(px, dtype=int) - 1
    mapping = subjet_idx.to_list()

    subjet_indices = []
    for subjet_idx, subjet in enumerate(subjets):
        subjet_indices.append([])
        for subjet_const in subjet.constituents():
            for idx, jet_const in enumerate(pseudojet):
                if (
                    subjet_const.px() == jet_const.px()
                    and subjet_const.py() == jet_const.py()
                    and subjet_const.pz() == jet_const.pz()
                    and subjet_const.E() == jet_const.E()
                ):
                    subjet_indices[-1].append(idx)


    for subjet_idx, subjet in enumerate(subjets):
        local_mapping = np.array(mapping)
        local_mapping[subjet_indices[subjet_idx]] = subjet_idx
        mapping = local_mapping

    # add the jet index
    jet["subjet_idx"] = ak.Array(mapping)

    subjet_vectors = [
        vector.obj(
            px=ak.sum(jet.px[jet.subjet_idx == j], axis=-1),
            py=ak.sum(jet.py[jet.subjet_idx == j], axis=-1),
            pz=ak.sum(jet.pz[jet.subjet_idx == j], axis=-1),
            E=ak.sum(jet.E[jet.subjet_idx == j], axis=-1),
        )
        for j in range(0, N_SUBJETS)
    ]

    return jet["subjet_idx"].to_numpy(), subjet_vectors


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable

# Define a function to plot the particles and attention lines
def plot_attention_with_particles(attention_head, jet, deta_all, dphi_all, pt_all, subjets_all, 
                                            layer_number, head_number, pf_features, output_filename):

    # Normalize pt values for alpha transparency
    print("Normalizing pt values...")
    norm_pt = mcolors.Normalize(vmin=pt_all.min(), vmax=pt_all.max())

    # Identify the lowest Pt subjet
    unique_subjets = np.unique(subjets_all)
    print(f"Unique subjets: {unique_subjets}")
    lowest_pt_subjet = min(np.unique(subjets_all), key=lambda s: np.sum(pt_all[subjets_all == s]))
    print(f"Lowest Pt subjet: {lowest_pt_subjet}")

    # Use a bright colormap
    colormap = plt.get_cmap('spring')

    # Normalize attention for line transparency/width
    print("Normalizing attention values...")
    norm_attention = mcolors.Normalize(vmin=attention_head.min(), vmax=attention_head.max())


    # Set figure size and DPI for faster rendering
    print("Setting up figure...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # Increase DPI for better rendering

    # Preprocess data to plot particles in batches based on their properties
    print("Categorizing particles...")
    particle_groups = {
        'charged_hadron': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': '^'},
        'neutral_hadron': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'v'},
        'photon': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'o'},
        'electron': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'P'},
        'muon': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'X'},
        'default': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'o'}
    }

    for i in range(len(deta_all)):
        # Alpha transparency based on pt with a minimum of 0.5 and scaling with normalized pt
        alpha = max(0.5, norm_pt(pt_all[i]))
#if subjets_all[i] == lowest_pt_subjet:
        #else:
        other_subjet_idx = np.where(unique_subjets == subjets_all[i])[0][0]
        color = colormap(other_subjet_idx / len(unique_subjets) * 0.7)  # Bright colors

        # Determine the marker based on the particle type in pf_features
        if pf_features[jet][6][i] == 1:  # Charged Hadron
            group = 'charged_hadron'
        elif pf_features[jet][7][i] == 1:  # Neutral Hadron
            group = 'neutral_hadron'
        elif pf_features[jet][8][i] == 1:  # Photon
            group = 'photon'
        elif pf_features[jet][9][i] == 1:  # Electron
            group = 'electron'
        elif pf_features[jet][10][i] == 1:  # Muon
            group = 'muon'
        else:
            group = 'default'

        # Store the particle data in the appropriate group for batch plotting
        particle_groups[group]['deta'].append(deta_all[i])
        particle_groups[group]['dphi'].append(dphi_all[i])
        particle_groups[group]['color'].append(color)
        particle_groups[group]['alpha'].append(alpha)

    print("Plotting particles...")
    # Batch plot particles for each group
    for group, data in particle_groups.items():
        if data['deta']:  # Only plot if there are particles in this group
            # Apply alpha transparency as an array for each particle
            ax.scatter(data['deta'], data['dphi'], color=data['color'], alpha=data['alpha'], s=250, zorder=3, marker=data['marker'],
                       edgecolors='black', linewidths=1.5, antialiased=False)  # Disable anti-aliasing for speed

    # Plot attention lines between particles with values above 0.9
    print("Plotting attention lines...")
    for i in range(attention_head.shape[0]):
        for j in range(attention_head.shape[1]):
            if i != j:  # No self-loops
                # Attention value between particles i and j
                attn_value = attention_head[i, j]
                linestyle = ''
                lineValue=1
                if attn_value > 0:  # Only plot lines for attention values >= 0.9
                    alpha = norm_attention(attn_value)  # Transparency based on attention
                    lineValue = 1
                    # Solid lines within the same subjet, dashed between different subjets
                    linestyle = ''
                    if subjets_all[i] == subjets_all[j]:
                        linestyle = 'solid'
                        lineValue = 1
                    else:
                        linestyle = 'dotted'
                        linestyle = (0, (2, 2))
                        lineValue = 1.3

                    # Plot lines
                    ax.plot([deta_all[i], deta_all[j]], [dphi_all[i], dphi_all[j]], color='black',
                            alpha=alpha, linewidth=lineValue * alpha, linestyle=linestyle, antialiased=False)  # Disable anti-aliasing for speed

    print("Adding legends...")

    # Add the subjet legend without Pt values
    legend1 = ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label=f'Subjet {int(subjet)}', markerfacecolor=colormap(subjet / len(unique_subjets) * 0.7), markersize=10)
        for subjet in unique_subjets
    ], loc='best', fontsize=12, title="Subjets")

    # Add the particle shape legend
    #legend2 = ax.legend(handles=[
    #    plt.Line2D([0], [0], marker=group['marker'], color='w', label=label, markerfacecolor='black', markersize=10)
    #    for label, group in zip(['Charged Hadron', 'Neutral Hadron', 'Photon', 'Electron', 'Muon'], particle_groups.values())
    #], loc='lower right', title="Particle Shapes")

    # Add all legends to the plot
    ax.add_artist(legend1)
    #ax.add_artist(legend2)

    # Set axis labels without bold styling
    ax.set_xlabel(r'$\Delta \eta$')
    ax.set_ylabel(r'$\Delta \phi$')

    # Add title with layer and head information
    ax.set_title(f'Layer {layer_number + 1} - Head {head_number + 1}')
    #ax.set_title('Untrained Model')

    #ax.set_xlim(-0.4, 0.7)
    #ax.set_xticks(np.arange(-0.4, 0.7, 0.2))

    # Add a colorbar for the attention weights at the bottom
    #cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.03])  # Add a colorbar axis below the plot
    cmap = plt.cm.gray_r  # Use a gray colormap to represent attention weights
    vmin=attention_head.min()
    vmax=attention_head.max()
    norm = plt.Normalize(vmin=attention_head.min(), vmax=attention_head.max())  # Normalize based on attention values
    #cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap))
    cb.set_label('Attention Weight')

    # Save the figure instead of showing it
    #print(f"Saving figure to {output_filename}...")
    #plt.savefig(output_filename, bbox_inches='tight')
    #print("Figure saved.")

def plot_attention_with_particles_and_ids(attention_head, jet, deta_all, dphi_all, pt_all, 
                                              subjets_all, layer_number, head_number, 
                                              pf_features, output_filename,):

    # Normalize pt values for alpha transparency
    print("Normalizing pt values...")
    norm_pt = mcolors.Normalize(vmin=pt_all.min(), vmax=pt_all.max())

    # Identify the lowest Pt subjet
    unique_subjets = np.unique(subjets_all)
    print(f"Unique subjets: {unique_subjets}")
    lowest_pt_subjet = min(np.unique(subjets_all), key=lambda s: np.sum(pt_all[subjets_all == s]))
    print(f"Lowest Pt subjet: {lowest_pt_subjet}")

    # Use a bright colormap
    colormap = plt.get_cmap('spring')

    # Normalize attention for line transparency/width
    print("Normalizing attention values...")
    norm_attention = mcolors.Normalize(vmin=attention_head.min(), vmax=attention_head.max())


    # Set figure size and DPI for faster rendering
    print("Setting up figure...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # Increase DPI for better rendering

    # Preprocess data to plot particles in batches based on their properties
    print("Categorizing particles...")
    particle_groups = {
        'charged_hadron': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': '^'},
        'neutral_hadron': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'v'},
        'photon': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'o'},
        'electron': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'P'},
        'muon': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'X'},
        'default': {'deta': [], 'dphi': [], 'pt': [], 'color': [], 'alpha': [], 'marker': 'o'}
    }

    for i in range(len(deta_all)):
        # Alpha transparency based on pt with a minimum of 0.5 and scaling with normalized pt
        alpha = max(0.5, norm_pt(pt_all[i]))
#if subjets_all[i] == lowest_pt_subjet:
        #else:
        other_subjet_idx = np.where(unique_subjets == subjets_all[i])[0][0]
        color = colormap(other_subjet_idx / len(unique_subjets) * 0.7)  # Bright colors


        if pf_features[jet][6][i] == 1:  # Charged Hadron
            group = 'charged_hadron'
        elif pf_features[jet][7][i] == 1:  # Neutral Hadron
            group = 'neutral_hadron'
        elif pf_features[jet][8][i] == 1:  # Photon
            group = 'photon'
        elif pf_features[jet][9][i] == 1:  # Electron
            group = 'electron'
        elif pf_features[jet][10][i] == 1:  # Muon
            group = 'muon'
        else:
            group = 'default'

        # Store the particle data in the appropriate group for batch plotting
        particle_groups[group]['deta'].append(deta_all[i])
        particle_groups[group]['dphi'].append(dphi_all[i])
        particle_groups[group]['color'].append(color)
        particle_groups[group]['alpha'].append(alpha)

    print("Plotting particles...")
    # Batch plot particles for each group
    for group, data in particle_groups.items():
        if data['deta']:  # Only plot if there are particles in this group
            # Apply alpha transparency as an array for each particle
            ax.scatter(data['deta'], data['dphi'], color=data['color'], alpha=data['alpha'], s=250, zorder=3, marker=data['marker'],
                       edgecolors='black', linewidths=1.5, antialiased=False)  # Disable anti-aliasing for speed

    # Plot attention lines between particles with values above 0.9
    print("Plotting attention lines...")
    for i in range(attention_head.shape[0]):
        for j in range(attention_head.shape[1]):
            if i != j:  # No self-loops
                # Attention value between particles i and j
                attn_value = attention_head[i, j]
                linestyle = ''
                lineValue=1
                if attn_value > 0:  # Only plot lines for attention values >= 0.9
                    alpha = norm_attention(attn_value)  # Transparency based on attention
                    lineValue = 1
                    # Solid lines within the same subjet, dashed between different subjets
                    linestyle = ''
                    if subjets_all[i] == subjets_all[j]:
                        linestyle = 'solid'
                        lineValue = 1
                    else:
                        linestyle = 'dotted'
                        linestyle = (0, (2, 2))
                        lineValue = 1.3

                    # Plot lines
                    ax.plot([deta_all[i], deta_all[j]], [dphi_all[i], dphi_all[j]], color='black',
                            alpha=alpha, linewidth=lineValue * alpha, linestyle=linestyle, antialiased=False)  # Disable anti-aliasing for speed

    print("Adding legends...")

    # Add the subjet legend without Pt values
    legend1 = ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label=f'Subjet {int(subjet)}', markerfacecolor=colormap(subjet / len(unique_subjets) * 0.7), markersize=10)
        for subjet in unique_subjets
    ], loc='best', fontsize=12, title="Subjets")

    # Add the particle shape legend
    #legend2 = ax.legend(handles=[
    #    plt.Line2D([0], [0], marker=group['marker'], color='w', label=label, markerfacecolor='black', markersize=10)
    #    for label, group in zip(['Charged Hadron', 'Neutral Hadron', 'Photon', 'Electron', 'Muon'], particle_groups.values())
    #], loc='upper left', title="Particle Shapes")

    # Add all legends to the plot
    ax.add_artist(legend1)
    #ax.add_artist(legend2)

    # Set axis labels without bold styling
    ax.set_xlabel(r'$\Delta \eta$')
    ax.set_ylabel(r'$\Delta \phi$')

    # Add title with layer and head information
    ax.set_title(f'Layer {layer_number + 1} - Head {head_number + 1}')
    #ax.set_title('Untrained Model')

    #ax.set_xlim(-0.4, 0.7)
    #ax.set_xticks(np.arange(-0.4, 0.7, 0.2))

    # Add a colorbar for the attention weights at the bottom
    #cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.03])  # Add a colorbar axis below the plot
    cmap = plt.cm.gray_r  # Use a gray colormap to represent attention weights
    vmin=attention_head.min()
    vmax=attention_head.max()
    norm = plt.Normalize(vmin=attention_head.min(), vmax=attention_head.max())  # Normalize based on attention values
    #cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap))
    cb.set_label('Attention Weight')

    # Save the figure instead of showing it
    #print(f"Saving figure to {output_filename}...")
    #plt.savefig(output_filename, bbox_inches='tight')
    #print("Figure saved.")

# Example usage based on your context (assuming pf_features, pf_mask, and attention are already defined)

# For top hadronic, jet = 3
jet = 3
Decay = 'TopHadronic'
number = jet
num = 0
for b in np.squeeze(jc_kin_top_hadronic_pf_mask[number]):
    if b == 0:
        break
    num += 1

print(f'Graphing for {Decay} jet')

# Extract the 4-momentum components for the valid particles
px = jc_kin_top_hadronic_pf_vectors[jet][0][0:num]
py = jc_kin_top_hadronic_pf_vectors[jet][1][0:num]
pz = jc_kin_top_hadronic_pf_vectors[jet][2][0:num]
e = jc_kin_top_hadronic_pf_vectors[jet][3][0:num]

# Get the subjets using the get_subjets function

N_SUBJETS = 3

subjets, subjet_vectors = get_subjets(px, py, pz, e, N_SUBJETS=N_SUBJETS, JET_ALGO="kt")

# Initialize and combine particle data from all types
deta_all = []
dphi_all = []
pt_all = []
subjets_all = []

# Append all particle types
def append_particles(deta, dphi, pt, subjets, deta_all, dphi_all, pt_all, subjets_all):
    deta_all.extend(deta)
    dphi_all.extend(dphi)
    pt_all.extend(pt)
    subjets_all.extend(subjets)

# Process the particles and combine them into one list
append_particles(jc_kin_top_hadronic_pf_features[jet][5][0:num], jc_kin_top_hadronic_pf_features[jet][6][0:num], 
                 jc_kin_top_hadronic_pf_features[jet][0][0:num], subjets,
                 deta_all, dphi_all, pt_all, subjets_all)

# Convert lists to numpy arrays for plotting
deta_all = np.array(deta_all)
dphi_all = np.array(dphi_all)
pt_all = np.array(pt_all)
subjets_all = np.array(subjets_all)

# softmax the pre-softmax attention matrix

#if not os.path.exists('./JetClasskin_attn_plots'):
#    os.makedirs('./JetClasskin_attn_plots')
#if not os.path.exists('./JetClassfull_attn_plots'):
#    os.makedirs('./JetClassfull_attn_plots')

layer_number = 7  # Choose the layer

jc_kin_pre_softmax_attentions = jc_kin_hooks.pre_softmax_attentions[layer_number][jet][:, :num, :num]
for head in range(jc_kin_pre_softmax_attentions.shape[0]):
    jc_kin_pre_softmax_attentions[head] = torch.nn.functional.softmax(jc_kin_pre_softmax_attentions[head], dim=-1)

# Example attention data, where `x` is the layer number
Decay = Decay
head_number = 0
plot_attention_with_particles_and_ids(attentions[layer_number][head_number, 0:num, 0:num], jet, deta_all, dphi_all, pt_all, 
                                            subjets_all, layer_number, head_number, pf_features=features, 
                                            output_filename=f'./JetClasskin_attn_plots/Jet_{jet}_Decay_{Decay}_Layer_{layer_number+1}_head_{head_number + 1}.pdf',
                                            )
  #plot_attention_with_particles(srandom_matrix[0][1][4, 0:num, 0:num], jet, deta_all, dphi_all, pt_all, subjets_all, layer_number, head_number, pf_features, '/content/drive/MyDrive/networks/Plots/Jet' + str(jet) + str(Decay) + '/' + 'randomAttentionMatrix' + str(Decay) + '-layer'+str(layer_number + 1) + '-head' + str(head_number + 1) + '.pdf')

# Extract the 4-momentum components for the valid particles
px = jc_full_top_hadronic_pf_vectors[jet][0][0:num]
py = jc_full_top_hadronic_pf_vectors[jet][1][0:num]
pz = jc_full_top_hadronic_pf_vectors[jet][2][0:num]
e = jc_full_top_hadronic_pf_vectors[jet][3][0:num]

# Get the subjets using the get_subjets function

subjets, subjet_vectors = get_subjets(px, py, pz, e, N_SUBJETS=N_SUBJETS, JET_ALGO="kt")

# Initialize and combine particle data from all types
deta_all = []
dphi_all = []
pt_all = []
subjets_all = []

# Append all particle types
def append_particles(deta, dphi, pt, subjets, deta_all, dphi_all, pt_all, subjets_all):
    deta_all.extend(deta)
    dphi_all.extend(dphi)
    pt_all.extend(pt)
    subjets_all.extend(subjets)

# Process the particles and combine them into one list
append_particles(jc_full_top_hadronic_pf_features[jet][15][0:num], 
                 jc_full_top_hadronic_pf_features[jet][16][0:num], 
                 jc_full_top_hadronic_pf_features[jet][0][0:num], subjets,
                 deta_all, dphi_all, pt_all, subjets_all)

# Convert lists to numpy arrays for plotting
deta_all = np.array(deta_all)
dphi_all = np.array(dphi_all)
pt_all = np.array(pt_all)
subjets_all = np.array(subjets_all)

#softmax the pre-softmax attention matrix for jc_full

layer_number = 7

jc_full_pre_softmax_attentions = jc_full_hooks.pre_softmax_attentions[layer_number][jet][:, :num, :num]
for head in range(jc_full_pre_softmax_attentions.shape[0]):
    jc_full_pre_softmax_attentions[head] = torch.nn.functional.softmax(jc_full_pre_softmax_attentions[head], dim=-1)

Decay = Decay
head_number = 6
plot_attention_with_particles(jc_full_pre_softmax_attentions[head_number, 0:num, 0:num], jet, deta_all, dphi_all, pt_all, subjets_all, 
                                  layer_number, head_number, pf_features=jc_full_top_hadronic_pf_features, 
                                  output_filename=f'./JetClassfull_attn_plots/Jet_{jet}_Decay_{Decay}_Layer_{layer_number+1}_head_{head_number + 1}.pdf')


jc_kin_model, _ = get_model('jck')
jc_full_model, _ = get_model('jc_full')

jc_kin_hooks = Pre_Softmax_Hook(model=jc_kin_model)
jc_full_hooks = Pre_Softmax_Hook(model=jc_full_model)