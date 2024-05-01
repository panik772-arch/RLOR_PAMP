import math

import torch
from torch import nn

# Use


class AttentionScore(nn.Module):
    r"""
    A helper class for attention operations.
    There are no parameters in this module.
    This module computes the alignment score with mask
    and return only the attention score.

    The default operation is

    .. math::
         \pmb{u} = \mathrm{Attention}(q,\pmb{k}, \mathrm{mask})

    where for each key :math:`k_j`, we have

    .. math::
        u_j =
        \begin{cases}
             &\frac{q^Tk_j}{\sqrt{\smash{d_q}}} & \text{ if } j \notin \mathrm{mask}\\
             &-\infty & \text{ otherwise. }
        \end{cases}

    If ``use_tanh`` is ``True``, apply clipping on the logits :math:`u_j` before masking:

    .. math::
        u_j =
        \begin{cases}
             &C\mathrm{tanh}\left(\frac{q^Tk_j}{\sqrt{\smash{d_q}}}\right) & \text{ if } j \notin \mathrm{mask}\\
             &-\infty & \text{ otherwise. }
        \end{cases}

    Args:
        use_tanh: if True, use clipping on the logits
        C: the range of the clipping [-C,C]
    Inputs: query, keys, mask
        * **query** : [..., 1, h_dim]
        * **keys**: [..., graph_size, h_dim]
        * **mask**: [..., graph_size] ``logits[...,j]==-inf`` if ``mask[...,j]==True``.
    Outputs: logits
        * **logits**: [..., 1, graph_size] The attention score for each key.
    """

    def __init__(self, use_tanh=True, C=10):
        super(AttentionScore, self).__init__()
        self.use_tanh = use_tanh
        self.C = C

    def forward(self, query, key, mask=torch.zeros([], dtype=torch.bool), state = None ):
        u = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if state is not None and state.problem == "cvrp_fleet_tw":
            self.use_tanh = False
            ratio, _,_,_ = state.tw_ratio()
            #Apply the transformation across the entire dist_matrix first
            # tw_ratio between [0,1]. if it >1 -> -inf, so not reachable.. if it 0, no time pressure. if it goes -> 1, time pressure increase..
            # so we want to attend to nodes with high time pressure
            time_ratio = torch.where(ratio > 1, float("-inf"), ratio)
            u += time_ratio

            #transformed_distances, _ = state.return_topK_closest()
            # Adjust u based on transformed_distances directly
            # This assumes transformed_distances have been processed as per DAR method logic in stateWrapper
            #u += transformed_distances

        if self.use_tanh:
            logits = torch.tanh(u) * self.C
        else:
            logits = u
            # Adjust mask if provided
        # in cvrp_env (8,1024, 51,51), with no mask and when mask (1024,50,51) with (False, True,True..) then Key (8,1024,51,16) and Query (8,1024,50,16)->
        # or when one head (sometimes too) -> key (1024, 51,128) and query (1024,50,128) -> logits (1024,50,51)
        logits[mask.expand_as(logits)] = float("-inf")  # masked after clipping

        return logits

        ''' 
        #######  only for cvrp_fleet_env ##########
        if len(u.size()) == 4 and len(mask.size()) == 3:
            # Assuming mask is initially of shape (1024, 50, 51)
            # Pad mask to match logits shape, i.e., (8, 1024, 51, 51)
            pad_value = float('-inf')
            # Pad the second dimension (trajectories) to go from 50 to 51
            padded_mask = torch.nn.functional.pad(mask, (0, 0, 1, 0), value=pad_value)
            # Expand mask dimensions for broadcasting, no need to explicitly match batch and head dimensions due to broadcasting
            expanded_mask = padded_mask.unsqueeze(0)  # Prepares for broadcasting by matching the head dimension

            # Apply the mask
            logits = torch.where(expanded_mask.expand_as(logits), logits,
                                 torch.tensor(float("-inf"), device=logits.device))
            print(f'logits from decoder attention core for each heads (note, I added one trajectory here) {logits}')
            return logits

        elif len(u.size()) == 3 and len(mask.size()) == 3:
            pad_value = float('-inf')
            # Pad the second dimension (trajectories) to go from 50 to 51
            padded_mask = torch.nn.functional.pad(mask, (0, 0, 1, 0), value=pad_value)

            # Apply the mask
            logits = torch.where(padded_mask.expand_as(logits), logits,
                                 torch.tensor(float("-inf"), device=logits.device))
            print(f'logits from decoder attention core for the whole batch (note, I added one trajectory here) {logits}')
            return logits

        else:

            logits[mask.expand_as(logits)] = float("-inf")  # masked after clipping
            return logits
        '''


class MultiHeadAttention(nn.Module):
    r"""
    Compute the multi-head attention.

    .. math::
        q^\prime = \mathrm{MultiHeadAttention}(q,\pmb{k},\pmb{v},\mathrm{mask})

    The following is computed:

    .. math::
        \begin{aligned}
        \pmb{a}^{(j)} &= \mathrm{Softmax}(\mathrm{AttentionScore}(q^{(j)},\pmb{k}^{(j)}, \mathrm{mask}))\\
        h^{(j)} &= \sum\nolimits_i \pmb{a}^{(j)}_i\pmb{v}_i \\
        q^\prime &= W^O \left[h^{(1)},...,h^{(J)}\right]
        \end{aligned}

    Args:
        embedding_dim: dimension of the query, keys, values
        n_head: number of heads
    Inputs: query, keys, value, mask
        * **query** : [batch, n_querys, embedding_dim]
        * **keys**: [batch, n_keys, embedding_dim]
        * **value**: [batch, n_keys, embedding_dim]
        * **mask**: [batch, 1, n_keys] ``logits[batch,j]==-inf`` if ``mask[batch, 0, j]==True``
    Outputs: logits, out
        * **out**: [batch, 1, embedding_dim] The output of the multi-head attention
    """

    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attentionScore = AttentionScore()
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, query, key, value, mask):
        query_heads = self._make_heads(query)
        key_heads = self._make_heads(key)
        value_heads = self._make_heads(value)

        # [n_heads, batch, 1, nkeys]
        compatibility = self.attentionScore(query_heads, key_heads, mask)

        # [n_heads, batch, 1, head_dim]
        out_heads = torch.matmul(torch.softmax(compatibility, dim=-1), value_heads)

        # from multihead [nhead, batch, 1, head_dim] -> [batch, 1, nhead* head_dim]
        out = self.project_out(self._unmake_heads(out_heads))
        return out

    def _make_heads(self, v):
        batch_size, nkeys, h_dim = v.shape
        #  [batch_size, ..., n_heads* head_dim] --> [n_heads, batch_size, ..., head_dim]
        out = v.reshape(batch_size, nkeys, self.n_heads, h_dim // self.n_heads).movedim(-2, 0)
        return out

    def _unmake_heads(self, v):
        #  [n_heads, batch_size, ..., head_dim] --> [batch_size, ..., n_heads* head_dim]
        out = v.movedim(0, -2).flatten(-2)
        return out


class MultiHeadAttentionProj(nn.Module):
    r"""
    Compute the multi-head attention with projection.
    Different from :class:`.MultiHeadAttention` which accepts precomputed query, keys, and values,
    this module computes linear projections from the inputs to query, keys, and values.

    .. math::
        q^\prime = \mathrm{MultiHeadAttentionProj}(q_0,\pmb{h},\mathrm{mask})

    The following is computed:

    .. math::
        \begin{aligned}
        q, \pmb{k}, \pmb{v} &= W^Qq_0, W^K\pmb{h}, W^V\pmb{h}\\
        \pmb{a}^{(j)} &= \mathrm{Softmax}(\mathrm{AttentionScore}(q^{(j)},\pmb{k}^{(j)}, \mathrm{mask}))\\
        h^{(j)} &= \sum\nolimits_i \pmb{a}^{(j)}_i\pmb{v}_i \\
        q^\prime &= W^O \left[h^{(1)},...,h^{(J)}\right]
        \end{aligned}

    if :math:`\pmb{h}` is not given. This module will compute the self attention of :math:`q_0`.

    .. warning::
        The results of the in-projection of query, key, value are
        slightly different (order of ``1e-6``) with the original implementation.
        This is due to the numerical accuracy.
        The two implementations differ by the way of multiplying matrix.
        Thus, different internal implementation libraries of pytorch are called
        and the results are slightly different.
        See the pytorch docs on `numerical accruacy <https://pytorch.org/docs/stable/notes/numerical_accuracy.html>`_ for detail.

    Args:
        embedding_dim: dimension of the query, keys, values
        n_head: number of heads
    Inputs: q, h, mask
        * **q** : [batch, n_querys, embedding_dim]
        * **h**: [batch, n_keys, embedding_dim]
        * **mask**: [batch, n_keys] ``logits[batch,j]==-inf`` if ``mask[batch,j]==True``
    Outputs: out
        * **out**: [batch, n_querys, embedding_dim] The output of the multi-head attention


    """

    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttentionProj, self).__init__()

        self.queryEncoder = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.keyEncoder = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.valueEncoder = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.MHA = MultiHeadAttention(embedding_dim, n_heads)

    def forward(self, q, h=None, mask=torch.zeros([], dtype=torch.bool)):

        if h is None:
            h = q  # compute self-attention

        query = self.queryEncoder(q)
        key = self.keyEncoder(h)
        value = self.valueEncoder(h)

        out = self.MHA(query, key, value, mask)

        return out
