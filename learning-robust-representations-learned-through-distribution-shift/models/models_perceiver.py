from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AttentionKVSplitted(nn.Module):
    def __init__(self, query_dim, buffer_dims, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 attn_retrieval="reps_for_v", embedding_from_query=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.buffer_dims = buffer_dims
        self.scale = dim_head ** -0.5
        self.heads = heads
        if attn_retrieval == "reps_for_k":
            self.k_dim = buffer_dims[0]
            self.v_dim = buffer_dims[1]
            self.reps_embedding_dim = buffer_dims[0]
        elif attn_retrieval == "reps_for_v":
            self.k_dim = buffer_dims[1]
            self.v_dim = buffer_dims[0]
            self.reps_embedding_dim = buffer_dims[0]
        else:
            self.k_dim = sum(buffer_dims)
            self.v_dim = sum(buffer_dims)
            self.reps_embedding_dim = buffer_dims[0]

        self.attn_retrieval = attn_retrieval

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(self.k_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(self.v_dim, inner_dim, bias=False)

        self.embedding_from_query = embedding_from_query
        if self.embedding_from_query:  # learn an embedding from query instead of using pretrained model
            self.q_to_embedding = nn.Linear(inner_dim, self.reps_embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, topk=2):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)

        if self.embedding_from_query:  # context is full retrieval buffer in this case
            embedding = self.q_to_embedding(q)
            context, _ = self.retrieve_nearest_neighbors(embedding, context, topk=topk)

        if self.attn_retrieval == "reps_for_v":
            # context_reps, context_labels = context.chunk(2, dim=-1)
            context_reps, context_labels = torch.split(context,
                                                       self.buffer_dims,
                                                       dim=-1)
            k = self.to_k(context_labels)
            v = self.to_v(context_reps)
        elif self.attn_retrieval == "reps_for_k":
            # context_reps, context_labels = context.chunk(2, dim=-1)
            context_reps, context_labels = torch.split(context,
                                                       self.buffer_dims,
                                                       dim=-1)
            k = self.to_k(context_reps)
            v = self.to_v(context_labels)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        if self.embedding_from_query:
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
            k, v = map(lambda t: rearrange(t, 'b n i (h d) -> (b h) n i d', h=h), (k, v))
            sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale


        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        if self.embedding_from_query:
            out = einsum('b i j, b i j d -> b i d', attn, v)
        else:
            out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

    def retrieve_nearest_neighbors(self, embedding, retrieval_data, topk=2):
        retrieved_data = torch.zeros((embedding.shape[0], embedding.shape[1], topk, retrieval_data.shape[-1]),
                                     device=embedding.device)
        retrieved_labels = None

        for i in range(embedding.shape[1]):
            input_reps = embedding[:, 0, :].flatten(start_dim=1)

            buffer_reps, _ = torch.split(retrieval_data,
                                         self.buffer_dims,
                                         dim=-1)

            input_reps = input_reps.view(input_reps.shape[0], 1, input_reps.shape[-1])
            input_reps = input_reps.expand(input_reps.shape[0], buffer_reps.shape[1], -1)

            dist = torch.norm(buffer_reps - input_reps, dim=-1, p=None)
            knn = dist.topk(topk, largest=False)
            # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))

            idx_knn = knn.indices
            idx_knn = idx_knn.view(idx_knn.shape[0], idx_knn.shape[1], 1)
            idx_knn = idx_knn.expand(-1, -1, retrieval_data.shape[-1])
            retrieval_data_for_emb = torch.gather(retrieval_data, dim=1, index=idx_knn)

            retrieved_data[:, i, :, :] = retrieval_data_for_emb

        return retrieved_data, retrieved_labels


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# main class

class PerceiverRetriever(nn.Module):
    def __init__(
            self,
            *,
            num_freq_bands,
            depth,
            max_freq,
            retrieval_buffer,
            input_channels=3,
            input_axis=2,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=1000,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
            fourier_encode_data=True,
            self_per_cross_attn=1,
            final_classifier_head=True,
            attn_retrieval="reps_as_k",
            retrieval_access="sampled",
            no_labels_from_reps=False,
            buffer_model=None,
            return_only_pred=False,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)
        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        assert attn_retrieval in ["reps_for_k", "reps_for_v", "reps_for_both"]
        assert retrieval_access in ["sampled", "knn"]
        self.retrieval_access = retrieval_access
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.return_only_pred = return_only_pred

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        # retrieval stuff
        self.buffer_model = buffer_model
        self.attn_retrieval = attn_retrieval
        self.retrieval_activated = True
        if retrieval_buffer is None:
            self.retrieval_activated = False
        else:
            self.stored_reps = retrieval_buffer["reps"]
            self.stored_labels = retrieval_buffer["labels"].float()
            if no_labels_from_reps:
                self.retrieval_data = self.stored_reps
                self.buffer_dims = [self.stored_reps.shape[-1]]
            else:
                self.retrieval_data = torch.cat((self.stored_reps, self.stored_labels), dim=1)
                self.buffer_dims = [self.stored_reps.shape[-1], self.stored_labels.shape[-1]]
            retrieval_data_dim = self.retrieval_data.shape[1]
            self.retrieval_data = torch.unsqueeze(self.retrieval_data, 0)
        self.no_labels_from_reps = no_labels_from_reps

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (
            get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))
        get_cross_attn_retrieval = None
        get_cross_ff_retrieval = None
        if self.retrieval_activated:
            get_cross_attn_retrieval = lambda: PreNorm(latent_dim,
                                                       AttentionKVSplitted(latent_dim,
                                                                           self.buffer_dims,
                                                                           retrieval_data_dim,
                                                                           heads=cross_heads,
                                                                           dim_head=cross_dim_head,
                                                                           dropout=attn_dropout,
                                                                           attn_retrieval=attn_retrieval,
                                                                           embedding_from_query=self.retrieval_access=="knn_from_learned_embedding"),
                                                       context_dim=retrieval_data_dim)
            get_cross_ff_retrieval = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
            get_cross_attn_retrieval, get_cross_ff_retrieval = map(cache_fn, (
                get_cross_attn_retrieval, get_cross_ff_retrieval))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=block_ind),
                    get_latent_ff(**cache_args, key=block_ind)
                ]))

            if self.retrieval_activated:
                self.layers.append(nn.ModuleList([
                    get_cross_attn(**cache_args),
                    get_cross_ff(**cache_args),
                    get_cross_attn_retrieval(**cache_args),
                    get_cross_ff_retrieval(**cache_args),
                    self_attns
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    get_cross_attn(**cache_args),
                    get_cross_ff(**cache_args),
                    self_attns
                ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def sample_retrieval_batch_idx(self, batch_size, buffer_size, device):
        idx = torch.zeros((batch_size, buffer_size), device=device)
        for sample in range(batch_size):
            idx[sample] = torch.randperm(self.retrieval_data.shape[1], device=device)[:buffer_size]
        return idx

    def retrieve_nearest_neighbors(self, obs, buffer_data_model, topk=2):

        with torch.no_grad():
            mean, logvar = buffer_data_model.encoder(obs)
            z = buffer_data_model.reparameterize(mean, logvar)
            input_reps = z.flatten(start_dim=1).to(obs.device)
            if self.no_labels_from_reps:
                buffer_reps = self.retrieval_data
            else:
                # buffer_reps, _ = self.retrieval_data.chunk(2, dim=-1)
                buffer_reps, _ = torch.split(self.retrieval_data,
                                             self.buffer_dims,
                                             dim=-1)
            buffer_reps = buffer_reps.expand(obs.shape[0], buffer_reps.shape[1], -1)
            retrieval_data = self.retrieval_data.expand(obs.shape[0], self.retrieval_data.shape[1], -1)


            input_reps = input_reps.view(input_reps.shape[0], 1, input_reps.shape[-1])
            input_reps = input_reps.expand(input_reps.shape[0], buffer_reps.shape[1], -1)

            dist = torch.norm(buffer_reps - input_reps, dim=-1, p=None)
            knn = dist.topk(topk, largest=False)
            # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))

            idx_knn = knn.indices
            idx_knn = idx_knn.view(idx_knn.shape[0], idx_knn.shape[1], 1)
            idx_knn = idx_knn.expand(-1, -1, retrieval_data.shape[-1])
            retrieval_data = torch.gather(retrieval_data, dim=1, index=idx_knn)

            retrieved_labels = None  # TODO: fix later

            return retrieval_data, retrieved_labels

    def forward(
            self,
            obs,
            mask=None,
            return_embeddings=False,
            sampled_buffer_size=20,
            buffer_data_model=None
    ):
        data = rearrange(obs, 'b c h w -> b h w c')
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if buffer_data_model is None:
            buffer_data_model = self.buffer_model

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            data = torch.cat((data, enc_pos), dim=-1)

        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        # sampled_retrieval_data = torch.randn((b, sampled_buffer_size, self.retrieval_data.shape[-1]), device=device)
        sampled_retrieval_data = None
        retrieved_labels = None
        # layers
        if self.retrieval_activated:
            retrieval_data = self.retrieval_data.expand(obs.shape[0], self.retrieval_data.shape[1], -1)
            if self.retrieval_access == "sampled":
                retrieval_idx = self.sample_retrieval_batch_idx(batch_size=b, buffer_size=sampled_buffer_size,
                                                                device=device)
                retrieval_idx = torch.unsqueeze(retrieval_idx, 2).type(torch.int64)
                retrieval_idx = torch.repeat_interleave(retrieval_idx, retrieval_data.shape[2], dim=2)
                retrieved_data = torch.gather(retrieval_data, 1, retrieval_idx)
                retrieved_labels = torch.argmax(retrieved_data[:, :, self.buffer_dims[0]:], dim=2)
            elif self.retrieval_access == "knn":
                retrieved_data, retrieved_labels = self.retrieve_nearest_neighbors(obs, buffer_data_model,
                                                                                   topk=sampled_buffer_size)
            else:
                raise Exception(f"retrieval access type {self.retrieval_access} not implemented")

            for cross_attn, cross_ff, cross_attn_retrieval, cross_ff_retrieval, self_attns in self.layers:
                x = cross_attn(x, context=data, mask=mask) + x
                x = cross_ff(x) + x

                x = cross_attn_retrieval(x, context=retrieved_data, mask=mask, topk=sampled_buffer_size) + x
                x = cross_ff_retrieval(x) + x

                for self_attn, self_ff in self_attns:
                    x = self_attn(x) + x
                    x = self_ff(x) + x
        else:
            for cross_attn, cross_ff, self_attns in self.layers:
                x = cross_attn(x, context=data, mask=mask) + x
                x = cross_ff(x) + x

                for self_attn, self_ff in self_attns:
                    x = self_attn(x) + x
                    x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        if self.return_only_pred:
            return self.to_logits(x)
        else:
            return {"prediction": self.to_logits(x),
                    "sampled_retrieval_labels": retrieved_labels}
