''' 
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn.functional as F
from torch import nn, einsum

from math import pi, log
from functools import wraps

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
    def __init__(self, dim, mult=1, dropout=0.):
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

    def forward(self, x, context=None, mask=None, topk=2, get_attn=False):
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

        if not get_attn:
            return self.to_out(out)
        else:
            return self.to_out(out), attn

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

class BasicBlockRetriever(nn.Module):
    expansion = 1


    def __init__(self, 
            in_planes, 
            planes, 
            retrieval_buffer,
            encoder_type,
            retrieval_depth,
            cross_heads=1,
            cross_dim_head=64,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
            buffer_model=None,
            attn_retrieval="reps_for_k",
            retrieval_access="sampled",
            no_labels_from_reps=False,
            retrieval_reps_size=None,
            eval_retrieval=True,
            stride=1
        ):
        super(BasicBlockRetriever, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.planes = planes

        # retrieval stuff
        self.buffer_model = buffer_model
        self.attn_retrieval = attn_retrieval
        self.retrieval_access = retrieval_access
        self.retrieval_activated = True
        self.eval_retrieval = eval_retrieval
        self.encoder_type = encoder_type
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

        get_cross_attn_retrieval = None
        get_cross_ff_retrieval = None
        if self.retrieval_activated:
            get_cross_attn_retrieval = lambda: PreNorm(planes,
                                                       AttentionKVSplitted(planes,
                                                                           self.buffer_dims,
                                                                           retrieval_data_dim,
                                                                           heads=cross_heads,
                                                                           dim_head=cross_dim_head,
                                                                           dropout=attn_dropout,
                                                                           attn_retrieval=attn_retrieval,
                                                                           embedding_from_query=self.retrieval_access=="knn_from_learned_embedding"),
                                                       context_dim=retrieval_data_dim)
            get_cross_ff_retrieval = lambda: PreNorm(planes, FeedForward(planes, dropout=ff_dropout))
            get_cross_attn_retrieval, get_cross_ff_retrieval = map(cache_fn, (
                get_cross_attn_retrieval, get_cross_ff_retrieval))
            
            self.retrieval_layers = nn.ModuleList([])

            for i in range(retrieval_depth):
                should_cache = i > 0 and weight_tie_layers
                cache_args = {'_cache': should_cache}
                
                self.retrieval_layers.append(nn.ModuleList([
                    get_cross_attn_retrieval(**cache_args),
                    get_cross_ff_retrieval(**cache_args)
                ]))
        else:
            self.retrieval_layers = None

    def sample_retrieval_batch_idx(self, batch_size, buffer_size, device):
        idx = torch.zeros((batch_size, buffer_size), device=device)
        for sample in range(batch_size):
            idx[sample] = torch.randperm(self.retrieval_data.shape[0], device=device)[:buffer_size]
        return idx

    def sampled_retrieval(self, obs, buffer_size, buffer_data_model, device):
        with torch.no_grad():
           retrieval_idx = self.sample_retrieval_batch_idx(batch_size=obs.shape[0], buffer_size=buffer_size,
                                                           device=device)
           retrieval_idx = torch.unsqueeze(retrieval_idx, 2).type(torch.int64)

           retrieval_data = self.retrieval_data.expand(obs.shape[0], self.retrieval_data.shape[1], -1)

           retrieval_idx = torch.repeat_interleave(retrieval_idx, retrieval_data.shape[2], dim=2)
           retrieved_data = torch.gather(retrieval_data, 1, retrieval_idx)
           try:
               retrieved_labels = torch.argmax(retrieved_data[:, :, self.buffer_dims[0]:], dim=2)
           except:
               retrieved_labels = None

        return retrieved_data, retrieved_labels
                
    def retrieve_nearest_neighbors(self, obs, buffer_data_model, topk=2):
        with torch.no_grad():
            if self.encoder_type=="vae":
                mean, logvar = buffer_data_model.encoder(obs)
                z = buffer_data_model.reparameterize(mean, logvar)
                input_reps = z.flatten(start_dim=1).to(obs.device)
            elif self.encoder_type=="resnet18":
                input_reps = buffer_data_model(obs)['feats'].to(obs.device)
            else:
                raise NotImplementedError

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
            retrieved_data = torch.gather(retrieval_data, dim=1, index=idx_knn)

            retrieved_labels = None  # TODO: fix later

        return retrieved_data, retrieved_labels


    def forward(self, fw_dict):

        x = fw_dict['inp']
        obs = fw_dict['obs']
        mask = fw_dict['mask'] if 'mask' in fw_dict.keys() else None
        return_embeddings = fw_dict['return_embeddings'] if 'return_embeddings' in fw_dict.keys() else False
        sampled_buffer_size = fw_dict['sampled_buffer_size'] if 'sampled_buffer_size' in fw_dict.keys() else 20
        buffer_data_model = fw_dict['buffer_data_model'] if 'buffer_data_model' in fw_dict.keys() else None
        retrieval_buffer = fw_dict['retrieval_buffer'] if 'retrieval_buffer' in fw_dict.keys() else None
        labels = fw_dict['labels'] if 'labels' in fw_dict.keys() else None

        device = x.device
        
        if buffer_data_model is None:
            buffer_data_model = self.buffer_model

        # pass observation through resnet block
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        b, d, h, w = out.shape

        out = out.reshape(b, h * w, d)
    
        # setup retrieval for forward pass
        sampled_retrieval_data = None
        retrieved_labels = None

        # layers
        if self.retrieval_activated:
            if self.retrieval_access == "sampled":
                retrieved_data, retrieved_labels = self.sampled_retrieval(obs, buffer_size=sampled_buffer_size, 
                                                                          buffer_data_model=buffer_data_model, device=device)
            elif self.retrieval_access == "knn":
                retrieved_data, retrieved_labels = self.retrieve_nearest_neighbors(obs, buffer_data_model,
                                                                                   topk=sampled_buffer_size)
            else:
                raise Exception(f"retrieval access type {self.retrieval_access} not implemented")

            for cross_attn_retrieval, cross_ff_retrieval in self.retrieval_layers:
                out = cross_attn_retrieval(out, context=retrieved_data, mask=mask, topk=sampled_buffer_size) + out
                out = cross_ff_retrieval(out) + out

        out = rearrange(out, 'b (m n) d -> b d m n', m=h, n=w) 

        fw_dict['inp'] = out

        return fw_dict

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self, 
            block, 
            num_blocks, 
            num_classes=1000,
            retrieval_activated=False,
            **kwargs):
        super(ResNet, self).__init__()

        # ResNet backbone setup
        self.in_planes = 64

        self.retrieval_activated = retrieval_activated

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, **kwargs)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if self.retrieval_activated:
            print(kwargs)
            for stride in strides:
                layers.append(block(
                    in_planes=self.in_planes, 
                    planes=planes, 
                    stride=stride, **kwargs))
                self.in_planes = planes * block.expansion
        else:
            for stride in strides:
                layers.append(block(
                    in_planes=self.in_planes, 
                    planes=planes, 
                    stride=stride))
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))

        if self.retrieval_activated:
            fw_dict = {'inp': out, 'obs': x}
            for k, v in zip(kwargs.keys(), kwargs.values()):
                fw_dict[str(k)] = v

            fw_dict = self.layer1(fw_dict)
            fw_dict = self.layer2(fw_dict)
            fw_dict = self.layer3(fw_dict)
            fw_dict = self.layer4(fw_dict)

            out = fw_dict['inp']
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        pred = self.linear(out)
        return {"prediction": pred, "rep": out}


def ResNetRetrieval18(**kwargs):
        return ResNet(BasicBlockRetriever, [2, 2, 2, 2], retrieval_activated=True, **kwargs)

def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
