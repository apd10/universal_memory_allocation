from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter
import math

import hashed_embedding_bag
import pdb

class HashedEmbeddingBagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hashed_weights, indices, offsets, mode, embedding_dim, signature, random_numbers, hmode, keymode, val_offset, norm, key_bits, keys_to_use, uma_chunk_size):
        if indices.dim() == 2:
            if offsets is not None:
                raise ValueError("if indices is 2D, then offsets has to be None"
                                ", as indices is treated is a mini-batch of"
                                " fixed length sequences. However, found "
                                "offsets of type {}".format(type(offsets)))
            offsets = torch.arange(0, indices.numel(), indices.size(1),
                                dtype=torch.long, device=indices.device)
            indices = indices.reshape(-1)
        elif indices.dim() == 1:
            if offsets is None:
                raise ValueError("offsets has to be a 1D Tensor but got None")
            if offsets.dim() != 1:
                raise ValueError("offsets has to be a 1D Tensor")
        else:
            raise ValueError("indices has to be 1D or 2D Tensor,"
                            " but got Tensor of dimension {}".format(indices.dim()))

        if mode == 'sum':
            mode_enum = 0
        elif mode == 'mean':
            mode_enum = 1
            raise ValueError("mean mode not supported")
        elif mode == 'max':
            mode_enum = 2
            raise ValueError("max mode not supported")

        if hmode == "rand_hash":
            hmode_enum = 0
        elif hmode == "lma_hash":
            hmode_enum = 1
        else:
            raise ValueError("hmode not defined")


        if keymode == "keymode_hashweight":
            keymode_enum = 0;
        elif keymode == "keymode_static_pm":
            keymode_enum = 1;
        else:
            raise ValueError("keymode not defined")

        if val_offset is not None:
            indices = indices + val_offset

        
        hashed_weights_size = hashed_weights.size(0)
        output, offset2bag, bag_size, max_indices, hashed_idx = \
            hashed_embedding_bag.forward(hashed_weights, indices, offsets, mode_enum, embedding_dim, signature, random_numbers, hmode_enum, keymode_enum, key_bits, keys_to_use, uma_chunk_size)
        if norm is not None:
            #assert(keymode_enum == 1)
            output = output/norm
        ctx.save_for_backward(indices, offsets, offset2bag, bag_size, max_indices, hashed_idx)
        ctx.mode_enum = mode_enum
        ctx.hashed_weights_size = hashed_weights_size
        ctx.keymode_enum = keymode_enum
        return output

    @staticmethod
    def backward(ctx, grad):
        indices, offsets, offset2bag, bag_size, max_indices, hashed_idx = ctx.saved_variables
        hashed_weights_size = ctx.hashed_weights_size
        mode_enum = ctx.mode_enum
        keymode_enum = ctx.keymode_enum
        embedding_dim = grad.size(1)
        if keymode_enum == 0:
            weight_grad = hashed_embedding_bag.backward(
                grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights_size, False, mode_enum, embedding_dim)
        elif keymode_enum == 1:
            weight_grad = None
        return weight_grad, None, None, None, None, None, None, None,None,None,None,None,None, None
    '''

    @staticmethod
    def backward(ctx, grad):
        keymode_enum = ctx.keymode_enum
        if keymode_enum == 0:
            indices, offsets, offset2bag, bag_size, max_indices, hashed_idx = ctx.saved_variables
            hashed_weights_size = ctx.hashed_weights_size
            if hashed_idx.is_contiguous():
                hashed_idx1 = hashed_idx.view(-1)
            else:
                hashed_idx1 = hashed_idx.reshape(-1)
            if grad.is_contiguous():
                grad1 = grad.view(-1)
            else:
                grad1 = grad.reshape(-1)
            weight_grad = torch.zeros(hashed_weights_size).to(indices.device)
            weight_grad.scatter_add_(0, hashed_idx1, grad1)
        elif keymode_enum == 1:
            weight_grad = None
        return weight_grad, None, None, None, None, None, None, None,None,None,None,None
    '''

class HashedEmbeddingBag(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        compression:float = 1. / 64., 
        mode:str = "sum", 
        _weight: Optional[torch.Tensor] = None,
        signature: Optional[torch.Tensor] = None,
        key_bits=4,
        keys_to_use=8,
        hmode = "rand_hash",
        keymode = "keymode_hashweight",
        val_offset = None,
        seed = 1024,
        uma_chunk_size = 1)->None:
        super(HashedEmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        memory = int(num_embeddings * embedding_dim * compression + 1)
        #memory = int(np.exp2(int(np.log2(memory)))) #  make sure it is power of 2
        self.weight_size = memory
        if keymode != "keymode_hashweight":
            assert(_weight is None)
        self.val_offset = val_offset
        self.mode = mode
        self.hmode = hmode
        self.keymode = keymode
        self.signature = signature
        self.norm = None
        self.key_bits = key_bits
        self.keys_to_use = keys_to_use
        self.uma_chunk_size = uma_chunk_size
        r = np.random.RandomState(seed)
        random_numbers = np.concatenate([np.array([2038074743]), r.randint(0, 2038074743, (50,))]) # set of 50 random numbers to use
        self.random_numbers = Parameter(torch.from_numpy(random_numbers.astype(np.int64)), requires_grad=False)
        print("RandomNumbers: ", self.random_numbers[:5])
        
        if self.signature is None:
                val = np.zeros(shape=(2,))
                self.signature = Parameter(torch.from_numpy(val.astype(np.int64)), requires_grad=False)
        if _weight is None :
            if keymode == "keymode_hashweight":
                low = -math.sqrt(1 / self.num_embeddings)
                high = math.sqrt(1 / self.num_embeddings)
                self.hashed_weight = Parameter(torch.rand(self.weight_size) * (high - low) + low)
            else:
                self.weight_size = 2
                val = np.random.uniform(low = -1, high = 1, size=(self.weight_size,))
                self.hashed_weight = Parameter(torch.from_numpy(val.astype(np.float32)), requires_grad=False)
                self.hashed_weight.requires_grad = False
                self.norm = (self.embedding_dim / 32)
                #self.norm = np.sqrt(self.embedding_dim)

            self.central = False
            #self.reset_parameters()
            print("Inside HashedEmbeddingBag (after reset): ", num_embeddings, embedding_dim, compression, self.weight_size, self.hashed_weight.shape)
        else:
            #assert len(_weight.shape) == 1 and _weight.shape[0] == weight_size, \
            #    'Shape of weight does not match num_embeddings and embedding_dim'
            print("Central weight", hmode, "val_offset", self.val_offset)
            self.hashed_weight = _weight
            self.weight_size = self.hashed_weight.numel()
            self.central = True
            assert(self.val_offset is not None)
        self.weight = self.hashed_weight
        print("HashedEmbeddingBag: ", num_embeddings, embedding_dim, "mode", mode,
              "hmode", hmode, "kmode", keymode, "central", self.central, "key_bits", self.key_bits,
              "keys_to_use", self.keys_to_use,
              "weight_size", self.weight_size,
              "uma_chunk_size", self.uma_chunk_size)
    """
    def reset_parameters(self) -> None:
        # init.normal_(self.weight)
        W = np.random.uniform(
                low=-np.sqrt(1 / self.num_embeddings), high=np.sqrt(1 / self.num_embeddings), size=(self.hashed_weight.shape[0], )
            ).astype(np.float32)
        self.hashed_weight.data = torch.tensor(W, requires_grad=True)
    """
    def forward(self, indices: torch.Tensor, offsets: Optional[torch.Tensor] = None, per_sample_weights=None) -> torch.Tensor:
        i_shape = indices.shape
        indices = indices.view(-1)
        if offsets is None:
            offsets  = torch.arange(len(indices)).to(indices.device)
        assert(per_sample_weights is None)
        embeddings =  HashedEmbeddingBagFunction.apply(
            self.hashed_weight,
            indices,
            offsets,
            self.mode,
            self.embedding_dim,
            self.signature,
            self.random_numbers,
            self.hmode,
            self.keymode,
            self.val_offset,
            self.norm,
            self.key_bits,
            self.keys_to_use,
            self.uma_chunk_size
        )
        embeddings = embeddings.view(*i_shape, embeddings.shape[-1])
        return embeddings

class SecondaryLearnedEmbedding(nn.Module):
    def __init__(self, underlying_embedding, learn_model):
        super(SecondaryLearnedEmbedding, self).__init__()
        self.underlying_embedding = underlying_embedding
        self.learn_model = learn_model
        self.weight = underlying_embedding.weight
      
    def forward(self, indices: torch.Tensor, offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        i_shape = indices.shape
        primary_embedding = self.underlying_embedding(indices, offsets)
        e_shape = primary_embedding.shape
        primary_embedding = primary_embedding.view(-1, e_shape[-1])
        secondary_embedding = self.learn_model(primary_embedding)
        secondary_embedding = secondary_embedding.view(*i_shape, secondary_embedding.shape[-1])
        return secondary_embedding


def get_mlplearned_embedding(underlying_embedding, str_mlp, dev="cuda:0"):
    ls = [ int(x) for x in str_mlp.split('-')]
    mlp_model = nn.ModuleList()
    for i in range(0, len(ls) - 2):
        mlp_model.append(nn.Linear(ls[i], ls[i+1]))
        mlp_model.append(nn.ReLU())
    mlp_model.append(nn.Linear(ls[len(ls)-2], ls[len(ls) - 1]))
    mlp_model = torch.nn.Sequential(*mlp_model).to(dev)
    return SecondaryLearnedEmbedding(underlying_embedding, mlp_model).to(dev)


class FunctionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, learn_model, val_offset):
        super(FunctionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.learn_model = learn_model
        self.val_offset = val_offset

        self.num_hashes = int((embedding_dim + 31) / 32)
        r = np.random.RandomState(1234)
        A = r.randint(0, 2**32-1, size=(1, self.num_hashes))*2-1 # odd
        B = r.randint(0, 2**32-1, size=(1, self.num_hashes))*2-1 # odd
        self.A = torch.from_numpy(A).to("cuda:0")
        self.B = torch.from_numpy(B).to("cuda:0")
        self.bits = 32
        mask = 2**torch.arange(32)
        self.mask = mask.to("cuda:0")
        
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # indices are N x 1
        hashes =  (indices * self.A + self.B) # no mod because a and b are odd  like taking mod 2^32
        bithashes = hashes.unsqueeze(-1).bitwise_and(self.mask).ne(0) * 2.0  - 1.0
        bithashes = bithashes.view(hashes.shape[0], -1)
        input_mlp = bithashes[:,:self.embedding_dim]
        if self.learn_model is not None:
            return self.learn_model(input_mlp)
        else:
            return input_mlp
        
        
def get_functional_embedding(embedding_dim, str_mlp, dev="cuda:0"):
    if str_mlp is None:
        return FunctionalEmbedding(embedding_dim, None, 0).to(dev)
    ls = [ int(x) for x in str_mlp.split('-')]
    mlp_model = nn.ModuleList()
    for i in range(0, len(ls) - 2):
        mlp_model.append(nn.Linear(ls[i], ls[i+1]))
        mlp_model.append(nn.ReLU())
    mlp_model.append(nn.Linear(ls[len(ls)-2], ls[len(ls) - 1]))
    mlp_model = torch.nn.Sequential(*mlp_model).to(dev)
    return FunctionalEmbedding(embedding_dim, mlp_model, 0).to(dev)
