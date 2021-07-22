from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter
import math


class UMAEmbeddingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hashed_weights, indices, embedding_dim, val_offset, P, A, B, C, hashed_weights_size, helper_E1sR, helper_Eidx):
        assert(indices.dim() == 1) # indices has tobe a one dimensional array of integers.

        #hashed_idx = (((((indices.view(-1,1) + val_offset) * helper_E1sR) % P + helper_Eidx * B)%P + A) % P) % hashed_weights_size
        hashed_idx = (((((indices.view(-1,1) + val_offset) * helper_E1sR) + helper_Eidx * B) + A) % P) % hashed_weights_size
        output = hashed_weights[hashed_idx] 
        #output, hashed_idx = \
        #    rma.forward(hashed_weights, indices, embedding_dim, random_numbers, val_offset)
        ctx.save_for_backward(indices, hashed_idx)
        ctx.hashed_weights_size = hashed_weights_size
        return output


    @staticmethod
    def backward(ctx, grad):
        indices, hashed_idx = ctx.saved_variables
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
        #weight_grad = rma.backward(
        #        grad, indices, hashed_idx, hashed_weights_size, embedding_dim)
        return weight_grad, None, None, None, None, None, None, None, None, None, None

class HashedEmbeddingCPU(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        _weight: torch.Tensor,
        val_offset: int,
        seed = 1024)->None:
        super(HashedEmbeddingCPU, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.val_offset = val_offset
        self.seed = seed
        self.weight = nn.Parameter(_weight, requires_grad = True) # add to parameter
        self.weights_size = self.weight.numel()


        r = np.random.RandomState(seed)
        random_numbers = np.concatenate([np.array([2038074743]), r.randint(0, 2038074743, (10,))]) # 10 random numbers
        random_numbers = torch.from_numpy(random_numbers.astype(np.int64))
        print("[Seed]", seed, "First 5 random numbers: ", random_numbers[:5])
        print("UMA Embedding Object: num_embeddings:{} dim:{} val_offset:{} seed:{} weights_size:{}".format(self.num_embeddings, self.embedding_dim,
                          self.val_offset, self.seed, self.weights_size), flush=True)

        # helpers to compute
        helper_Eidx = torch.LongTensor(np.arange(self.embedding_dim))
        helper_E1sR = torch.LongTensor(np.ones(self.embedding_dim) * int(random_numbers[3])) # A

        # adding to parameters
        self.random_numbers = nn.Parameter(random_numbers, requires_grad=False)
        self.helper_Eidx = nn.Parameter(helper_Eidx, requires_grad=False)
        self.helper_E1sR = nn.Parameter(helper_E1sR, requires_grad=False)


    def forward(self, indices: torch.Tensor) -> torch.Tensor:

        #def forward(ctx, hashed_weights, indices, embedding_dim, val_offset, P, A, B, hashed_weights_size, helper_E1sR, helper_Eidx):
        embeddings =  UMAEmbeddingFunc.apply(
            self.weight,
            indices,
            self.embedding_dim,
            self.val_offset,
            self.random_numbers[0],
            self.random_numbers[1],
            self.random_numbers[2],
            self.random_numbers[3],
            self.weights_size,
            self.helper_E1sR,
            self.helper_Eidx
        )
        return embeddings
