# Description
Code for Embeddings with Random Offset Block Embedding Array (ROBE)
- The code was modified from the original pytorch Embedding bag code
- the code is only tested on embedding style scenarios. So beware while using embedding bag

# How to run 
after checkout run the following to install the ROBE/UMA
```
python3 setup.py install
```

Usage Multiple Embedding Tables sharing the same underlying set of parameters: 
```
import hashedEmbeddingBag
import torch
import torch.nn as nn
import numpy as np


robe_size = 1000
_weight = nn.Parameter( torch.from_numpy( np.random.uniform(low = -0.001, high=0.001, size=((robe_size, ))).astype(np.float32)))

n1 = 100000
m1 = 16
E1 = hashedEmbeddingBag.HashedEmbeddingBag(n1, m1, _weight=_weight, val_offset=0).cuda(0)

n2 = 200000
m2 = 32
E2 = hashedEmbeddingBag.HashedEmbeddingBag(n2, m2, _weight=_weight, val_offset=n1).cuda(0) # note the offset

indices = torch.arange(5).cuda(0)
embeddings1 = E1(indices)
embeddings2 = E2(indices)
```

#While running with CPU

hashedEmbeddingBag is written for GPU.
In order to use for CPU/GPU , you can directly use
hashedEmbeddingCPU module. Everything remains the same


