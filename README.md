Code for Embeddings with UMA - Universal Memory Allocation
- The code was modified from the original pytorch Embedding bag code
- the code is only tested on embedding style scenarios. So beware while using embedding bag

after checkout run the following to install the UMA
```
python3 setup.py install
```

Usage Multiple Embedding Tables sharing the same underlying set of parameters: 
```
import hashedEmbeddingBag
import torch
import torch.nn as nn


_weight = nn.Parameter( torch.from_numpy( np.random.uniform(low = -np.sqrt(1/n), high=np.sqrt(1/n), size=((uma_size, ))).astype(np.float32)))

n1 = 100000
m1 = 16
E1 = hashedEmbeddingBag.HashedEmbeddingBag(n1, m1, _weight=_weight, val_offset=0)

n2 = 200000
m2 = 32
E2 = hashedEmbeddingBag.HashedEmbeddingBag(n2, m2, _weight=_weight, val_offset=n1) # note the offset
```
