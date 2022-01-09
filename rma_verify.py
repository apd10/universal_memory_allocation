import hashedEmbeddingBag
import torch
import torch.nn as nn
import numpy as np
import time

import pickle
with open("check.pkg", "rb") as f:
    quick_check_pkg = pickle.load(f)

n1 = quick_check_pkg["n"]
m1 = quick_check_pkg["m"]
_weight = nn.Parameter(torch.from_numpy( quick_check_pkg["weight"].astype(np.float32)))
robe_size = _weight.shape[0]
uma_chunk_size = quick_check_pkg["uma_chunk_size"]
val_offset = quick_check_pkg["val_offset"]

E1 = hashedEmbeddingBag.HashedEmbeddingBag(n1, m1, _weight=_weight, val_offset=val_offset, uma_chunk_size=uma_chunk_size, no_bag=True, sparse=True).cuda(0)

indices = torch.from_numpy(quick_check_pkg["input"]).to("cuda:0")

output_v = E1(indices)
norm = torch.norm(output_v)
norm.backward()
torch.cuda.synchronize()
out = np.array(output_v.detach().cpu())
if E1.sparse :
    grad = np.array(_weight.grad.detach().cpu().to_dense()) # sparse
else:
    grad = np.array(_weight.grad.detach().cpu())
if np.linalg.norm(out - quick_check_pkg["out"])  < 1e-7:
    print("out check OK")
else:
    print("out check [NOT OK]")
if np.linalg.norm(grad - quick_check_pkg["grad"])  < 1e-7:
    print("grad check OK")
else:
    print("grad check [NOT OK]")

