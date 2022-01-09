import hashedEmbeddingBag
import torch
import torch.nn as nn
import numpy as np
import time
import pdb
import yaml
import sys

M=1000000
config = {
    "robez" : 1,
    "robez_compression" : 10,
    "n": 32*1000000,
    "m" : 128,
    "batch" : 10240,
    "Iter" : 1000,
    "uma_chunk_size" : 32,
    "sparse" : True,
    "no_bag" : True
}
with open(sys.argv[1], "r") as f:
    config = yaml.load(f)


batch = config["batch"]
Iter = config["Iter"]
n1 = config["n"]
m1 = config["m"]
uma_chunk_size = config["uma_chunk_size"]

print("ORIG: ", n1* m1*4/1000000000, "GB", "Iter:", Iter, "Batch:", batch)
if config["robez"]:
    robe_size = int(n1*m1/config["robez_compression"])
    print("robez: ", robe_size*4/1000000, "MB")

    _weight = nn.Parameter( torch.from_numpy( np.random.uniform(low = -0.001, high=0.001, size=((robe_size, ))).astype(np.float32)))
    E1 = hashedEmbeddingBag.HashedEmbeddingBag(n1, m1, _weight=_weight, val_offset=0, uma_chunk_size=uma_chunk_size, no_bag=config["no_bag"], sparse=config["sparse"]).cuda(0)
    _weight = _weight * 1.1 # touch all to warmup cache with all weights
else:
    E1 = nn.Embedding(n1, m1, sparse=config["sparse"]).cuda(0)
E1.train()

# put indices on the gpu. we dont want to measure transferring of indices.
indices = torch.randint(low=0, high=n1, size=(Iter, batch)).to("cuda:0")

# cache warmup. I am not sure how many are needed. But law of diminishing returns
output_v = E1(indices[1,:])
torch.cuda.synchronize()
output_v = E1(indices[10,:])
torch.cuda.synchronize()
output_v = E1(indices[5,:])

optimizer = torch.optim.SGD(E1.parameters(), lr=0.001)
# timers for timing.
start = time.time()
inf_time = []
bk_time = []

for i in range(Iter):

    optimizer.zero_grad() ## this is very important aparrently. Without this, if we just increase the number of iterations
    # time / itr wil keep on blowing up. SOmething weird happens
    torch.cuda.synchronize()
    t1 = time.time()
    # simple model . get embeddings of some indices
    # simple operation of norm
    # compute backward operation on norm. in order to (hypothetically) minimize the norm
  
    # forward
    output_v = E1(indices[i,:])
    norm = torch.norm(output_v)
    torch.cuda.synchronize()
    t2 = time.time()
    # backward
    norm.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t3 = time.time()
    inf_time_ = (t2 - t1)
    bk_time_ = (t3 - t2)
    inf_time.append(inf_time_)
    bk_time.append(bk_time_)
    
    
end = time.time()
print(*(config.items()), "time in (ms/itr) ||  inf: ", 1000* np.percentile(inf_time, [25,50,75]))
print(*(config.items()), "time in (ms/itr) ||   bk: ", 1000* np.percentile(bk_time, [25,50,75]))
