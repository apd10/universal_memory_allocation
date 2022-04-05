#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/TensorAccessor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <vector>
#include <stdio.h>


constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

constexpr int NWEIGHT_PER_THREAD = 128;
constexpr int BIT4MASK = 15;
constexpr int64_t BIT32MASK = ((1u <<31u) - 1u);

constexpr int HMODE_RANDOMHASH = 0;
constexpr int HMODE_LMAHASH = 1;

constexpr int KEYMODE_HASHWEIGHT = 0;
constexpr int KEYMODE_STATIC_PM = 1;

// Fast ceil division (no overflow checking)
__host__ __device__ __forceinline__
int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

__global__
void krn_partials_per_segment(int64_t *ret, const int64_t *segment_offsets,
                              int64_t num_segments, int64_t numel) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_segments) {
    const int64_t idx_start = segment_offsets[id];
    const int64_t idx_end = (id == num_segments-1)?numel:segment_offsets[id+1];
    const int64_t size = idx_end - idx_start;
    ret[id] = ceil_div(size, NWEIGHT_PER_THREAD);
  }
}

__global__
void krn_partial_segment_offset(
        int64_t *ret,
        const int64_t *partials_per_segment,
        const int64_t *partials_per_segment_offset,
        const int64_t *segment_offsets,
        int64_t num_segments) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_segments) {
    int64_t idx = partials_per_segment_offset[id];
    const int64_t num_partials = partials_per_segment[id];
    const int64_t segment_offset = segment_offsets[id];
    for (int64_t i=0; i<num_partials; ++i) {
      ret[idx++] = segment_offset + i * NWEIGHT_PER_THREAD;
    }
  }
}


__device__ __host__ int64_t hash_func_backup(int64_t a, int64_t b) {
    return a + b;
}

__device__ __host__ int64_t hash_func(int64_t a, int64_t b, int64_t * random_numbers) {
    return (a * random_numbers[3] + b * random_numbers[2] + random_numbers[1]) % random_numbers[0];
    //return a * 16 + b;
}

__device__ __host__ int64_t hash_func1(int64_t x) {
    return (x * 575796319u  + 3013888363u) & BIT32MASK;
    //return a * 16 + b;
}

template<typename scalar_t>
__device__ __host__ scalar_t keymode_static_pm(int64_t a) {
    int64_t val =  (a * 71371560971u + 46023704752u) % 100000004987u % 2u;
    if (val == 0) {
        return -1.0;
    } else {
        return 1.0;
    }
    
    //return a * 16 + b;
}

/* fast way to map to +1.-1 */
template<typename scalar_t>
__device__ __host__ scalar_t keymode_static_pm_parity(int64_t a) {
    int64_t val =  (a * 71371560971u + 46023704752u) % 100000004987u;
    int64_t val1 = val ^ (val>>1); // get parity to decide between +1/-1
    val1 = val1 ^ (val1 >> 2);
    val1 = val1 ^ (val1 >> 4);
    val1 = val1 ^ (val1 >> 8);
    val1 = val1 ^ (val1 >> 16);
    val1 = val1 ^ (val1 >> 32);
    if (val1 & 1)  {
      return 1.0;
    }else{
      return -1.0;
    }
    //return a * 16 + b;
}


__device__ __host__ int64_t lma_hash_func(int64_t v, int64_t i, int64_t signature) {
        // input is value, embedding_location, signature 4x16 bit representation which is 
        // drawn from signature array[value]
        // a and b for hashing % 17 % 15  to choose from 16 minhashes
        // 9,  1, 14,  2, 10, 10,  2,  8, 
        // 14,  6, 10,  4,  1, 14, 12, 12])
        // P = 100000004987
        // 27099547127,  2699391407, 46970219979, 16806825237, 74212261504, 93432047494, 16220329892, 82313724554, 
        // 50469911173, 52271898367, 98939193954, 94293094042, 96314459732,  2349378832,  1727459397, 48438134705

        int64_t extracted = ((signature >> (4*((82313724554*i+48438134705)% 100000004987 %15))) & BIT4MASK) // 4 bit number
                             ^  (((signature >> (4*((27099547127*i+50469911173 )% 100000004987%15))) & BIT4MASK) << 4)
                             ^  (((signature >> (4*((2699391407*i+52271898367)% 100000004987 %15))) & BIT4MASK) << 8)
                             ^  (((signature >> (4*((46970219979*i+98939193954)% 100000004987 %15))) & BIT4MASK) << 12)
                             ^  (((signature >> (4*((16806825237*i+94293094042)% 100000004987 %15))) & BIT4MASK) << 16)
                             ^  (((signature >> (4*((74212261504*i+96314459732)% 100000004987 %15))) & BIT4MASK) << 20)
                             ^  (((signature >> (4*((93432047494*i+2349378832)% 100000004987 %15))) & BIT4MASK) << 24)
                             ^  (((signature >> (4*((16220329892*i+1727459397)% 100000004987 %15))) & BIT4MASK) << 28);
        return (int64_t) extracted; // extracted is a 32 bit number
}


__device__ __host__ int64_t lma_hash_func_e1(int64_t v, int64_t i, int64_t signature, // still assuming signature is 64 bit
                                              int64_t key_bits, int64_t keys_to_use, int64_t * random_numbers) {
        /*
            Memory based optimizations:
            put random_numbers into __constant__ memory

            code  based
            make keys_to_use into template parameter and foward declare it with all different values
            
            
            
        */
        CUDA_KERNEL_ASSERT(keys_to_use == 1 or keys_to_use == 2 or keys_to_use == 4 or keys_to_use == 6 or keys_to_use == 8 or keys_to_use == 12 or keys_to_use == 16);
        int64_t total_bits = key_bits * keys_to_use;
        CUDA_KERNEL_ASSERT(total_bits < 60);
        int64_t bitmask = (1 << key_bits) - 1;
        int64_t numkeys = 64/key_bits -1;
        int64_t extracted = ((signature >> (key_bits*((random_numbers[11]*i+random_numbers[12])% random_numbers[0] %numkeys))) & bitmask) ;// key_bits bit number
        if (keys_to_use >= 2)
              extracted ^=  (((signature >> (key_bits*((random_numbers[13]*i+random_numbers[14] )% random_numbers[0]%numkeys))) & bitmask) << key_bits);
        if (keys_to_use >=4) {
              extracted ^=  (((signature >> (key_bits*((random_numbers[15]*i+random_numbers[16])% random_numbers[0] %numkeys))) & bitmask) << 2*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[17]*i+random_numbers[18])% random_numbers[0] %numkeys))) & bitmask) << 3*key_bits);
        }
        if (keys_to_use >= 6) {
              extracted ^=  (((signature >> (key_bits*((random_numbers[19]*i+random_numbers[20])% random_numbers[0] %numkeys))) & bitmask) << 4*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[21]*i+random_numbers[22])% random_numbers[0] %numkeys))) & bitmask) << 5*key_bits);
        }
        if (keys_to_use >= 8) {
              extracted ^=  (((signature >> (key_bits*((random_numbers[23]*i+random_numbers[24])% random_numbers[0] %numkeys))) & bitmask) << 6*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[25]*i+random_numbers[26])% random_numbers[0] %numkeys))) & bitmask) << 7*key_bits);
        }
      /*
        array([[22406334177, 63792722443],
       [75791256117, 15202366190],
       [40623773873,  8640139384],
       [13655260797, 99959231757],
       [21577857905, 50989087799],
       [ 8043429682, 29709184765],
       [95200260355, 49014991094],
       [36941582829, 21960689983]])
      */
        if (keys_to_use >= 12) {
              extracted ^=  (((signature >> (key_bits*((random_numbers[27]*i+random_numbers[28])% random_numbers[0] %numkeys))) & bitmask) << 8*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[29]*i+random_numbers[30])% random_numbers[0] %numkeys))) & bitmask) << 9*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[31]*i+random_numbers[32])% random_numbers[0] %numkeys))) & bitmask) << 10*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[33]*i+random_numbers[34])% random_numbers[0] %numkeys))) & bitmask) << 11*key_bits);
        }
        if (keys_to_use >= 16) {
              extracted ^=  (((signature >> (key_bits*((random_numbers[35]*i+random_numbers[36])% random_numbers[0] %numkeys))) & bitmask) << 12*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[37]*i+random_numbers[38])% random_numbers[0] %numkeys))) & bitmask) << 13*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[39]*i+random_numbers[40])% random_numbers[0] %numkeys))) & bitmask) << 14*key_bits);
              extracted ^=  (((signature >> (key_bits*((random_numbers[41]*i+random_numbers[42])% random_numbers[0] %numkeys))) & bitmask) << 15*key_bits);
        }

        // extracted uses i for consistent usage from small storage offered by signature
        // return value has to be a hash of (extracted, i)
        int64_t hash = hash_func(extracted, i, random_numbers);
        //int64_t hash = (1<<total_bits)*i + extracted;
        return hash;
}

__device__ __host__ int64_t lma_hash_func_e2(int64_t v, int64_t i, int64_t signature, // still assuming signature is 64 bit
                                              int64_t key_bits, int64_t keys_to_use, int64_t * random_numbers) {
        int64_t total_bits = key_bits * keys_to_use;
        CUDA_KERNEL_ASSERT(total_bits < 60);
        int64_t bitmask = (1 << key_bits) - 1;
        int64_t numkeys = 64/key_bits -1;
        int64_t extracted = ((signature >> (key_bits*((random_numbers[11]*i+random_numbers[12])% random_numbers[0] %numkeys))) & bitmask) ;// key_bits bit number
        for (int k=1; k < keys_to_use;k++) {
              extracted ^=  (((signature >> (key_bits*((random_numbers[10 + 2*k+1]*i+random_numbers[10+2*k+2] )% random_numbers[0]%numkeys))) & bitmask) << k*key_bits);
        }
        // extracted uses i for consistent usage from small storage offered by signature
        // return value has to be a hash of (extracted, i)
        int64_t hash = hash_func(extracted, i, random_numbers);
        //int64_t hash = (1<<total_bits)*i + extracted;
        return hash;
}


template<typename scalar_t>
__global__ void hashed_embedding_bag_update_output_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> offsets,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> hashed_weights,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> offset2bag,
    int64_t numIndices,
    int64_t numBags,
    int64_t embedding_dim,
    int64_t hashedWeightSize,
    int mode,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> hashed_index,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> bag_size,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> max_indices,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> signature,
    int64_t * random_numbers,
    int hmode,
    int keymode,
    int key_bits,
    int keys_to_use,
    int uma_chunk_size)
{
    /*
        optimizations. modes into template paramters
        accessor to pointers?
        
    */
    // the strategy here is that each bag x feature is handled by a single thread

    int64_t chunksPerBag = (embedding_dim + (int64_t)blockDim.x - 1) / (int64_t)blockDim.x;
    int64_t numChunks = numBags * chunksPerBag;
    int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t chunkStride = gridDim.x * blockDim.y;

    for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
        int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
        if (featureDim < embedding_dim) {
            int64_t bag = chunk / chunksPerBag;
            int64_t begin = bag == 0 ? 0 : offsets[bag]; // forces first offset to be 0 instead of asserting on it
            int64_t end = (bag < numBags - 1) ? (offsets[bag + 1]) : numIndices;
            CUDA_KERNEL_ASSERT(end >= begin);

            scalar_t weightFeatSum = 0;
            scalar_t weightFeatMax;

            int64_t bag_size_ = 0;
            int64_t maxWord = -1;
            // from start of bag to end of bag.
            int64_t hfd = featureDim / uma_chunk_size;
            int64_t hfd_shift =  featureDim % uma_chunk_size;
            for (int64_t emb = begin; emb < end; emb++) {
                const int64_t weightRow = input[emb];
                
                int64_t hashKey = 0;
                int64_t hashedWeightIdx = 0;
                scalar_t weightValue = 0;

                switch (hmode) {
                    case HMODE_LMAHASH:
                        hashKey = lma_hash_func_e2(weightRow, hfd, signature[weightRow], key_bits, keys_to_use, random_numbers); // expects a val_offset + value
                        break;
                    default: // HMODE_RANDOMHASH
                        // this will be recomputed within uma_chunk_size. But i think if we want to not do that we need a better grid layout
                        hashKey = hash_func(weightRow, hfd, random_numbers); // expects a val_offset + value if central
                        break;
                }

                switch (keymode) {
                    case KEYMODE_STATIC_PM:
                        weightValue = keymode_static_pm_parity<scalar_t>(hashKey);
                        break;
                    default: // KEYMODE_HASHWEIGHT
                        hashedWeightIdx = hashKey % (hashedWeightSize - uma_chunk_size + 1)+ hfd_shift;
                        hashed_index[emb][featureDim] = hashedWeightIdx;
                        weightValue = hashed_weights[hashedWeightIdx];
                        break;
                }
                

                if (mode == MODE_MAX) {
                    if (emb == begin || weightValue > weightFeatMax) {
                        weightFeatMax = weightValue;
                        maxWord = input[emb];
                    }
                } else {
                    weightFeatSum += static_cast<scalar_t>(weightValue);
                }

                bag_size_++;
                if (featureDim == 0) {
                offset2bag[emb] = bag;
                }
            }
            if (mode == MODE_MEAN) {
                    if (end == begin) {
                    bag_size[bag] = 0;
                    } else {
                        weightFeatSum = weightFeatSum / static_cast<scalar_t>(bag_size_);
                        bag_size[bag] = bag_size_;
                    }
            }

            if (mode == MODE_MEAN || mode == MODE_SUM) {
                output[bag][featureDim] = static_cast<scalar_t>(weightFeatSum);
            }
            else if (mode == MODE_MAX) {
                if (end == begin) {
                    // If bag is empty, set output to 0.
                    weightFeatMax = 0;
                }
                max_indices[bag][featureDim] = maxWord;
                output[bag][featureDim] = weightFeatMax;
            }
        }
    }
}

template<typename scalar_t>
__global__ void compute_grad_weight_bags(
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> orig_hash_idx_idx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_grad,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> offset2bag,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> bag_size,
    int64_t embedding_dim,
    int64_t numel,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> partial_segment_offset,
    int64_t num_of_partial_segments,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_weight_per_partial,
    int64_t mode
) 
{
    const int partial_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (partial_id >= num_of_partial_segments) {
        return;
    }
    const int idx_begin = partial_segment_offset[partial_id];
    const int idx_end = (partial_id == num_of_partial_segments - 1) ? numel : partial_segment_offset[partial_id + 1];

    scalar_t grad_acc = 0;
    for (int idx = idx_begin; idx < idx_end; ++idx) {
        const int orig_hash_idx = orig_hash_idx_idx[idx];    // orig_idx in range [0, |indices| x embedding_dim)
        const int orig_cat_idx = orig_hash_idx / embedding_dim; // in range [0, |indices|)
        const int feature_idx =  orig_hash_idx % embedding_dim;     // in range [0, embedding_dim)
        const int bag_idx = offset2bag[orig_cat_idx];     
        if (mode == MODE_SUM) {
            grad_acc += output_grad[bag_idx][feature_idx];
        } else if(mode == MODE_MEAN) {
            grad_acc += output_grad[bag_idx][feature_idx] / (float) (bag_size[bag_idx]); 
        }
    }
    grad_weight_per_partial[partial_id] = grad_acc;
}

template<typename scalar_t>
__global__ void sum_and_scatter(
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_unique_weight_idx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_weight_per_segment,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> partical_per_segment_offset,
    int64_t num_segments,
    int64_t num_of_partial_segments,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> weight_grad
)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_segments) {
        return;
    }
    const int weight_idx = sorted_unique_weight_idx[gid];

    const int idx_begin = partical_per_segment_offset[gid];
    const int idx_end = (gid == num_segments - 1) ? num_of_partial_segments : partical_per_segment_offset[gid + 1];
    scalar_t grad_acc = 0;
    for (int idx = idx_begin; idx < idx_end; ++idx) {
        grad_acc += grad_weight_per_segment[idx];
    }
    weight_grad[weight_idx] = grad_acc;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> hashed_embedding_bag_cuda_forward(
    const torch::Tensor& hashed_weights,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const int64_t mode,
    const int64_t embedding_dim,
    const torch::Tensor& signature,
    const torch::Tensor& random_numbers,
    const int64_t hmode,
    const int64_t keymode,
    const int64_t key_bits,
    const int64_t keys_to_use,
    const int64_t uma_chunk_size)
{
    int64_t numIndices = indices.size(0);
    int64_t numBags = offsets.size(0);
    
    int64_t hashedWeightSize = 0;
    if (keymode  == KEYMODE_HASHWEIGHT) {
        hashedWeightSize = hashed_weights.size(0);
    }
    auto bag_size = at::empty(offsets.sizes(), indices.options());
    auto offset2bag = 
        at::empty({indices.size(0)}, indices.options());
    auto hashed_index = at::empty({indices.size(0), embedding_dim}, indices.options());
    auto output = at::empty({numBags, embedding_dim}, hashed_weights.options());  // this gets initialized on CUDA:0 even if hashed_weights is on CUDA:1 why??
    torch::Tensor max_indices;
    if (mode == MODE_MAX) {
        max_indices = at::empty({numBags, embedding_dim}, indices.options());
    } else {
        max_indices = at::empty({0, 0}, indices.options());
    } 
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(indices.device().index()); 

#ifdef __HIP_PLATFORM_HCC__
    dim3 block = dim3(64, 4);
#else
    dim3 block = dim3(32, 8);
#endif
    int grid = 1024; //  TODO 2: fix grid size as per size of the index. maybe have a max cap. But having 1024 direclty will be sub-optimial

    AT_DISPATCH_FLOATING_TYPES(hashed_weights.type(), "hashed_embedding_bag_cuda", ([&] {
        hashed_embedding_bag_update_output_kernel<scalar_t><<<grid, block, 0, stream>>>(
            indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            offsets.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            hashed_weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            offset2bag.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            numIndices,
            numBags,
            embedding_dim,
            hashedWeightSize,
            mode,
            hashed_index.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            bag_size.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            max_indices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            signature.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            random_numbers.data_ptr<int64_t>(),
            hmode,
            keymode,
            key_bits,
            keys_to_use,
            uma_chunk_size);
    }));
    cudaDeviceSynchronize(); // TODO 1: remove this. this will wait for all sreams to synchronize. we dont want that.
                             // instead use cudaStreamSynchronize

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
        output, offset2bag, bag_size, max_indices, hashed_index);
}

torch::Tensor hashed_embedding_bag_sum_backward(
    const torch::Tensor& output_grad,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const torch::Tensor& offset2bag,
    const torch::Tensor& bag_size,
    const torch::Tensor& hash_index,
    int64_t num_weights,
    int64_t embedding_dim,
    int64_t mode)
{
    int64_t numIndices = indices.size(0);
    int64_t numBags = offsets.size(0);
    torch::Tensor weight_grad = torch::zeros({num_weights}, output_grad.options());

    if (numIndices == 0) {
        return weight_grad;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(indices.device().index());
    torch::Tensor flattened_hash_index = hash_index.flatten();
    int64_t numel = flattened_hash_index.size(0);

    // hash_index is a |indices| x embedding_dim Tensor, contains the index in hashed weight for each input indices x embedding dim.
    // the hash_index is flattened, and then we want to sort it, we use orig_hash_idx_idx to keep track of its orignal indices.
    auto sorted_hash_idx = at::empty_like(flattened_hash_index, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto orig_hash_idx_idx = at::empty_like(flattened_hash_index, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    using device_ptr = thrust::device_ptr<int64_t>;
    {
        sorted_hash_idx.copy_(flattened_hash_index);


        auto count_iter = thrust::counting_iterator<int64_t>(0);
        auto orig_hash_idx_idx_data = device_ptr(orig_hash_idx_idx.data_ptr<int64_t>());
        thrust::copy(count_iter, count_iter + numel, orig_hash_idx_idx_data);

        auto sorted_hash_idx_data = device_ptr(sorted_hash_idx.data_ptr<int64_t>());
        thrust::sort_by_key(
            sorted_hash_idx_data, 
            sorted_hash_idx_data + numel, 
            orig_hash_idx_idx_data);
    }

    // There may be many duplicates in the hash_index, now it's sorted, we find the start index for each hash_index value.
    // then we can get the count for each hash_index_value.
    auto segment_offsets = at::empty({numel}, orig_hash_idx_idx.options());
    auto sorted_unique_weight_idx = at::empty_like(sorted_hash_idx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    int64_t num_segments;
    {
        auto sorted_hash_idx_data = device_ptr(sorted_hash_idx.data_ptr<int64_t>());
        auto sorted_unique_weight_idx_data = device_ptr(sorted_unique_weight_idx.data_ptr<int64_t>());
        auto iter_end_pair = thrust::unique_by_key_copy(
            sorted_hash_idx_data,
            sorted_hash_idx_data + numel,
            thrust::make_counting_iterator(0),
            sorted_unique_weight_idx_data,
            thrust::device_ptr<int64_t>(segment_offsets.data_ptr<int64_t>())
        );
        num_segments = thrust::get<0>(iter_end_pair) - sorted_unique_weight_idx_data;
    }

    // We split the segments up into sizes of `NROWS_PER_THREAD`
    // Compute the number partial-segments per segment (some partial-segments 
    // may not be the full `NROWS_PER_THREAD` number of rows)
    auto partials_per_segment = at::empty({num_segments}, orig_hash_idx_idx.options());
    {
        krn_partials_per_segment<<<ceil_div(num_segments, 32), 32, 0, stream>>> (
            partials_per_segment.data_ptr<int64_t>(),
            segment_offsets.data_ptr<int64_t>(),
            num_segments,
            numel);
    }


    // In order to compute `partial_segment_offset`, which is the start index
    // of each partial-segment in `sorted_indices`, we need to compute the
    // start position of each _segment_ in `partial_segment_offset`.
    // Unit: index in `partial_segment_offset`
    auto partials_per_segment_offset = at::empty({num_segments}, orig_hash_idx_idx.options());
    thrust::exclusive_scan(
        device_ptr(partials_per_segment.data_ptr<int64_t>()),
        device_ptr(partials_per_segment.data_ptr<int64_t>() + num_segments),
        device_ptr(partials_per_segment_offset.data_ptr<int64_t>())
    );

    // The total number of partial-segments is the sum of `partials_per_segment_offset`
    const int num_of_partial_segments = partials_per_segment[num_segments - 1].item<int64_t>() +
        partials_per_segment_offset[num_segments - 1].item<int64_t>();

    // Now we can compute the start position of each partial-segment
    // Unit: index in `sorted_indices` and `orig_indices`
    auto partial_segment_offset = at::empty({num_of_partial_segments}, orig_hash_idx_idx.options());
    {
        krn_partial_segment_offset<<<ceil_div(num_segments, 32), 32, 0, stream>>> (
            partial_segment_offset.data_ptr<int64_t>(),
            partials_per_segment.data_ptr<int64_t>(),
            partials_per_segment_offset.data_ptr<int64_t>(),
            segment_offsets.data_ptr<int64_t>(),
            num_segments);
    }
    auto grad_weight_per_segment = at::empty({num_of_partial_segments}, weight_grad.options());

    const int block = NWEIGHT_PER_THREAD;
    const int grid = ceil_div(num_of_partial_segments, block);
    AT_DISPATCH_ALL_TYPES(weight_grad.scalar_type(), "hashed_embedding_bag_backward_cuda", ([&] {
        compute_grad_weight_bags<scalar_t><<<grid, block, 0, stream>>>(
            orig_hash_idx_idx.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            output_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            offset2bag.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            bag_size.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            embedding_dim,
            numel,
            partial_segment_offset.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            num_of_partial_segments,
            grad_weight_per_segment.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            mode
        );
        const int grid2 = ceil_div(num_segments, block);
        sum_and_scatter<scalar_t><<<grid2, block, 0, stream>>>(
            sorted_unique_weight_idx.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            grad_weight_per_segment.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            partials_per_segment_offset.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            num_segments,
            num_of_partial_segments,
            weight_grad.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));
    

    return weight_grad;
}

torch::Tensor hashed_embedding_bag_cuda_backward(
    const torch::Tensor& grad_,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const torch::Tensor& offset2bag,
    const torch::Tensor& bag_size_,
    const torch::Tensor& max_indices_,
    const torch::Tensor& hashed_index,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    int64_t embedding_dim)
{
    torch::Tensor grad = grad_.contiguous();
    switch (mode) {
        case MODE_SUM:
        case MODE_MEAN:
            return hashed_embedding_bag_sum_backward(
                grad_,
                indices,
                offsets,
                offset2bag,
                bag_size_,
                hashed_index,
                num_weights,
                embedding_dim,
                mode);
        case MODE_MAX:
            //return hashed_embedding_bag_cuda_max()
        default:
            return torch::Tensor();
    }
}

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> hashed_embedding_bag_forward(
    const torch::Tensor& hashed_weights,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    //const bool scale_grad_by_freq,
    const int64_t mode,
    const int64_t embedding_dim,
    const torch::Tensor& signature,
    const torch::Tensor& random_numbers,
    const int64_t hmode,
    const int64_t keymode,
    const int64_t key_bits, 
    const int64_t keys_to_use,
    const int64_t uma_chunk_size) 
{
  
    if (keymode == KEYMODE_HASHWEIGHT) {
        CHECK_INPUT(hashed_weights);
    }
    CHECK_INPUT(indices);
    CHECK_INPUT(offsets);
    if(hmode == HMODE_LMAHASH) {
        CHECK_INPUT(signature);
    }

    return hashed_embedding_bag_cuda_forward(hashed_weights, indices, offsets, mode, embedding_dim, signature, random_numbers, hmode, keymode, key_bits, keys_to_use, uma_chunk_size);
}


torch::Tensor hashed_embedding_bag_backward(
    const torch::Tensor& grad,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const torch::Tensor& offset2bag,
    const torch::Tensor& bag_size_,
    const torch::Tensor& max_indices_,
    const torch::Tensor& hashed_index_,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    int64_t embedding_dim)
{
    CHECK_CUDA(grad);
    CHECK_INPUT(indices);
    CHECK_INPUT(offsets);
    CHECK_INPUT(offset2bag);
    CHECK_INPUT(bag_size_);
    CHECK_INPUT(max_indices_);
    return hashed_embedding_bag_cuda_backward(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size_,
        max_indices_,
        hashed_index_,
        num_weights,
        scale_grad_by_freq,
        mode,
        embedding_dim
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hashed_embedding_bag_forward, "hash embedding forward (CUDA)");
  m.def("backward", &hashed_embedding_bag_backward, "hash embedding backward (CUDA)");
  m.def("hash", &hash_func, "hash function");
  m.def("lma_hash", &lma_hash_func, "lma hash function");
}
