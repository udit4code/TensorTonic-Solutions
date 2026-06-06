import torch

# We can visualize a tensor = {storage_pointer, storage_offset, dtype, shape, stride}, where 
# storage_pointer → points to the underlying memory block (base_address), 
# storage_offset → where this tensor starts within that storage (Why ? Because in case of slicing the tensor, both the slice and the original tensor point to the same backend storage with offset altered), 
# dtype → float32, int64, etc., 
# shape → dimensions
# stride → how many elements to jump in storage for each dimension
# 
# address of an element within the tensor = base_address + (offset + storage_offset) * sizeof(dtype),
# where offset = i * stride[0] + j * stride[1] , {i, j, ...} are the indices to access that given element

def flatten_v1(x):
    # Default implementation using torch.flatten()
    return torch.flatten(x)

def flatten(x):
    # Total number of elements in the tensor.
    num_elements = x.numel()
    # view() only works when the tensor's memory layout is compatible with the requested shape.
    # view() , unlike reshape(), reinterprets the same underlying memory as a different shape.
    # Example: x = torch.arange(6).reshape(2, 3)
    # y = x.T and in this case, y is non-contiguous because elements are no longer stored in row-major order.
    # Attempting y.view(-1) raises an error.
    # -1 in view() means: "PyTorch, please infer this dimension for me."
    # PyTorch computes the missing dimension so that the total number of elements remains unchanged.
    if x.is_contiguous():
        return x.view(num_elements)
    # For non-contiguous tensors we must first create a contiguous copy of the underlying data.
    # This physically rearranges bytes in memory into row-major order so that a flat view becomes possible.
    x = x.contiguous()
    return x.view(num_elements)


def squeeze_v1(x):
     # Default implementation using torch.squeeze()
    return torch.squeeze(x)

def squeeze(x):
    # squeeze under the hood is basically to remove size-1 dimensions, keep storage and keep indexing semantics.
    # So, implementation-wise, it is akin to create new tensor object, reuse storage pointer, adjust shape metadata and adjust stride metadata than to performing a general reshape and involving a data copy.
    # Time Complexity of squeeze = O(d) where d = number of dimensions (tensor rank)
    # Because no tensor data is touched, PyTorch does not iterate over and hence, time complexity is not O(n), n = number of elements in the vector.
    # It only modifies shape, stride, storage pointer and dtype (if needed).
    new_shape = []
    new_stride = []
    # Inspect each dimension, keep dimensions whose size ≠ 1 and construct a new tensor metadata object.
    for size, stride in zip(x.shape, x.stride()):
        if size != 1:
            new_shape.append(size)
            new_stride.append(stride)

    # Special case: A tensor like shape=(1,1,1) becomes a scalar tensor shape=()
    if len(new_shape) == 0:
        new_shape = ()
        new_stride = ()

    # torch.as_strided() is one of the lowest-level tensor view primitives in PyTorch.
    # It lets you manually specify: Storage (reused from another tensor), Shape and Stride.
    # PyTorch creates a new tensor view without copying data.
    # The appropriate model would be Tensor = (storage pointer, storage offset, shape, stride)
    # And, as_strided() : "Construct a tensor by manually specifying shape and stride metadata."
    # as_strided() is essentially the primitive that exposes this machinery directly. It's the closest thing to saying: "Here's a block of memory. I know exactly how I want to interpret it."
    return torch.as_strided(x, size=new_shape, stride=new_stride)


def transpose_v1(x):
    return x.T

def transpose(x):
    # x = torch.tensor([[1,2,3],[4,5,6]])
    # Under the hood, in row-major form, it is stored as a 1-d backend array : [1 2 3 4 5 6]
    # Its shape is (2, 3) and its stride is (3, 1). 
    # x[i][j] = base_address + (i*3 + j*1) * element_size
    # In transpose, shape should become (3, 2) from (2, 3).
    # Then, what should be the new stride of the transpose ? In this case, the new stride is (1, 3)/
    # So, all we have done is swapped the shape and the stride.
    shape = list(x.shape)
    stride = list(x.stride())
    # Swap the shape and the stride. Time Complexity : O(d), not O(n). We are only modifying the metadata, not creating copies of each element.
    shape[0], shape[1] = shape[1], shape[0]
    stride[0], stride[1] = stride[1], stride[0]
    return torch.as_strided(x,size=shape,stride=stride)


STRATEGIES = {
    "flatten": flatten,
    "squeeze": squeeze,
    "transpose": transpose,
}


def reshape_tensor(x, op):
    tensor = torch.tensor(x, dtype=torch.float32)

    try:
        strategy = STRATEGIES[op]
    except KeyError:
        raise ValueError(f"invalid op {op}")

    return strategy(tensor).tolist()
    
def reshape_tensor_v1(x, op):
    """
    Returns: list
    """
    result = None 
    x = torch.tensor(x, dtype=torch.float32)
    if op == "flatten":
        # .flatten() collapses all dimensions into a single 1D tensor. A 2x3 tensor becomes a length-6 vector.
        result = torch.flatten(x)  
    elif op == "squeeze":
        # .squeeze() removes every dimension that has size 1. A tensor of shape (1, 1, 3) becomes shape (3,).
        result  = torch.squeeze(x)
    elif op == "transpose":
        # .T swaps the rows and columns of a 2D tensor. A 2x3 tensor becomes 3x2.
        result = x.T
    else:
        raise Exception(f"invalid op {op}")
    return result.tolist()
