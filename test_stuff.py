from fixations import as_tensor
import torch

def unflatten(index, shape):
    shape = as_tensor(shape)
    if index < 0:
        index = shape.prod().item() + index
    assert index < shape.prod(), "Index is out of range for this given shape"
    assert index >= 0, "Index is out of range for this given shape"
    indices = []
    for i in range(len(shape) - 1):
        value = shape[i + 1:].prod()
        partial = int(index / value)
        indices.append(int(index / value))
        index -= partial * value
    indices.append(index % shape[-1].item())
    return torch.LongTensor(indices)

def flatten(index, shape):
    shape = as_tensor(shape)
    index = as_tensor(index)
    flat_index = 0
    for i, val in enumerate(index):
        if val < 0:
            index[i] = shape[i] + val
    assert index.prod() < shape.prod(), "Index out of range for this given shape"
    assert all(index >= 0), "Index out of range for this given shape"
    for i in range(len(shape) - 1):
        flat_index += index[i] * shape[i + 1:].prod()
    flat_index += index[-1]
    return flat_index