import cnn_fixations.utils as U
import torch
import torch.nn as nn
import torch.nn.modules as M
from collections import defaultdict
from contextlib import contextmanager


def to_cpu(elements):
    if type(elements) == tuple or type(elements) == list:
        return [x.cpu() for x in elements]
    return elements.cpu()


class LayerInfo():
    KERNEL_TYPES = [M.conv._ConvNd, M.pooling._MaxPoolNd]
    IGNORE_MODULES = [M.container.Sequential, M.ReLU, M.batchnorm._BatchNorm]

    def __init__(self, name, in_data=None, out_data=None, sub_layers=[]):
        self.name = name
        self.in_data = in_data
        self.out_data = out_data

    def __str__(self):
        string = f"Info{self.name}("
        string += f"in_shape: {(*self.get_in_shape().tolist(),)}, "
        string += f"out_shape: {(*self.get_out_shape().tolist(),)})"
        return string

    def __repr__(self):
        return f"Info({self.name})"

    def get_in_shape(self):
        if self.in_data is None:
            return None
        return torch.LongTensor((len(self.in_data), *self.in_data[0].shape))

    def get_out_shape(self):
        if self.out_data is None:
            return None
        return torch.LongTensor((len(self.out_data), *self.out_data[0].shape))


class Fixations:
    def __init__(self):
        self._layer_info_dict = None

    @staticmethod
    def _is_any_instance(module, types):
        return any([isinstance(module, t) for t in types])

    @staticmethod
    def _int_to_tuple(param, dims):
        if isinstance(param, int):
            return tuple(param for _ in range(dims))
        return param

    @staticmethod
    def _expand_kernel_params(module, in_data):
        if not Fixations._is_any_instance(module, LayerInfo.KERNEL_TYPES):
            return
        if in_data is None:
            print("in_data is None, kernel hyperparameter expansion not done.")
            return
        dims = len(in_data[0][0][0].shape)
        m = module
        m.kernel_size = Fixations._int_to_tuple(m.kernel_size, dims)
        m.padding = Fixations._int_to_tuple(m.padding, dims)
        m.stride = Fixations._int_to_tuple(m.stride, dims)
        m.dilation = Fixations._int_to_tuple(m.dilation, dims)

    def register_hook(self, module, handles, seen):

        def store_data(module, in_data, out_data):
            Fixations._expand_kernel_params(module, in_data)
            layer = LayerInfo(module.__class__.__name__, to_cpu(in_data), to_cpu(out_data))
            self._layer_info_dict[module].append(layer)

        if module not in seen:
            handles.append(module.register_forward_hook(store_data))
            seen.append(module)
        for sub_module in module.children():
            self.register_hook(sub_module, handles, seen)
        return handles

    @contextmanager
    def record_activations(self, model):
        self._layer_info_dict = defaultdict(list)
        # Important to pass in empty lists instead of initializing
        # them in the function as it needs to be reset each time.
        handles = self.register_hook(model, [], [])
        yield
        for handle in handles:
            handle.remove()

    '''
    ------------------------------------------------------------------------
    FIXATIONS FUNCTIONS
    ------------------------------------------------------------------------
    '''

    def fully_connected(self, fixations, module):
        layer_info = self._layer_info_dict[module].pop()
        # Fixation shape: (B, d1)
        weights = module.weight
        activations = layer_info.in_data[0]
        next_fixations = []
        #C = torch.stack([activations[i] * weights for i in range(len(activations))], 0)
        for neuron_pos in fixations:
            C = activations * weights[neuron_pos[1]]
            next_fixations += list((C > 0).nonzero())
        assert len(next_fixations) > 0, "Found no contributing connections, no visualization can be done"
        return U.unique(torch.stack(next_fixations, 0))

    def convolution(self, fixations, module):
        layer_info = self._layer_info_dict[module].pop()
        fixations = U.convert_flat_fixations(fixations, layer_info)
        # Fixation shape: (B, C, d1, d2, ..., dn) 
        weights = module.weight
        padding = tuple(U.as_tensor(module.padding).repeat_interleave(2))
        activations = nn.functional.pad(layer_info.in_data[0], padding)
        kernel_size = torch.LongTensor((activations.shape[1], *module.kernel_size))
        stride = torch.LongTensor((1, *module.stride))
        dilation = torch.LongTensor((1, *module.dilation))
        next_fixations = []
        for neuron_pos in fixations:
            batch = neuron_pos[0]
            weight = weights[neuron_pos[1]]  # Select the relevant filter for neuron
            slicer, lower_bound = U.get_slicer(neuron_pos[1:], kernel_size, stride, dilation)
            a_selection = activations[batch][slicer]  # Get activations going into neuron
            C = (a_selection * weight)
            ch = torch.argmax(C.sum(dim=tuple(range(1, len(weight.shape)))))  # Select the channel with greatest impact from previous layer
            pos = U.unflatten(torch.argmax(C[ch]), weight[ch].shape)  # Position of neuron with greatest impact in channel ch
            pos += lower_bound[1:] - U.as_tensor(module.padding)
            ch.unsqueeze_(0)  # Making channel into a 1D tensor with one element (for the concatination)
            batch.unsqueeze_(0) # Same as above
            new_fixation = torch.cat((batch, ch, pos), 0)
            assert all(new_fixation < U.as_tensor(activations.shape)), f"Point {new_fixation} doesn't fit in shape: {activations.shape}"
            next_fixations.append(new_fixation)  # Storing full position (B, C, d1, d2, ..., dn)
        return U.unique(torch.stack(next_fixations, 0))

    def maxpool(self, fixations, module):
        layer_info = self._layer_info_dict[module].pop()
        fixations = U.convert_flat_fixations(fixations, layer_info)
        # Fixation shape: (B, C, d1, d2, ..., dn)
        max_pool = module
        activations = layer_info.in_data[0]

        return_indices = max_pool.return_indices
        max_pool.return_indices = True
        _, max_indices = max_pool(activations)
        max_pool.return_indices = return_indices

        flat_fixations = U.flatten(fixations, max_indices.shape)
        next_fixations = max_indices.view(-1)[flat_fixations]
        next_fixations = U.unflatten(next_fixations, activations.shape[2:])
        next_fixations = torch.cat([fixations[:, :2], next_fixations], dim=1)
        return next_fixations

    def pass_on(self, fixations, layer_info):
        return fixations
