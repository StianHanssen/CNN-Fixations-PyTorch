import utils as U
import torch
import torch.nn as nn
from math import floor
import torch.nn.modules as M
from conv2_1d import Conv2_1d
from torch.nn import functional as F

class LayerInfo():
    KERNEL_TYPES = [M.conv._ConvNd, M.pooling._MaxPoolNd]
    def __init__(self, module, in_data=None, out_data=None, sub_layers=[]):
        self.module = module
        self.in_data = in_data
        self.out_data = out_data
        self.sub_layers = sub_layers
        self._expand_kernel_params()
    
    def __str__(self):
        string = "Info" + str(self.module)[:-1]
        string += ", in_shape: " + str((*self.get_in_shape().tolist(),)) + ", "
        string += "out_shape: " + str((*self.get_out_shape().tolist(),)) + ")"
        return string
    
    def __repr__(self):
        return "Info(" + self.module.__class__.__name__ + ")"
    
    @staticmethod
    def __int_to_tuple(param, dims):
        if isinstance(param, int):
            return tuple(param for _ in range(dims))
        return param

    def _expand_kernel_params(self):
        if not any([isinstance(self.module, t) for t in LayerInfo.KERNEL_TYPES]):
            return
        if self.in_data is None:
            print("in_data is None, kernel hyperparameter expansion not done.")
            return
        dims = len(self.in_data[0][0][0].shape)
        m = self.module
        m.kernel_size = LayerInfo.__int_to_tuple(m.kernel_size, dims)
        m.padding = LayerInfo.__int_to_tuple(m.padding, dims)
        m.stride = LayerInfo.__int_to_tuple(m.stride, dims)
        m.dilation = LayerInfo.__int_to_tuple(m.dilation, dims)
    
    def get_in_shape(self):
        if self.in_data is None:
            return None
        return torch.LongTensor((len(self.in_data), *self.in_data[0].shape))
    
    def get_out_shape(self):
        if self.out_data is None:
            return None
        return torch.LongTensor((len(self.out_data), *self.out_data[0].shape))

    @staticmethod
    def register_hook(module, storage=[]):
        
        def store_data(module, in_data, out_data):
            layer = LayerInfo(module, in_data, out_data)
            storage.append(layer)
        
        module.register_forward_hook(store_data)
        for sub_module in module.children():
            LayerInfo.register_hook(sub_module, storage)
        return storage

    @staticmethod
    def link_sub_layers(storage):
        storage = list(reversed(storage))
        for layer in storage:
            sub_modules = list(layer.module.children())
            if sub_modules:
                layer.sub_layers = [x for x in storage if x.module in sub_modules]
        return storage[0].sub_layers  

'''
------------------------------------------------------------------------
FIXATIONS FUNCTIONS
------------------------------------------------------------------------
'''

def fixations_fc(fixations, layer_info):
    # Fixation shape: (B, d1)
    weights = layer_info.module.weight
    activations = layer_info.in_data[0]
    next_fixations = []
    #C = torch.stack([activations[i] * weights for i in range(len(activations))], 0)
    for neuron_pos in fixations:
        C = activations * weights[neuron_pos[1]]
        next_fixations += list((C > 0).nonzero())
    assert len(next_fixations) > 0, "Found no contributing connections, no visualization can be done"
    return U.unique(torch.stack(next_fixations, 0))

def fixations_conv(fixations, layer_info):
    fixations = U.convert_flat_fixations(fixations, layer_info)
    # Fixation shape: (B, C, d1, d2, ..., dn) 
    weights = layer_info.module.weight
    padding = tuple(U.as_tensor(layer_info.module.padding).repeat_interleave(2))
    activations = nn.functional.pad(layer_info.in_data[0], padding)
    kernel_size = torch.LongTensor((activations.shape[1], *layer_info.module.kernel_size))
    stride = torch.LongTensor((1, *layer_info.module.stride))
    dilation = torch.LongTensor((1, *layer_info.module.dilation))
    next_fixations = []
    for neuron_pos in fixations:
        batch = neuron_pos[0]
        weight = weights[neuron_pos[1]]  # Select the relevant filter for neuron
        slicer, lower_bound = U.get_slicer(neuron_pos[1:], kernel_size, stride, dilation)
        a_selection = activations[batch][slicer]  # Get activations going into neuron
        C = (a_selection * weight)
        ch = torch.argmax(C.sum(dim=tuple(range(1, len(weight.shape)))))  # Select the channel with greatest impact from previous layer
        pos = U.unflatten(torch.argmax(C[ch]), weight[ch].shape)  # Position of neuron with greatest impact in channel ch
        pos += lower_bound[1:] - U.as_tensor(layer_info.module.padding)
        ch.unsqueeze_(0)  # Making channel into a 1D tensor with one element (for the concatination)
        batch.unsqueeze_(0) # Same as above
        new_fixation = torch.cat((batch, ch, pos), 0)
        assert all(new_fixation < U.as_tensor(activations.shape)), "Point " + str(new_fixation) + " doesn't fit in shape: " + str(activations.shape)
        next_fixations.append(new_fixation)  # Storing full position (B, C, d1, d2, ..., dn)
    return U.unique(torch.stack(next_fixations, 0))

def fixations_maxpool(fixations, layer_info):
    fixations = U.convert_flat_fixations(fixations, layer_info)
    # Fixation shape: (B, C, d1, d2, ..., dn)
    max_pool = layer_info.module
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

def fixations_pass(fixations, layer_info):
    return fixations

def fixations_conv2_1d(fixations, layer_info): # NEEDS FIXING: Unflatten to wrong shape
    fixations = U.convert_flat_fixations(fixations, layer_info)
    # Fixations shape: (B, C, D, H, W)
    shape = U.as_tensor((len(layer_info.out_data), *layer_info.out_data[0].shape))
    conv_2d, conv_1d = layer_info.sub_layers
    # Rearranging points to (B, H, W, C, D)
    shape = shape[torch.LongTensor([0, 3, 4, 1, 2])]
    fixations = fixations[:, torch.LongTensor([0, 3, 4, 1, 2])]

    # 3D points -> 1D points, points for shape (B*H*W, C, D)
    fixations[:, 2] = U.flatten(fixations[:, :3], shape[:3])
    
    # Getting 1D fixations
    fixations_1d = fixations_conv(fixations[:, 2:], conv_1d)
    # Back to 3D points for rearrangment to (B, D, C, H, W)
    expanded = U.unflatten(fixations_1d[:, 0], shape[:3])
    fixations_1d = torch.cat([expanded, fixations_1d[:, 1:]], dim=1)
    shape = shape[torch.LongTensor([0, 4, 3, 1, 2])]
    fixations_1d = fixations_1d[:, torch.LongTensor([0, 4, 3, 1, 2])]
    # 3D points -> 2D points, points for shape (B*D, C, H, W)
    fixations_1d[:, 1] = U.flatten(fixations_1d[:, :2], shape[:2])
    
    # Getting 2D fixations
    fixations_2d = fixations_conv(fixations_1d[:, 1:], conv_2d)
    # Back to 3D points for rearrangment to (B, C, D, H, W)
    expanded = U.unflatten(fixations_2d[:, 0], shape[:2])
    fixations_2d = torch.cat([expanded, fixations_2d[:, 1:]], dim=1)
    fixations_2d = fixations_2d[:, torch.LongTensor([0, 4, 3, 1, 2])]
    return fixations_2d

'''
------------------------------------------------------------------------
FIXATIONS EXECUTOR
------------------------------------------------------------------------
'''

fixation_functions = [(M.conv._ConvNd, fixations_conv), 
                      (M.pooling._MaxPoolNd, fixations_maxpool), 
                      (M.batchnorm._BatchNorm, fixations_pass), 
                      (M.Linear, fixations_fc)]

def fixations(fixations, layer_info_list, debug=False):
    layer_info_list = LayerInfo.link_sub_layers(layer_info_list)
    print("Network:\n" + str(layer_info_list))
    for layer_info in layer_info_list:
        next_fixations = None
        for module_type, fixation_func in fixation_functions:
            if isinstance(layer_info.module, module_type):
                next_fixations = fixation_func(fixations, layer_info)
                if(next_fixations.shape[1] > 2 and debug):
                    print("Running:", layer_info.__repr__())
                    print("Shape:", (*layer_info.in_data[0].shape,))
                    print(next_fixations[next_fixations[:, 2:].sum(dim=1).max(dim=0)[1]])
                    print()
        assert next_fixations is not None, "Found no function to handle: " + str(layer_info)
        fixations = next_fixations
    return fixations
        

if __name__ == '__main__':
    sel = 0
    if sel == 0: # Convolution
        fixations = torch.LongTensor([[0, 1, 2, 2], [0, 0, 3, 1]])
        in_channels, out_channels = 3, 2
        batch_size = 1
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        info = LayerInfo(conv)
        info.in_data = (torch.Tensor([[[2, 4, 8, 6],
                                       [7, 1, 0, 2],
                                       [1, 1, 3, 6],
                                       [7, 8, 10, 1]], 
                                      [[0, 6, 0, 6],
                                       [1, 0, 5, 4],
                                       [1, 2, 2, 4],
                                       [7, 1, 3, 4]], 
                                      [[9, 1, 2, 1],
                                       [3, 2, 1, 11],
                                       [10, 10, 4, 3],
                                       [6, 12, 6, 8]]]).unsqueeze(0),)
        info.out_data = conv(info.in_data[0])
        info._expand_kernel_params()
        conv.weight.data = torch.Tensor([[[[1, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 1]], 
                                           [[2, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2]], 
                                           [[3, 3, 3],
                                            [3, 3, 3],
                                            [3, 3, 3]]],
                                          [[[1, 2, 3],
                                            [4, 5, 6],
                                            [7, 8, 9]], 
                                           [[10, 11, 12],
                                            [13, 14, 15],
                                            [16, 17, 18]], 
                                           [[19, 20, 21],
                                            [22, 23, 24],
                                            [25, 26, 27]]]])
        #init_weights(conv)

        print(fixations_conv(fixations, info))
    elif sel == 1: # Fully Connected
        fixations = torch.LongTensor([[0, 1]])
        batch_size = 5
        in_size = 100
        out_size = 200
        fc = nn.Linear(in_size, out_size)
        info = LayerInfo(fc)
        info.in_data = (torch.randint(-10, 2, (5, 100), dtype=torch.float),)
        info.out_data = fc(info.in_data[0])
        info._expand_kernel_params()
        U.init_weights(fc)
        fc.weight.data = torch.abs(fc.weight.data)

        print(fixations_fc(fixations, info))
    elif sel == 2: # Max Pool
        fixations = torch.LongTensor([[3, 4, 5, 5, 5], [0, 1, 3, 0, 1], [1, 6, 2, 4, 5]])
        in_channels, out_channels = 8, 16
        batch_size = 5
        max_pool = nn.MaxPool3d(2, 2)
        info = LayerInfo(max_pool)
        info.in_data = (torch.rand(batch_size, in_channels, 12, 12, 12),)
        info.out_data = max_pool(info.in_data[0])
        info._expand_kernel_params()

        print(fixations_maxpool(fixations, info))
    if sel == 3: # (2+1)D Convolution
        fixations = torch.LongTensor([[4, 15, 1, 11, 3]])
        in_channels = 8
        conv = Conv2_1d(in_channels, out_channels=16, kernel_size=3, padding=1)
        mid_size = conv.hidden_size
        info = LayerInfo(conv)
        for layer in conv.children():
            info.sub_layers.append(LayerInfo(layer))
        info2d, info1d = info.sub_layers

        info.in_data = (torch.rand(5, in_channels, 12, 12, 12),)
        info.out_data = conv(info.in_data[0])

        b, c, d, h, w = info.in_data[0].size()
        x = info.in_data[0].permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*d, c, h, w)
        info2d.in_data = (x,)
        info2d.out_data = F.relu(info2d.module(x))

        c, h, w = info2d.out_data.size()[1:]
        x = info2d.out_data.view(b, d, c, h, w)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*h*w, c, d)
        info1d.in_data = (x,)
        info1d.out_data = info1d.module(x)

        final_c, final_d = info1d.out_data.size()[1:]
        x = info1d.out_data.view(b, h, w, final_c, final_d)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        
        U.init_weights(conv)

        print(fixations_conv2_1d(fixations, info))
