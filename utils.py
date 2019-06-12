'''
Credits: https://github.com/val-iisc/cnn-fixations/blob/master/demo/utils.py
The following functions are taken from the original implementation:
 - outlier_removal
 - heatmap (small modification allowing to set k, adaptable sigma)
 - visualize (small modification allowing to set k, red dots)
'''
import numpy as np
import torch
from torch.cuda import current_device, get_device_capability, is_available
from conv2_1d import Conv2_1d
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from numpy import ravel_multi_index, unravel_index, unique as npunique

def as_tensor(iterable):
    return torch.LongTensor(list(iterable))

def slice_weight(lower_bound, upper_bound, shape, padding, dilation, act_shape): # UNFINISHED
    act_shape += padding
    lower_crop = -(lower_bound - padding) / dilation
    lower_crop = torch.where(lower_crop > 0, lower_crop, torch.zeros(lower_crop.shape, dtype=torch.long))
    upper_crop = (act_shape - upper_bound) / dilation
    upper_crop = torch.where(upper_crop > 0, upper_crop, shape)

def get_slicer(neuron_pos, kernel_size, stride, dilation):
    lower_bound = neuron_pos * stride
    lower_bound[0] = 0
    size = kernel_size + (kernel_size - 1) * (dilation - 1)
    upper_bound = lower_bound + size
    slicer = tuple(slice(lower_bound[i].item(), upper_bound[i].item(), dilation[i].item()) for i in range(len(upper_bound)))
    return slicer, lower_bound

def unflatten(indices, shape):
    return torch.LongTensor(unravel_index(indices, list(shape))).t()

def flatten(points, shape):
    return torch.from_numpy(ravel_multi_index(points.t().numpy(), list(shape)))

def convert_flat_fixations(fixations, layer_info):
    if len(fixations[0]) == 2:
        shape = as_tensor((len(layer_info.out_data), *layer_info.out_data[0].shape))
        expanded = unflatten(fixations[:, 1], shape[1:])
        fixations = torch.cat([fixations[:, 0].unsqueeze(1), expanded], dim=1)
    return fixations

def unique(fixations, d=False):
    return torch.from_numpy(npunique(fixations.numpy(), axis=0))

def init_weights(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, Conv2_1d):
        for mm in (m.conv2d, m.conv1d):
            torch.nn.init.xavier_normal_(mm.weight)
            if mm.bias is not None:
                mm.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.Linear) and 0:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def chunk_batch(points, remove_chanels=True):
    if remove_chanels:
        points = torch.cat([points[:, :1], points[:, 2:]], dim=1)
    ordered = points[points[:, 0].argsort(dim=0)]
    chunks = ordered[:, 0].bincount().tolist()
    return ordered[:, 1:].split(chunks, dim=0)

def outlier_removal(points, diag):
    neighbors = np.zeros((points.shape[0]))
    selPoints = np.empty((1, 2))
    for i in range(points.shape[0]):
        diff = np.sqrt(np.sum(np.square(points-points[i]), axis=1))
        neighbors[i] = np.sum(diff < diag)
    for i in range(points.shape[0]):
        if neighbors[i] > 0.05*points.shape[0]:
            selPoints = np.append(selPoints, points[i:i+1, :], axis=0)
    selPoints = selPoints[1:, :]
    selPoints = selPoints.astype(int)
    return selPoints

def heatmap(img, points, sigma=None, k=None):
    if k is None:
        k = (np.min(img.shape[:2])) if (
            np.min(img.shape[:2]) % 2 == 1) else (np.min(img.shape[:2])-1)
    if sigma is None:
        sigma= k / 20.45
    mask = np.zeros(img.shape[:2])
    shape = mask.shape
    for i in range(points.shape[0]):
        # Check if inside the image
        if points[i, 0] < shape[0] and points[i, 1] < shape[1]:
            mask[points[i, 0], points[i, 1]] += 1
    # Gaussian blur the points to get a nice heatmap
    blur = cv2.GaussianBlur(mask, (k, k), sigma)
    blur = blur*255/np.max(blur)
    return blur

def visualize(img, points, diag_percent, image_label, k=None):
    shape = img.shape
    if len(shape) == 3 and shape[0] == 1:
        img.squeeze_(0)
    elif len(shape) == 3 and shape[0] == 3:
        img = img.permute(1, 2, 0).contiguous()
    vmin, vmax = img.min(), img.max()
    diag = np.sqrt(sum([dim**2 for dim in img.shape]))*diag_percent
    values = points.numpy()
    selPoints = outlier_removal(values, diag)
    # Make heatmap and show images
    hm = heatmap(np.copy(img), selPoints, k=k)
    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax), ax[0].axis('off'), ax[0].set_title(image_label)
    ax[1].imshow(img, cmap='gray', vmin=vmin, vmax=vmax), ax[1].axis('off'),
    ax[1].scatter(selPoints[:, 1], selPoints[:, 0], c='r'),
    ax[1].set_title('CNN Fixations')
    ax[2].imshow(img, cmap='gray', vmin=vmin, vmax=vmax), ax[2].imshow(hm, 'jet', alpha=0.6)
    ax[2].axis('off'), ax[2].set_title('Heatmap')
    plt.show()