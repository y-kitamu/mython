# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import numpy as np


def check_loss_scale(n_classes, loss):
    """Estimate initial softmax cross entropy loss
    https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21
    Args:
        n_classes : number of cnn class
        loss : calculated loss (mean)
    """
    print("Estimated loss : {:.3f}, Calculated loss : {:.3f}".format(-math.log(1.0 / n_classes), loss))


def gradient_zero():
    pass


def gradient_infinit():
    pass


def weight_update():
    pass


def compare_gradient(numerical, analytical):
    """
    https://cs231n.github.io/neural-networks-3/#gradcheck
    """
    pass


def numerical_gradient(model, img, n_classes, channel_idx, x_idx, y_idx, h, device):
    """
    """
    assert isinstance(img, torch.Tensor)
    assert isinstance(model, nn.Module)
    img = img.to(device)[None, ...]
    assert len(img.shape) == 4

    val = img[0][channel_idx][y_idx][x_idx]
    img[0][channel_idx][y_idx][x_idx] += h
    z_plus = model(img)
    img[0][channel_idx][y_idx][x_idx] -= 2 * h
    z_minus = model(img)
    img[0][channel_idx][y_idx][x_idx] = val
    return ((z_plus - z_minus) / (2 * h)).detach().cpu().numpy()[0]


def analytical_gradient(model, x, n_classes, device):
    """
    """
    assert isinstance(x, torch.Tensor)
    assert isinstance(model, nn.Module)
    if len(x.shape) == 3:
        x = x[None, ...]
    assert len(x.shape) == 4

    x = x.to(device).requires_grad_(True)
    grads = np.zeros((x.shape[0], n_classes, x.shape[-3], x.shape[-2], x.shape[-1]))
    for batch_idx in range(x.shape[0]):
        for class_idx in range(n_classes):
            x.grad = None
            model.zero_grad()
            z = model(x)
            z.backward()
            grads[batch_idx, class_idx, ...] = x.grad[0].detach().cpu().numpy()
    return grads


def calc_relative_error(num_grad, ana_grad):
    assert num_grad.shape == ana_grad.shape
    return np.abs(num_grad - ana_grad) / np.maximum(np.abs(num_grad), np.abs(ana_grad))
