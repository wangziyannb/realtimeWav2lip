# #!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/30/2024 1:32 AM
# @Author  : Wang Ziyan
# @Email   : 1269586767@qq.com
# @File    : FID.py
# @Software: PyCharm
import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F


def get_activations(images, model, batch_size=1, dims=2048, cuda=False):
    model.eval()
    if cuda:
        model.cuda()
    act = np.empty((len(images), dims))
    for i in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[i:i+batch_size]).type(torch.FloatTensor).permute(0, 3, 1, 2)
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)
        act[i:i+batch_size] = pred.cpu().data.numpy().reshape(batch_size, -1)
    return act


def calculate_fid(mu1, sigma1, mu2, sigma2):
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def getFID(real_images, gen_images, model):

    real_activations = get_activations(real_images, model, batch_size=1, cuda=True)
    gen_activations = get_activations(gen_images, model, batch_size=1, cuda=True)

    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)

    mu_gen = np.mean(gen_activations, axis=0)
    sigma_gen = np.cov(gen_activations, rowvar=False)


    fid_score = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid_score
