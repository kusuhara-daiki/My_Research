import cv2
import numpy as np
cimport numpy as cnp
import torch
import torch.nn.functional as F
cimport cython
from math import exp

@cython.boundscheck(False)
cdef gaussian(int window_size, float sigma):
    cdef: 
        int x
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

@cython.boundscheck(False)
def create_window(int window_size, int channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

@cython.boundscheck(False)
cdef cnp.ndarray[cnp.float32_t, ndim=2] _ssim(img1, img2, window, int window_size, int channel, bint size_average = True):
    cdef:
        float C1, C2
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().cpu().numpy()
    else:
        return ssim_map.mean(1).mean(1).mean(1).cpu().numpy()


@cython.boundscheck(False)
cdef cnp.ndarray[cnp.float32_t, ndim=2] _ssim_batch_cython(images, window, int window_size=11):
    cdef:
        int bs, channel, width, height
        float C1, C2
    (bs, channel, width, height) = images.size()
    mu = F.conv2d(images, window, padding = window_size//2, groups = channel)
    mu_sq = mu.pow(2)
    # #bbox x #bbox x channel x height x width になるように軸を増やして要素積を取る
    mu1_mu2 = (mu.unsqueeze(0)*mu.unsqueeze(1)).view(-1, channel, width, height)
    sigma_sq = F.conv2d(images.pow(2), window, padding = window_size//2, groups = channel) - mu_sq
    images_12 = (images.unsqueeze(0)*images.unsqueeze(1)).view(-1, channel, width, height)
    sigma12 = F.conv2d(images_12, window, padding = window_size//2, groups = channel
        ) - mu1_mu2
    mu1_mu2 = mu1_mu2.view(bs, bs, channel, width, height)
    sigma12 = sigma12.view(bs, bs, channel, width, height)
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/(
        (mu_sq.unsqueeze(0) + mu_sq.unsqueeze(1) + C1)
        *(sigma_sq.unsqueeze(0) + sigma_sq.unsqueeze(1) + C2))

    ssim_matrix = ssim_map.mean(-1).mean(-1).mean(-1)
    ssim_matrix[range(bs), range(bs)] = 0.0
    return ssim_matrix.cpu().numpy()

@cython.boundscheck(False)
def _appearance_feature_ssim_batch_cython_hybrid(
    cnp.ndarray[cnp.uint8_t, ndim=3] image,
    cnp.ndarray[cnp.int64_t, ndim=2] all_bbox, 
    window, 
    int window_size, 
    int channel, 
    int _resize,
    int q_matrix_threshold=100, 
    str device="cuda:0"):
    cdef:
        int n_candidates, i
        cnp.ndarray bbox
        cnp.ndarray[cnp.float32_t, ndim=2] similarity
    prepr = lambda x: torch.from_numpy(
        np.rollaxis(cv2.resize(x, (_resize, _resize)), 2)
        ).float()/255.0
    bbox_list = torch.stack([prepr(image[bbox[1] : bbox[3], bbox[0] : bbox[2]]) for bbox in all_bbox])
    n_candidates = len(all_bbox)
    if n_candidates > q_matrix_threshold: # should determine based on GPU memory
        similarity = np.zeros((n_candidates, n_candidates), dtype=np.float32)
        for i in range(1, n_candidates):
            _base_bbox = bbox_list[i].expand(i, -1, -1, -1)
            similarity[i, :i] = _ssim(_base_bbox.to(device), bbox_list[:i].to(device), window.to(device), window_size=window_size, channel=channel, size_average=False)
    else: # 行列を分解して計算するのもあり
        similarity = _ssim_batch_cython(bbox_list.to(device), window.to(device), window_size=window_size)
    return similarity
