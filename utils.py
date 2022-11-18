import cv2
import torch
import numpy as np
from numpy import log
import math
from PIL import Image 

def get_kernel(path):
    kernel = torch.from_numpy(np.array(Image.open(path).convert('L'))).float()
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    kernel /= torch.sum(kernel)
    pad=kernel.shape[-1]//2
    return kernel, pad

def read_img(path):
    img = cv2.imread(path)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(x)

    return torch.from_numpy(y).view(1,1,y.shape[-2],y.shape[-1]), cr, cb

def torch_to_np(x):
    return x.detach().cpu().numpy()[0,0]

def save_img(x, cr, cb, path, bw=True):
    out_x_np = torch_to_np(x)
    out_x_np = np.uint8(255 * out_x_np)

    if not bw:
        out_x_np = cv2.merge([out_x_np, cr, cb])
        out_x_np = cv2.cvtColor(out_x_np, cv2.COLOR_YCrCb2BGR)
    
    cv2.imwrite(path, out_x_np)

def Gauss(sigma):
    sigma = np.array(sigma,dtype = 'float32')
    s=sigma.size
    if s==1:
       sigma=[sigma,sigma]
    sigma = np.array(sigma,dtype = 'float32')
    psfN = np.ceil(sigma / math.sqrt(8 * log(2)) * math.sqrt(-2 * log(0.0002))) + 1
    N = psfN * 2 + 1
    sigma = sigma / (2 * math.sqrt(2 * log(2)))
    dim = len(N)
    if dim > 1:
        N[1] = np.maximum(N[0], N[1])
        N[0] = N[1]
    if dim == 2:
        m = N[0]
        n = N[1]
        x = np.arange(-np.fix((n / 2)), np.ceil((n / 2)),dtype='float32')
        y = np.arange(-np.fix((m / 2)), np.ceil((m / 2)),dtype='float32')
        X, Y = np.meshgrid(x, y)
        s1 = sigma[0]
        s2 = sigma[1]
        PSF = np.exp(-(X * X) / (2 * np.dot(s1, s1)) - (Y * Y) / (2 * np.dot(s2, s2)))
        PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        center = [m / 2 + 1, n / 2 + 1]
        return PSF

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]