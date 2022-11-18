from utils import Gauss
from fft_conv_pytorch import fft_conv
import numpy as np
import torch

def preprocess(y, pad):
    pad2 = pad*2

    # inital padding (torch standard)
    yy = torch.nn.functional.pad(y, (pad2,pad2,pad2,pad2), mode='reflect')

    # blurring PSF for padded space
    psf = Gauss(10)
    psf /= np.sum(psf)

    # mask
    mask = np.zeros((1,1,y.shape[-2] + pad2*2, y.shape[-1] + pad2*2))
    mask[:,:,pad2:-pad2,pad2:-pad2] = np.ones(y.shape)
    
    mask = torch.from_numpy(mask).cuda()
    psf = torch.from_numpy(psf).unsqueeze(0).unsqueeze(0).cuda()

    # blurred mask
    mask = fft_conv(mask, psf, padding=psf.shape[-1]//2,padding_mode='reflect')

    # blur padded image
    yy_blur = fft_conv(yy, psf, padding=psf.shape[-1]//2,padding_mode='reflect')
    yy_blur = torch.clip(yy_blur, 0, 1)

    # combine based on mask
    combined = (1-mask)*yy_blur + mask*yy
    combined = combined[:,:,pad:-pad,pad:-pad]

    # crop additional padding
    # combined2 = torch.from_numpy(combined[pad:-pad,pad:-pad])
    # combined2 = combined2.view(1,1,combined2.shape[-2],combined2.shape[-1])

    return combined