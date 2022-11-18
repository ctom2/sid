import torch
from ssim import SSIM 
from hessian import hessian_loss
from fft_conv_pytorch import fft_conv

def deconvolve(y, k, num_iter=2000, lam=1e-2, beta=0.08,pad=None):
    xi = torch.nn.Parameter(torch.ones(y.shape)*0.4).float()

    ssim = SSIM().cuda()
    optimizer = torch.optim.Adam([xi], lr=lam)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.2)

    for it in range(num_iter):
        optimizer.zero_grad()

        y_out = fft_conv(xi.cuda(), k.cuda(),padding=k.shape[-1]//2,padding_mode='reflect')
        
        loss = -ssim(y_out[:,:,pad:-pad,pad:-pad], y[:,:,pad:-pad,pad:-pad]) \
                +hessian_loss(xi[0,:,:,:])*beta \
                +torch.mean(torch.abs(y_out[:,:,pad:-pad,pad:-pad] - y[:,:,pad:-pad,pad:-pad]))
        
        if (it % 50 == 0) or (it == num_iter - 1):
            print(' Iteration: {}, loss: {}'.format(it, round(loss.item(),4)))

        loss.backward()
        optimizer.step()
        scheduler.step()

    return torch.clip(xi,0,1).detach()