import torch 
from utils import get_kernel, read_img, save_img
from deconvolve import deconvolve

torch.manual_seed(1)
torch.use_deterministic_algorithms(False)

def main():

    im_path='im'+str(i)+'_kernel'+str(j)+'_img.png'
    k_path='kernel'+str(j)+'.png'
    save_path1=im_path.split('/')[-1].split('.')[0] + '_out.png'

    k, _ = get_kernel(k_path)
    y, cr, cb = read_img(im_path)
    y = y.float().cuda()/255.

    pad = k.shape[-1]
    y2 = torch.nn.functional.pad(y, (pad,pad,pad,pad), mode='reflect')

    lw_x2 = deconvolve(y2.float().cuda(),k.cuda(),num_iter=300,lam=1e-2,pad=pad,beta=.1)
    lw_x2 = lw_x2[:,:,pad:-pad,pad:-pad]

    save_img(torch.clip(lw_x2,0,1), cr, cb, save_path1, bw=True)

if __name__ == '__main__':
    main()