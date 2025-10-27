from networks import swinIR_1_denoising as swinIR_s
from networks import swinIR_2_denoising as swinIR_l
import torch
import torch.nn as nn
class SwinIR(nn.Module):
    def __init__(self, in_chans=1, embed_dim=60, **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim*2, num_out_ch, 3, 1, 1)
        self.swinIR_s = swinIR_s()
        self.swinIR_l = swinIR_l()
        self.p_criterion = nn.MSELoss()

    def forward(self,x):
        x_first=self.conv_first(x)
        x_s=self.swinIR_s(x_first)
        x_l=self.swinIR_l(x_first)
        x_cat=torch.cat((x_s,x_l),dim=1)
        x_res=self.conv_last(x_cat)
        x_last=x+x_res
        # print(self.swinIR_l.flops())
        return x_last



