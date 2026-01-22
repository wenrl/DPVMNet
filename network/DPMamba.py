import torch
from torch import nn

from mamba_ssm import Mamba
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F



class LearnableScanner(nn.Module):
    def __init__(
        self,
        in_dim=64,
        reduction_ratio=8,
        num_tokens=64,  
        return_tokens_layout="BMC", 
        eps=1e-6,
    ):
        super().__init__()
        assert return_tokens_layout in ("BMC", "BCM")

        self.in_dim = in_dim
        self.num_tokens = num_tokens
        self.return_tokens_layout = return_tokens_layout
        self.eps = float(eps)

        mid = max(1, in_dim // reduction_ratio)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, mid, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(mid, in_dim, kernel_size=1),
            nn.Sigmoid()
        )

        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, groups=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(1, 4, kernel_size=3, dilation=1, padding=1, bias=False),
            nn.Conv2d(4, 4, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.Conv2d(4, 4, kernel_size=3, dilation=4, padding=4, bias=False),
            nn.Conv2d(4, 1, kernel_size=1),
            nn.Sigmoid()
        )

       
        self.assign = nn.Conv2d(in_dim, num_tokens, kernel_size=1, bias=False)

    

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == self.in_dim, f"channel mismatch: got {C}, expected {self.in_dim}"
        N = H * W
        ch = self.channel_attn(x)  # (B,C,1,1)
        x_ch = x * ch + x          # (B,C,H,W)
        x_weighted = x_ch.sum(dim=1, keepdim=True)          # (B,1,H,W)
        sp = self.spatial_attn(x_weighted)                  # (B,1,H,W)
        x_att = x_ch * sp + x_ch                             # (B,C,H,W)
        logits = self.assign(x_att).reshape(B, self.num_tokens, N)  # (B,M,N)
        A = F.softmax(logits, dim=1)  # (B,M,N)
        feats = x_att.reshape(B, C, N)                       # (B,C,N)
        tokens = torch.bmm(feats, A.transpose(1, 2))          # (B,C,M)

        if self.return_tokens_layout == "BMC":
            tokens_out = tokens.transpose(1, 2).contiguous()  # (B,M,C)
        else:
            tokens_out = tokens                               # (B,C,M)

        ctx = {"A": A, "H": H, "W": W}
        return tokens_out, ctx

    @torch.no_grad()
    def _check_ctx(self, ctx):
        if not isinstance(ctx, dict):
            raise TypeError("ctx must be the dict returned by forward")
        for k in ("A", "H", "W"):
            if k not in ctx:
                raise KeyError(f"ctx missing key {k}")

    def inverse(self, tokens_out: torch.Tensor, ctx: dict, tokens_layout="BMC"):
        self._check_ctx(ctx)
        A = ctx["A"]               # (B,M,N)
        H, W = int(ctx["H"]), int(ctx["W"])

        if tokens_layout == "BMC":
            t = tokens_out.transpose(1, 2).contiguous()      # (B,C,M)
        elif tokens_layout == "BCM":
            t = tokens_out                                    # (B,C,M)
        else:
            raise ValueError("tokens_layout must be 'BMC' or 'BCM'")

        feats_rec = torch.bmm(t, A)                            # (B,C,N)
        B, C, N = feats_rec.shape
        x_rec = feats_rec.reshape(B, C, H, W)
        return x_rec


class patch_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=in_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.ds = nn.Sequential(nn.AvgPool2d(2, 2, padding=0),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(out_channels))
        self.fusion = nn.Sequential(nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU(out_channels)
                                    )
    def _channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self,x):
        x = torch.concat([self.ds(x), self.step(x)],dim=1)
        x = self._channel_shuffle(x,2)
        x = self.fusion(x)
        return x

class DPMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False, use_att=False, stride=1):
        super().__init__()
        self.att = use_att
        self.stride = stride

        self.DWconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.PWconv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU()
        self.DWconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.PWconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if first == True:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.residual = nn.Identity()
        if self.att == True:
            self.ln_att0 = nn.LayerNorm(out_channels)
            self.ln_att01 = nn.LayerNorm(out_channels)
            self.ln_att = nn.LayerNorm(out_channels)
            self.bn_att =  nn.Sequential(nn.BatchNorm2d(out_channels*3),
                nn.Conv2d(out_channels*3, out_channels, kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(out_channels))
            self.resM = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(out_channels))
            

        self.mamba = Mamba(
            d_model=out_channels,       
            d_state=16,      
            d_conv=4,         
            expand=2,         
        )
       
        self.mamba0 = Mamba(
            d_model=out_channels,      
            d_state=16,     
            d_conv=4,       
            expand=2,      
        )
        self.mamba01 = Mamba(
            d_model=out_channels,     
            d_state=16,     
            d_conv=4,      
            expand=2,      
        )
        self.scanner = LearnableScanner(in_dim=out_channels)


    def _channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        residual = x
        if self.att == True:
            x = self.DWconv1(x)
            x = self.PWconv1(x)
            x = self.bn1(x)
            B, C, H, W = x.shape
            x0, x01 = x, x.permute(0, 1, 3, 2)

            x0 = x0.reshape(B, C, -1).permute(0, 2, 1)  # [B, 256, 32]
            x0 = self.ln_att0(x0)
            x0 = self.mamba0(x0)  # [B, 256, 32]
            x0 = x0.permute(0,2,1).reshape(B,C,H,W)

            x01 = x01.reshape(B, C, -1).permute(0, 2, 1)  # [B, 256, 32]
            x01 = self.ln_att01(x01)
            x01 = self.mamba01(x01)  # [B, 256, 32]
            x01 = x01.permute(0,2,1).reshape(B,C,H,W)
            x01 = x01.permute(0,1,3,2)
            x1, inv_order = self.scanner(x)###
            x1 = self.ln_att(x1)
            x1 = self.mamba(x1)  # [B, 256, 32]
            x1 = self.scanner.inverse(x1, inv_order, tokens_layout="BMC")###
            x1 = self.bn_att(torch.concat([x0, x01, x1], dim=1))
            x = self.resM(x) + x1
            
            x = self.DWconv2(x)
            x = self.PWconv2(x)
            x = self.bn2(x)
        else:
            x = self.DWconv1(x)
            x = self.PWconv1(x)
            x = self.bn1(x)
            x = self.DWconv2(x)
            x = self.PWconv2(x)
            x = self.bn2(x)
        return x + residual


class UPDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, up=True, last=True, att = True):
        super().__init__()
        self.att = att
        if last ==True:
            wrl = in_channels
            wrl1 = in_channels * 2
        else:
            wrl = in_channels // 2
            wrl1= in_channels
        self.step1 = nn.Sequential(
            nn.Conv2d(wrl1, in_channels // 2, kernel_size=3, stride=1, padding=1, groups=in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 2),
            nn.PReLU())
        self.DWconv1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, groups=in_channels // 2)
        self.PWconv1 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Sequential(nn.Conv2d(wrl, out_channels, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(out_channels))
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(in_channels, wrl, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(wrl))
        
         
        if self.att == True:
            self.ln_att0 = nn.LayerNorm(in_channels // 2)
            self.ln_att01 = nn.LayerNorm(in_channels // 2)
            self.ln_att = nn.LayerNorm(in_channels // 2)
            self.bn_att =  nn.Sequential(nn.BatchNorm2d(in_channels // 2*3),
                nn.Conv2d(in_channels // 2*3, in_channels // 2, kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(in_channels // 2))
            self.resM = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(in_channels // 2))
            
        # Mamba
        self.mamba = Mamba(
            d_model=in_channels // 2,
            d_state=16, 
            d_conv=4,     
            expand=2,    
        )   
        # Mamba0
        self.mamba0 = Mamba(
            d_model=in_channels // 2,       
            d_state=16, 
            d_conv=4,         
            expand=2,        
        )
        
        self.mamba01 = Mamba(
            d_model=in_channels // 2,      
            d_state=16, 
            d_conv=4,  
            expand=2, 
        )
        self.scanner = LearnableScanner(in_dim=in_channels // 2)
    def _channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x2, x1):

        x2 = self.upsample(x2)
        residual = x2
        x = torch.cat([x2, x1], dim=1)
        x = self._channel_shuffle(x, 2)
        x = self.step1(x)
        if self.att == True:
            
            B, C, H, W = x.shape
            x0, x01 = x, x.permute(0, 1, 3, 2)
            x0 = x0.reshape(B, C, -1).permute(0, 2, 1)  # [B, 256, 32]
            x0 = self.ln_att0(x0)
            x0 = self.mamba0(x0)  # [B, 256, 32]
            x0 = x0.permute(0,2,1).reshape(B,C,H,W)
            x01 = x01.reshape(B, C, -1).permute(0, 2, 1)  # [B, 256, 32]
            x01 = self.ln_att01(x01)
            x01 = self.mamba01(x01)  # [B, 256, 32]
            x01 = x01.permute(0,2,1).reshape(B,C,H,W)
            x01 = x01.permute(0,1,3,2)
            
            x1, inv_order = self.scanner(x)###
            x1 = self.ln_att(x1)
            x1 = self.mamba(x1)  # [B, 256, 32]
            x1 = self.scanner.inverse(x1, inv_order, tokens_layout="BMC")###
            x1 = self.bn_att(torch.concat([x0,x01,x1], dim=1))
            x = self.resM(x) + x1
        x = self.DWconv1(x)
        x = self.PWconv1(x)
        x = self.bn1(x)
        return x + self.residual(residual)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=False, blocks=None, block1=DPMambaBlock,
                 block2=UPDoubleConv):
        super().__init__()
        init_channels = 64
        self.with_bn = with_bn
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(init_channels))

        self.en_1 = self._make_layer(block1, 1 * init_channels, 1 * init_channels, blocks=blocks[0], use_att=True)
        self.en_2 = self._make_layer(block1, 1 * init_channels, 2 * init_channels, blocks=blocks[1], use_att=True)
        self.en_3 = self._make_layer(block1, 2 * init_channels, 4 * init_channels, blocks=blocks[2], use_att=True)
        self.en_4 = self._make_layer(block1, 4 * init_channels, 8 * init_channels, blocks=blocks[3], use_att=True)

        self.de_1 = UPDoubleConv(8 * init_channels, 4 * init_channels, last=False, att=True)
        self.de_2 = UPDoubleConv(4 * init_channels, 2 * init_channels, last=False, att=True)
        self.de_3 = UPDoubleConv(2 * init_channels, 1 * init_channels, last=False, att=True)
        self.de_4 = UPDoubleConv(1 * init_channels, out_channels, last=True, att=True)


        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(in_channels=4 * init_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(in_channels=2 * init_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=1 * init_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, use_att=False):
        layers = []
        layers.append(patch_embedding(inplanes, planes))
        for i in range(0, blocks):
            layers.append(block(planes, planes, first=False, use_att=use_att, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        e1 = self.en_1(x)
        e2 = self.en_2(e1)
        e3 = self.en_3(e2)
        e4 = self.en_4(e3)

        d1 = self.de_1(e4, e3)
        d2 = self.de_2(d1, e2)
        d3 = self.de_3(d2, e1)
        d4 = self.de_4(d3, x)

        return [self.up1(d1), self.up2(d2), self.up3(d3), d4]



def DPVMNet(in_channels=1, out_channels=9):
    blocks = [2, 3, 12, 3]
    return UNet(in_channels=in_channels, out_channels=out_channels, with_bn=True, blocks=blocks)
