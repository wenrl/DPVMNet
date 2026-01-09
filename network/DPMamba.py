import torch
from torch import nn
from network.former import MHSAblock
from mamba_ssm import Mamba
from functools import partial

class patch_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=in_channels),
            # nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            # nn.PReLU(),
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
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self,x):
        x = torch.concat([self.ds(x), self.step(x)],dim=1)
        x = self._channel_shuffle(x,2)
        x = self.fusion(x)
        return x


class DownDoubleConv(nn.Module):
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
            
        # Mamba
        self.mamba = Mamba(
            d_model=out_channels,       # 输入特征维度（与CNN输出通道一致）
            d_state=out_channels // 2,       # SSM状态维度
            d_conv=4,         # 局部卷积维度
            expand=2,         # 扩展因子
        )


    def _channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        # B, C, H, W = x
        residual = x

        if self.att == True:

            x = self.DWconv1(x)
            x = self.PWconv1(x)
            x = self.bn1(x)
            
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

            
        # Mamba
        self.mamba = Mamba(
            d_model=in_channels // 2,       # 输入特征维度（与CNN输出通道一致）
            d_state=in_channels // 4,       # SSM状态维度
            d_conv=4,         # 局部卷积维度
            expand=2,         # 扩展因子
        )

    def _channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x2, x1):
        # B, C, H, W = x
        x2 = self.upsample(x2)
        residual = x2
        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        x = self._channel_shuffle(x, 2)
        x = self.step1(x)
        if self.att == True:
            x1, inv_order = self.scanner(x)
            x1 = x1.permute(0, 2, 1)
            x1 = self.ln_att(x1)
            x1 = self.mamba(x1)
            x = x + x1
        x = self.DWconv1(x)
        x = self.PWconv1(x)
        x = self.bn1(x)
        return x + self.residual(residual)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=False, blocks=None, block1=DownDoubleConv):
        super().__init__()
        init_channels = 64
        self.with_bn = with_bn
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(init_channels))

        # self.en_1 = DownDoubleConv(1*init_channels, 1*init_channels, with_bn)
        self.en_1 = self._make_layer(block1, 1 * init_channels, 1 * init_channels, blocks=blocks[0], use_att=True)
        # self.en_2 = DownDoubleConv(1*init_channels, 2*init_channels, with_bn)
        self.en_2 = self._make_layer(block1, 1 * init_channels, 2 * init_channels, blocks=blocks[1], use_att=True)
        # self.en_3 = DownDoubleConv(2*init_channels, 4*init_channels, with_bn)
        self.en_3 = self._make_layer(block1, 2 * init_channels, 4 * init_channels, blocks=blocks[2], use_att=True)
        # self.en_4 = DownDoubleConv(4*init_channels, 8*init_channels, with_bn)
        self.en_4 = self._make_layer(block1, 4 * init_channels, 8 * init_channels, blocks=blocks[3], use_att=True)
        self.cross_attention = cross_attention(init_channels, att=True)
        self.de_1 = UPDoubleConv(8 * init_channels, 4 * init_channels, last=False, att=True)
        self.de_2 = UPDoubleConv(4 * init_channels, 2 * init_channels, last=False, att=True)
        self.de_3 = UPDoubleConv(2 * init_channels, 1 * init_channels, last=False, att=True)
        self.de_4 = UPDoubleConv(1 * init_channels, out_channels, last=True, att=True)

        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
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
            # print(i)
            layers.append(block(planes, planes, first=False, use_att=use_att, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        e1 = self.en_1(x)
        e2 = self.en_2(e1)
        e3 = self.en_3(e2)
        e4 = self.en_4(e3)
        e1, e2, e3, e4 = self.cross_attention(e1, e2, e3, e4)
        d1 = self.de_1(e4, e3)
        d2 = self.de_2(d1, e2)
        d3 = self.de_3(d2, e1)
        d4 = self.de_4(d3, x)
        return [self.up1(d1), self.up2(d2), self.up3(d3), d4]

def CNNlike36(in_channels=1, out_channels=9):
    blocks = [3, 4, 6, 3]
    return UNet(in_channels=in_channels, out_channels=out_channels, with_bn=True, blocks=blocks)


def CNNlike50(in_channels=1, out_channels=9):
    blocks = [2, 3, 12, 3]
    return UNet(in_channels=in_channels, out_channels=out_channels, with_bn=True, blocks=blocks)
