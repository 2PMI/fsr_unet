import skimage.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model1 = nn.Sequential(

            # 先通过平均池化进行降采样，scale_factor为4
            nn.AvgPool3d(kernel_size=(4, 1, 1)),


            nn.Conv3d(1, 240, kernel_size=(5, 13, 13), padding=(5//2, 13//2, 13//2)),
            nn.PReLU(),

            nn.Conv3d(240, 48, kernel_size=(1, 1, 1)),
            nn.PReLU(),

            nn.Conv3d(48, 48, kernel_size=(3, 9, 9), padding=(3//2, 9//2, 9//2)),
            nn.PReLU(),

            nn.Conv3d(48, 48, kernel_size=(3, 9, 9), padding=(3//2, 9//2, 9//2)),
            nn.PReLU(),

            nn.Conv3d(48, 240, kernel_size=(1, 1, 1)),
            nn.PReLU(),
            # 进行上采样过程，只对z轴进行上采样
            nn.ConvTranspose3d(240, 1, kernel_size=(13, 13, 13), stride=(4, 1, 1), padding=(13//2), output_padding=(3, 0, 0))

        )



    def forward(self, x):
        x = self.model1(x)
        return x



