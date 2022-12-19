import torch
import torch.nn as nn



# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            # 对应参数d取3，所以进行三次卷积操作？
            nn.Conv3d(C_in, C_out, kernel_size=(3, 3, 3), padding=(3//2)),
            # batchnorm和Dropout是否保留？防止过拟合
            nn.BatchNorm3d(C_out),
            nn.Dropout(0.3),
            nn.PReLU(),

            nn.Conv3d(C_out, C_out, kernel_size=(3, 3, 3), padding=(3//2)),
            # batchnorm和Dropout是否保留？防止过拟合
            nn.BatchNorm3d(C_out),
            nn.Dropout(0.4),
            nn.PReLU(),

            nn.Conv3d(C_out, C_out, kernel_size=(3, 3, 3), padding=(3//2)),
            # batchnorm和Dropout是否保留？防止过拟合
            nn.BatchNorm3d(C_out),
            nn.Dropout(0.3),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用最大池化进行下采样，且只对横向维度进行，轴向不变.通道数不变
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            # Conv(C_in, C_out)
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C_in, C_out):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        # 采用反卷积进行上采样
        self.Up = nn.ConvTranspose3d(C_in, C_in//2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3//2),
                                     output_padding=(1, 1, 1))
        self.conv = Conv(C_in, C_out)
    # 需要进行拼接操作，使用torch.cat
    # inputs1:上采样数据；inputs2：前面的桥接传过来的数据
    def forward(self, inputs1, inputs2):
        # 先进行一次反卷积上采样操作
        inputs1 = self.Up(inputs1)
        # 拼接，当前上采样的，和之前下采样过程中的
        # 进行特征融合
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, C_in, C_out):
        super(LastConv, self).__init__()

        self.conv = nn.Conv3d(C_in, C_out, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# 主干网络
class UNet3DSR(nn.Module):

    def __init__(self):
        super(UNet3DSR, self).__init__()
        # 进行四倍降采样
        self.down = nn.AvgPool3d(kernel_size=(4, 1, 1))

        # 2次下采样
        self.C1 = Conv(1, 32)
        # 每个卷积之后跟了一个反卷积模块，在这里定义这，用于得到inputs2，但是在forward连接前馈时要注意不要弄错了地方
        self.skip1 = nn.ConvTranspose3d(32, 32, kernel_size=(3, 3, 3), stride=(4, 1, 1), padding=3//2,
                                        output_padding=(3, 0, 0))
        self.D1 = DownSampling(32, 32)
        self.C2 = Conv(32, 64)
        self.skip2 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=3 // 2,
                                        output_padding=(1, 0, 0))
        self.D2 = DownSampling(64, 64)
        self.C3 = Conv(64, 128)

        # 2次上采样，上采样之后跟的卷积操作写在了upsampling块中了
        self.U1 = UpSampling(128, 64)
        self.U2 = UpSampling(64, 32)

        self.Last = LastConv(32, 1)

    def forward(self, x):
        # 下采样部分
        DOWN = self.down(x)
        R1 = self.C1(DOWN)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))

        # 卷积之后跟的反卷积的前馈计算
        Y1 = self.skip1(R1)
        Y2 = self.skip2(R2)

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.U1(R3, Y2)
        O2 = self.U2(O1, Y1)

        return self.Last(O2)




