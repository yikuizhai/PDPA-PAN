import jittor.nn as nn
import jittor as jt
from jdet.models.utils.weight_init import normal_init, bias_init_with_prob

class cSKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(cSKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        # 由于在检测头中使用，因此不使用bn归一化

        if in_channels != out_channels:
            self.exo = True
            self.exo_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(),
            )
            in_channels = out_channels
        else:
            self.exo = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )

        self.mean = nn.AdaptiveAvgPool2d(output_size=1)

        # 由于在检测头中使用，因此不使用bn归一化
        self.fc1 = nn.Conv2d(out_channels, d, 1, bias=False)

        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.init_weights()

    def init_weights(self):
        if self.exo:
            normal_init(self.exo_conv[0], std=0.01)
        normal_init(self.conv1[0], std=0.01)
        normal_init(self.conv2[0], std=0.01)
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2, std=0.01)
        normal_init(self.conv_out[0], std=0.01)

    def execute(self, input):
        if self.exo:
            input = self.exo_conv(input)
        batch_size = input.size(0)
        f1 = self.conv1(input)
        f2 = self.conv2(input)

        U = jt.add(f1, f2)
        s = self.mean(U)

        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)

        atn1, atn2 = jt.chunk(a_b, self.M, 1)
        atn1 = jt.squeeze(atn1, dim=1).unsqueeze(dim=-1)
        atn2 = jt.squeeze(atn2, dim=1).unsqueeze(dim=-1)
        atn1 = atn1.expand_as(f1)
        atn2 = atn2.expand_as(f2)
        out1 = jt.multiply(f1, atn1)
        out2 = jt.multiply(f2, atn2)
        out = jt.add(out1, out2)
        out = jt.add(input, out)
        res = self.norm1(out)

        out = self.conv_out(res)
        res = jt.add(res, out)
        res = self.norm2(res)

        return res

class cSKConv1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(cSKConv1, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        # 由于在检测头中使用，因此不使用bn归一化

        if in_channels != out_channels:
            self.exo = True
            self.exo_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(),
            )
            in_channels = out_channels
        else:
            self.exo = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )

        self.mean = nn.AdaptiveAvgPool2d(output_size=1)

        # 由于在检测头中使用，因此不使用bn归一化
        self.fc1 = nn.Conv2d(out_channels, d, 1, bias=False)

        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )
        self.init_weights()

    def init_weights(self):
        if self.exo:
            normal_init(self.exo_conv[0], std=0.01)
        normal_init(self.conv1[0], std=0.01)
        normal_init(self.conv2[0], std=0.01)
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2, std=0.01)
        normal_init(self.conv_out[0], std=0.01)

    def execute(self, input):
        if self.exo:
            input = self.exo_conv(input)
        batch_size = input.size(0)
        f1 = self.conv1(input)
        f2 = self.conv2(input)

        U = jt.add(f1, f2)
        s = self.mean(U)

        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)

        atn1, atn2 = jt.chunk(a_b, self.M, 1)
        atn1 = jt.squeeze(atn1, dim=1).unsqueeze(dim=-1)
        atn2 = jt.squeeze(atn2, dim=1).unsqueeze(dim=-1)
        atn1 = atn1.expand_as(f1)
        atn2 = atn2.expand_as(f2)
        out1 = jt.multiply(f1, atn1)
        out2 = jt.multiply(f2, atn2)
        out = jt.add(out1, out2)
        out = jt.add(input, out)
        res = self.norm1(out)
        res = self.conv_out(res)

        return res

class sSKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(sSKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        # 由于在检测头中使用，因此不使用bn归一化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )

        # 由于在检测头中使用，因此不使用bn归一化
        self.fc1_1 = nn.Sequential(
            nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU()
        )

        self.fc1_2 = nn.Sequential(
            nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU()
        )

        self.fc2 = nn.Conv2d(1, 2, 3, 1, padding=2, dilation=2, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        normal_init(self.conv1[0], std=0.01)
        normal_init(self.conv2[0], std=0.01)
        normal_init(self.fc1_1[0], std=0.01)
        normal_init(self.fc1_2[0], std=0.01)
        normal_init(self.fc2, std=0.01)

    def execute(self, input):
        f1 = self.conv1(input)
        f2 = self.conv2(input)
        f_1 = self.fc1_1(f1)
        f_2 = self.fc1_2(f2)

        U = jt.add(f_1, f_2)
        z = U.mean(dim=1, keepdims=True)

        a_b = self.fc2(z)
        a_b = self.softmax(a_b)

        atn1, atn2 = jt.chunk(a_b, self.M, 1)
        atn1 = atn1.expand_as(f1)
        atn2 = atn2.expand_as(f2)
        out1 = jt.multiply(f1, atn1)
        out2 = jt.multiply(f2, atn2)
        out = jt.add(out1, out2)

        return out

class scSKConv_v1(nn.Module):   # 全并接
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(scSKConv_v1, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        # 由于在检测头中使用，因此不使用bn归一化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, dilation=1, groups=32, bias=False),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, groups=32, bias=False),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, groups=32, bias=False),
            nn.ReLU(),
        )

        # 由于在检测头中使用，因此不使用bn归一化
        self.mean = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.ReLU())
        self.fc2c = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)

        self.fc1_1 = nn.Sequential(nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
                                 nn.ReLU())
        self.fc1_2 = nn.Sequential(nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
                                   nn.ReLU())

        self.fc2s = nn.Conv2d(1, 2, 3, 1, padding=2, dilation=2, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        normal_init(self.conv1, std=0.01)
        normal_init(self.conv2, std=0.01)
        normal_init(self.conv3, std=0.01)
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2c, std=0.01)
        normal_init(self.fc1_1, std=0.01)
        normal_init(self.fc1_2, std=0.01)
        normal_init(self.fc2s, std=0.01)

    def execute(self, input):
        batch_size = input.size(0)
        f1 = self.conv1(input)
        f2 = self.conv2(input)
        f3 = self.conv3(input)

        c_U = jt.add(f1, f2)
        c_s = self.mean(c_U)

        c_z = self.fc1(c_s)
        c_t = self.fc2c(c_z)
        c_t = c_t.reshape(batch_size, self.M, self.out_channels, -1)
        c_t = self.softmax(c_t)

        c_atn1, c_atn2 = jt.chunk(c_t, self.M, 1)
        c_atn1 = jt.squeeze(c_atn1, dim=1).unsqueeze(dim=-1)
        c_atn2 = jt.squeeze(c_atn2, dim=1).unsqueeze(dim=-1)
        c_atn1 = c_atn1.expand_as(f1)
        c_atn2 = c_atn2.expand_as(f2)
        c_out1 = jt.multiply(f1, c_atn1)
        c_out2 = jt.multiply(f2, c_atn2)
        c_out = jt.add(c_out1, c_out2)

        f_1 = self.fc1_1(f1)
        f_3 = self.fc1_2(f3)

        s_U = jt.add(f_1, f_3)
        s_z = s_U.mean(dim=1, keepdims=True)

        s_t = self.fc2s(s_z)
        s_t = self.softmax(s_t)

        s_atn1, s_atn2 = jt.chunk(s_t, self.M, 1)
        out1 = jt.multiply(f1, s_atn1)
        out2 = jt.multiply(f2, s_atn2)
        s_out = jt.add(out1, out2)

        out = jt.add(c_out, s_out)

        return out


class scSKConv_v2(nn.Module): # 全串接
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(scSKConv_v2, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        # 由于在检测头中使用，因此不使用bn归一化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, dilation=1, groups=32, bias=False),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, groups=32, bias=False),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, groups=32, bias=False),
            nn.ReLU(),
        )

        # 由于在检测头中使用，因此不使用bn归一化
        self.mean = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.ReLU())
        self.fc2c = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)

        self.fc1_1 = nn.Sequential(nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
                                   nn.ReLU())
        self.fc1_2 = nn.Sequential(nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
                                   nn.ReLU())

        self.fc2s = nn.Conv2d(1, 2, 3, 1, padding=2, dilation=2, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        normal_init(self.conv1, std=0.01)
        normal_init(self.conv2, std=0.01)
        normal_init(self.conv3, std=0.01)
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2c, std=0.01)
        normal_init(self.fc1_1, std=0.01)
        normal_init(self.fc1_2, std=0.01)
        normal_init(self.fc2s, std=0.01)

    def execute(self, input):
        batch_size = input.size(0)
        f1 = self.conv1(input)
        f2 = self.conv2(f1)
        f3 = self.conv3(f1)

        c_U = jt.add(f1, f2)
        c_s = self.mean(c_U)

        c_z = self.fc1(c_s)
        c_t = self.fc2c(c_z)
        c_t = c_t.reshape(batch_size, self.M, self.out_channels, -1)
        c_t = self.softmax(c_t)

        c_atn1, c_atn2 = jt.chunk(c_t, self.M, 1)
        c_atn1 = jt.squeeze(c_atn1, dim=1).unsqueeze(dim=-1)
        c_atn2 = jt.squeeze(c_atn2, dim=1).unsqueeze(dim=-1)
        c_atn1 = c_atn1.expand_as(f1)
        c_atn2 = c_atn2.expand_as(f2)
        c_out1 = jt.multiply(f1, c_atn1)
        c_out2 = jt.multiply(f2, c_atn2)
        c_out = jt.add(c_out1, c_out2)

        f_1 = self.fc1_1(f1)
        f_3 = self.fc1_2(f3)

        s_U = jt.add(f_1, f_3)
        s_z = s_U.mean(dim=1, keepdims=True)

        s_t = self.fc2s(s_z)
        s_t = self.softmax(s_t)

        s_atn1, s_atn2 = jt.chunk(s_t, self.M, 1)
        out1 = jt.multiply(f1, s_atn1)
        out2 = jt.multiply(f2, s_atn2)
        s_out = jt.add(out1, out2)

        out = jt.add(c_out, s_out)

        return out

class scSKConv_v3(nn.Module): # s并接 c串接
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(scSKConv_v3, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        # 由于在检测头中使用，因此不使用bn归一化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, dilation=1, groups=32, bias=False),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, groups=32, bias=False),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, groups=32, bias=False),
            nn.ReLU(),
        )

        # 由于在检测头中使用，因此不使用bn归一化
        self.mean = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.ReLU())
        self.fc2c = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)

        self.fc1_1 = nn.Sequential(nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
                                   nn.ReLU())
        self.fc1_2 = nn.Sequential(nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
                                   nn.ReLU())

        self.fc2s = nn.Conv2d(1, 2, 3, 1, padding=2, dilation=2, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        normal_init(self.conv1, std=0.01)
        normal_init(self.conv2, std=0.01)
        normal_init(self.conv3, std=0.01)
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2c, std=0.01)
        normal_init(self.fc1_1, std=0.01)
        normal_init(self.fc1_2, std=0.01)
        normal_init(self.fc2s, std=0.01)

    def execute(self, input):
        batch_size = input.size(0)
        f1 = self.conv1(input)
        f2 = self.conv2(f1)
        f3 = self.conv3(input)

        c_U = jt.add(f1, f2)
        c_s = self.mean(c_U)

        c_z = self.fc1(c_s)
        c_t = self.fc2c(c_z)
        c_t = c_t.reshape(batch_size, self.M, self.out_channels, -1)
        c_t = self.softmax(c_t)

        c_atn1, c_atn2 = jt.chunk(c_t, self.M, 1)
        c_atn1 = jt.squeeze(c_atn1, dim=1).unsqueeze(dim=-1)
        c_atn2 = jt.squeeze(c_atn2, dim=1).unsqueeze(dim=-1)
        c_atn1 = c_atn1.expand_as(f1)
        c_atn2 = c_atn2.expand_as(f2)
        c_out1 = jt.multiply(f1, c_atn1)
        c_out2 = jt.multiply(f2, c_atn2)
        c_out = jt.add(c_out1, c_out2)

        f_1 = self.fc1_1(f1)
        f_3 = self.fc1_2(f3)

        s_U = jt.add(f_1, f_3)
        s_z = s_U.mean(dim=1, keepdims=True)

        s_t = self.fc2s(s_z)
        s_t = self.softmax(s_t)

        s_atn1, s_atn2 = jt.chunk(s_t, self.M, 1)
        out1 = jt.multiply(f1, s_atn1)
        out2 = jt.multiply(f2, s_atn2)
        s_out = jt.add(out1, out2)

        out = jt.add(c_out, s_out)

        return out

class scSKConv_v4(nn.Module): # c并接 s串接
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(scSKConv_v4, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        # 由于在检测头中使用，因此不使用bn归一化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, dilation=1, groups=32, bias=False),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, groups=32, bias=False),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=2, dilation=2, groups=32, bias=False),
            nn.ReLU(),
        )

        # 由于在检测头中使用，因此不使用bn归一化
        self.mean = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.ReLU())
        self.fc2c = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)

        self.fc1_1 = nn.Sequential(nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
                                   nn.ReLU())
        self.fc1_2 = nn.Sequential(nn.Conv2d(out_channels, d, 3, stride, padding=2, dilation=2, bias=False),
                                   nn.ReLU())

        self.fc2s = nn.Conv2d(1, 2, 3, 1, padding=2, dilation=2, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        normal_init(self.conv1[0], std=0.01)
        normal_init(self.conv2[0], std=0.01)
        normal_init(self.conv3, std=0.01)
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2c, std=0.01)
        normal_init(self.fc1_1, std=0.01)
        normal_init(self.fc1_2, std=0.01)
        normal_init(self.fc2s, std=0.01)

    def execute(self, input):
        batch_size = input.size(0)
        f1 = self.conv1(input)
        f2 = self.conv2(input)
        f3 = self.conv3(f1)

        c_U = jt.add(f1, f2)
        c_s = self.mean(c_U)

        c_z = self.fc1(c_s)
        c_t = self.fc2c(c_z)
        c_t = c_t.reshape(batch_size, self.M, self.out_channels, -1)
        c_t = self.softmax(c_t)

        c_atn1, c_atn2 = jt.chunk(c_t, self.M, 1)
        c_atn1 = jt.squeeze(c_atn1, dim=1).unsqueeze(dim=-1)
        c_atn2 = jt.squeeze(c_atn2, dim=1).unsqueeze(dim=-1)
        c_atn1 = c_atn1.expand_as(f1)
        c_atn2 = c_atn2.expand_as(f2)
        c_out1 = jt.multiply(f1, c_atn1)
        c_out2 = jt.multiply(f2, c_atn2)
        c_out = jt.add(c_out1, c_out2)

        f_1 = self.fc1_1(f1)
        f_3 = self.fc1_2(f3)

        s_U = jt.add(f_1, f_3)
        s_z = s_U.mean(dim=1, keepdims=True)

        s_t = self.fc2s(s_z)
        s_t = self.softmax(s_t)

        s_atn1, s_atn2 = jt.chunk(s_t, self.M, 1)
        out1 = jt.multiply(f1, s_atn1)
        out2 = jt.multiply(f2, s_atn2)
        s_out = jt.add(out1, out2)

        out = jt.add(c_out, s_out)

        return out

if __name__ == "__main__":
    t = jt.ones((32, 256, 24, 24))
    sk = sSKConv(256, 256)
    out = sk(t)
    print(out.shape)
