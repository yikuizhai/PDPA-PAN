import jittor
import jittor as jt
from jittor import nn
import warnings
import random
from jdet.utils.registry import NECKS
from jdet.models.utils.modules import ConvModule
from jdet.models.utils.weight_init import xavier_init


@NECKS.register_module()
class PDPAPAN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs="on_input",
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 upsample_div_factor=1,
                 beta=2,
                 attention=True):
        super(PDPAPAN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_div_factor = upsample_div_factor
        self.beta = beta
        self.attention = attention

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_output', 'on_input')

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        if self.attention:
            self.attens = nn.ModuleList()
            self.atten_fs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            atten_channel = in_channels[i]
            if self.attention:
                atten = PA(atten_channel)
                atten_f = PA(out_channels)
                self.atten_fs.append(atten_f)
                self.attens.append(atten)

            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                3,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg)

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if add_extra_convs == "on_output":
                    in_channels = out_channels
                if add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    5,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.fpn_convs.append(extra_fpn_conv)

        self.down_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level - 1):
            down_conv = ConvModule(
                out_channels,
                out_channels,
                5,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.down_convs.append(down_conv)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        if self.attention:
            laterals = [
                lateral_conv(atten(inputs[i + self.start_level]))
                for i, lateral_conv, atten in zip(range(len(self.lateral_convs)), self.lateral_convs, self.attens)
            ]
        else:
            laterals = [
                lateral_conv(inputs[i + self.start_level])
                for i, lateral_conv in zip(range(len(self.lateral_convs)), self.lateral_convs)
            ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 2, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += nn.interpolate(laterals[i], size=prev_shape, mode='bilinear')

        # build down-top path second
        for i, down_conv in zip(range(1, used_backbone_levels), self.down_convs):
            laterals[i] = down_conv(laterals[i - 1]) + (self.beta * laterals[i])

        # build outputs
        # part 1: from original levels
        if self.attention:
            outs = [
                atten_f(self.fpn_convs[i](laterals[i]))
                for i, atten_f in zip(range(used_backbone_levels), self.atten_fs)
            ]
        else:
            outs = [
                self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels)
            ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(nn.pool(outs[-1], 1, stride=2, op="maximum"))
            else:
                if self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                elif self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                else:
                    raise NotImplementedError

                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](nn.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class SAM_mix_mix7(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(SAM_mix_mix7, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid(),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        avg_out = x.mean(dim=1, keepdims=True)
        max_out = x.max(dim=1, keepdims=True)
        merge = avg_out + max_out
        return jt.multiply(x, self.mlp(jt.concat([merge, avg_out, max_out], dim=1)))


class SPCAM_mix2_add(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SPCAM_mix2_add, self).__init__()
        self.p1 = nn.AdaptiveMaxPool2d(1)
        self.p2 = nn.AdaptiveMaxPool2d(2)
        self.p3 = nn.AdaptiveAvgPool2d(1)
        self.p4 = nn.AdaptiveAvgPool2d(2)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel * 10, 10 * channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(10 * channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        b, c, h, w = x.shape
        p1 = self.p1(x)
        p4 = self.p4(x)
        feats = jt.concat([p1, p4], dim=1)
        return jt.multiply(x, self.mlp(feats))


class PA(nn.Module):
    def __init__(self, channel):
        super(PA, self).__init__()
        self.c_attention = SPCAM_mix2_add(channel)
        self.s_attention = SAM_mix_mix7(channel)

    def execute(self, x):
        return jt.concat(self.c_attention(x), self.s_attention(x))
