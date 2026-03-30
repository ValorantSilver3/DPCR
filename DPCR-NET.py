import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        x_flat = x_flat + self.attn(self.norm1(x_flat))
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape


        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = self.proj(x)
        return x


class WaveletTransformModule(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.dwt_h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5),
                               stride=1, padding=(0, 2), groups=in_channels, bias=False)

        self.dwt_v = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1),
                               stride=1, padding=(2, 0), groups=in_channels, bias=False)


        haar_h = torch.tensor([1 / 4, 1 / 2, 1 / 4, -1 / 4, -1 / 2, -1 / 4]).view(1, 1, 1, 6)
        haar_v = haar_h.permute(0, 1, 3, 2)


        haar_h = haar_h.repeat(in_channels, 1, 1, 1)
        haar_v = haar_v.repeat(in_channels, 1, 1, 1)

        with torch.no_grad():
            self.dwt_h.weight = nn.Parameter(haar_h[:, :, :, :5])
            self.dwt_v.weight = nn.Parameter(haar_v[:, :, :5, :])


        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):

        x_ll = x
        x_lh = self.dwt_h(x)
        x_hl = self.dwt_v(x)
        x_hh = self.dwt_h(self.dwt_v(x))

        x_wavelet = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return self.fusion(x_wavelet)


class AdaptiveDeQuantization(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=8, stride=8, padding=0, groups=channels // 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.restorer = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape


        block_pattern = F.unfold(x, kernel_size=8, stride=8)
        block_pattern = block_pattern.view(B, C, 8, 8, -1).permute(0, 1, 4, 2, 3)
        block_pattern = block_pattern.reshape(B, C, -1, 8, 8)

        block_features = self.dct_analyzer(x)

        enhanced = self.restorer(x * block_features)


        return x + enhanced


class NonLocalContrastEnhancement(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))


        self.contrast_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        mean_local = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)

        contrast_features = torch.cat([x, mean_local], dim=1)

        contrast_weight = self.contrast_mlp(contrast_features)

        enhanced = x * contrast_weight + self.gamma * out
        return enhanced


class AdaptiveNormFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.batch_norm = nn.BatchNorm2d(channels, affine=False)
        self.layer_norm = nn.GroupNorm(1, channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, 3, kernel_size=1),
            nn.Sigmoid()
        )


        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):

        in_out = self.instance_norm(x)
        bn_out = self.batch_norm(x)
        ln_out = self.layer_norm(x)

        stacked_features = torch.cat([in_out, bn_out, ln_out], dim=1)

        weights = self.fusion(stacked_features)

        w_in = weights[:, 0:1]
        w_bn = weights[:, 1:2]
        w_ln = weights[:, 2:3]

        norm_out = w_in * in_out + w_bn * bn_out + w_ln * ln_out

        return norm_out * self.gamma + self.beta


class PhysicalModelConstraint(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.A_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 4, 3, 1),
            nn.Sigmoid()
        )

        self.t_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        )


        self.enhancer = nn.Sequential(
            nn.Conv2d(channels + 4, channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x, hazy_img=None):

        A = self.A_estimator(x)
        t = self.t_estimator(x)

        t = torch.clamp(t, min=0.1)

        if hazy_img is not None:
            B, C, H, W = hazy_img.shape

            A_expanded = A.expand(B, 3, H, W)
            t_expanded = t.expand(B, 3, H, W)

            recovered = (hazy_img - A_expanded * (1 - t_expanded)) / t_expanded
            recovered = torch.clamp(recovered, 0, 1)

            enhanced_features = torch.cat([x, A.expand_as(t), t, recovered], dim=1)
        else:
            enhanced_features = torch.cat([x, A.expand_as(t), t], dim=1)

        return self.enhancer(enhanced_features)


class MultiScaleRecursiveResBlock(nn.Module):

    def __init__(self, channels, recursions=3):
        super().__init__()
        self.recursions = recursions

        self.main_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

        self.down_branch = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

        self.recursive_weight = nn.Parameter(torch.ones(recursions) / recursions)

    def forward(self, x):
        outputs = []

        current = x

        for i in range(self.recursions):
            main_out = self.main_branch(current)

            down_out = self.down_branch(current)
            up_out = self.up(down_out)

            current = main_out + up_out + current
            outputs.append(current)

        weights = F.softmax(self.recursive_weight, dim=0)
        final_output = sum(w * out for w, out in zip(weights, outputs))

        return final_output


class FogAwareAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.dark_attention = nn.Sequential(
            nn.Conv2d(1, channels // 8, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        self.bright_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        self.grad_attention = nn.Sequential(
            nn.Conv2d(2, channels // 8, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.Sigmoid()
        )

        self.register_buffer('sobel_x',
                             torch.tensor([[[-1.0, 0.0, 1.0],
                                            [-2.0, 0.0, 2.0],
                                            [-1.0, 0.0, 1.0]]]).view(1, 1, 3, 3))

        self.register_buffer('sobel_y',
                             torch.tensor([[[-1.0, -2.0, -1.0],
                                            [0.0, 0.0, 0.0],
                                            [1.0, 2.0, 1.0]]]).view(1, 1, 3, 3))

    def forward(self, x, hazy_img=None):
        if hazy_img is None:
            hazy_img = x

        if hazy_img.shape[2:] != x.shape[2:]:
            hazy_img = F.interpolate(hazy_img, size=x.shape[2:],
                                     mode='bilinear', align_corners=False)

        dark_channel = -F.max_pool2d(-hazy_img.min(dim=1, keepdim=True)[0],
                                     kernel_size=7, stride=1, padding=3)
        dark_attn = self.dark_attention(dark_channel)

        bright_attn = self.bright_attention(x)

        gray = hazy_img.mean(dim=1, keepdim=True)

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        grad_attn = self.grad_attention(torch.cat([grad_x, grad_y], dim=1))

        combined_attn = self.fusion(torch.cat([dark_attn, bright_attn, grad_attn], dim=1))

        return x * combined_attn


class TexturePreservationModule(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.texture_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=k, padding=k // 2),
                nn.LeakyReLU(0.2)
            ) for k in [3, 5, 7]
        ])

        self.texture_attention = nn.Sequential(
            nn.Conv2d(channels // 4 * 3, channels // 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

        self.high_pass = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):

        texture_feats = [extractor(x) for extractor in self.texture_extractor]
        texture_combined = torch.cat(texture_feats, dim=1)

        texture_attn = self.texture_attention(texture_combined)

        high_freq = x - F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        enhanced_high_freq = self.high_pass(high_freq)

        return x + texture_attn * enhanced_high_freq

class ResnetBlock(nn.Module):

    def __init__(self, dim, first=False, levels=1, down=True, bn=False):
        super(ResnetBlock, self).__init__()
        self.first = first
        self.levels = levels
        self.down = down
        self.bn = bn
        self.dim = dim

        if first:

            self.conv1 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, dim, kernel_size=7, padding=0),
                nn.LeakyReLU(0.2, True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.InstanceNorm2d(dim) if not bn else nn.BatchNorm2d(dim),
                nn.LeakyReLU(0.2, True)
            )
            return


        blocks = []
        in_dim = dim
        out_dim = dim

        for i in range(levels):
            if i == 0 and down:

                out_dim = dim * 2
                blocks.append(nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_dim) if not bn else nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(0.2, True)
                ))
            else:

                blocks.append(nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_dim) if not bn else nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(0.2, True)
                ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.first:
            x = self.conv1(x)
            x = self.conv2(x)
            return x
        else:
            return self.blocks(x)


class PALayer(nn.Module):

    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class Attention(nn.Module):

    def __init__(self, channel):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DenseConnection(nn.Module):

    def __init__(self, channels):
        super(DenseConnection, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        return x2


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        size = x.size()

        x1 = self.lrelu(self.norm(self.conv1(x)))
        x2 = self.lrelu(self.norm(self.conv2(x)))
        x3 = self.lrelu(self.norm(self.conv3(x)))
        x4 = self.lrelu(self.norm(self.conv4(x)))
        x5 = self.lrelu(self.norm(self.conv5(self.pool(x))))

        x5 = F.interpolate(x5, size[2:], mode='bilinear', align_corners=True)

        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.lrelu(self.norm(self.conv_out(out)))
        return out


class FrequencyAttention(nn.Module):

    def __init__(self, channels):
        super(FrequencyAttention, self).__init__()

        self.conv_mag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv_phase = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):

        fft = torch.fft.rfft2(x)
        mag = torch.abs(fft)
        phase = torch.angle(fft)


        mag_att = self.conv_mag(mag)
        phase_att = self.conv_phase(phase)

        real = mag_att * mag * torch.cos(phase_att * phase)
        imag = mag_att * mag * torch.sin(phase_att * phase)
        fft_complex = torch.complex(real, imag)

        out = torch.fft.irfft2(fft_complex, s=x.shape[-2:])
        return out + x

class ResnetBlockV2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResnetBlockV2, self).__init__()
        self.same_channels = in_channels == out_channels
        padding = kernel_size // 2

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels)
        )

        if not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.blocks(x)

        if not self.same_channels:
            residual = self.shortcut(residual)

        return out + residual

class UpSamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DualAttention(nn.Module):

    def __init__(self, channels, mode="both"):
        super(DualAttention, self).__init__()
        self.channels = channels
        self.mode = mode

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        x = x + skip

        if self.mode in ["both", "channel"]:
            ca = self.channel_attention(x)
            x = x * ca

        if self.mode in ["both", "spatial"]:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            sa_in = torch.cat([avg_out, max_out], dim=1)
            sa = self.spatial_attention(sa_in)
            x = x * sa

        return x


class DarkChannelExtractor(nn.Module):

    def __init__(self, kernel_size=15):
        super(DarkChannelExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=self.pad)

    def forward(self, x):
        dark_channel = torch.min(x, dim=1, keepdim=True)[0]
        dark_channel = -self.pool(-dark_channel)
        return dark_channel

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down1 = ResnetBlock(64, first=True)
        self.down2 = ResnetBlock(64, levels=2, down=True)
        self.down3 = ResnetBlock(128, levels=2, down=True)

        self.res = nn.Sequential(
            ResnetBlockV2(256, 256),
            ResnetBlockV2(256, 256),
            ResnetBlockV2(256, 256),
            ResnetBlockV2(256, 256),
            ResnetBlockV2(256, 256),
            ResnetBlockV2(256, 256),
            ResnetBlockV2(256, 256),
            ResnetBlockV2(256, 256),
            ResnetBlockV2(256, 256)
        )

        self.up1 = UpSamplingBlock(256, 128)
        self.up2 = UpSamplingBlock(128, 64)

        self.outconv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )

        self.dark_extractor = DarkChannelExtractor()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y=None):
        x_down1 = self.down1(x)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x_res = self.res(x_down3)

        x_up1 = self.up1(x_res) + x_down2
        x_up2 = self.up2(x_up1) + x_down1

        out = self.outconv(x_up2)

        if self.training and y is not None:
            x_dark = self.dark_extractor(x)
            y_dark = self.dark_extractor(y)
            dark_loss = self.l1_loss(x_dark, y_dark)

            return out, {'dark_loss': dark_loss}

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.add_module(
                f'layer{n+1}',
                nn.Sequential(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.add_module(
            f'layer{n_layers+1}',
            nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        self.add_module(
            f'layer{n_layers+2}',
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input):
        x = input
        for name, layer in self.named_children():
            x = layer(x)
        return x
