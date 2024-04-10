# Code taken from HW2 handout of 10-423/623 Spring 2024, CMU
# Implements a UNet

import math
from functools import partial

from einops import rearrange, reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F


# helpers functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def l2norm(t):
    return F.normalize(t, dim=-1)


# helper modules
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q, k = map(l2norm, (q, k))

        sim = einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

# Loss
def weighted_l1_loss(pred, labels):
    # ratio of images with non-zero pixels
    val_masks = {
        'endrow': [1755, 18334], 
        'water': [987, 18334], 
        'waterway': [696, 18334], 
        'double_plant': [2322, 18334], 
        'drydown': [5800, 18334], 
        'storm_damage': [89, 18334], 
        'nutrient_deficiency': [3883, 18334], 
        'planter_skip': [1197, 18334], 
        'weed_cluster': [2834, 18334]
    }
    train_masks = {
        'endrow': [4481, 56944], 
        'water': [2155, 56944], 
        'waterway': [3899, 56944], 
        'double_plant': [6234, 56944], 
        'drydown': [16806, 56944], 
        'storm_damage': [356, 56944], 
        'nutrient_deficiency': [13308, 56944], 
        'planter_skip': [2599, 56944], 
        'weed_cluster': [11111, 56944]
    }
    train_ratios = {
        'endrow': 0.07869134588367518, 
        'water': 0.037844197808373135, 
        'waterway': 0.06847077830851363, 
        'double_plant': 0.10947597639786456, 
        'drydown': 0.29513205956729416, 
        'storm_damage': 0.00625175611126721, 
        'nutrient_deficiency': 0.23370328744029223, 
        'planter_skip': 0.045641331834785054, 
        'weed_cluster': 0.19512152289969092
    }
    # ratio of black to white pixels in the labels
    val_masks = {
        'drydown': [674846434, 4806148096], 
        'weed_cluster': [201170131, 4806148096], 
        'waterway': [20129377, 4806148096], 
        'endrow': [60007223, 4806148096], 
        'planter_skip': [10885482, 4806148096], 
        'nutrient_deficiency': [343483024, 4806148096], 
        'double_plant': [44044947, 4806148096], 
        'storm_damage': [4224793, 4806148096], 
        'water': [74630112, 4806148096]
    }
    train_masks = {
        'drydown': [1838660590, 14927527936], 
        'weed_cluster': [921419728, 14927527936], 
        'waterway': [141159490, 14927527936], 
        'endrow': [135527899, 14927527936], 
        'planter_skip': [28304311, 14927527936], 
        'nutrient_deficiency': [1108061290, 14927527936], 
        'double_plant': [126330289, 14927527936], 
        'storm_damage': [27771992, 14927527936], 
        'water': [149637355, 14927527936]
    }
    train_ratios = {
        'drydown': 0.12317247690863742, 
        'weed_cluster': 0.061726210257349874, 
        'waterway': 0.009456320604805063, 
        'endrow': 0.009079058473784792, 
        'planter_skip': 0.001896115091618074, 
        'nutrient_deficiency': 0.07422938980591301, 
        'double_plant': 0.008462907558547275, 
        'storm_damage': 0.0018604548669457603, 
        'water': 0.010024255566062403
    }
    # adds up to 1.070332256251756
    # L1 loss with ratios
        # check where the target comes from and see what the ratio is for that 
        # class
    drydown_ratio = train_ratios["drydown"]
    
    mask = torch.max(labels, dim=1, keepdim=True)[0] > 0
    loss_non_zero = torch.mean((1 / drydown_ratio) * torch.abs(pred - labels), dim=1)
    loss_zero = torch.mean((1 / (1 - drydown_ratio)) * torch.abs(pred - labels), dim=1)
    loss = torch.where(mask.squeeze(), loss_non_zero, loss_zero)
    
    return torch.mean(loss)

# model
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(dim_in, dim_in, time_emb_dim=time_dim),
                        block_class(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_class(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        # Sigmoid to push the output to [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x)
        x = self.final_conv(x)

        return self.sigmoid(x)

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from dataset import AgricultureVisionDataset
if __name__ == "__main__":
    transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.functional.invert]
    )
    target_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            lambda x: torch.clamp(x, 0.0, 1.0),
        ]
    )
    dataset = AgricultureVisionDataset(
        "./data/Agriculture-Vision-2021/train/",
        transform=transform,
        target_transform=target_transform,
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    imgs, labels = next(iter(data_loader))

    pred = torch.randint(0, 2, size=(1, 128, 128))
    print("standard l1 loss:", F.l1_loss(pred,labels))
    print("weighted l1 loss:", weighted_l1_loss(pred, labels))