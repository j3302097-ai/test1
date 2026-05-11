import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


class Mlp(nn.Module):
    """
    MLP of the transformer layer
    input : x(B,N,C)
    Args:
        in_features: the dim of the input features
        hidden_features: the hidden dim of the MLP
        out_features: the dim of the output features
        act_layers: the act function of the MLP, default GELU
        drop: dropout ratio
    returns: (B,N,C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvFFN(nn.Module):
    """CNN for the transformer FFN module 

    input: x(B,C,H,W)
    Args:
        in_features: the dim of the input features
        hidden_features: the hidden dim of the MLP
        out_features: the dim of the output features
        act_layers: the act function of the MLP, default GELU
        drop: dropout ratio
    returns: (B,C,H,W)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv = nn.Conv2d(in_features, in_features, 3, 1, 3, groups=in_features)
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()   # B H W C
        X = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    window partion function
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., drop_key=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.drop_key = drop_key

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,
                                                                                         0, 3, 1, 4).contiguous()  # (3,B*num_windows,heads,N,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    # q:(B*windows, heads, N, C), att:(B*windows, heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # positional enbedding

        # Drop key
        if self.drop_key:
            m_r = torch.ones_like(attn)*0.1
            attn = attn+torch.bernoulli(m_r)*-1e12

        if mask is not None:
            nW = mask.shape[0]  # mask(nW,N,N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)   # att(bs,nW,heads,N,N); mask(1,nW,1,N,N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)    # x:(B*windows, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Channel_Adapter(nn.Module):
    """The Channel Adapter module
    args:
        num_channels: the channel number of the feature map
        resolution: the resolution of the feature map
    """

    def __init__(self, num_channels, resolution):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(num_channels, 4*num_channels, 1),
            nn.GELU(),
            nn.Conv2d(4*num_channels, num_channels, 1)
        )
        self.pool = nn.AvgPool2d(resolution)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, 4*num_channels),
            nn.Linear(4*num_channels, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.proj(x)
        avg_x = self.pool(x_proj).reshape(B, C)   # (B,C)
        attn = self.mlp(avg_x)
        attn = attn.reshape(B, C, 1, 1)*x_proj
        x = x+attn
        return x



    def __init__(self, num_channels, resolution):
        super().__init__()
        super().__init__()
        self.proj = nn.Sequential(
        self.proj = nn.Sequential(
            nn.Conv2d(num_channels, 4*num_channels, 1),
            nn.Conv2d(num_channels, 4*num_channels, 1),
            nn.GELU(),
            nn.GELU(),
            nn.Conv2d(4*num_channels, num_channels, 1)
            nn.Conv2d(4*num_channels, num_channels, 1)
        )
        )
        self.pool = nn.AvgPool2d(resolution)
        self.pool = nn.AvgPool2d(resolution)
        self.mlp = nn.Sequential(
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, 4*num_channels),
            nn.Linear(num_channels, 4*num_channels),
            nn.Linear(4*num_channels, num_channels),
            nn.Linear(4*num_channels, num_channels),
            nn.Sigmoid()
            nn.Sigmoid()
        )
        )


    def forward(self, x):
    def forward(self, x):
        B, C, H, W = x.shape
        B, C, H, W = x.shape
        x_proj = self.proj(x)
        x_proj = self.proj(x)
        avg_x = self.pool(x_proj).reshape(B, C)   # (B,C)
        avg_x = self.pool(x_proj).reshape(B, C)   # (B,C)
        attn = self.mlp(avg_x)
        attn = self.mlp(avg_x)
        attn = attn.reshape(B, C, 1, 1)*x_proj
        attn = attn.reshape(B, C, 1, 1)*x_proj
        x = x+attn
        x = x+attn
        return x
        return x




class Mamba3MIMOChannelMixer(nn.Module):
    """Lightweight Mamba3-MIMO style mixer for cover-secret channel matching.

    It treats every channel as a token and scans the cover descriptor, secret
    descriptor and their difference jointly to predict a per-channel hiding
    strength.  The implementation intentionally avoids an external Mamba
    dependency while keeping the MIMO channel sequence interface needed by the
    steganography controller.
    """

    def __init__(self, num_channels, hidden_ratio=4, kernel_size=5):
        super().__init__()
        hidden_dim = max(num_channels // hidden_ratio, 8)
        padding = kernel_size // 2
        self.in_proj = nn.Linear(3, hidden_dim)
        self.scan = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, groups=hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, cover_desc, secret_desc):
        relation = torch.stack([cover_desc, secret_desc, cover_desc-secret_desc], dim=-1)
        x = self.in_proj(relation)
        scanned = self.scan(x.transpose(1, 2)).transpose(1, 2)
        x = scanned*self.gate(relation)
        return torch.sigmoid(self.out_proj(x).squeeze(-1))


class SecretAwareChannelController(nn.Module):
    """Secret-aware Channel Adapter for steganographic channel allocation.

    Compared with the original Channel_Adapter, this controller can explicitly
    compare cover and secret feature statistics and outputs the hiding strength
    of every channel. If no cover/secret condition is supplied it falls back to
    the original self-adaptive channel behavior, keeping backward compatibility.
    """

    def __init__(self, num_channels, resolution):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(num_channels, 4*num_channels, 1),
            nn.GELU(),
            nn.Conv2d(4*num_channels, num_channels, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.self_mlp = nn.Sequential(
            nn.Linear(num_channels, 4*num_channels),
            nn.GELU(),
            nn.Linear(4*num_channels, num_channels),
            nn.Sigmoid()
        )
        self.cover_proj = nn.Linear(num_channels, num_channels)
        self.secret_proj = nn.Linear(num_channels, num_channels)
        self.mimo = Mamba3MIMOChannelMixer(num_channels)

    def forward(self, x, cover_feat=None, secret_feat=None):
        B, C, H, W = x.shape
        x_proj = self.proj(x)
        if cover_feat is None or secret_feat is None:
            avg_x = self.pool(x_proj).reshape(B, C)
            strength = self.self_mlp(avg_x)
        else:
            cover_desc = self.pool(cover_feat).reshape(B, C)
            secret_desc = self.pool(secret_feat).reshape(B, C)
            cover_desc = self.cover_proj(cover_desc)
            secret_desc = self.secret_proj(secret_desc)
            strength = self.mimo(cover_desc, secret_desc)
        return x + strength.reshape(B, C, 1, 1)*x_proj


class VMamba2DScan(nn.Module):
    """VMamba-style two-dimensional scan for exchanging information across windows.

    The block performs horizontal and vertical bidirectional selective scans
    with lightweight cumulative state propagation, then gates the scanned state
    before projecting it back to the feature space.
    """

    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Conv2d(dim, dim*2, 1)
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.act = nn.SiLU()

    def _bidirectional_scan(self, x, dim):
        length = x.shape[dim]
        forward = torch.cumsum(x, dim=dim)
        backward = torch.flip(torch.cumsum(torch.flip(x, dims=(dim,)), dim=dim), dims=(dim,))
        return (forward + backward) / max(length, 1)

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, dim=1)
        value = self.dwconv(self.act(value))
        horizontal = self._bidirectional_scan(value, dim=3)
        vertical = self._bidirectional_scan(value, dim=2)
        scanned = 0.5*(horizontal + vertical)
        return self.out_proj(scanned*torch.sigmoid(gate))


class PEG_Conv(nn.Module):
    """Conditional position encode generator, using depth-wise convolution

    Args:
        input: (B,C,H,W), based on patch
        dim: the dim of the heads' token
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x):
        pe = self.conv(x)
        return pe


class ConditinoalAttention(nn.Module):
    r""" Global attention with conditional positional encoding.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_key (bool, optional): Using dropkey or not. Default: False
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., drop_key=False):

        super().__init__()
        self.dim = dim
        self.head_dim = dim//num_heads
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.drop_key = drop_key

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # 生成 QKV
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力的 dropout
        self.proj = nn.Linear(dim, dim)  # 输出映射矩阵
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的 dropout

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,
                                                                                         0, 3, 1, 4).contiguous()  # (3,B*num_windows,heads,N,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    # q:(B, heads, N, C), att:(B, heads, N, N)

        # Drop key
        if self.drop_key:
            m_r = torch.ones_like(attn)*0.1
            attn = attn+torch.bernoulli(m_r)*-1e12

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)    # x:(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def PatchReverse(x, patch_size):
    """Transform the tokens back to image

    input:
        x: (B, N, C), N = (ph pw): Token representation.
    returns:
        x: (B,C,H,W)

    """
    _, N, _ = x.shape
    p1 = p2 = int(N**0.5)    # 有多少个 patch
    x = rearrange(x, 'b (p1 p2) (ph pw c) -> b c (p1 ph) (p2 pw)', p1=p1, p2=p2, ph=patch_size, pw=patch_size).contiguous()
    return x


def PatchDivide(x, patch_size):
    """Transform the image to tokens.

    input:
        x: (B,C,H,W)
    returns:
        x: (B, N, C)
    """
    x = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', h=patch_size, w=patch_size).contiguous()
    return x


class DownSampler(nn.Module):
    """Down sample the feature map.
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSampler(nn.Module):
    """Up sample the feature map."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion, based on patch.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.LeakyReLU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn_type (string): FFN module,using Convolution or MLP, Default: Mlp
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
        drop_key (bool, optional): Using dropkey or not. Default: False

        输入的 x 为 (B,N,C) 表示，在过程中进行窗口划分
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,   # 你妈的，这里写错了，弄得窗口变得没有滑动了
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.LeakyReLU, norm_layer=nn.LayerNorm, ffn_type='Mlp',   # 消融：修改为 LeakyReLU
                 fused_window_process=False, drop_key=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ffn_type = ffn_type
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, drop_key=drop_key)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.ffn_type == 'Mlp':
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)    # 注册为 buffer
        self.fused_window_process = fused_window_process    # 是否执行 merge 操作

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # turn to 2-D reprensentation

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # (B*windows,Wh,Ww,C)

        # reverse cyclic shift 恢复循环移位操作
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        # FFN
        if self.ffn_type == 'Mlp':
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            shortcut2 = x
            x = rearrange(x, 'bs (h w) c -> bs c h w', bs=B, h=H, w=W).contiguous()  # B C H W
            x = self.mlp(x)
            x = rearrange(x, 'bs c h w -> bs (h w) c').contiguous()  # B N C
            x = self.drop_path(x)
            x = shortcut2+x
        return x


class Global_Enhanced_BottleNeck_Block(nn.Module):
    r""" Global Enhanced BottleNeck Block, using global attention to model global information.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion. 这个 resolution 是 patch 的分辨率
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn_type (string): FFN module,using Convolution or MLP, Default: Mlp
        drop_key (bool, optional): Using dropkey or not. Default: False

        输入的 x 为 (B,N,C) 表示，不再划分窗口
    """

    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ffn_type='Mlp', drop_key=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.ffn_type = ffn_type

        self.norm1 = norm_layer(dim)
        self.attn = ConditinoalAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, drop_key=drop_key)
        self.mamba_scan = VMamba2DScan(dim)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.branch_fusion = nn.Conv2d(dim*3, dim, 1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.ffn_type == 'Mlp':
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x_norm = self.norm1(x)

        # Hybrid Mamba3-GEB: global self-attention + VMamba scan + local convolution.
        attn_x = self.attn(x_norm)  # B, N, C
        img_x = rearrange(x_norm, 'bs (h w) c -> bs c h w', bs=B, h=H, w=W).contiguous()
        mamba_x = self.mamba_scan(img_x)
        conv_x = self.conv_branch(img_x)
        fused = self.branch_fusion(torch.cat([rearrange(attn_x, 'bs (h w) c -> bs c h w', bs=B, h=H, w=W).contiguous(),
                                              mamba_x, conv_x], dim=1))
        x = rearrange(fused, 'bs c h w -> bs (h w) c').contiguous()

        x = shortcut + self.drop_path(x)

        # FFN
        if self.ffn_type == 'Mlp':
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # 使用 ConvNext block 替换
            shortcut2 = x
            x = rearrange(x, 'bs (h w) c -> bs c h w', bs=B, h=H, w=W).contiguous()  # B C H W
            x = self.mlp(x)
            x = rearrange(x, 'bs c h w -> bs (h w) c').contiguous()  # B N C
            x = self.drop_path(x)
            x = shortcut2+x
        return x


class GEB(nn.Module):
    r""" Global Enhanced BottleNeck.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion. 这个 resolution 是 patch 的分辨率
        num_heads (int): Number of attention heads.
        patch_size (int): Patch size o
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn_type (string): FFN module,using Convolution or MLP, Default: Mlp
        drop_key (bool, optional): Using dropkey or not. Default: False

        输入的 x 为 (B,N,C) 表示，不再划分窗口
    """

    def __init__(self, dim, input_resolution,  num_heads, patch_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, drop_key=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.blocks = nn.ModuleList([
            Global_Enhanced_BottleNeck_Block(dim=dim, input_resolution=to_2tuple(input_resolution),
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop, attn_drop=attn_drop,
                                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                             norm_layer=norm_layer, drop_key=drop_key)
            for i in range(depth)])
        self.pe_generator = PEG_Conv(dim)

    def forward(self, x):
        x = PatchDivide(x, self.patch_size)
        B_, N, C = x.shape
        x = rearrange(x, 'b (h w) c -> b c h w ', h=int(N**0.5), w=int(N**0.5)).contiguous()  # 转换为图像表示
        pe = self.pe_generator(x)   # 条件位置编码
        x = x+pe  # 消融: PEG
        x = rearrange(x, ' b c h w -> b (h w) c').contiguous()  # 转换成序列表示
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = PatchReverse(x, self.patch_size)
        return x


class Swin_Transformer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    input:
        x: (B, N, C)

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
        drop_key (bool, optional): Using dropkey or not. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 fused_window_process=False, drop_key=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        # 构建一个 Swin Transformer Block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    shift_size=0 if (i % 2 == 0) else window_size // 2,   # 消融：滑动窗口
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,
                                    fused_window_process=fused_window_process, drop_key=drop_key)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class Channel_Adaptive_Transformer_Block(nn.Module):
    """Channel Adaptive Transformer Block (CATB).

    input:
        x: (B, C, H, W), feature map

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution. based on patch
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        patch_size (int): the size of the patch
        depth (int): Number of Swin Transformer blocks, default 2.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
        drop_key (bool, optional): Using dropkey or not. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size, patch_size,
                 depth=2, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 fused_window_process=False, drop_key=False, use_vmamba=False,
                 use_secret_controller=False):

        super().__init__()
        self.transformer_block = Swin_Transformer(dim, to_2tuple(input_resolution), depth, num_heads, window_size, mlp_ratio, qkv_bias,
                                                  qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, fused_window_process, drop_key=drop_key)
        channel_dim = dim//(patch_size**2)
        self.CA = SecretAwareChannelController(channel_dim, input_resolution*patch_size) if use_secret_controller else Channel_Adapter(channel_dim, input_resolution*patch_size)
        self.vmamba_scan = VMamba2DScan(channel_dim) if use_vmamba else None
        self.patch_size = patch_size
        self.window_size = window_size
        self.input_resolution = input_resolution
        self.use_secret_controller = use_secret_controller

    def forward(self, x, cover_feat=None, secret_feat=None):
        if self.use_secret_controller:
            x = self.CA(x, cover_feat, secret_feat)
        else:
            x = self.CA(x)  # 消融: CA
        x = PatchDivide(x, self.patch_size)
        x = self.transformer_block(x)
        x = PatchReverse(x, self.patch_size)
        if self.vmamba_scan is not None:
            x = x + self.vmamba_scan(x)
        return x


class CATB_Layer(nn.Module):
    """Channel Adaptive Transformer Block Layer, compose with numbers of CATBs.

    input:
        x: (B, C, H, W), feature map

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution. based on patch
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        patch_size (int): the size of the patch
        depth (int): Number of CATB, default 1.
        depth_tr (int): Number of Swin Transformer blocks in each CATB, default 2.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
        drop_key (bool, optional): Using dropkey or not. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size, patch_size, depth=1,
                 depth_tr=2, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                  fused_window_process=False, drop_key=False, use_vmamba=False,
                 use_secret_controller=False):

        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            Channel_Adaptive_Transformer_Block(dim, input_resolution, num_heads, window_size, patch_size, depth_tr, mlp_ratio, qkv_bias,
                                              qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, fused_window_process, drop_key,
                                               use_vmamba=use_vmamba, use_secret_controller=use_secret_controller)for i in range(depth)])

    def forward(self, x, cover_feat=None, secret_feat=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, cover_feat, secret_feat)
            else:
                x = blk(x, cover_feat, secret_feat)
        return x


class StegFormer(nn.Module):
    """StegFormer

    Args:
        img_resolution (int): Resolution of images
        input_dim(int): Dim of the input
        output_dim (int): Dim of the finnal output
        cnn_emb_dim (int): Embedding dim using convolution
        output_act (nn.Module): The act function in the end of StegFormer, Default: None
        patch_size(int): the size of the patch
        num_heads (list): Number of attention heads.
        window_size (int): Local window size
        depth (list): Number of CATB in each CATB Layer
        depth_tr (list): Number of the Swin Transformer Block in each CATB
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        drop_key (bool, optional): Using dropkey or not. Default: False
    """

    def __init__(self, img_resolution, input_dim=3, output_dim=3, cnn_emb_dim=16, output_act=None, patch_size=2, num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2], window_size=8,
                 depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, drop_key=False):
        super().__init__()
        self.dim = cnn_emb_dim
        self.token_dim = (patch_size**2)*cnn_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_conceal_network = input_dim > output_dim and output_dim == 3
        self.embedding = nn.Conv2d(input_dim, self.dim, 3, 1, 1)
        self.cover_embedding = nn.Conv2d(output_dim, self.dim, 3, 1, 1) if self.is_conceal_network else None
        self.secret_embedding = nn.Conv2d(input_dim-output_dim, self.dim, 3, 1, 1) if self.is_conceal_network else None

        self.patch_size = patch_size
        self.patch_resolution = img_resolution//patch_size
        if output_act:
            self.output_act = output_act()
        else:
            self.output_act = None

        # encoder
        self.encoderlayer_0 = CATB_Layer(self.token_dim, self.patch_resolution, num_heads[0],
                                         window_size, patch_size, depth[0], depth_tr[0], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key)
        self.downsampler_0 = DownSampler(self.dim, self.dim*2)

        self.cover_downsampler_0 = DownSampler(self.dim, self.dim*2) if self.is_conceal_network else None
        self.secret_downsampler_0 = DownSampler(self.dim, self.dim*2) if self.is_conceal_network else None

        self.encoderlayer_1 = CATB_Layer(self.token_dim*2, self.patch_resolution//(2**1),
                                         num_heads[1], window_size, patch_size, depth[1], depth_tr[1], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key,
                                         use_vmamba=True)
        self.downsampler_1 = DownSampler(self.dim*2, self.dim*4)
        self.cover_downsampler_1 = DownSampler(self.dim*2, self.dim*4) if self.is_conceal_network else None
        self.secret_downsampler_1 = DownSampler(self.dim*2, self.dim*4) if self.is_conceal_network else None

        self.encoderlayer_2 = CATB_Layer(self.token_dim*4, self.patch_resolution//(2**2),
                                         num_heads[2], window_size, patch_size, depth[2], depth_tr[2], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key,
                                         use_vmamba=True, use_secret_controller=self.is_conceal_network)
        self.downsampler_2 = DownSampler(self.dim*4, self.dim*8)
        self.cover_downsampler_2 = DownSampler(self.dim*4, self.dim*8) if self.is_conceal_network else None
        self.secret_downsampler_2 = DownSampler(self.dim*4, self.dim*8) if self.is_conceal_network else None

        self.encoderlayer_3 = CATB_Layer(self.token_dim*8, self.patch_resolution//(2**3),
                                         num_heads[3], window_size, patch_size, depth[3], depth_tr[3], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key,
                                         use_secret_controller=self.is_conceal_network)
        self.downsampler_3 = DownSampler(self.dim*8, self.dim*16)

        # bottleneck
        self.bottleneck = GEB(self.token_dim*16, self.patch_resolution//(2**4),
                              num_heads[4], patch_size, depth[4], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key)

        # decoder
        self.upsampler_0 = UpSampler(self.dim*16, self.dim*8)
        self.decoderlayer_0 = CATB_Layer(self.token_dim*16, self.patch_resolution//(2**3),
                                         num_heads[5], window_size, patch_size, depth[5], depth_tr[4], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key,
                                         use_vmamba=True)

        self.upsampler_1 = UpSampler(self.dim*16, self.dim*4)
        self.decoderlayer_1 = CATB_Layer(self.token_dim*8, self.patch_resolution//(2**2),
                                         num_heads[6], window_size, patch_size, depth[6], depth_tr[5], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key,
                                         use_vmamba=True)


        self.upsampler_2 = UpSampler(self.dim*8, self.dim*2)
        self.decoderlayer_2 = CATB_Layer(self.token_dim*4, self.patch_resolution//(2**1),
                                         num_heads[7], window_size, patch_size, depth[7], depth_tr[6], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key)

        self.upsampler_3 = UpSampler(self.dim*4, self.dim)
        self.decoderlayer_3 = CATB_Layer(self.token_dim*2, self.patch_resolution,
                                         num_heads[8], window_size, patch_size, depth[8], depth_tr[7], mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, use_checkpoint, drop_key=drop_key)

        self.output_proj = nn.Conv2d(self.dim*2, output_dim, 3, 1, 1)

    def forward(self, x):
        cover_img = x[:, :self.output_dim, :, :] if self.is_conceal_network else None
        secret_img = x[:, self.output_dim:, :, :] if self.is_conceal_network else None
        x = self.embedding(x)

        if self.is_conceal_network:
            cover_cond0 = self.cover_embedding(cover_img)
            secret_cond0 = self.secret_embedding(secret_img)
        else:
            cover_cond0 = secret_cond0 = None

        # encode: high resolution keeps the original CATB for pixel-detail protection.
        conv0 = self.encoderlayer_0(x)
        pool0 = self.downsampler_0(conv0)
        if self.is_conceal_network:
            cover_cond1 = self.cover_downsampler_0(cover_cond0)
            secret_cond1 = self.secret_downsampler_0(secret_cond0)
        else:
            cover_cond1 = secret_cond1 = None

        # medium resolution adds VMamba 2-D scan for cross-window communication.
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.downsampler_1(conv1)
        if self.is_conceal_network:
            cover_cond2 = self.cover_downsampler_1(cover_cond1)
            secret_cond2 = self.secret_downsampler_1(secret_cond1)
        else:
            cover_cond2 = secret_cond2 = None

        # low resolution uses the secret-aware channel controller.
        conv2 = self.encoderlayer_2(pool1, cover_cond2, secret_cond2)
        pool2 = self.downsampler_2(conv2)
        if self.is_conceal_network:
            cover_cond3 = self.cover_downsampler_2(cover_cond2)
            secret_cond3 = self.secret_downsampler_2(secret_cond2)
        else:
            cover_cond3 = secret_cond3 = None

        conv3 = self.encoderlayer_3(pool2, cover_cond3, secret_cond3)
        pool3 = self.downsampler_3(conv3)

        # bottleneck: Hybrid Mamba3-GEB plans the global secret embedding distribution.
        bottle = self.bottleneck(pool3)

        # decode with skip connections and generate a stego residual for concealment.

        up0 = self.upsampler_0(bottle)
        deconv0 = torch.cat([up0, conv3], 1)
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsampler_1(deconv0)
        deconv1 = torch.cat([up1, conv2], 1)
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsampler_2(deconv1)
        deconv2 = torch.cat([up2, conv1], 1)
        deconv2 = self.decoderlayer_2(deconv2)

        up3 = self.upsampler_3(deconv2)
        deconv3 = torch.cat([up3, conv0], 1)
        deconv3 = self.decoderlayer_3(deconv3)

        output = self.output_proj(deconv3)

        if self.is_conceal_network:
            output = cover_img + 0.1*torch.tanh(output)
        if self.output_act:
            output = self.output_act(output)
        return output


class MambaStegFormer(StegFormer):
    """Named entry point for the optimized MambaStegFormer architecture.

    The implementation inherits the upgraded StegFormer backbone that includes
    VMamba scanning, secret-aware channel control, and the hybrid Mamba3-GEB.
    """

    pass
