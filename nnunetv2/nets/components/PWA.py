import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from monai.networks.layers import DropPath
from typing import Sequence
from .attention_utils import FFN, LayerNorm, PositionalEmbedding, PatchMerging
from math import ceil


class Paired_Windows_Attention(nn.Module):
    """
    Shape-safe paired-window attention.

    This version keeps the original PWA branch enabled, but makes window
    gathering/scattering robust to dynamic nnU-Net patch sizes. Runtime feature
    maps are padded to valid window multiples before attention and cropped back
    afterwards.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        in_channels: int,
        min_big_window_size: Sequence[int] = (3, 3, 3),
        min_small_window_size: Sequence[int] = (1, 1, 1),
        scale_factor: int = 2,
        num_heads: int = 1,
        min_dim_head: int = 4,
        dropout: float = 0.1,
        dim: int = 3,
        use_pos_embed: bool = True,
    ):
        super().__init__()

        self.input_size = [int(i) for i in input_size]
        self.in_channels = int(in_channels)

        self.channels_qk = int(in_channels)
        self.channels_v = int(in_channels)

        self.min_big_window_size = [int(i) for i in min_big_window_size]
        self.min_small_window_size = [int(i) for i in min_small_window_size]
        self.scale_factor = int(scale_factor)
        self.num_heads = int(num_heads)
        self.min_dim_head = int(min_dim_head)
        self.dim = int(dim)
        self.use_pos_embed = bool(use_pos_embed)

        if self.num_heads > 0:
            self.big_window_size, self.small_window_size = self.get_window_sizes()

            self.n_hwd = [
                max(1, int(self.min_big_window_size[i]) // max(1, int(self.min_small_window_size[i])))
                for i in range(self.dim)
            ]

            self.num_bswin = len(self.big_window_size)
            self.num_heads_bswin = self.num_heads * self.num_bswin
            self.mid_channels_perhead = max(1, self.in_channels // max(1, self.num_heads_bswin))

            if self.use_pos_embed:
                self.position_embedding = PositionalEmbedding(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    window_size=self.n_hwd,
                )

            self.window_gathering = self.window_gathering_3d if self.dim == 3 else self.window_gathering_2d
            self.window_scattering = self.window_scattering_3d if self.dim == 3 else self.window_scattering_2d

            self.softmax = nn.Softmax(dim=-1)
            self.dropout_weight = nn.Dropout(dropout)

    @staticmethod
    def _gcd(a: int, b: int) -> int:
        a = abs(int(a))
        b = abs(int(b))
        while b:
            a, b = b, a % b
        return max(1, a)

    @classmethod
    def _lcm(cls, a: int, b: int) -> int:
        a = max(1, int(a))
        b = max(1, int(b))
        return abs(a * b) // cls._gcd(a, b)

    @classmethod
    def _lcm_list(cls, values: Sequence[int]) -> int:
        result = 1
        for value in values:
            result = cls._lcm(result, int(value))
        return max(1, int(result))

    @staticmethod
    def _ceil_to_multiple(value: int, divisor: int) -> int:
        divisor = max(1, int(divisor))
        return int(ceil(int(value) / divisor) * divisor)

    def _get_common_divisibility(self) -> Sequence[int]:
        divisibility = []
        for axis in range(self.dim):
            axis_windows = [int(w[axis]) for w in self.big_window_size]
            divisibility.append(self._lcm_list(axis_windows))
        return divisibility

    def _get_safe_spatial_shape(self, spatial_shape: Sequence[int]) -> Sequence[int]:
        divisibility = self._get_common_divisibility()
        safe_shape = []
        for size, divisor in zip(spatial_shape, divisibility):
            safe_shape.append(self._ceil_to_multiple(int(size), int(divisor)))
        return safe_shape

    @staticmethod
    def _pad_spatial(x: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        current_shape = list(x.shape[2:])
        target_shape = [int(i) for i in target_shape]

        if current_shape == target_shape:
            return x

        pads = []
        for current, target in zip(reversed(current_shape), reversed(target_shape)):
            diff = int(target) - int(current)
            if diff <= 0:
                pads.extend([0, 0])
            else:
                left = diff // 2
                right = diff - left
                pads.extend([left, right])

        if any(pads):
            x = F.pad(x, pads)

        return x

    @staticmethod
    def _crop_spatial(x: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        target_shape = [int(i) for i in target_shape]

        if list(x.shape[2:]) == target_shape:
            return x

        slices = [slice(None), slice(None)]
        for current, target in zip(x.shape[2:], target_shape):
            current = int(current)
            target = int(target)
            if current > target:
                start = (current - target) // 2
                end = start + target
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))

        x = x[tuple(slices)]

        if list(x.shape[2:]) != target_shape:
            x = Paired_Windows_Attention._pad_spatial(x, target_shape)

        return x

    def get_window_sizes(self):
        input_size = torch.tensor(self.input_size, dtype=torch.long)
        min_big_window_size = torch.tensor(self.min_big_window_size, dtype=torch.long)
        min_small_window_size = torch.tensor(self.min_small_window_size, dtype=torch.long)

        bw_sizes = []
        sw_sizes = []

        bw = min_big_window_size.clone()
        sw = min_small_window_size.clone()

        # The original code used .any(), which can create windows larger than
        # one or more axes. .all() is safer for static planning, while runtime
        # padding below handles remaining dynamic cases.
        while (bw <= input_size).all():
            bw_sizes.append([int(i) for i in bw.tolist()])
            sw_sizes.append([int(i) for i in sw.tolist()])

            bw = bw * self.scale_factor
            sw = sw * self.scale_factor

        # If the static input_size is smaller than the minimum window, keep the
        # minimum window and rely on runtime padding.
        if len(bw_sizes) == 0:
            bw_sizes.append([int(i) for i in min_big_window_size.tolist()])
            sw_sizes.append([int(i) for i in min_small_window_size.tolist()])

        channels_need = max(1, len(bw_sizes) * self.num_heads * self.min_dim_head)
        channels_qk = channels_need
        channels_v = ceil(self.channels_v / channels_need) * channels_need

        self.channels_qk = int(channels_qk)
        self.channels_v = int(channels_v)

        return bw_sizes, sw_sizes

    def attention_operation(self, query, key, value):
        l, c = query.shape[-2:]

        scores = torch.einsum("bhNmc,bhNnc->bhNmn", [query, key]) / (c ** 0.5)

        if self.use_pos_embed:
            try:
                relative_position_bias = self.position_embedding.get_relative_position_bias(l=l)[None, :, None]
                scores = scores + relative_position_bias
            except Exception:
                # Runtime padding may change l. If the relative-position table
                # does not match, skip position bias instead of failing.
                pass

        weights = self.softmax(scores)
        weights = self.dropout_weight(weights)

        attention = torch.einsum("bhNmn,bhNnc->bhNmc", [weights, value])
        return attention

    def window_gathering_3d(self, x):
        b, _, h, w, d = x.size()

        x = rearrange(
            x,
            "b (bswin head c) h w d -> b bswin head c h w d",
            bswin=self.num_bswin,
            head=self.num_heads,
        )

        n = None
        Ns = []
        xs = []

        for i in range(self.num_bswin):
            b_win_h, b_win_w, b_win_d = [int(v) for v in self.big_window_size[i]]
            s_win_h, s_win_w, s_win_d = [int(v) for v in self.small_window_size[i]]

            Nh, Nw, Nd = h // b_win_h, w // b_win_w, d // b_win_d
            nh, nw, nd = b_win_h // s_win_h, b_win_w // s_win_w, b_win_d // s_win_d

            if Nh <= 0 or Nw <= 0 or Nd <= 0:
                raise RuntimeError(
                    "Invalid 3D PWA window configuration. "
                    "feature_shape=({}, {}, {}), big_window=({}, {}, {}).".format(
                        h, w, d, b_win_h, b_win_w, b_win_d
                    )
                )

            xi = rearrange(
                x[:, i],
                "b head c (Nh winh) (Nw winw) (Nd wind) -> b (head Nh Nw Nd c) winh winw wind",
                winh=b_win_h,
                winw=b_win_w,
                wind=b_win_d,
            )

            xi = F.max_pool3d(
                xi,
                kernel_size=(s_win_h, s_win_w, s_win_d),
                stride=(s_win_h, s_win_w, s_win_d),
            )

            xi = rearrange(
                xi,
                "b (head Nh Nw Nd c) nh nw nd -> b head (Nh Nw Nd) (nh nw nd) c",
                head=self.num_heads,
                Nh=Nh,
                Nw=Nw,
                Nd=Nd,
            )

            xs.append(xi)
            Ns.append([Nh, Nw, Nd])

            if n is None:
                n = [nh, nw, nd]
            elif not (n[0] == nh and n[1] == nw and n[2] == nd):
                raise RuntimeError(
                    "PWA requires the same number of small windows per big window. "
                    "Expected {}, got {}.".format(n, [nh, nw, nd])
                )

        x = torch.cat(xs, dim=2)
        return x, Ns, n

    def window_gathering_2d(self, x):
        b, _, h, w = x.size()

        x = rearrange(
            x,
            "b (bswin head c) h w -> b bswin head c h w",
            bswin=self.num_bswin,
            head=self.num_heads,
        )

        n = None
        Ns = []
        xs = []

        for i in range(self.num_bswin):
            b_win_h, b_win_w = [int(v) for v in self.big_window_size[i]]
            s_win_h, s_win_w = [int(v) for v in self.small_window_size[i]]

            Nh, Nw = h // b_win_h, w // b_win_w
            nh, nw = b_win_h // s_win_h, b_win_w // s_win_w

            if Nh <= 0 or Nw <= 0:
                raise RuntimeError(
                    "Invalid 2D PWA window configuration. "
                    "feature_shape=({}, {}), big_window=({}, {}).".format(
                        h, w, b_win_h, b_win_w
                    )
                )

            xi = rearrange(
                x[:, i],
                "b head c (Nh winh) (Nw winw) -> b (head Nh Nw c) winh winw",
                winh=b_win_h,
                winw=b_win_w,
            )

            xi = F.max_pool2d(
                xi,
                kernel_size=(s_win_h, s_win_w),
                stride=(s_win_h, s_win_w),
            )

            xi = rearrange(
                xi,
                "b (head Nh Nw c) nh nw -> b head (Nh Nw) (nh nw) c",
                head=self.num_heads,
                Nh=Nh,
                Nw=Nw,
            )

            xs.append(xi)
            Ns.append([Nh, Nw])

            if n is None:
                n = [nh, nw]
            elif not (n[0] == nh and n[1] == nw):
                raise RuntimeError(
                    "PWA requires the same number of small windows per big window. "
                    "Expected {}, got {}.".format(n, [nh, nw])
                )

        x = torch.cat(xs, dim=2)
        return x, Ns, n

    def window_scattering_3d(self, outs, Ns, n):
        nh, nw, nd = n

        outs = rearrange(
            outs,
            "b head Ns (nh nw nd) c -> b head Ns c nh nw nd",
            nh=nh,
            nw=nw,
            nd=nd,
        )

        idx = 0
        outs_ = []

        for i in range(self.num_bswin):
            Nh, Nw, Nd = Ns[i]
            N = Nh * Nw * Nd

            out = rearrange(
                outs[:, :, idx:idx + N],
                "b head N c nh nw nd -> b (head N c) nh nw nd",
                nh=nh,
                nw=nw,
                nd=nd,
            )

            out = F.interpolate(
                out,
                scale_factor=tuple(int(v) for v in self.small_window_size[i]),
                mode="trilinear",
                align_corners=True,
            )

            out = rearrange(
                out,
                "b (head Nh Nw Nd c) winh winw wind -> b 1 head c (Nh winh) (Nw winw) (Nd wind)",
                head=self.num_heads,
                Nh=Nh,
                Nw=Nw,
                Nd=Nd,
            )

            outs_.append(out)
            idx += N

        out = torch.cat(outs_, dim=1)
        out = rearrange(out, "b bswin head c h w d -> b (bswin head c) h w d")
        return out

    def window_scattering_2d(self, outs, Ns, n):
        nh, nw = n

        outs = rearrange(
            outs,
            "b head Ns (nh nw) c -> b head Ns c nh nw",
            nh=nh,
            nw=nw,
        )

        idx = 0
        outs_ = []

        for i in range(self.num_bswin):
            Nh, Nw = Ns[i]
            N = Nh * Nw

            out = rearrange(
                outs[:, :, idx:idx + N],
                "b head N c nh nw -> b (head N c) nh nw",
                nh=nh,
                nw=nw,
            )

            out = F.interpolate(
                out,
                scale_factor=tuple(int(v) for v in self.small_window_size[i]),
                mode="bilinear",
                align_corners=True,
            )

            out = rearrange(
                out,
                "b (head Nh Nw c) winh winw -> b 1 head c (Nh winh) (Nw winw)",
                head=self.num_heads,
                Nh=Nh,
                Nw=Nw,
            )

            outs_.append(out)
            idx += N

        out = torch.cat(outs_, dim=1)
        out = rearrange(out, "b bswin head c h w -> b (bswin head c) h w")
        return out

    def forward(self, query, key, value):
        if self.num_heads == 0:
            return query

        original_shape = tuple(query.shape[2:])
        safe_shape = self._get_safe_spatial_shape(original_shape)

        query = self._pad_spatial(query, safe_shape)
        key = self._pad_spatial(key, safe_shape)
        value = self._pad_spatial(value, safe_shape)

        q, Ns, n = self.window_gathering(query)
        k, _, _ = self.window_gathering(key)
        v, _, _ = self.window_gathering(value)

        attn = self.attention_operation(q, k, v)
        attn = self.window_scattering(attn, Ns, n)
        attn = self._crop_spatial(attn, original_shape)

        return attn


class MultiModal_Paired_Windows_Attention(Paired_Windows_Attention):

    def __init__(
        self,
        input_size: Sequence[int],
        in_channels: Sequence[int],
        min_big_window_size: Sequence[int] = (3, 3, 3),
        min_small_window_size: Sequence[int] = (1, 1, 1),
        scale_factor: int = 2,
        num_heads: int = 1,
        min_dim_head: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        norm_layer=LayerNorm,
        dim: int = 3,
        use_pos_embed: bool = True,
    ):
        self.mid_channels = max(int(i) for i in in_channels)

        super().__init__(
            input_size=input_size,
            in_channels=self.mid_channels,
            min_big_window_size=min_big_window_size,
            min_small_window_size=min_small_window_size,
            scale_factor=scale_factor,
            num_heads=num_heads,
            min_dim_head=min_dim_head,
            dropout=attn_drop,
            dim=dim,
            use_pos_embed=use_pos_embed,
        )

        if self.num_heads > 0:
            self.in_channels = [int(i) for i in in_channels]
            self.num_modalities = len(in_channels)

            conv = nn.Conv3d if self.dim == 3 else nn.Conv2d

            input_norms = []
            qkv_proj = []
            mix_channels = []
            dropout_attns = []

            for m in range(self.num_modalities):
                input_norms.append(
                    norm_layer(self.in_channels[m], data_format="channels_first", dim=self.dim)
                )

                qkv_proj.append(
                    nn.ModuleList(
                        [
                            conv(self.in_channels[m], self.channels_qk, kernel_size=1, bias=qkv_bias),
                            conv(self.in_channels[m], self.channels_qk, kernel_size=1, bias=qkv_bias),
                            conv(self.in_channels[m], self.channels_v, kernel_size=1, bias=qkv_bias),
                        ]
                    )
                )

                mix_channels.append(conv(self.channels_v, self.in_channels[m], kernel_size=1))
                dropout_attns.append(nn.Dropout(proj_drop))

            self.input_norms = nn.ModuleList(input_norms)
            self.qkv_proj = nn.ModuleList(qkv_proj)
            self.mix_channels = nn.ModuleList(mix_channels)
            self.dropout_attns = nn.ModuleList(dropout_attns)

            self.window_gathering = self.window_gathering_3d if self.dim == 3 else self.window_gathering_2d
            self.window_scattering = self.window_scattering_3d if self.dim == 3 else self.window_scattering_2d

    def attention_operation(self, query, key, value):
        ml, c = query.shape[-2:]
        l = ml // self.num_modalities

        scores = torch.einsum("bhNmc,bhNnc->bhNmn", [query, key]) / (c ** 0.5)

        if self.use_pos_embed:
            try:
                relative_position_bias = self.position_embedding.get_relative_position_bias(l=l)
                for i in range(self.num_modalities):
                    for j in range(self.num_modalities):
                        scores[:, :, :, i * l:(i + 1) * l, j * l:(j + 1) * l] = (
                            scores[:, :, :, i * l:(i + 1) * l, j * l:(j + 1) * l]
                            + relative_position_bias[None, :, None]
                        )
            except Exception:
                pass

        weights = self.softmax(scores)
        weights = self.dropout_weight(weights)

        attention = torch.einsum("bhNmn,bhNnc->bhNmc", [weights, value])
        return attention

    def forward(self, inputs):
        if self.num_heads == 0:
            return inputs

        if len(inputs) != self.num_modalities:
            raise RuntimeError(
                "The number of modalities should be {}, but got {}.".format(
                    self.num_modalities, len(inputs)
                )
            )

        original_shape = tuple(inputs[0].shape[2:])
        safe_shape = self._get_safe_spatial_shape(original_shape)

        querys, keys, values = [], [], []

        for m in range(self.num_modalities):
            if tuple(inputs[m].shape[2:]) != original_shape:
                raise RuntimeError(
                    "All modalities must have the same spatial shape. "
                    "Expected {}, got {} at modality {}.".format(
                        original_shape, tuple(inputs[m].shape[2:]), m
                    )
                )

            x_norm = self.input_norms[m](inputs[m])

            q = self.qkv_proj[m][0](x_norm)
            k = self.qkv_proj[m][1](x_norm)
            v = self.qkv_proj[m][2](x_norm)

            q = self._pad_spatial(q, safe_shape)
            k = self._pad_spatial(k, safe_shape)
            v = self._pad_spatial(v, safe_shape)

            querys.append(q)
            keys.append(k)
            values.append(v)

        q, k, v = None, None, None
        n, Ns = None, None
        l = None

        for m in range(self.num_modalities):
            q, Ns, n = self.window_gathering(querys[m])
            k, _, _ = self.window_gathering(keys[m])
            v, _, _ = self.window_gathering(values[m])

            querys[m] = q
            keys[m] = k
            values[m] = v

            if l is None:
                l = q.shape[-2]
            elif l != q.shape[-2]:
                raise RuntimeError(
                    "The seq length in all modalities should be equal, "
                    "but got {} and {}.".format(l, q.shape[-2])
                )

        querys = torch.cat(querys, dim=-2)
        keys = torch.cat(keys, dim=-2)
        values = torch.cat(values, dim=-2)

        attn = self.attention_operation(querys, keys, values)
        attn = rearrange(attn, "b head Ns (m l) c -> b head Ns m l c", l=l)

        attns = []
        for m in range(self.num_modalities):
            attn_m = self.window_scattering(attn[:, :, :, m], Ns, n)
            attn_m = self._crop_spatial(attn_m, original_shape)
            attn_m = inputs[m] + self.dropout_attns[m](self.mix_channels[m](attn_m))
            attns.append(attn_m)

        return attns


class Paired_Windows_TransformerBlock(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        in_channels: Sequence[int],
        min_big_window_size: Sequence[int] = (3, 3, 3),
        min_small_window_size: Sequence[int] = (1, 1, 1),
        scale_factor: int = 2,
        num_heads: int = 1,
        min_dim_head: int = 4,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0.0,
        ffn_expansion_ratio: int = 4,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = LayerNorm,
        qkv_bias: bool = True,
        dim: int = 3,
    ) -> None:

        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)

        self.attn = MultiModal_Paired_Windows_Attention(
            input_size=input_size,
            in_channels=in_channels,
            min_big_window_size=min_big_window_size,
            min_small_window_size=min_small_window_size,
            scale_factor=scale_factor,
            num_heads=num_heads,
            min_dim_head=min_dim_head,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            dim=dim,
            use_pos_embed=True,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ffns = nn.ModuleList()
        self.norms = nn.ModuleList()
        for m in range(self.num_modalities):
            self.ffns.append(
                FFN(
                    in_channels[m],
                    expansion_ratio=ffn_expansion_ratio,
                    dropout_rate=proj_drop,
                    act=act_layer,
                    dim=dim,
                )
            )
            self.norms.append(norm_layer(in_channels[m], data_format="channels_first", dim=dim))

    def forward(self, xs: Sequence[torch.Tensor]):
        attns = self.attn(xs)
        attns = [xs[m] + self.drop_path(attns[m]) for m in range(self.num_modalities)]
        attns = [
            attns[m] + self.drop_path(self.ffns[m](self.norms[m](attns[m])))
            for m in range(self.num_modalities)
        ]

        return attns


class Transformer_BasicLayer(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        in_channels: Sequence[int],
        depth: int = 2,
        min_big_window_size: Sequence[int] = (3, 3, 3),
        min_small_window_size: Sequence[int] = (1, 1, 1),
        scale_factor: int = 2,
        num_heads: int = 1,
        min_dim_head: int = 4,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        ffn_expansion_ratio: int = 4,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = LayerNorm,
        qkv_bias: bool = True,
        do_downsample: bool = True,
        dim: int = 3,
    ):

        super().__init__()

        self.num_modalities = len(in_channels)
        self.blocks = nn.ModuleList(
            [
                Paired_Windows_TransformerBlock(
                    input_size=input_size,
                    in_channels=in_channels,
                    min_big_window_size=min_big_window_size,
                    min_small_window_size=min_small_window_size,
                    scale_factor=scale_factor,
                    num_heads=num_heads,
                    min_dim_head=min_dim_head,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    ffn_expansion_ratio=ffn_expansion_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    qkv_bias=qkv_bias,
                    dim=dim,
                )
                for i in range(depth)
            ]
        )
        self.downs = None
        if do_downsample:
            self.downs = nn.ModuleList(
                [
                    PatchMerging(in_ch=in_channels[m], norm_layer=norm_layer, dim=dim)
                    for m in range(self.num_modalities)
                ]
            )

    def attn_forward(self, xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        for blk in self.blocks:
            xs = blk(xs)
        return xs

    def down_forward(self, xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        down = None
        if self.downs is not None:
            down = [self.downs[m](xs[m]) for m in range(self.num_modalities)]
        return down

    def forward(self, xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        xs = self.attn_forward(xs)
        down = self.down_forward(xs)
        return xs, down


class Cross_Channel_Attention(nn.Module):

    def __init__(
        self,
        ch1: Sequence[int],
        ch2: int,
        channel_reduction: int = 4,
        spatial_dim: int = 3,
        output_both: bool = False,
    ):

        super().__init__()

        self.chs1 = ch1
        self.ch2 = ch2
        self.spatial_dim = spatial_dim
        self.output_both = output_both

        if spatial_dim == 3:
            avp = nn.AdaptiveAvgPool3d
            conv = nn.Conv3d
        elif spatial_dim == 2:
            avp = nn.AdaptiveAvgPool2d
            conv = nn.Conv2d
        else:
            raise RuntimeError("spatial_dim must be 2 or 3.")

        self.ch1 = sum(ch1)
        reduced_ch1 = max(1, self.ch1 // channel_reduction)
        reduced_ch2 = max(1, ch2 // channel_reduction)

        self.squeeze_extract_1 = nn.Sequential(
            avp(1),
            conv(self.ch1, reduced_ch1, kernel_size=1),
            nn.GELU(),
            conv(reduced_ch1, self.ch1, kernel_size=1),
            nn.Flatten(2),
        )
        self.squeeze_extract_2 = nn.Sequential(
            avp(1),
            conv(ch2, reduced_ch2, kernel_size=1),
            nn.GELU(),
            conv(reduced_ch2, ch2, kernel_size=1),
            nn.Flatten(2),
        )

    def forward(self, x1, x2) -> Sequence[torch.Tensor]:
        x1 = torch.cat(x1, dim=1)
        qkv_attn = self.squeeze_extract_1(x1)
        qkv_conv = self.squeeze_extract_2(x2)

        scores = torch.einsum("bmd,bnd->bmn", qkv_attn, qkv_conv)
        weight_1_to_2 = F.softmax(scores, dim=1) / (self.ch1 ** 0.5)

        if self.output_both:
            weight_2_to_1 = F.softmax(scores, dim=2) / (self.ch2 ** 0.5)

        if self.spatial_dim == 3:
            x2_ = torch.einsum("bmn,bmhwd->bnhwd", weight_1_to_2, x1) + x2

            if self.output_both:
                x1_ = torch.einsum("bmn,bnhwd->bmhwd", weight_2_to_1, x2) + x1

                xs = []
                c = 0
                for c1 in self.chs1:
                    xs.append(x1_[:, c:c + c1])
                    c += c1
                return xs, x2_
            return x2_

        x2_ = torch.einsum("bmn,bmhw->bnhw", weight_1_to_2, x1) + x2

        if self.output_both:
            x1_ = torch.einsum("bmn,bnhw->bmhw", weight_2_to_1, x2) + x1

            xs = []
            c = 0
            for c1 in self.chs1:
                xs.append(x1_[:, c:c + c1])
                c += c1
            return xs, x2_

        return x2_
