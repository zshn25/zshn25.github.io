---
layout: notebook
title: "Inference Speedup Benchmark"
description: "Guide to CUDA GPU inference optimizations, benchmarked on a state-of-the-art Pytorch depth estimation model."
image: images/2026-03-24-inference_optimizations_benchmark/output_80_1.png
date: 2026-03-24 15:21:23 -0700
author: Zeeshan Khan Suri
comments: true
categories: runtime-optimization gpu deep-learning
published: true
nb_path: _notebooks/2026-03-24-inference_optimizations_benchmark.ipynb
---

How fast can we push a single forward pass of a modern depth estimator on a CUDA GPU? Let's find out using an extensive collection of optimizations:


| # | Method | Key idea |
|---|--------|----------|
| 0 | Eager baseline | all extras disabled; clean reference point |
| 1 | `cudnn.benchmark` | auto-tune conv algorithms |
| 2 | `channels_last` | NHWC memory layout |
| 3 | `inference_mode` | stricter than `no_grad` |
| 4 | TF32 matmul | Ampere fast-math for float32 |
| 5 | AMP FP16 | autocast to half precision |
| 6 | AMP BF16 | autocast to bfloat16 |
| 7 | Static FP16 | `model.half()` permanently |
| 7b | SDPA backends | Flash / MemEfficient / Math attention |
| 8–11 | `torch.compile` | Dynamo + Inductor + Triton (4 modes) |
| 12 | JIT Trace + optimize | TorchScript graph fusion |
| 13 | Manual CUDA Graphs | replay captured kernel sequence |
| 14–15 | torchao int8 | GPU-native weight-only quantization |
| 16 | torchao int4 weight-only | 4-bit Triton kernels |
| 17 | ONNX Runtime CUDA EP | ORT graph optimizations |
| 18 | ORT CUDA EP + IO Binding | zero-copy GPU inference |
| 19 | ORT TensorRT EP | TRT engine via ORT |
| 20 | Torch-TensorRT | direct TRT compilation from PyTorch |
| 21 | Triton fused softmax | custom Triton kernel replacing attention softmax |
| 22 | compile fullgraph + freeze | zero graph breaks + Inductor tuning |
| 23–24 | Weight pruning 50 % | unstructured L1 sparsity ± compile |
| 25 | torch.export + AOTInductor | ahead-of-time compiled .so artifact |
| 26 | Multi-stream pipelining | batch splitting across CUDA streams |
| 27 | bitsandbytes int8 | LLM.int8() quantization |
| 28 | bitsandbytes NF4 | 4-bit NormalFloat from QLoRA |
| 29 | FP16 + compile reduce-overhead | the "GPT, Fast" recipe |
| 30 | FP16 + TF32 + cudnn.benchmark | simplest no-compile combo |
| 31 | FP16 + compile reduce-overhead + TF32 | strongest combo |

**Methodology:**

- We don't just use a feature encoder model, but a full-fledged model in order to be able to compare outputs (important for some optimizations like quantization)
- **Sanity checks:** Every optimization is automatically checked against the baseline
depth map. `bench()` computes MSE on a real-image depth prediction - if the output
diverges, we see a warning and a side-by-side depth visualization.
- Every benchmark cell calls `reset_backends()` first so optimizations never accidentally stack. 
- A GPU warmup pass runs before each timed section. 
- Each method runs **5 independent rounds** of `blocked_autorange`; we report the **median of medians**.


```python
# ── Self-contained installs for Colab ──────────────────────────────────────────────
# Tested with PyTorch 2.5 + CUDA 12.1.

%pip install -q torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# NOTE: install depth_anything_3 with --no-deps to avoid pulling a newer
# PyTorch that may be incompatible with your CUDA driver.
%pip install -q "depth_anything_3 @ git+https://github.com/ByteDance-Seed/depth-anything-3" --no-deps
%pip install -q e3nn einops "evo>=0.3" omegaconf safetensors addict pillow opencv-python-headless
%pip install -q torchao bitsandbytes onnxruntime-gpu matplotlib
print("All dependencies assumed installed.")
```

    All dependencies assumed installed.
    

System info:


<details class="cell-collapse" markdown="1">
<summary>Show code</summary>

```python
from __future__ import annotations

import gc
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

warnings.filterwarnings("ignore")

# ── Environment setup for TensorRT EP + AOTInductor ──────────────────────
_trt_libs = os.path.join(
    os.path.dirname(torch.__file__),
    os.pardir,
    "tensorrt_libs",
)
_trt_libs = os.path.abspath(_trt_libs)
if os.path.isdir(_trt_libs):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _trt_libs not in _ld:
        os.environ["LD_LIBRARY_PATH"] = f"{_trt_libs}:{_ld}"
        import ctypes

        for _lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10"):
            _p = os.path.join(_trt_libs, _lib)
            if os.path.exists(_p):
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

for _cand in ("/usr/local/cuda", "/usr/local/cuda-12.1", "/usr/local/cuda-12"):
    if os.path.isdir(_cand) and os.path.isfile(os.path.join(_cand, "bin", "nvcc")):
        os.environ.setdefault("CUDA_HOME", _cand)
        break

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    cap = torch.cuda.get_device_capability(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {mem_gb:.1f} GB")
```

</details>
    PyTorch 2.5.1+cu121
    CUDA available: True
    CUDA version: 12.1
    VRAM: 34.1 GB
    

## Model: Depth Anything v3 - Metric-Large

ByteDance's monocular metric depth estimator - **334 M params**:
- **Backbone:** DinoV2 ViT-L/14 (304 M) - self-supervised vision transformer
- **Head:** DPT (30 M) - dense prediction transformer for depth regression

Input: a single RGB image `(B, 1, 3, 504, 504)`.
Output: metric depth map `(B, 1, 504, 504)`.

Since the backbone is a **Vision Transformer** the optimization landscape is matmul-heavy:
- **SDPA / FlashAttention** is a primary target
- **FP16 / TF32** have outsized impact
- **`channels_last`** has less impact (mostly linear layers)
- **`torch.compile`** can fuse attention + FFN blocks effectively

Load the model:


<details class="cell-collapse" markdown="1">
<summary>Show code</summary>

```python
from depth_anything_3.api import DepthAnything3
from pathlib import Path

BATCH = 1
IMG_SIZE = 504
MODEL_NAME = "da3-large"
WEIGHTS_PATH = Path("model.safetensors")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DA3Depth(nn.Module):
    """Benchmark-friendly wrapper: returns just the depth tensor."""

    def __init__(self, da3_api):
        super().__init__()
        self.net = da3_api.model  # DepthAnything3Net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x, export_feat_layers=[])
        return out["depth"]


# ── Build model & load weights ───────────────────────────────────────────
_api = DepthAnything3(model_name=MODEL_NAME)
_weights_loaded = False

# Try from_pretrained (needs HuggingFace Hub connectivity)
if not _weights_loaded:
    try:
        _api_pre = DepthAnything3.from_pretrained(f"depth-anything/{MODEL_NAME}")
        _api = _api_pre
        _weights_loaded = True
        print("Weights: loaded via from_pretrained (HuggingFace Hub)")
    except Exception as _e:
        print(f"from_pretrained unavailable: {type(_e).__name__}")

# If not, load from local safetensors file
if not _weights_loaded and WEIGHTS_PATH.exists():
    from safetensors.torch import load_file as _load_safetensors

    _sd = _load_safetensors(str(WEIGHTS_PATH))
    _sd = {k.replace("model.", "", 1): v for k, v in _sd.items()}
    _result = _api.model.load_state_dict(_sd, strict=False)
    _weights_loaded = True
    print(
        f"Weights: loaded from {WEIGHTS_PATH}  "
        f"(matched {len(_sd) - len(_result.unexpected_keys)}/{len(_sd)} keys, "
        f"{len(_result.missing_keys)} missing)"
    )

if not _weights_loaded:
    print("Weights: random init (benchmark timings are still valid)")

model = DA3Depth(_api).to(device).eval()

dummy_input = torch.randn(BATCH, 1, 3, IMG_SIZE, IMG_SIZE, device=device)
with torch.no_grad():
    baseline_output = model(dummy_input)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model  : Depth Anything v3 ({MODEL_NAME})")
print(f"Params : {n_params:.1f} M")
print(f"Input  : {tuple(dummy_input.shape)} - (B, N_views, C, H, W)")
print(f"Output : {tuple(baseline_output.shape)} - metric depth map")
```

</details>
    [INFO ] using MLP layer as FFN
    Weights: loaded from model.safetensors  (matched 637/637 keys, 6 missing)
    Model  : Depth Anything v3 (da3-large)
    Params : 410.9 M
    Input  : (1, 1, 3, 504, 504) - (B, N_views, C, H, W)
    Output : (1, 1, 504, 504) - metric depth map
    

Load the test image and test the model:


<details class="cell-collapse" markdown="1">
<summary>Show code</summary>

```python
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image as _PILImage
from torchvision import transforms as T
from urllib.request import urlopen
import io as _io

# Download a real test image (GitHub-hosted)
_IMG_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
try:
    _resp = urlopen(_IMG_URL, timeout=15)
    SANITY_IMG = _PILImage.open(_io.BytesIO(_resp.read())).convert("RGB")
    print(f"Downloaded test image: {SANITY_IMG.size}")
except Exception:
    print("Image download failed - using synthetic test image")
    SANITY_IMG = _PILImage.fromarray(
        np.random.default_rng(42).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    )

# Prepare real image as model input
_sanity_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
SANITY_INPUT = _sanity_transform(SANITY_IMG).unsqueeze(0).unsqueeze(0).to(device)
# shape: (1, 1, 3, 504, 504)

with torch.no_grad():
    SANITY_DEPTH_REF = model(SANITY_INPUT)

# ── Show the input image and baseline depth ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(SANITY_IMG)
axes[0].set_title("Input image")
axes[0].axis("off")

depth_np = SANITY_DEPTH_REF[0, 0].cpu().numpy()
print(f"Depth range: [{depth_np.min():.4f}, {depth_np.max():.4f}]")
im = axes[1].imshow(depth_np, cmap="inferno")
axes[1].set_title("Baseline depth")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
plt.close(fig)
```

</details>
    Downloaded test image: (1546, 1213)
    Depth range: [0.8860, 2.6241]
    


    
![png](/images/2026-03-24-inference_optimizations_benchmark/output_7_1.png)
    


## Benchmarking utilities

Every method gets timed the same way: `torch.utils.benchmark.Timer` with
`blocked_autorange`, repeated for `N_ROUNDS` independent rounds. We report
the **median of medians**.

Key helpers:
- **`bench()`** - timing with GPU clock warmup + 5 rounds of `blocked_autorange`
- **`sanity_check()`** - compare depth map vs baseline; show side-by-side on failure
- **`fresh_model()`** - clean model copy so no optimization leaks between cells
- **`reset_backends()`** - disable all global toggles (TF32, cudnn.benchmark, etc.)


<details class="cell-collapse" markdown="1">
<summary>Show code</summary>

```python
import time
from contextlib import contextmanager
from statistics import median
import torch.utils.benchmark as _bench_mod

# ── Shared state ─────────────────────────────────────────────────────────
GPU_RESULTS: dict[str, float] = {}
SETUP_TIMES: dict[str, float] = {}
MIN_RUN_TIME: float = 3.0
N_ROUNDS: int = 5

# Cache base model weights for fresh_model()
_BASE_STATE_DICT = {k: v.cpu().clone() for k, v in model.state_dict().items()}


def reset_backends() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


_TORCH_COMPILE_OK: bool | None = None


def _check_compile_support() -> bool:
    global _TORCH_COMPILE_OK
    if _TORCH_COMPILE_OK is not None:
        return _TORCH_COMPILE_OK
    try:
        _m = torch.compile(torch.nn.Linear(1, 1).to(device), mode="default")
        with torch.no_grad():
            _m(torch.randn(1, 1, device=device))
        _TORCH_COMPILE_OK = True
    except Exception:
        _TORCH_COMPILE_OK = False
    return _TORCH_COMPILE_OK


def check_output(name, output, reference, atol=0.05):
    if isinstance(output, np.ndarray):
        diff = float(np.max(np.abs(output - reference.cpu().numpy())))
    else:
        diff = (output.float().cpu() - reference.cpu()).abs().max().item()
    status = "OK" if diff < atol else "WARN"
    print(f"  [{status}] {name}: max diff = {diff:.6f} (atol={atol})")


def _gpu_warmup(stmt, globals_dict, n=10):
    t = _bench_mod.Timer(stmt=stmt, globals=globals_dict)
    t.timeit(n)


def bench(
    label,
    stmt,
    globals_dict,
    results=None,
    min_run_time=MIN_RUN_TIME,
    n_rounds=N_ROUNDS,
    sanity=True,
):
    if results is None:
        results = GPU_RESULTS

    _gpu_warmup(stmt, globals_dict)

    t = _bench_mod.Timer(stmt=stmt, globals=globals_dict, label=label)
    round_medians = []
    for _ in range(n_rounds):
        m = t.blocked_autorange(min_run_time=min_run_time)
        round_medians.append(m.median * 1e3)
    ms = median(round_medians)
    results[label] = ms
    spread = max(round_medians) - min(round_medians)
    print(
        f"  {label}: {ms:.3f} ms  "
        f"(spread: {spread:.3f} ms  "
        f"rounds: {', '.join(f'{r:.2f}' for r in round_medians)})"
    )

    if sanity and SANITY_INPUT is not None:
        _mdl = globals_dict.get("model")
        if _mdl is not None and isinstance(_mdl, nn.Module):
            try:
                _dtype = None
                _ac = None
                try:
                    _fp = next(_mdl.parameters())
                    if _fp.dtype == torch.float16:
                        _dtype = torch.float16
                except StopIteration:
                    pass
                if "autocast" in stmt and "float16" in stmt:
                    _ac = torch.float16
                    _dtype = None
                elif "autocast" in stmt and "bfloat16" in stmt:
                    _ac = torch.bfloat16
                    _dtype = None
                sanity_check(label, _mdl, input_dtype=_dtype, autocast_dtype=_ac)
            except Exception as e:
                print(f"  ⚠ Sanity skip: {type(e).__name__}: {e}")


@contextmanager
def timed(label):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    SETUP_TIMES[label] = elapsed
    print(f"  [{label}] setup: {elapsed:.2f} s")


def fresh_model(
    patch_head_fp16: bool = False, patch_rope: bool = False, fullgraph: bool = False
) -> nn.Module:
    """Return a fresh DA3 depth model with cached weights."""
    _api = DepthAnything3(model_name=MODEL_NAME)
    m = DA3Depth(_api)
    m.load_state_dict(_BASE_STATE_DICT, strict=True)
    if patch_head_fp16:
        _patch_da3_head_fp16(m.net)
    m = m.to(device).eval()
    if patch_rope:
        _patch_rope_for_compile(m, dummy_input)
    if fullgraph:
        _patch_fullgraph_forward(m)
    return m


# ── DA3 head FP16 patch ──────────────────────────────────────────────────
from depth_anything_3.model.da3 import DepthAnything3Net as _DA3Net

_da3_orig_forward = _DA3Net.forward
_dpt_orig_activation = None


def _safe_activation(self, x, activation="linear"):
    act = activation.lower() if isinstance(activation, str) else activation
    if act in ("exp", "expp1", "expm1", "softplus"):
        x32 = x.float()
        if act == "exp":
            out = torch.exp(x32)
        elif act == "expp1":
            out = torch.exp(x32) + 1
        elif act == "expm1":
            out = torch.expm1(x32)
        else:
            out = torch.nn.functional.softplus(x32)
        return out.to(x.dtype)
    return _dpt_orig_activation(self, x, activation)


def _da3_forward_fp16(self, x, **kw):
    extrinsics = kw.get("extrinsics")
    intrinsics = kw.get("intrinsics")
    export_feat_layers = kw.get("export_feat_layers", [])
    ref_view_strategy = kw.get("ref_view_strategy", "saddle_balanced")
    use_ray_pose = kw.get("use_ray_pose", False)
    infer_gs = kw.get("infer_gs", False)

    if extrinsics is not None:
        with torch.autocast(device_type=x.device.type, enabled=False):
            cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
    else:
        cam_token = None

    feats, aux_feats = self.backbone(
        x,
        cam_token=cam_token,
        export_feat_layers=export_feat_layers,
        ref_view_strategy=ref_view_strategy,
    )
    H, W = x.shape[-2], x.shape[-1]

    output = self._process_depth_head(feats, H, W)
    if use_ray_pose:
        output = self._process_ray_pose_estimation(output, H, W)
    else:
        output = self._process_camera_estimation(feats, H, W, output)
    if infer_gs:
        output = self._process_gs_head(
            feats, H, W, output, x, kw.get("extrinsics"), kw.get("intrinsics")
        )
    output = self._process_mono_sky_estimation(output)
    output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)
    return output


def _patch_da3_head_fp16(net):
    global _dpt_orig_activation
    import types
    from depth_anything_3.model.dpt import DPT as _DPTClass

    if _dpt_orig_activation is None:
        _dpt_orig_activation = _DPTClass._apply_activation_single
    net.forward = types.MethodType(_da3_forward_fp16, net)
    net.head._apply_activation_single = types.MethodType(_safe_activation, net.head)


# ── RoPE compile patch ───────────────────────────────────────────────────
def _patch_rope_for_compile(m, x):
    """Pre-compute RoPE caches and eliminate graph breaks for torch.compile."""
    import types
    from depth_anything_3.model.dinov2.layers.rope import (
        PositionGetter,
        RotaryPositionEmbedding2D,
    )

    net = m.net if hasattr(m, "net") else m

    # Pre-warm all caches
    with torch.no_grad():
        _ = m(x)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _ = m(x)

    # 1. Freeze PositionGetter
    for mod in net.modules():
        if isinstance(mod, PositionGetter):
            (h, w), pos = next(iter(mod.position_cache.items()))
            _frozen_pos = pos.clone()
            _fh, _fw = h, w

            def _frozen_pg(
                self, batch_size, height, width, device, _p=_frozen_pos, _h=_fh, _w=_fw
            ):
                return (
                    _p.to(device).view(1, _h * _w, 2).expand(batch_size, -1, -1).clone()
                )

            mod.__call__ = types.MethodType(_frozen_pg, mod)

    # 2. Freeze RoPE freq cache
    for mod in net.modules():
        if isinstance(mod, RotaryPositionEmbedding2D):
            _fc = {}
            for cache_key, (cos_c, sin_c) in mod.frequency_cache.items():
                dim, seq, dev, dt = cache_key
                norm_dev = torch.device(
                    dev.type, dev.index if dev.index is not None else 0
                )
                _fc[(dim, seq, norm_dev, dt)] = (cos_c.clone(), sin_c.clone())
            _max_seq = max(k[1] for k in _fc)
            # Pre-compute float16 versions (autocast keeps RoPE in FP32)
            for (dim, seq, dev, dt), (cos_c, sin_c) in list(_fc.items()):
                fp16_key = (dim, seq, dev, torch.float16)
                if fp16_key not in _fc:
                    _fc[fp16_key] = (cos_c.to(torch.float16), sin_c.to(torch.float16))

            def _frozen_rope_fwd(self, tokens, positions, _cache=_fc, _seq=_max_seq):
                feature_dim = tokens.size(-1) // 2
                dev = tokens.device
                norm_dev = torch.device(
                    dev.type, dev.index if dev.index is not None else 0
                )
                ck = (feature_dim, _seq, norm_dev, tokens.dtype)
                cos_comp, sin_comp = _cache[ck]
                v, h = tokens.chunk(2, dim=-1)
                v = self._apply_1d_rope(v, positions[..., 0], cos_comp, sin_comp)
                h = self._apply_1d_rope(h, positions[..., 1], cos_comp, sin_comp)
                return torch.cat((v, h), dim=-1)

            mod.forward = types.MethodType(_frozen_rope_fwd, mod)

    # 3. Freeze _prepare_rope - avoid CPU tensor creation during CUDA graph capture
    vit = net.backbone.pretrained
    if vit.rope is not None:
        B, S, _, H, W = x.shape
        _fpos, _fpos_nd = vit._prepare_rope(B, S, H, W, x.device)
        # Positions are int64 indices (for F.embedding) - keep them as-is
        _fpos_frozen = _fpos.clone()
        _fpos_nd_frozen = _fpos_nd.clone()

        def _frozen_prepare_rope(
            self, B, S, H, W, device, _p=_fpos_frozen, _pnd=_fpos_nd_frozen
        ):
            return _p.to(device).clone(), _pnd.to(device).clone()

        vit._prepare_rope = types.MethodType(_frozen_prepare_rope, vit)

    print("  RoPE patched: caches frozen, graph breaks eliminated")


# ── Fullgraph forward patch ──────────────────────────────────────────────
def _patch_fullgraph_forward(m):
    """Replace DA3Depth.forward to avoid addict.Dict for fullgraph compile."""
    import types
    from depth_anything_3.model.utils.head_utils import (
        create_uv_grid,
        position_grid_to_embed,
    )

    def _add_pos_embed_safe(self, x, W, H, ratio=0.1):
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.to(x.dtype)
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    m.net.head._add_pos_embed = types.MethodType(_add_pos_embed_safe, m.net.head)

    def _fullgraph_fwd(self, x):
        net = self.net
        feats, _ = net.backbone(
            x,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy="saddle_balanced",
        )
        H, W = x.shape[-2], x.shape[-1]
        B, S, N, C = feats[0][0].shape
        flat_feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        out_dict = net.head._forward_impl(flat_feats, H, W, patch_start_idx=0)
        depth = out_dict["depth"].reshape(B, S, *out_dict["depth"].shape[1:])
        if net.cam_dec is not None:
            cam_feat = feats[-1][1]
            _pose = net.cam_dec.backbone(cam_feat)
        return depth

    m.forward = types.MethodType(_fullgraph_fwd, m)
    print("  Fullgraph forward patched: addict.Dict bypassed")


def cleanup():
    torch._dynamo.reset()
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


def sanity_check(
    name, model_or_out, *, input_dtype=None, autocast_dtype=None, atol_mse=0.5
):
    if isinstance(model_or_out, (torch.Tensor, np.ndarray)):
        out = torch.as_tensor(model_or_out).float()
    else:
        inp = SANITY_INPUT
        if input_dtype is not None:
            inp = inp.to(dtype=input_dtype)
        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    out = model_or_out(inp)
            else:
                out = model_or_out(inp)
        out = out.float()

    ref = SANITY_DEPTH_REF.float()
    mse = ((out.cpu() - ref.cpu()) ** 2).mean().item()

    if mse < atol_mse:
        print(f"  ✓ Sanity ({name}): depth MSE = {mse:.6f}")
    elif mse < 5.0:
        print(
            f"  ⚠ Sanity ({name}): depth MSE = {mse:.6f}  (above {atol_mse} threshold)"
        )
    else:
        print(f"  ✗ Sanity ({name}): depth MSE = {mse:.6f}  - DEPTH DIVERGED")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(SANITY_IMG)
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(ref[0, 0].cpu().numpy(), cmap="inferno")
        axes[1].set_title("Baseline depth")
        axes[1].axis("off")
        axes[2].imshow(out[0, 0].cpu().numpy(), cmap="inferno")
        axes[2].set_title(f"{name} depth")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()
        plt.close(fig)


print("Ready.")
```

</details>
    Ready.
    

### GPU warmup

GPUs start in a low-power state with throttled clocks. We run 50 forward passes
to force clocks up and stabilize cuDNN plan caches before any timing begins.
`bench()` also runs 10 untimed iterations before each timed section.


```python
print("Warming up GPU...")
with torch.no_grad():
    for _ in range(50):
        _ = model(dummy_input)
torch.cuda.synchronize()
print("GPU warm - clocks boosted and stable.")
```

    Warming up GPU...
    GPU warm — clocks boosted and stable.
    

---
## 0 – Baseline (eager)

All extras disabled: no TF32, no cudnn.benchmark, no compilation. Pure eager
PyTorch with `torch.no_grad()`. This is our reference point.


```python
reset_backends()

with torch.no_grad():
    _out = model(dummy_input)
print(f"Baseline output shape: {_out.shape}")

bench(
    "0 - Baseline (eager)",
    "with torch.no_grad(): model(x)",
    {"model": model, "x": dummy_input},
)
```

    Baseline output shape: torch.Size([1, 1, 504, 504])
      0 - Baseline (eager): 173.970 ms  (spread: 0.489 ms  rounds: 174.01, 173.97, 174.06, 173.57, 173.77)
      ✓ Sanity (0 - Baseline (eager)): depth MSE = 0.000000
    

---
## 1 – cudnn.benchmark

Auto-tune cuDNN convolution algorithms. Mostly helps CNNs; the DA3
depth head has a few convs so there's a small benefit.


```python
reset_backends()
torch.backends.cudnn.benchmark = True

bench(
    "1 - cudnn.benchmark",
    "with torch.no_grad(): model(x)",
    {"model": model, "x": dummy_input},
)
reset_backends()
```

      1 - cudnn.benchmark: 174.301 ms  (spread: 0.404 ms  rounds: 174.16, 173.99, 174.32, 174.30, 174.40)
      ✓ Sanity (1 - cudnn.benchmark): depth MSE = 0.000000
    

---
## 2 – channels_last

NHWC memory layout. Helps cuDNN conv kernels on Tensor-Core GPUs.
ViTs are mostly linear layers, so the gain is limited to the DPT head's
convolutions and the patch embedding.


```python
reset_backends()

_m = fresh_model()
# channels_last only applies to 4D tensors; DA3 input is 5D.
_m = _m.to(memory_format=torch.channels_last)


bench(
    "2 - channels_last",
    "with torch.no_grad(): model(x)",
    {"model": _m, "x": dummy_input},
)
cleanup()
```

    [INFO ] using MLP layer as FFN
      2 - channels_last: 175.483 ms  (spread: 0.285 ms  rounds: 175.49, 175.28, 175.52, 175.48, 175.24)
      ✓ Sanity (2 - channels_last): depth MSE = 0.000000
    

---
## 3 – inference_mode

Stricter than `no_grad`: also disables autograd version counters
and view tracking. A few percent faster, zero accuracy cost.


```python
reset_backends()

bench(
    "3 - inference_mode",
    "with torch.inference_mode(): model(x)",
    {"model": model, "x": dummy_input},
)
```

      3 - inference_mode: 171.552 ms  (spread: 0.563 ms  rounds: 171.53, 171.55, 171.80, 171.59, 171.24)
      ✓ Sanity (3 - inference_mode): depth MSE = 0.000000
    

---
## 4 – TF32 matmul

On Ampere+ GPUs, TF32 rounds float32 mantissa from 23 to 10 bits
before the hardware tensor core matmul. ~2× faster for free.
On V100 (SM 7.0) this is a **no-op**.


```python
reset_backends()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

bench(
    "4 - TF32 matmul",
    "with torch.no_grad(): model(x)",
    {"model": model, "x": dummy_input},
)
reset_backends()
```

      4 - TF32 matmul: 174.350 ms  (spread: 0.524 ms  rounds: 174.29, 174.63, 174.11, 174.47, 174.35)
      ✓ Sanity (4 - TF32 matmul): depth MSE = 0.000000
    

---
## 5 – AMP FP16

Autocast to half precision. V100 Tensor Cores do FP16 × FP16 → FP32
accumulation. This is usually the single biggest easy win.


```python
reset_backends()

bench(
    "5 - AMP FP16",
    "with torch.no_grad():\n"
    "  with torch.amp.autocast('cuda', dtype=torch.float16):\n"
    "    model(x)",
    {"model": model, "x": dummy_input},
)
```

      5 - AMP FP16: 115.303 ms  (spread: 0.174 ms  rounds: 115.28, 115.28, 115.46, 115.33, 115.30)
      ✓ Sanity (5 - AMP FP16): depth MSE = 0.000000
    

### 5b - AMP FP16 (full model, patched head)
DA3 wraps the DPT depth head in `autocast(enabled=False)`, forcing FP32 for ~28% of compute.
We monkey-patch the forward to **let AMP run through the head** while keeping `exp()` in FP32.


```python
reset_backends()

_m = fresh_model(patch_head_fp16=True)
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=torch.float16):
        _out = _m(dummy_input)
check_output("AMP FP16 (full)", _out, baseline_output, atol=0.5)

bench(
    "5b - AMP FP16 (full, patched head)",
    "with torch.no_grad():\n"
    "  with torch.amp.autocast('cuda', dtype=torch.float16):\n"
    "    model(x)",
    {"model": _m, "x": dummy_input},
)
cleanup()
```

    [INFO ] using MLP layer as FFN
      [OK] AMP FP16 (full): max diff = 0.000670 (atol=0.5)
      5b - AMP FP16 (full, patched head): 101.142 ms  (spread: 0.188 ms  rounds: 101.14, 101.12, 101.23, 101.14, 101.31)
      ✓ Sanity (5b - AMP FP16 (full, patched head)): depth MSE = 0.000000
    

── Profiling: backbone vs head compute split ─────────────────────────────



<details class="cell-collapse" markdown="1">
<summary>Show code</summary>

```python
# ── Profiling: backbone vs head compute split ─────────────────────────────
import torch.utils.benchmark as _b

_net = model.net  # DepthAnything3Net


def _run_backbone():
    with torch.no_grad():
        return _net.backbone(
            dummy_input,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy="saddle_balanced",
        )


def _run_head(feats):
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=False):  # DA3's default
            return _net._process_depth_head(feats, 504, 504)


def _run_head_amp(feats):
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16):  # Allow FP16 in head
            return _net.head(feats, 504, 504, patch_start_idx=0)


def _run_bb_amp():
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            return _net.backbone(
                dummy_input,
                cam_token=None,
                export_feat_layers=[],
                ref_view_strategy="saddle_balanced",
            )


# Warm up
_feats32 = _run_backbone()[0]
_ = _run_head(_feats32)
_feats16 = _run_bb_amp()[0]
_ = _run_head_amp(_feats16)
torch.cuda.synchronize()

# Measure FP32 components
bb32 = (
    _b.Timer(stmt="f()", globals={"f": _run_backbone})
    .blocked_autorange(min_run_time=3)
    .median
    * 1e3
)
hd32 = (
    _b.Timer(stmt="f(x)", globals={"f": _run_head, "x": _feats32})
    .blocked_autorange(min_run_time=3)
    .median
    * 1e3
)

# Measure AMP FP16 components
bb16 = (
    _b.Timer(stmt="f()", globals={"f": _run_bb_amp})
    .blocked_autorange(min_run_time=3)
    .median
    * 1e3
)
hd16_off = (
    _b.Timer(stmt="f(x)", globals={"f": _run_head, "x": _feats16})
    .blocked_autorange(min_run_time=3)
    .median
    * 1e3
)  # autocast OFF (DA3 default)
hd16_on = (
    _b.Timer(stmt="f(x)", globals={"f": _run_head_amp, "x": _feats16})
    .blocked_autorange(min_run_time=3)
    .median
    * 1e3
)  # autocast ON (patched)

print("=" * 70)
print("COMPONENT PROFILING - backbone vs depth head")
print("=" * 70)
print(f"{'Component':<35} {'FP32':>8} {'AMP FP16':>8} {'Speedup':>8}")
print("-" * 70)
print(f"{'Backbone (DinoV2-L)':<35} {bb32:>7.1f}ms {bb16:>7.1f}ms {bb32 / bb16:>7.2f}x")
print(
    f"{'Head (autocast disabled - DA3 dflt)':<35} {hd32:>7.1f}ms {hd16_off:>7.1f}ms {hd32 / hd16_off:>7.2f}x"
)
print(
    f"{'Head (autocast enabled - patched)':<35} {hd32:>7.1f}ms {hd16_on:>7.1f}ms {hd32 / hd16_on:>7.2f}x"
)
print("-" * 70)
tot32 = bb32 + hd32
tot_cur = bb16 + hd16_off
tot_pat = bb16 + hd16_on
print(
    f"{'E2E current (head FP32)':<35} {tot32:>7.1f}ms {tot_cur:>7.1f}ms {tot32 / tot_cur:>7.2f}x"
)
print(
    f"{'E2E patched (head FP16)':<35} {tot32:>7.1f}ms {tot_pat:>7.1f}ms {tot32 / tot_pat:>7.2f}x"
)
print("=" * 70)
print(f"\nHead fraction of FP32 total: {hd32 / tot32 * 100:.1f}%")
print(f"Head FP16 savings: {hd16_off - hd16_on:.1f} ms")
```

</details>
    ======================================================================
    COMPONENT PROFILING — backbone vs depth head
    ======================================================================
    Component                               FP32 AMP FP16  Speedup
    ----------------------------------------------------------------------
    Backbone (DinoV2-L)                   124.3ms    65.1ms    1.91x
    Head (autocast disabled — DA3 dflt)    48.7ms    48.8ms    1.00x
    Head (autocast enabled — patched)      48.7ms    31.7ms    1.54x
    ----------------------------------------------------------------------
    E2E current (head FP32)               173.1ms   113.9ms    1.52x
    E2E patched (head FP16)               173.1ms    96.8ms    1.79x
    ======================================================================
    
    Head fraction of FP32 total: 28.2%
    Head FP16 savings: 17.1 ms
    

---
## 6 – AMP BF16

BFloat16 has the same exponent range as FP32 (no overflow issues) but
less precision. Requires **Ampere+** Tensor Cores. On V100 this falls
back to FP32.


```python
reset_backends()

if not torch.cuda.is_bf16_supported():
    print("BF16 not supported on this GPU - skipping")
    GPU_RESULTS["6 - AMP BF16"] = float("nan")
else:
    bench(
        "6 - AMP BF16",
        "with torch.no_grad():\n"
        "  with torch.amp.autocast('cuda', dtype=torch.bfloat16):\n"
        "    model(x)",
        {"model": model, "x": dummy_input},
    )
```

      6 - AMP BF16: 250.965 ms  (spread: 0.281 ms  rounds: 250.89, 250.97, 251.09, 250.85, 251.14)
      ✓ Sanity (6 - AMP BF16): depth MSE = 0.000000
    

---
## 7 – Static FP16 (model.half())

Convert all model weights to FP16 permanently. No autocast overhead
per forward pass, but less precision control than AMP.


```python
reset_backends()

# Static FP16: DA3 has multiple forced-float32 paths (pos embed, cam_dec).
# Use fullgraph forward + pos embed patch to stay fully in float16.
try:
    _m = fresh_model(patch_head_fp16=True, fullgraph=True).half()
    _x = dummy_input.half()
    with torch.no_grad():
        _out = _m(_x)
    check_output("static fp16", _out, baseline_output, atol=0.5)
    bench("7 - Static FP16", "with torch.no_grad(): model(x)", {"model": _m, "x": _x})
except RuntimeError:
    import traceback

    traceback.print_exc()
    GPU_RESULTS["7 - Static FP16"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
      Fullgraph forward patched: addict.Dict bypassed
      [OK] static fp16: max diff = 0.000732 (atol=0.5)
      7 - Static FP16: 89.383 ms  (spread: 0.173 ms  rounds: 89.38, 89.25, 89.41, 89.24, 89.41)
      ✓ Sanity (7 - Static FP16): depth MSE = 0.000000
    

---
## 7b – SDPA backends

PyTorch's `scaled_dot_product_attention` dispatches to FlashAttention,
Memory-Efficient, or Math backends. We test each one. DinoV2's attention
already uses SDPA, so this shows which backend the GPU selects.


```python
reset_backends()

# Test individual SDPA backends
_sdpa_backends = {
    "Flash": torch.backends.cuda.flash_sdp_enabled,
    "MemEfficient": torch.backends.cuda.mem_efficient_sdp_enabled,
    "Math": torch.backends.cuda.math_sdp_enabled,
}
print(f"SDPA backends: { {k: v() for k, v in _sdpa_backends.items()} }")

for backend_name, (en_flash, en_mem, en_math) in [
    ("flash", (True, False, False)),
    ("mem_efficient", (False, True, False)),
    ("math", (False, False, True)),
]:
    try:
        torch.backends.cuda.enable_flash_sdp(en_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(en_mem)
        torch.backends.cuda.enable_math_sdp(en_math)

        with torch.no_grad():
            _ = model(dummy_input)

        bench(
            f"7b - SDPA ({backend_name})",
            "with torch.no_grad(): model(x)",
            {"model": model, "x": dummy_input},
        )
    except Exception as e:
        print(f"  SDPA {backend_name}: {type(e).__name__} - {e}")
        GPU_RESULTS[f"7b - SDPA ({backend_name})"] = float("nan")

# Restore all backends
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

    SDPA backends: {'Flash': True, 'MemEfficient': True, 'Math': True}
      SDPA flash: RuntimeError — No available kernel. Aborting execution.
      7b - SDPA (mem_efficient): 174.502 ms  (spread: 0.275 ms  rounds: 174.57, 174.58, 174.31, 174.36, 174.50)
      ✓ Sanity (7b - SDPA (mem_efficient)): depth MSE = 0.000000
      7b - SDPA (math): 210.918 ms  (spread: 0.204 ms  rounds: 210.92, 210.89, 210.92, 210.92, 211.10)
      ✓ Sanity (7b - SDPA (math)): depth MSE = 0.000000
    

---
## 8 – torch.compile (default)

Fast compile, good speed. Best compile-time / run-time trade-off.
Requires Triton (Linux only).


```python
reset_backends()

if not _check_compile_support():
    print("torch.compile not supported - skipping")
    GPU_RESULTS["8 - compile (default)"] = float("nan")
else:
    _m = fresh_model()
    with timed("compile default"):
        _m = torch.compile(_m, mode="default")
        with torch.no_grad():
            _ = _m(dummy_input)  # trigger compilation

    bench(
        "8 - compile (default)",
        "with torch.no_grad(): model(x)",
        {"model": _m, "x": dummy_input},
    )
    cleanup()
```

    [INFO ] using MLP layer as FFN
    

    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0] Graph break from `Tensor.item()`, consider setting:
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     torch._dynamo.config.capture_scalar_outputs = True
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0] or:
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0] to include these operations in the captured graph.
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0] 
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0] Graph break: from user code at:
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/tmp/ipykernel_668996/181713102.py", line 19, in forward
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     out = self.net(x, export_feat_layers=[])
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/da3.py", line 132, in forward
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     feats, aux_feats = self.backbone(
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/dinov2.py", line 60, in forward
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     return self.pretrained.get_intermediate_layers(
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 379, in get_intermediate_layers
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 338, in _get_intermediate_layers_not_chunked
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     x = self.process_attention(x, blk, "local", pos=l_pos)
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 364, in process_attention
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     x = block(x, pos=pos, attn_mask=attn_mask)
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/block.py", line 101, in forward
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/block.py", line 79, in attn_residual_func
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/attention.py", line 57, in forward
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     q = self.rope(q, pos)
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]   File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/rope.py", line 183, in forward
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0]     max_position = int(positions.max()) + 1
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0] 
    W0327 18:10:43.018000 668996 torch/_dynamo/variables/tensor.py:776] [1/0] 
    

      [compile default] setup: 21.42 s
      8 - compile (default): 145.039 ms  (spread: 0.448 ms  rounds: 144.73, 144.87, 145.04, 145.14, 145.17)
      ✓ Sanity (8 - compile (default)): depth MSE = 0.000000
    

---
## 9 – torch.compile (reduce-overhead)

Uses CUDA Graphs internally - best for batch=1.
Requires Triton (Linux only).


```python
reset_backends()

if not _check_compile_support():
    print("torch.compile not supported - skipping")
    GPU_RESULTS["9 - compile (reduce-overhead)"] = float("nan")
else:
    _m = fresh_model(patch_rope=True)
    try:
        with timed("compile reduce-overhead"):
            _m = torch.compile(_m, mode="reduce-overhead")
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                _ = _m(dummy_input)  # trigger compilation

        bench(
            "9 - compile (reduce-overhead)",
            "torch.compiler.cudagraph_mark_step_begin()\n"
            "with torch.no_grad(): model(x)",
            {"model": _m, "x": dummy_input, "torch": torch},
        )
    except Exception as e:
        print(f"compile reduce-overhead failed: {type(e).__name__}: {e}")
        GPU_RESULTS["9 - compile (reduce-overhead)"] = float("nan")
    cleanup()
```

    [INFO ] using MLP layer as FFN
      RoPE patched: caches frozen, graph breaks eliminated
      [compile reduce-overhead] setup: 53.92 s
      9 - compile (reduce-overhead): 138.564 ms  (spread: 0.235 ms  rounds: 138.38, 138.53, 138.56, 138.62, 138.59)
      ✓ Sanity (9 - compile (reduce-overhead)): depth MSE = 0.000000
    

---
## 10 – torch.compile (max-autotune)

Slowest compile; tries more kernel variants.
Requires Triton (Linux only).


```python
reset_backends()

if not _check_compile_support():
    print("torch.compile not supported - skipping")
    GPU_RESULTS["10 - compile (max-autotune)"] = float("nan")
else:
    _m = fresh_model(patch_rope=True)
    try:
        with timed("compile max-autotune"):
            _m = torch.compile(_m, mode="max-autotune")
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                _ = _m(dummy_input)  # trigger compilation

        bench(
            "10 - compile (max-autotune)",
            "torch.compiler.cudagraph_mark_step_begin()\n"
            "with torch.no_grad(): model(x)",
            {"model": _m, "x": dummy_input, "torch": torch},
        )
    except Exception as e:
        print(f"compile max-autotune failed: {type(e).__name__}: {e}")
        GPU_RESULTS["10 - compile (max-autotune)"] = float("nan")
    cleanup()
```

    [INFO ] using MLP layer as FFN
      RoPE patched: caches frozen, graph breaks eliminated
      [compile max-autotune] setup: 59.69 s
      10 - compile (max-autotune): 138.306 ms  (spread: 0.231 ms  rounds: 138.22, 138.23, 138.44, 138.31, 138.45)
      ✓ Sanity (10 - compile (max-autotune)): depth MSE = 0.000000
    

---
## 11 – compile fullgraph + freeze

Zero graph breaks + Inductor tuning. If the model has no dynamic
control flow, the compiler can optimize the entire graph at once.


```python
reset_backends()

if not _check_compile_support():
    print("torch.compile not supported - skipping")
    GPU_RESULTS["11 - compile fullgraph+freeze"] = float("nan")
else:
    _m = fresh_model(patch_rope=True, fullgraph=True)
    try:
        with timed("compile fullgraph"):
            _m = torch.compile(_m, mode="reduce-overhead", fullgraph=True)
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                _ = _m(dummy_input)

        bench(
            "11 - compile fullgraph+freeze",
            "torch.compiler.cudagraph_mark_step_begin()\n"
            "with torch.no_grad(): model(x)",
            {"model": _m, "x": dummy_input, "torch": torch},
        )
    except Exception as e:
        print(f"fullgraph compile failed (graph breaks): {e}")
        GPU_RESULTS["11 - compile fullgraph+freeze"] = float("nan")
    cleanup()
```

    [INFO ] using MLP layer as FFN
      RoPE patched: caches frozen, graph breaks eliminated
      Fullgraph forward patched: addict.Dict bypassed
      [compile fullgraph] setup: 51.40 s
      11 - compile fullgraph+freeze: 117.110 ms  (spread: 1.954 ms  rounds: 116.36, 117.09, 117.11, 117.31, 118.32)
      ✓ Sanity (11 - compile fullgraph+freeze): depth MSE = 0.000000
    

---
## 12 – JIT Trace + optimize

TorchScript graph fusion via `torch.jit.trace`. Records the computation
graph and applies graph-level optimizations. May fail on models with
dynamic control flow or dict outputs.


```python
reset_backends()

_m = fresh_model()
try:
    with torch.no_grad():
        _traced = torch.jit.trace(_m, dummy_input)
    _traced = torch.jit.optimize_for_inference(_traced)
    with torch.no_grad():
        _out = _traced(dummy_input)
    check_output("jit trace", _out, baseline_output, atol=0.5)
    bench(
        "12 - JIT Trace",
        "with torch.no_grad(): model(x)",
        {"model": _traced, "x": dummy_input},
    )
except Exception as e:
    print(f"JIT Trace failed: {type(e).__name__}: {e}")
    GPU_RESULTS["12 - JIT Trace"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
      [OK] jit trace: max diff = 0.000000 (atol=0.5)
      12 - JIT Trace: 127.224 ms  (spread: 0.168 ms  rounds: 127.07, 127.22, 127.23, 127.24, 127.10)
      ✓ Sanity (12 - JIT Trace): depth MSE = 0.000000
    

---
## 13 – Manual CUDA Graphs

Pre-record a GPU kernel sequence and replay it with zero CPU overhead.
Best for static input shapes. Requires capturing the computation graph
in a controlled environment.


```python
reset_backends()

# Manual CUDA Graphs: needs fullgraph forward (no addict.Dict), patched RoPE,
# and patched head + pos embed for clean graph capture.
try:
    _m = fresh_model(patch_head_fp16=True, patch_rope=True, fullgraph=True).half()
    _x16 = dummy_input.half()

    # Warm up - run a few times to stabilize allocations
    for _ in range(3):
        with torch.no_grad():
            _ = _m(_x16)
    torch.cuda.synchronize()

    # Capture CUDA Graph with private pool to avoid allocator conflicts
    _static_input = _x16.clone()
    _graph = torch.cuda.CUDAGraph()
    _s = torch.cuda.Stream()
    _s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_s):
        with torch.no_grad():
            _graph.capture_begin(pool=torch.cuda.graph_pool_handle())
            _static_output = _m(_static_input)
            _graph.capture_end()

    # Verify
    torch.cuda.current_stream().wait_stream(_s)
    _static_input.copy_(dummy_input.half())
    _graph.replay()
    torch.cuda.synchronize()
    check_output("CUDA Graphs", _static_output, baseline_output, atol=0.5)

    bench(
        "13 - CUDA Graphs",
        "static_input.copy_(x); graph.replay(); torch.cuda.synchronize()",
        {"static_input": _static_input, "x": _x16, "graph": _graph, "torch": torch},
    )
except Exception as e:
    print(f"CUDA Graphs failed: {type(e).__name__}: {e}")
    GPU_RESULTS["13 - CUDA Graphs"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
      RoPE patched: caches frozen, graph breaks eliminated
      Fullgraph forward patched: addict.Dict bypassed
      [OK] CUDA Graphs: max diff = 0.000648 (atol=0.5)
      13 - CUDA Graphs: 68.995 ms  (spread: 0.084 ms  rounds: 69.07, 69.00, 68.98, 68.98, 68.99)
    

---
## 14 – torchao int8 weight-only

Store weights as int8, dequantize on-the-fly in the matmul kernel.
2× memory reduction. With `torch.compile` the dequant fuses into
the matmul for near-FP16 speed.


```python
reset_backends()

try:
    from torchao.quantization import quantize_, int8_weight_only

    _m = fresh_model()
    quantize_(_m, int8_weight_only())
    with torch.no_grad():
        _out = _m(dummy_input)
    check_output("torchao int8wo", _out, baseline_output, atol=1.0)

    bench(
        "14 - torchao int8 weight-only",
        "with torch.no_grad(): model(x)",
        {"model": _m, "x": dummy_input},
    )
except Exception as e:
    print(f"torchao int8 weight-only failed: {e}")
    GPU_RESULTS["14 - torchao int8 weight-only"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
      [OK] torchao int8wo: max diff = 0.002756 (atol=1.0)
      14 - torchao int8 weight-only: 186.573 ms  (spread: 0.343 ms  rounds: 186.57, 186.57, 186.33, 186.67, 186.62)
      ✓ Sanity (14 - torchao int8 weight-only): depth MSE = 0.000005
    

---
## 15 – torchao int8 dynamic quantization

Quantize both weights and activations to int8 dynamically.
Stronger accuracy preservation than weight-only, with potential
speedup from int8 matmul on supported hardware.


```python
reset_backends()

try:
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight

    _m = fresh_model()
    quantize_(_m, int8_dynamic_activation_int8_weight())
    with torch.no_grad():
        _out = _m(dummy_input)
    check_output("torchao int8dyn", _out, baseline_output, atol=1.0)

    bench(
        "15 - torchao int8 dynamic",
        "with torch.no_grad(): model(x)",
        {"model": _m, "x": dummy_input},
    )
except Exception as e:
    print(f"torchao int8 dynamic failed: {e}")
    GPU_RESULTS["15 - torchao int8 dynamic"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
      [OK] torchao int8dyn: max diff = 0.012926 (atol=1.0)
      15 - torchao int8 dynamic: 368.720 ms  (spread: 1.399 ms  rounds: 369.83, 368.72, 368.43, 368.90, 368.66)
      ✓ Sanity (15 - torchao int8 dynamic): depth MSE = 0.000020
    

---
## 16 – torchao int4 weight-only

4-bit weight quantization via custom Triton kernels. 4× memory reduction
vs FP16. Best for memory-bound workloads.


```python
reset_backends()

try:
    from torchao.quantization import quantize_, int4_weight_only

    # int4_weight_only uses TensorCoreTiledLayout which requires SM >= 8.0 (Ampere+).
    # On V100 (SM 7.0), try with bfloat16 model (V100 emulates bf16 via FP32 datapath).
    cap = torch.cuda.get_device_capability()
    if cap < (8, 0):
        print(f"GPU SM {cap[0]}.{cap[1]}: int4 TensorCoreTiledLayout needs SM >= 8.0")
        print("  Trying with bf16 model (emulated on V100)...")
        _m = fresh_model().to(torch.bfloat16)
    else:
        _m = fresh_model()
    quantize_(_m, int4_weight_only())
    _x = dummy_input.to(_m.net.backbone.pretrained.cls_token.dtype)
    with torch.no_grad():
        _out = _m(_x)
    check_output("torchao int4wo", _out, baseline_output, atol=2.0)

    bench(
        "16 - torchao int4 weight-only",
        "with torch.no_grad(): model(x)",
        {"model": _m, "x": _x},
    )
except Exception as e:
    print(f"torchao int4 weight-only failed: {type(e).__name__}: {e}")
    GPU_RESULTS["16 - torchao int4 weight-only"] = float("nan")
cleanup()
```

    GPU SM 7.0: int4 TensorCoreTiledLayout needs SM >= 8.0
      Trying with bf16 model (emulated on V100)...
    [INFO ] using MLP layer as FFN
    torchao int4 weight-only failed: RuntimeError: CUDA error: named symbol not found
    CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
    For debugging consider passing CUDA_LAUNCH_BLOCKING=1
    Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
    
    

---
## 17 – ONNX Runtime CUDA EP

Export to ONNX, run via ONNX Runtime with CUDA Execution Provider.
ORT applies graph-level optimizations (constant folding, op fusion).


```python
reset_backends()

try:
    import onnxruntime as ort

    _m = fresh_model()

    # Export to ONNX on CPU to avoid CUDA issues during tracing
    with timed("ONNX export"):
        _cpu_m = _m.cpu()
        _dummy_cpu = dummy_input.cpu()
        torch.onnx.export(
            _cpu_m,
            _dummy_cpu,
            "model.onnx",
            input_names=["input"],
            output_names=["depth"],
            dynamic_axes={"input": {0: "batch"}, "depth": {0: "batch"}},
            opset_version=17,
        )
        del _cpu_m
    _m.to(device)

    with timed("ORT session"):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess = ort.InferenceSession(
            "model.onnx",
            opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    _np_input = dummy_input.cpu().numpy()
    _ort_out = sess.run(None, {"input": _np_input})[0]
    check_output(
        "ORT CUDA", torch.tensor(_ort_out).to(device), baseline_output, atol=0.5
    )

    bench(
        "17 - ONNX Runtime CUDA EP",
        "sess.run(None, {'input': np_input})",
        {"sess": sess, "np_input": _np_input},
        sanity=False,
    )
except Exception as e:
    print(f"ORT CUDA EP failed: {type(e).__name__}: {e}")
    GPU_RESULTS["17 - ONNX Runtime CUDA EP"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
    ORT CUDA EP failed: UnsupportedOperatorError: Exporting the operator 'aten::cartesian_prod' to ONNX opset version 17 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues.
    

---
## 18 – ORT CUDA EP + IO Binding

Zero-copy GPU inference: bind CUDA tensors directly to ORT inputs/outputs
so data never leaves the GPU.


```python
reset_backends()

try:
    import onnxruntime as ort

    if not os.path.exists("model.onnx"):
        print("model.onnx not found - run cell 17 first")
        GPU_RESULTS["18 - ORT CUDA EP + IO Bind"] = float("nan")
    else:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess = ort.InferenceSession(
            "model.onnx", opts, providers=["CUDAExecutionProvider"]
        )

        _io = sess.io_binding()
        _gpu_input = dummy_input.contiguous()
        _gpu_output = torch.empty_like(baseline_output)
        _io.bind_input(
            "input",
            "cuda",
            0,
            np.float32,
            list(_gpu_input.shape),
            _gpu_input.data_ptr(),
        )
        _io.bind_output(
            "depth",
            "cuda",
            0,
            np.float32,
            list(_gpu_output.shape),
            _gpu_output.data_ptr(),
        )

        sess.run_with_iobinding(_io)
        check_output("ORT IO Bind", _gpu_output, baseline_output, atol=0.5)

        bench(
            "18 - ORT CUDA EP + IO Bind",
            "sess.run_with_iobinding(io)",
            {"sess": sess, "io": _io},
            sanity=False,
        )
except Exception as e:
    print(f"ORT IO Binding failed: {type(e).__name__}: {e}")
    GPU_RESULTS["18 - ORT CUDA EP + IO Bind"] = float("nan")
cleanup()
```

    model.onnx not found — run cell 17 first
    

---
## 19 – ORT TensorRT EP

TRT engine via ORT. Requires TensorRT libraries and **SM ≥ 8.0** (Ampere+).
On V100 (SM 7.0) this may fail.


```python
reset_backends()

try:
    import onnxruntime as ort

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        print("TensorRT EP not available")
        GPU_RESULTS["19 - ORT TensorRT EP"] = float("nan")
    elif not os.path.exists("model.onnx"):
        print("model.onnx not found - run cell 17 first")
        GPU_RESULTS["19 - ORT TensorRT EP"] = float("nan")
    else:
        with timed("TRT engine build"):
            sess = ort.InferenceSession(
                "model.onnx",
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            )
        _np_in = dummy_input.cpu().numpy()
        _out = sess.run(None, {"input": _np_in})[0]
        check_output(
            "ORT TRT", torch.tensor(_out).to(device), baseline_output, atol=1.0
        )
        bench(
            "19 - ORT TensorRT EP",
            "sess.run(None, {'input': np_input})",
            {"sess": sess, "np_input": _np_in},
            sanity=False,
        )
except Exception as e:
    print(f"ORT TensorRT EP failed: {type(e).__name__}: {e}")
    GPU_RESULTS["19 - ORT TensorRT EP"] = float("nan")
cleanup()
```

    model.onnx not found — run cell 17 first
    

---
## 20 – Torch-TensorRT

Direct TRT compilation from PyTorch. Requires `torch_tensorrt` and
**SM ≥ 8.0**. On V100 this typically fails.


```python
reset_backends()

try:
    import torch_tensorrt

    _m = fresh_model(patch_head_fp16=True).half()
    _x16 = dummy_input.half()
    with timed("Torch-TRT compile"):
        _trt = torch_tensorrt.compile(
            _m,
            inputs=[torch_tensorrt.Input(shape=dummy_input.shape, dtype=torch.float16)],
            enabled_precisions={torch.float16},
        )
    with torch.no_grad():
        _out = _trt(_x16)
    check_output("Torch-TRT", _out, baseline_output, atol=1.0)
    bench(
        "20 - Torch-TensorRT",
        "with torch.no_grad(): model(x)",
        {"model": _trt, "x": _x16},
    )
except Exception as e:
    print(f"Torch-TRT failed: {type(e).__name__}: {e}")
    GPU_RESULTS["20 - Torch-TensorRT"] = float("nan")
cleanup()
```

    Unable to import quantization op. Please install modelopt library (https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#installation) to add support for compiling quantized models
    

    [INFO ] using MLP layer as FFN
    

    W0327 18:21:33.784000 668996 torch/fx/experimental/symbolic_shapes.py:5124] [0/0] failed during evaluate_expr(u0 + 1, hint=None, size_oblivious=False, forcing_spec=False
    E0327 18:21:33.786000 668996 torch/fx/experimental/recording.py:298] [0/0] failed while running evaluate_expr(*(u0 + 1, None), **{'fx_node': False})
    

    Torch-TRT failed: UserError: Consider annotating your code using torch._check*(). Could not extract specialized integer from data-dependent expression u0 + 1 (unhinted: u0 + 1).  (Size-like symbols: none)
    
    Potential framework code culprit (scroll up for full backtrace):
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/_dynamo/variables/tensor.py", line 1117, in evaluate_expr
        return guard_scalar(self.sym_num)
    
    For more information, run with TORCH_LOGS="dynamic"
    For extended logs when we create symbols, also add TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u0"
    If you suspect the guard was triggered from C++, add TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
    For more debugging help, see https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit?usp=sharing
    
    User Stack (most recent call last):
      (snipped, see stack below for prefix)
      File "/tmp/ipykernel_668996/181713102.py", line 19, in forward
        out = self.net(x, export_feat_layers=[])
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/tmp/ipykernel_668996/1460543228.py", line 152, in _da3_forward_fp16
        feats, aux_feats = self.backbone(
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/dinov2.py", line 60, in forward
        return self.pretrained.get_intermediate_layers(
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 379, in get_intermediate_layers
        outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 338, in _get_intermediate_layers_not_chunked
        x = self.process_attention(x, blk, "local", pos=l_pos)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 364, in process_attention
        x = block(x, pos=pos, attn_mask=attn_mask)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/block.py", line 101, in forward
        x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/block.py", line 79, in attn_residual_func
        return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/attention.py", line 57, in forward
        q = self.rope(q, pos)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/rope.py", line 184, in forward
        cos_comp, sin_comp = self._compute_frequency_components(
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/rope.py", line 102, in _compute_frequency_components
        if cache_key not in self.frequency_cache:
    
    For C++ stack trace, run with TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
    For more information about this error, see: https://pytorch.org/docs/main/generated/exportdb/index.html#constrain-as-size-example
    
    from user code:
       File "/tmp/ipykernel_668996/181713102.py", line 19, in forward
        out = self.net(x, export_feat_layers=[])
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/tmp/ipykernel_668996/1460543228.py", line 152, in _da3_forward_fp16
        feats, aux_feats = self.backbone(
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/dinov2.py", line 60, in forward
        return self.pretrained.get_intermediate_layers(
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 379, in get_intermediate_layers
        outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 338, in _get_intermediate_layers_not_chunked
        x = self.process_attention(x, blk, "local", pos=l_pos)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/vision_transformer.py", line 364, in process_attention
        x = block(x, pos=pos, attn_mask=attn_mask)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/block.py", line 101, in forward
        x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/block.py", line 79, in attn_residual_func
        return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/attention.py", line 57, in forward
        q = self.rope(q, pos)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/rope.py", line 184, in forward
        cos_comp, sin_comp = self._compute_frequency_components(
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dinov2/layers/rope.py", line 102, in _compute_frequency_components
        if cache_key not in self.frequency_cache:
    
    Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
    
    

---
## 21 – Triton fused softmax

Replace the attention softmax in every transformer block with a custom
Triton kernel. Reduces memory traffic by fusing the softmax into a
single kernel launch.


```python
reset_backends()

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        inp = tl.load(
            input_ptr + row * n_cols + offsets, mask=mask, other=-float("inf")
        )
        row_max = tl.max(inp, axis=0)
        safe = inp - row_max
        num = tl.exp(safe)
        den = tl.sum(num, axis=0)
        out = num / den
        tl.store(output_ptr + row * n_cols + offsets, out, mask=mask)

    class _TritonSoftmax(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            n_cols = x.shape[-1]
            BLOCK = triton.next_power_of_2(n_cols)
            out = torch.empty_like(x)
            _fused_softmax_kernel[(x.numel() // n_cols,)](out, x, n_cols, BLOCK=BLOCK)
            return out

    def _triton_softmax(x, dim=-1):
        if dim != -1 and dim != x.ndim - 1:
            return torch.nn.functional.softmax(x, dim=dim)
        return _TritonSoftmax.apply(x)

    _m = fresh_model()
    # Patch DinoV2 attention blocks
    _patched = 0
    for module in _m.modules():
        if hasattr(module, "attn_drop") and hasattr(module, "qkv"):

            def _make_patched(mod):
                def _fwd(x, **kwargs):  # Accept extra kwargs like pos
                    B, N, C = x.shape
                    qkv = (
                        mod.qkv(x)
                        .reshape(B, N, 3, mod.num_heads, C // mod.num_heads)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv.unbind(0)
                    scale = (C // mod.num_heads) ** -0.5
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = _triton_softmax(attn, dim=-1)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = mod.proj(x)
                    if hasattr(mod, "proj_drop") and isinstance(
                        mod.proj_drop, nn.Module
                    ):
                        x = mod.proj_drop(x)
                    return x

                return _fwd

            module.forward = _make_patched(module)
            _patched += 1

    print(f"Patched {_patched} attention blocks with Triton softmax")
    with torch.no_grad():
        _out = _m(dummy_input)
    check_output("triton softmax", _out, baseline_output, atol=0.5)

    bench(
        "21 - Triton fused softmax",
        "with torch.no_grad(): model(x)",
        {"model": _m, "x": dummy_input},
    )
except Exception as e:
    print(f"Triton fused softmax failed: {type(e).__name__}: {e}")
    GPU_RESULTS["21 - Triton fused softmax"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
    Patched 28 attention blocks with Triton softmax
      [OK] triton softmax: max diff = 0.093633 (atol=0.5)
      21 - Triton fused softmax: 165.728 ms  (spread: 0.407 ms  rounds: 165.32, 165.47, 165.73, 165.73, 165.73)
      ⚠ Sanity (21 - Triton fused softmax): depth MSE = 0.559205  (above 0.5 threshold)
    

---
## 22 – compile fullgraph + FP16

Combine `torch.compile(fullgraph=True)` with FP16 for maximum
compiler optimization on half-precision model.


```python
reset_backends()

if not _check_compile_support():
    print("torch.compile not supported - skipping")
    GPU_RESULTS["22 - compile fullgraph+FP16"] = float("nan")
else:
    _m = fresh_model(patch_head_fp16=True, patch_rope=True, fullgraph=True)
    try:
        with timed("compile fullgraph+FP16"):
            _m = torch.compile(_m, mode="reduce-overhead", fullgraph=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    torch.compiler.cudagraph_mark_step_begin()
                    _ = _m(dummy_input)
        bench(
            "22 - compile fullgraph+FP16",
            "torch.compiler.cudagraph_mark_step_begin()\n"
            "with torch.no_grad():\n"
            "  with torch.amp.autocast('cuda', dtype=torch.float16):\n"
            "    model(x)",
            {"model": _m, "x": dummy_input, "torch": torch},
        )
    except Exception as e:
        print(f"compile fullgraph+FP16 failed: {e}")
        GPU_RESULTS["22 - compile fullgraph+FP16"] = float("nan")
    cleanup()
```

    [INFO ] using MLP layer as FFN
      RoPE patched: caches frozen, graph breaks eliminated
      Fullgraph forward patched: addict.Dict bypassed
      [compile fullgraph+FP16] setup: 234.53 s
      22 - compile fullgraph+FP16: 45.036 ms  (spread: 0.019 ms  rounds: 45.02, 45.04, 45.04, 45.03, 45.04)
      ✓ Sanity (22 - compile fullgraph+FP16): depth MSE = 0.000000
    

---
## 23 – Weight pruning 50 %

Unstructured L1 pruning: set 50 % of weights to zero per layer.
On V100 there's no hardware sparse tensor core support, so the
dense matmul kernels still process every element.


```python
reset_backends()
import torch.nn.utils.prune as prune

_m = fresh_model()
for name, module in _m.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        prune.l1_unstructured(module, name="weight", amount=0.5)
        prune.remove(module, "weight")

_nonzero = sum((p == 0).sum().item() for p in _m.parameters())
_total = sum(p.numel() for p in _m.parameters())
print(f"Sparsity: {_nonzero / _total * 100:.1f}% zeros")

with torch.no_grad():
    _out = _m(dummy_input)
check_output("pruned 50%", _out, baseline_output, atol=2.0)

bench(
    "23 - Pruning 50%",
    "with torch.no_grad(): model(x)",
    {"model": _m, "x": dummy_input},
)
cleanup()
```

    [INFO ] using MLP layer as FFN
    

    Sparsity: 49.5% zeros
      [OK] pruned 50%: max diff = 0.084996 (atol=2.0)
      23 - Pruning 50%: 172.651 ms  (spread: 0.384 ms  rounds: 172.40, 172.65, 172.77, 172.79, 172.62)
      ✓ Sanity (23 - Pruning 50%): depth MSE = 0.331261
    

---
## 24 – Weight pruning 50 % + compile

Pruned model + `torch.compile`. The compiler might exploit sparsity
patterns for better kernel selection.


```python
reset_backends()

if not _check_compile_support():
    print("torch.compile not supported - skipping")
    GPU_RESULTS["24 - Pruning 50% + compile"] = float("nan")
else:
    import torch.nn.utils.prune as prune

    _m = fresh_model()
    for name, module in _m.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=0.5)
            prune.remove(module, "weight")

    with timed("compile pruned"):
        _m = torch.compile(_m, mode="default")
        with torch.no_grad():
            _ = _m(dummy_input)

    bench(
        "24 - Pruning 50% + compile",
        "with torch.no_grad(): model(x)",
        {"model": _m, "x": dummy_input},
    )
    cleanup()
```

    [INFO ] using MLP layer as FFN
      [compile pruned] setup: 17.96 s
      24 - Pruning 50% + compile: 154.146 ms  (spread: 0.511 ms  rounds: 153.77, 154.06, 154.15, 154.21, 154.28)
      ✓ Sanity (24 - Pruning 50% + compile): depth MSE = 0.331261
    

---
## 25 – torch.export + AOTInductor

Ahead-of-time compiled `.so` artifact via `torch.export` + AOTInductor.
Produces a standalone shared library that can be loaded without Python.


```python
reset_backends()

try:
    _m = fresh_model(patch_rope=True)
    with timed("export + aot"):
        ep = torch.export.export(_m, (dummy_input,))
        so_path = torch._inductor.aot_compile(ep.module(), (dummy_input,))
    print(f"AOT artifact: {so_path}")

    _aot = torch._export.aot_load(so_path, device=str(device))
    with torch.no_grad():
        _out = _aot(dummy_input)
    check_output("AOTInductor", _out, baseline_output, atol=0.5)

    bench(
        "25 - AOTInductor", "model(x)", {"model": _aot, "x": dummy_input}, sanity=False
    )
except Exception as e:
    print(f"AOTInductor failed: {type(e).__name__}: {e}")
    GPU_RESULTS["25 - AOTInductor"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
      RoPE patched: caches frozen, graph breaks eliminated
    AOTInductor failed: Unsupported: call_function UserDefinedClassVariable(<class 'addict.addict.Dict'>) [ConstDictVariable()] {}
    
    from user code:
       File "/tmp/ipykernel_668996/181713102.py", line 19, in forward
        out = self.net(x, export_feat_layers=[])
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/da3.py", line 140, in forward
        output = self._process_depth_head(feats, H, W)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/da3.py", line 209, in _process_depth_head
        return self.head(feats, H, W, patch_start_idx=0)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/nn_inference/.venv/lib/python3.11/site-packages/depth_anything_3/model/dualdpt.py", line 186, in forward
        return Dict(out_dict)
    
    Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
    
    

---
## 26 – Multi-stream pipelining

Split a batch across CUDA streams. At batch=1 there's nothing to split,
so we test with batch=4 to show the concept.


```python
reset_backends()

# Multi-stream only makes sense with batch>1
_MULTI_BATCH = 4
_multi_x = torch.randn(_MULTI_BATCH, 1, 3, IMG_SIZE, IMG_SIZE, device=device)

N_STREAMS = 2
streams = [torch.cuda.Stream() for _ in range(N_STREAMS)]

_m = fresh_model()


def _multi_stream_infer():
    chunks = _multi_x.chunk(N_STREAMS, dim=0)
    outs = [None] * N_STREAMS
    for i, (chunk, stream) in enumerate(zip(chunks, streams)):
        with torch.cuda.stream(stream):
            with torch.no_grad():
                outs[i] = _m(chunk)
    torch.cuda.synchronize()
    return torch.cat(outs, dim=0)


_out = _multi_stream_infer()
print(f"Multi-stream output: {_out.shape}")

bench(
    "26 - Multi-stream (batch={})".format(_MULTI_BATCH),
    "fn()",
    {"fn": _multi_stream_infer},
)
cleanup()
```

    [INFO ] using MLP layer as FFN
    Multi-stream output: torch.Size([4, 1, 504, 504])
      26 - Multi-stream (batch=4): 587.485 ms  (spread: 1.667 ms  rounds: 586.88, 587.80, 587.49, 588.55, 587.36)
    

---
## 27 – bitsandbytes int8 (LLM.int8())

Mixed-precision decomposition: outlier features go through FP16,
the rest through int8. From Dettmers et al. (2022).


```python
reset_backends()

try:
    import bitsandbytes as bnb

    _m = fresh_model()

    # Replace Linear layers with bnb.nn.Linear8bitLt
    def _replace_with_bnb_int8(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                has_bias = child.bias is not None
                new = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                new.weight = bnb.nn.Int8Params(
                    child.weight.data,
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                if has_bias:
                    new.bias = nn.Parameter(child.bias.data)
                setattr(module, name, new.to(device))
            else:
                _replace_with_bnb_int8(child)

    _replace_with_bnb_int8(_m)
    _m = _m.to(device).eval()

    with torch.no_grad():
        _out = _m(dummy_input)
    check_output("bnb int8", _out, baseline_output, atol=2.0)

    bench(
        "27 - bitsandbytes int8",
        "with torch.no_grad(): model(x)",
        {"model": _m, "x": dummy_input},
    )
except Exception as e:
    print(f"bitsandbytes int8 failed: {type(e).__name__}: {e}")
    GPU_RESULTS["27 - bitsandbytes int8"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
      [OK] bnb int8: max diff = 0.005113 (atol=2.0)
      27 - bitsandbytes int8: 222.538 ms  (spread: 0.911 ms  rounds: 222.54, 222.48, 222.85, 222.60, 221.94)
      ✓ Sanity (27 - bitsandbytes int8): depth MSE = 0.000024
    

---
## 28 – bitsandbytes NF4

4-bit NormalFloat from QLoRA. Saves 8× memory vs FP32 but the dequant
kernel adds overhead per matmul.


```python
reset_backends()

try:
    import bitsandbytes as bnb

    _m = fresh_model()

    def _replace_with_bnb_nf4(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                has_bias = child.bias is not None
                new = bnb.nn.LinearNF4(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data,
                    requires_grad=False,
                    quant_type="nf4",
                    blocksize=64,
                )
                if has_bias:
                    new.bias = nn.Parameter(child.bias.data)
                setattr(module, name, new.to(device))
            else:
                _replace_with_bnb_nf4(child)

    _replace_with_bnb_nf4(_m)
    _m = _m.to(device).eval()

    with torch.no_grad():
        _out = _m(dummy_input)
    check_output("bnb nf4", _out, baseline_output, atol=5.0)

    bench(
        "28 - bitsandbytes NF4",
        "with torch.no_grad(): model(x)",
        {"model": _m, "x": dummy_input},
    )
except Exception as e:
    print(f"bitsandbytes NF4 failed: {type(e).__name__}: {e}")
    GPU_RESULTS["28 - bitsandbytes NF4"] = float("nan")
cleanup()
```

    [INFO ] using MLP layer as FFN
      [OK] bnb nf4: max diff = 0.024452 (atol=5.0)
      28 - bitsandbytes NF4: 177.700 ms  (spread: 0.609 ms  rounds: 177.76, 177.47, 177.87, 177.70, 177.26)
      ✓ Sanity (28 - bitsandbytes NF4): depth MSE = 0.000475
    

---
## 29 – FP16 + compile reduce-overhead

The "GPT, Fast" recipe: half-precision model + `torch.compile` with
`reduce-overhead` mode (internal CUDA Graphs).


```python
reset_backends()

if not _check_compile_support():
    print("torch.compile not supported - skipping")
    GPU_RESULTS["29 - AMP FP16 + compile (full)"] = float("nan")
else:
    _m = fresh_model(patch_head_fp16=True)
    with timed("compile AMP-FP16 (full)"):
        _m = torch.compile(_m, mode="default")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = _m(dummy_input)

    bench(
        "29 - AMP FP16 + compile (full)",
        "with torch.no_grad():\n"
        "  with torch.amp.autocast('cuda', dtype=torch.float16):\n"
        "    model(x)",
        {"model": _m, "x": dummy_input},
    )
    cleanup()
```

    [INFO ] using MLP layer as FFN
      [compile AMP-FP16 (full)] setup: 210.91 s
      29 - AMP FP16 + compile (full): 66.190 ms  (spread: 0.455 ms  rounds: 66.08, 66.19, 66.22, 66.13, 66.54)
      ✓ Sanity (29 - AMP FP16 + compile (full)): depth MSE = 0.000000
    

---
## 30 – FP16 + TF32 + cudnn.benchmark

The simplest no-compile combo: just flip switches and call `.half()`.
Anyone can do this in two lines of code.


```python
reset_backends()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_m = fresh_model(patch_head_fp16=True)

with torch.inference_mode():
    with torch.amp.autocast("cuda", dtype=torch.float16):
        _out = _m(dummy_input)
check_output("amp-fp16+tf32+cudnn (full)", _out, baseline_output, atol=0.5)

bench(
    "30 - AMP FP16 + TF32 + cudnn (full)",
    "with torch.inference_mode():\n"
    "  with torch.amp.autocast('cuda', dtype=torch.float16):\n"
    "    model(x)",
    {"model": _m, "x": dummy_input},
)
cleanup()
```

    [INFO ] using MLP layer as FFN
      [OK] amp-fp16+tf32+cudnn (full): max diff = 0.000670 (atol=0.5)
      30 - AMP FP16 + TF32 + cudnn (full): 97.184 ms  (spread: 0.075 ms  rounds: 97.17, 97.24, 97.19, 97.18, 97.16)
      ✓ Sanity (30 - AMP FP16 + TF32 + cudnn (full)): depth MSE = 0.000000
    

---
## 31 – FP16 + compile reduce-overhead + TF32

The strongest combo: half precision + TF32 matmul + `torch.compile`
with reduce-overhead mode (internal CUDA Graphs).


```python
reset_backends()

if not _check_compile_support():
    print("torch.compile not supported - skipping")
    GPU_RESULTS["31 - AMP FP16 + compile + TF32 (full)"] = float("nan")
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    _m = fresh_model(patch_head_fp16=True)
    with timed("compile AMP-FP16+TF32 (full)"):
        _m = torch.compile(_m, mode="default")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = _m(dummy_input)

    bench(
        "31 - AMP FP16 + compile + TF32 (full)",
        "with torch.no_grad():\n"
        "  with torch.amp.autocast('cuda', dtype=torch.float16):\n"
        "    model(x)",
        {"model": _m, "x": dummy_input},
    )
    cleanup()
```

    [INFO ] using MLP layer as FFN
      [compile AMP-FP16+TF32 (full)] setup: 26.57 s
      31 - AMP FP16 + compile + TF32 (full): 64.838 ms  (spread: 0.087 ms  rounds: 64.84, 64.77, 64.80, 64.84, 64.85)
      ✓ Sanity (31 - AMP FP16 + compile + TF32 (full)): depth MSE = 0.000000
    

---
## Profiler deep-dive

Kernel-level breakdown of where time goes in the baseline model.


```python
try:
    torch.cuda.synchronize()
except RuntimeError:
    pass
gc.collect()
try:
    torch.cuda.empty_cache()
except RuntimeError:
    pass

prof_model = fresh_model()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    with_flops=True,
) as prof:
    with torch.no_grad():
        for _ in range(10):
            prof_model(dummy_input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
cleanup()
```

    [INFO ] using MLP layer as FFN
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls   Total FLOPs  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                               aten::linear         1.17%      19.354ms         6.18%     102.001ms     100.991us       0.000us         0.00%     660.179ms     653.643us           0 b           0 b      10.69 Gb           0 b          1010            --  
                                                aten::addmm         2.63%      43.336ms         3.96%      65.431ms      64.783us     660.179ms        41.69%     660.179ms     653.643us           0 b           0 b      10.69 Gb      10.69 Gb          1010  7833785835520.000  
                                      volta_sgemm_128x64_tn         0.00%       0.000us         0.00%       0.000us       0.000us     592.664ms        37.43%     592.664ms     823.144us           0 b           0 b           0 b           0 b           720            --  
                                          aten::convolution         0.28%       4.704ms         5.36%      88.404ms     121.101us       0.000us         0.00%     378.394ms     518.348us           0 b           0 b       9.62 Gb           0 b           730            --  
                                         aten::_convolution         0.68%      11.302ms         5.07%      83.699ms     114.657us       0.000us         0.00%     378.394ms     518.348us           0 b           0 b       9.62 Gb    -405.00 Mb           730            --  
                                               aten::conv2d         0.21%       3.417ms         5.35%      88.286ms     124.347us       0.000us         0.00%     370.496ms     521.825us           0 b           0 b       9.33 Gb           0 b           710  7529198469120.000  
                                    aten::cudnn_convolution         1.78%      29.338ms         2.94%      48.599ms      68.449us     343.384ms        21.68%     343.384ms     483.640us           0 b           0 b       9.33 Gb       9.33 Gb           710            --  
    _5x_cudnn_volta_scudnn_winograd_128x128_ldg1_ldg4_re...         0.00%       0.000us         0.00%       0.000us       0.000us     289.983ms        18.31%     289.983ms     527.241us           0 b           0 b           0 b           0 b           550            --  
                         aten::scaled_dot_product_attention         0.21%       3.417ms         1.53%      25.287ms     105.363us       0.000us         0.00%     201.669ms     840.287us          24 b      -3.70 Kb       1.19 Gb           0 b           240            --  
              aten::_scaled_dot_product_efficient_attention         0.23%       3.871ms         1.32%      21.870ms      91.125us       0.000us         0.00%     201.669ms     840.287us       3.75 Kb           0 b       1.19 Gb           0 b           240            --  
                         aten::_efficient_attention_forward         0.34%       5.636ms         0.84%      13.808ms      57.533us     201.669ms        12.73%     201.669ms     840.287us       3.75 Kb           0 b       1.19 Gb           0 b           240            --  
    fmha_cutlassF_f32_aligned_64x64_rf_sm70(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us     201.669ms        12.73%     201.669ms     840.287us           0 b           0 b           0 b           0 b           240            --  
                                   aten::upsample_bicubic2d         0.02%     252.067us         0.04%     628.529us      62.853us     121.202ms         7.65%     121.273ms      12.127ms           0 b           0 b      50.62 Mb      50.62 Mb            10            --  
    void at::native::(anonymous namespace)::upsample_bic...         0.00%       0.000us         0.00%       0.000us       0.000us     121.202ms         7.65%     121.202ms      12.120ms           0 b           0 b           0 b           0 b            10            --  
    void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      73.278ms         4.63%      73.278ms      97.703us           0 b           0 b           0 b           0 b           750            --  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 1.651s
    Self CUDA time total: 1.584s
    
    

---
## Results (batch=1)

All GPU methods ranked by latency. Green = faster than baseline, red = slower.
N/A = requires Docker / specific hardware.


```python
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage


def _print_table(results, baseline_key):
    bl = results.get(baseline_key, 1.0)
    print(f"{'Method':<48} {'ms':>8}  {'Speedup':>8}")
    print("-" * 68)
    for name, ms in sorted(
        results.items(), key=lambda kv: kv[1] if kv[1] == kv[1] else 1e9
    ):
        if ms != ms:
            print(f"{name:<48} {'N/A':>8}  {'N/A':>8}")
        else:
            arrow = "  <-- baseline" if name == baseline_key else ""
            print(f"{name:<48} {ms:>7.3f}   {bl / ms:>7.2f}x{arrow}")
    print()


print(
    f"=== GPU batch={BATCH} | model=DA3 {MODEL_NAME} | {N_ROUNDS} rounds / median of medians ===\n"
)
_print_table(GPU_RESULTS, "0 - Baseline (eager)")

if SETUP_TIMES:
    print("=== One-time setup / conversion costs ===")
    print(f"{'Step':<48} {'seconds':>8}")
    print("-" * 58)
    for name, s in SETUP_TIMES.items():
        print(f"{name:<48} {s:>7.2f}s")
    print()
```

    === GPU batch=1 | model=DA3 da3-large | 5 rounds / median of medians ===
    
    Method                                                 ms   Speedup
    --------------------------------------------------------------------
    22 - compile fullgraph+FP16                       45.036      3.86x
    31 - AMP FP16 + compile + TF32 (full)             64.838      2.68x
    29 - AMP FP16 + compile (full)                    66.190      2.63x
    13 - CUDA Graphs                                  68.995      2.52x
    7 - Static FP16                                   89.383      1.95x
    30 - AMP FP16 + TF32 + cudnn (full)               97.184      1.79x
    5b - AMP FP16 (full, patched head)               101.142      1.72x
    5 - AMP FP16                                     115.303      1.51x
    11 - compile fullgraph+freeze                    117.110      1.49x
    12 - JIT Trace                                   127.224      1.37x
    10 - compile (max-autotune)                      138.306      1.26x
    9 - compile (reduce-overhead)                    138.564      1.26x
    8 - compile (default)                            145.039      1.20x
    24 - Pruning 50% + compile                       154.146      1.13x
    21 - Triton fused softmax                        165.728      1.05x
    3 - inference_mode                               171.552      1.01x
    23 - Pruning 50%                                 172.651      1.01x
    0 - Baseline (eager)                             173.970      1.00x  <-- baseline
    1 - cudnn.benchmark                              174.301      1.00x
    4 - TF32 matmul                                  174.350      1.00x
    7b - SDPA (mem_efficient)                        174.502      1.00x
    2 - channels_last                                175.483      0.99x
    28 - bitsandbytes NF4                            177.700      0.98x
    14 - torchao int8 weight-only                    186.573      0.93x
    7b - SDPA (math)                                 210.918      0.82x
    27 - bitsandbytes int8                           222.538      0.78x
    6 - AMP BF16                                     250.965      0.69x
    15 - torchao int8 dynamic                        368.720      0.47x
    26 - Multi-stream (batch=4)                      587.485      0.30x
    7b - SDPA (flash)                                     N/A       N/A
    16 - torchao int4 weight-only                         N/A       N/A
    17 - ONNX Runtime CUDA EP                             N/A       N/A
    18 - ORT CUDA EP + IO Bind                            N/A       N/A
    19 - ORT TensorRT EP                                  N/A       N/A
    20 - Torch-TensorRT                                   N/A       N/A
    25 - AOTInductor                                      N/A       N/A
    
    === One-time setup / conversion costs ===
    Step                                              seconds
    ----------------------------------------------------------
    compile default                                    21.42s
    compile reduce-overhead                            53.92s
    compile max-autotune                               59.69s
    compile fullgraph                                  51.40s
    compile fullgraph+FP16                            234.53s
    compile pruned                                     17.96s
    compile AMP-FP16 (full)                           210.91s
    compile AMP-FP16+TF32 (full)                       26.57s
    
    


```python
def _speedup_chart(results, baseline_key, title, filename):
    bl = results.get(baseline_key, 1.0)
    items = list(results.items())
    names = [k for k, _ in items]
    speedups = [bl / v if (v == v and v > 0) else 0 for _, v in items]
    colors = [
        ("#2196F3" if n == baseline_key else "#4CAF50" if s >= 1.0 else "#F44336")
        for n, s in zip(names, speedups)
    ]

    fig, ax = plt.subplots(figsize=(11, max(5, len(names) * 0.38)))
    bars = ax.barh(names, speedups, color=colors)
    ax.axvline(x=1.0, color="black", linewidth=0.9, linestyle="--", label="baseline")
    ax.set_xlabel("Speedup vs baseline (higher is better)")
    ax.set_title(title)
    ax.invert_yaxis()

    for bar, s, (n, _) in zip(bars, speedups, items):
        lbl = f"{s:.2f}x" if s > 0 else "N/A"
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved {filename}")


_speedup_chart(
    GPU_RESULTS,
    "0 - Baseline (eager)",
    f"GPU Inference Speedup - DA3 {MODEL_NAME}  (batch={BATCH}, 504×504)",
    "inference_benchmark_gpu.png",
)
IPImage(filename="inference_benchmark_gpu.png")
```


    
![png](/images/2026-03-24-inference_optimizations_benchmark/output_86_0.png)
    


    Saved inference_benchmark_gpu.png
    




    
![png](/images/2026-03-24-inference_optimizations_benchmark/output_86_2.png)
    



---
## What we learned

### The free wins (everyone should use these)

- **`TF32 matmul`** - On Ampere+ GPUs (A100, RTX 30xx, H100), TF32 rounds float32
  mantissa bits from 23 to 10 before the hardware tensor core matmul, then rounds back.
  About 2× faster for free. On V100 (pre-Ampere) there are no TF32 tensor cores so
  this is a no-op.

- **`inference_mode`** - Stricter than `no_grad`: also disables autograd version counters
  and view tracking. A few percent faster, zero accuracy cost.

- **`model.half()` (FP16)** - Halves memory bandwidth. On V100 the Tensor Cores do
  FP16 × FP16 → FP32 accumulation, so this is the single biggest easy win.

### The compiler wins (need Triton / Linux)

- **`torch.compile`** - TorchDynamo traces your model, TorchInductor compiles it via
  Triton. `reduce-overhead` uses CUDA Graphs internally - best for batch=1. Combined
  with FP16 this is the "GPT, Fast" recipe.

- **`fullgraph=True`** forces zero graph breaks. If it works (no dynamic control flow),
  the compiler optimizes the entire model as one fused kernel graph.

### Depth estimation gotchas

- **DA3's inner model uses `autocast(enabled=False)`** for the DPT depth head: this
  means AMP FP16 only accelerates the backbone (DinoV2), while the head always runs
  in FP32. Static FP16 (`model.half()`) is more effective but may lose precision in
  the depth head's convolutions.

- **`addict.Dict` output** - DA3 returns an `addict.Dict` internally, which can cause
  graph breaks in `torch.compile` and issues with JIT trace. The `DA3Depth` wrapper
  extracts just the depth tensor for compatibility.

### Why some techniques are slow or fail

- **`channels_last`** - Designed for CNNs (NHWC layout). DA3's backbone is mostly
  `nn.Linear`; only the DPT head's convolutions benefit, so the gain is tiny.

- **TensorRT** - TRT 10.x requires **SM ≥ 8.0** (Ampere). On V100 (SM 7.0) it fails.

- **bitsandbytes int8/NF4** - Custom dequantization kernels on V100 don't match
  hardware matmul throughput. Best for memory-constrained scenarios.

- **Pruning 50 %** - No hardware sparse support on V100 (that's Ampere 2:4 sparsity).

### The quantization landscape

- **torchao int8 weight-only** - 2× memory reduction; dequant fuses with `torch.compile`.
- **torchao int4** - 4× reduction vs FP16; custom Triton kernels.
- **ONNX int8 static** - Calibrated QDQ format for ORT deployment.

### References

1. **Depth Anything v3**: ByteDance-Seed, 2025. https://github.com/ByteDance-Seed/depth-anything-3
2. **DINOv2**: Oquab et al., 2023. https://arxiv.org/abs/2304.07193
3. **GPT, Fast**: PyTorch Blog, 2023. https://pytorch.org/blog/accelerating-generative-ai-2/
4. **LLM.int8()**: Dettmers et al., 2022. https://arxiv.org/abs/2208.07339
5. **QLoRA (NF4)**: Dettmers et al., 2023. https://arxiv.org/abs/2305.14314
6. **torch.compile**: PyTorch 2.0 blog, 2023.
7. **gpu-mode lectures**: https://github.com/gpu-mode/lectures
8. **FlashAttention**: Dao et al., 2022. https://arxiv.org/abs/2205.14135
9. **ONNX Runtime**: https://onnxruntime.ai/
10. **TensorRT**: https://developer.nvidia.com/tensorrt
11. **torchao**: https://github.com/pytorch/ao
12. **PyTorch Serve performance checklist**: https://docs.pytorch.org/serve/performance_checklist.html
13. **NVIDIA inference optimization**: https://developer.nvidia.com/blog/tag/inference-performance/

---
## Cleanup

Remove temporary files generated during the benchmark run.


```python
import glob

_temp_patterns = [
    "model.onnx",
    "model_prep.onnx",
    "model_int8.onnx",
    "model_scripted.pt",
    "inference_benchmark_gpu.png",
]

for _pat in _temp_patterns:
    for _f in glob.glob(_pat):
        try:
            os.remove(_f)
            print(f"  removed {_f}")
        except OSError:
            pass

print("Cleanup done.")
```

      removed inference_benchmark_gpu.png
    Cleanup done.
    
