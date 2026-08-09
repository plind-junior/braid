from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_CSRC = Path(__file__).parent / "csrc"


@lru_cache(maxsize=1)
def load_gdn():
    """JIT-build the GDN extension for sm_120a.

    `12.0a` (arch-conditional) is requested rather than plain `12.0`: consumer
    Blackwell exposes instructions under the `a` variant that plain sm_120 does
    not. torch 2.11's own arch list reports only `sm_120`, but that constrains
    torch's prebuilt kernels, not what nvcc will emit for ours -- sm_120a
    landed in CUDA 12.8, which is the toolkit on this box.
    """
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    os.environ.setdefault("MAX_JOBS", "8")  # nvcc is memory-hungry
    from torch.utils.cpp_extension import load

    # `BRAID_KERNEL_VERBOSE=1` adds `-Xptxas=-v`, which prints registers per
    # thread and **spill stores/loads** per kernel. That is the number that
    # decides whether `gdn_prefill` is doing what it claims: it holds the whole
    # [128] state column in registers across the chunk, and if nvcc spills it to
    # local memory the arithmetic is still exact but the entire point is gone.
    # Off by default because it changes the flags, and changing the flags
    # invalidates ninja's cache and forces a rebuild.
    verbose = os.environ.get("BRAID_KERNEL_VERBOSE") == "1"
    flags = ["-O3", "-lineinfo"]
    if verbose:
        flags.append("-Xptxas=-v")

    return load(
        name="braid_gdn",
        sources=[str(_CSRC / "bindings.cpp"), str(_CSRC / "gdn_decode.cu"),
                 str(_CSRC / "conv1d_decode.cu")],
        # NO --use_fast_math: it turns rsqrtf into an approximation and breaks
        # fp32 parity against the oracle at the tolerances the tests assert.
        extra_cuda_cflags=flags,
        verbose=verbose,
    )
