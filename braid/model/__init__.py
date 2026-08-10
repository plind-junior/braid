"""The braid runtime: config, checkpoint loading, and the conventional sublayers.

The GDN half of the model is verified against HF in `tests/test_hf_parity.py`
and implemented in `braid/kernels`. This package is the other half — the parts
that are ordinary transformer, plus the loader that feeds both.

Everything on the decode path here is subject to the capture-safety
contract — no allocation, no host synchronisation, no CPU-dependent
control flow inside anything the CUDA graph captures. The rules live in
CONTRIBUTING.md.
"""
