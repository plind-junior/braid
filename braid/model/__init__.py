"""The braid runtime: config, checkpoint loading, and the conventional sublayers.

The GDN half of the model is verified against HF in `tests/test_hf_parity.py`
and implemented in `braid/kernels`. This package is the other half — the parts
that are ordinary transformer, plus the loader that feeds both.
"""
