# M3 split SwiGLU-OAI support

This change adds correctness support to the existing MoE interface. It does
not change GEMM tiling, accumulator types, quantization schedules or shared
expert scheduling.

Use `activation="swigluoai_uninterleave"` and explicitly provide finite real
`gemm1_alpha`, `gemm1_beta` and nonnegative `gemm1_clamp_limit` to
`fused_experts_impl`, `inplace_fused_experts` or `outplace_fused_experts`.
Boolean, missing, nonfinite and overflowing scalar parameters are rejected
before tensor work. SiLU remains the default; its fused activation shortcut
is not used for the new split OAI activation. Existing activation names keep
their meaning.

The packed intermediate is `[all gates; all ups]`. The kernel loads in FP32,
clamps gate only above the limit and up on both sides, computes
`gate * sigmoid(alpha * gate) * (up + beta)`, then rounds once to the existing
FP16/BF16 activation workspace. GEMM2's original quantizer consumes that
workspace. This differs from the staged eager dense/shared OAI chain, which
rounds at each intermediate; the separate `swiglu_oai` optimization does not
replace this MoE formula. Strided input/output and empty outputs are covered.

The fixed launch retains the original split OAI kernel schedule. Performance
experiments are separate, and no operator or serving speedup is claimed here.

## Verification

```sh
VLLM_PLUGINS= PYTHONPATH=src python -m pytest -q tests/test_fused_moe_oai.py
VLLM_PLUGINS= PYTHONPATH=src python -m pytest -q tests/test_fused_experts_impl.py -k 'test_fused_moe_int8 and not w8a16' --quick
```

The focused tests compare FP16/BF16 against independent FP32 activation and
literal INT8 full-expert references, including clamp boundaries, tail widths,
noncontiguous strides, changed CUDA Graph inputs, invalid parameters and the
inplace/outplace wrappers. The original SiLU W8A8 regression is tested
separately. Full-expert comparisons use elementwise tolerances; they do not
claim that parallel GEMM reductions equal a different reference reduction
bit for bit.
