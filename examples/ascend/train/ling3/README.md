# Ling3 Ascend FSDP2 LoRA SFT

The Ling3 loader installs Swift adapters for packed document boundaries, short KDA training, and ordinary left/right padding in cache-free training and evaluation. MLA uses the configured attention backend. Official Ling3 modeling, weights, and FLA kernels remain unchanged. No private fork or manual model patch is required.

Use official `inclusionAI/Ling-3.0-tiny` revision
`9a98e35fe1c9ee255f78dd64771c7ae15a799481` (modeling SHA256
`c2509bf7ac580c262e2581d34d6403aa21682d2e10beb9ad85ad8820a7e33a40`).
Adapter installation is restricted to the tiny configuration; other Ling variants
keep their existing non-packing implementation and reject unsupported packing.

The validated dependency lane is Transformers 4.57.1, PyTorch 2.9.0, torch_npu 2.9.0.post6, CANN 9.1.0, Triton 3.2.0, triton-ascend 3.2.2, Accelerate 1.14.0, PEFT 0.20.0, and official FLA revision `35dceaee5408e69a555fec34cb215c93c375dabe`. This example does not install dependencies. Transformers 5.x is outside this change's validated scope.

Before changing attention forwards, the installer checks Transformers 4.57.1 and the required legacy FLA convolution and KDA callable interfaces. If these are unavailable, ordinary model loading keeps its upstream implementation and emits a warning: this does not certify MLA causality, padding, or short-sequence training. Explicit unsupported packing fails with the required dependency lane. Interface checks do not establish correctness for arbitrary unpinned FLA revisions.

Activate a compatible environment and set:

- `CANN_SET_ENV`: the installed CANN `set_env.sh`.
- `MODEL_DIR`: the official Ling3-tiny model directory.
- `DATASET`: an SFT JSONL file supported by Swift.
- `OUTPUT_DIR` and `CACHE_ROOT`: writable locations outside the source tree and historical artifacts.
- `TMPDIR`: an existing short scratch path, suitable for Unix sockets.
- `DEVICE_SET`: eight distinct physical device numbers selected by the user.
- `MASTER_PORT`: an unused port selected by the user.
- `FLA_ROOT`, optionally: an existing official FLA checkout when it is not installed as a package.

After assigning these variables:

```bash
bash examples/ascend/train/ling3/ling3_tiny_lora_fsdp2.sh
```

`PACKING=false` is the default. Use `PER_DEVICE_TRAIN_BATCH_SIZE=2` or greater with variable-length samples to exercise ordinary padding; the batch-size default remains 1. `PADDING_SIDE=right` is the default, and `PADDING_SIDE=left` selects left padding. `GRADIENT_ACCUMULATION_STEPS` defaults to 2.

Set `PACKING=true` to use packed documents, with optional `PACKING_LENGTH=2048`. This path requires native `flash_attention_2` varlen support and no historical cache, MTP, or context-parallel metadata. Do not combine PAD tokens with a flattened packed input. Ordinary padding and packed document boundaries have separate contracts.

`GRADIENT_CHECKPOINTING=false` is the D1 default for this example only. Set it to `true` to re-enable ordinary model activation checkpointing when memory is insufficient. Swift's global defaults, FSDP2 reshard settings, and FLA's internal autograd recomputation are unchanged. `MAX_STEPS` optionally limits the training duration.

Disabling activation checkpointing requires more memory. Choose this example's profile only when the memory budget permits it; no speed or training-quality guarantee is made. No MoE dispatch optimization or Task Queue override is included.

Ordinary padded KDA supports cache-free chunk-mode training and evaluation. Set `use_cache=False` explicitly for evaluation when the configuration defaults to caching. An omitted or `None` value follows the layer configuration; a historical state or a request to cache is rejected. Padded decoding/cache, empty sequences, internal mask gaps, conflicting packed metadata, and MTP combinations remain outside the supported contract.

The saved full-model BF16 padding comparison had a B1/B2 loss delta of 0.0664996624, above its frozen 0.02 budget. Local KDA/MLA padding isolation and a real 10-step padding SFT passed, but they do not establish strict numerical equivalence or explain the entire loss difference. The original numerical BAD verdict is retained. Device regression of this rebased and simplified candidate has not been run. Historical runs do not certify this diff.

`python tests/models/test_ling3_compat.py` runs portable CPU adapter tests without model files or FLA kernels. For the additional official-source lifecycle and checkpoint regression, set `LING3_MODEL_SOURCE` to a local official model source directory and run `python tests/models/test_ling3_packing.py`. That optional suite explicitly skips when no source is supplied; it does not download or load weights. Actual FLA, NPU, and FSDP2 validation requires separate device tests.

`python tests/models/test_ling3_maintainer_fixes.py` adds CPU fixture contracts for dependency preflight, loader preservation, cache-free padded evaluation, and test-module isolation. Supply `LING3_MODEL_SOURCE` for its same-process packing-suite check. These stubs and oracles do not execute real FLA kernels or certify device training/evaluation.

This compatibility lane uses the official remote-code `BailingMoeV3ForCausalLM`,
not the native Transformers `BailingHybridForCausalLM` implementation for Flash.
The tiny checkpoint uses low-rank Q projections, so the native Flash class is
not a drop-in replacement. No checkpoint weight conversion is performed here.

Keep the exact FLA revision above: its ShortConvolution routes singleton inputs
to an inference path even during training, which is why the local training-only
adapter is needed. Newer FLA has fixed this behavior and changed the internal
convolution package path. Updating that dependency requires removing this adapter
and rerunning the relevant tests; arbitrary FLA main revisions are not supported.

To prepare public data with Swift, download the public
`AI-ModelScope/alpaca-gpt4-data-zh` dataset and export examples to JSONL using the
standard `messages` format: one user message and one assistant response per row.
Set `DATASET` to that file. This is a reproducible data preparation route, not the
identity of the historical fixed-order 2000-step dataset.
