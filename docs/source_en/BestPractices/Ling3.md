# Ling3 Packing and Ascend LoRA SFT

See [the Ling3 example](https://github.com/modelscope/ms-swift/tree/main/examples/ascend/train/ling3) for pinned
dependencies, default RoPE, FA2, cache-free padding/packing, and save/restore limits.

The unified candidate was validated with Transformers 4.57.1 / 5.14.1 and
Torch/torch-npu 2.10 on two-rank FSDP2. The eight-device example was not
re-certified. Historical numerical BAD results remain; cross-version checkpoint
interchangeability is unsupported. Grouped MM is outside this change.
