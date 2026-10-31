# Ling3 Packing 与 Ascend LoRA SFT

固定依赖、默认 RoPE、FA2、无缓存 Padding/Packing 及保存恢复限制见
[Ling3 示例](https://github.com/modelscope/ms-swift/tree/main/examples/ascend/train/ling3)。

统一候选的验收范围为 Transformers 4.57.1 / 5.14.1、Torch/torch-npu 2.10 和两卡 FSDP2；八卡示例未重新认证。历史数值 BAD 保留，不支持跨版本 checkpoint 互换。Grouped MM 不在本次范围内。
