# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import AutoTokenizer, PretrainedConfig
from typing import Any, Dict

from swift.template import TemplateType
from swift.utils import Processor, get_logger, safe_snapshot_download
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, SentenceTransformersLoader, register_model

logger = get_logger()


class GrokLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer_dir = safe_snapshot_download('AI-ModelScope/grok-1-tokenizer', download_model=False, check_local=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        return tokenizer


register_model(
    ModelMeta(
        LLMModelType.grok, [
            ModelGroup([
                Model('colossalai/grok-1-pytorch', 'hpcai-tech/grok-1'),
            ]),
        ],
        GrokLoader,
        template=TemplateType.default,
        architectures=['Grok1ModelForCausalLM'],
        model_arch=ModelArch.llama))


class PolyLMLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=True)


register_model(
    ModelMeta(
        LLMModelType.polylm,
        [
            ModelGroup(
                [
                    # base
                    Model('damo/nlp_polylm_13b_text_generation', 'DAMO-NLP-MT/polylm-13b'),
                ], ),
        ],
        PolyLMLoader,
        template=TemplateType.default,
        architectures=['GPT2LMHeadModel'],
        model_arch=ModelArch.qwen))


class YuanLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=True)
        addi_tokens = [
            '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
            '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>',
            '<empty_output>'
        ]
        tokenizer.add_tokens(addi_tokens, special_tokens=True)
        return tokenizer


register_model(
    ModelMeta(
        LLMModelType.yuan2,
        [
            ModelGroup([
                Model('IEITYuan/Yuan2.0-2B-hf', 'IEITYuan/Yuan2-2B-hf'),
                Model('IEITYuan/Yuan2.0-51B-hf', 'IEITYuan/Yuan2-51B-hf'),
                Model('IEITYuan/Yuan2.0-102B-hf', 'IEITYuan/Yuan2-102B-hf'),
                Model('IEITYuan/Yuan2-2B-Janus-hf', 'IEITYuan/Yuan2-2B-Janus-hf'),
            ]),
            ModelGroup([
                Model('IEITYuan/Yuan2-M32-hf', 'IEITYuan/Yuan2-M32-hf'),
            ]),
        ],
        YuanLoader,
        template=TemplateType.yuan,
        model_arch=ModelArch.llama,
        architectures=['YuanForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.orion,
        [
            ModelGroup([
                Model('OrionStarAI/Orion-14B-Chat', 'OrionStarAI/Orion-14B-Chat'),
                Model('OrionStarAI/Orion-14B-Base', 'OrionStarAI/Orion-14B-Base'),
            ]),
        ],
        template=TemplateType.orion,
        model_arch=ModelArch.llama,
        architectures=['OrionForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.dbrx, [
            ModelGroup([
                Model('AI-ModelScope/dbrx-base', 'databricks/dbrx-base'),
                Model('AI-ModelScope/dbrx-instruct', 'databricks/dbrx-instruct'),
            ]),
        ],
        template=TemplateType.dbrx,
        model_arch=ModelArch.dbrx,
        architectures=['DbrxForCausalLM'],
        requires=['transformers>=4.36']))

register_model(
    ModelMeta(
        LLMModelType.bluelm,
        [
            ModelGroup([
                Model('vivo-ai/BlueLM-7B-Chat-32K', 'vivo-ai/BlueLM-7B-Chat-32K'),
                Model('vivo-ai/BlueLM-7B-Chat', 'vivo-ai/BlueLM-7B-Chat'),
                Model('vivo-ai/BlueLM-7B-Base-32K', 'vivo-ai/BlueLM-7B-Base-32K'),
                Model('vivo-ai/BlueLM-7B-Base', 'vivo-ai/BlueLM-7B-Base'),
            ]),
        ],
        template=TemplateType.bluelm,
        model_arch=ModelArch.llama,
        architectures=['BlueLMForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.seggpt,
        [
            ModelGroup([
                Model('damo/nlp_seqgpt-560m', 'DAMO-NLP/SeqGPT-560M'),
            ]),
        ],
        template=TemplateType.default,
        model_arch=None,
        architectures=['BloomForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.xverse,
        [
            ModelGroup([
                Model('xverse/XVERSE-7B-Chat', 'xverse/XVERSE-7B-Chat'),
                Model('xverse/XVERSE-7B', 'xverse/XVERSE-7B'),
                Model('xverse/XVERSE-13B', 'xverse/XVERSE-13B'),
                Model('xverse/XVERSE-13B-Chat', 'xverse/XVERSE-13B-Chat'),
                Model('xverse/XVERSE-65B', 'xverse/XVERSE-65B'),
                Model('xverse/XVERSE-65B-2', 'xverse/XVERSE-65B-2'),
                Model('xverse/XVERSE-65B-Chat', 'xverse/XVERSE-65B-Chat'),
                Model('xverse/XVERSE-13B-256K', 'xverse/XVERSE-13B-256K', ms_revision='v1.0.0'),
            ]),
        ],
        template=TemplateType.xverse,
        model_arch=ModelArch.llama,
        architectures=['XverseForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.xverse_moe,
        [
            ModelGroup([
                Model('xverse/XVERSE-MoE-A4.2B', 'xverse/XVERSE-MoE-A4.2B'),
            ]),
        ],
        template=TemplateType.xverse,
        model_arch=ModelArch.llama,
        architectures=['XverseForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.c4ai,
        [
            ModelGroup([
                Model('AI-ModelScope/c4ai-command-r-v01', 'CohereForAI/c4ai-command-r-v01'),
                Model('AI-ModelScope/c4ai-command-r-plus', 'CohereForAI/c4ai-command-r-plus'),
            ]),
        ],
        template=TemplateType.c4ai,
        model_arch=ModelArch.llama,
        architectures=['CohereForCausalLM'],
        requires=['transformers>=4.39'],
    ))

register_model(
    ModelMeta(
        LLMModelType.aya, [
            ModelGroup([
                Model('AI-ModelScope/aya-expanse-8b', 'CohereForAI/aya-expanse-8b'),
                Model('AI-ModelScope/aya-expanse-32b', 'CohereForAI/aya-expanse-32b'),
            ]),
        ],
        template=TemplateType.aya,
        model_arch=ModelArch.llama,
        architectures=['CohereForCausalLM'],
        requires=['transformers>=4.44.0']))

register_model(
    ModelMeta(
        LLMModelType.ling,
        [
            ModelGroup([
                Model('inclusionAI/Ling-lite', 'inclusionAI/Ling-lite'),
                Model('inclusionAI/Ling-plus', 'inclusionAI/Ling-plus'),
                Model('inclusionAI/Ling-lite-base', 'inclusionAI/Ling-lite-base'),
                Model('inclusionAI/Ling-plus-base', 'inclusionAI/Ling-plus-base'),
            ]),
        ],
        template=TemplateType.ling,
        architectures=['BailingMoeForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen2_gte, [
            ModelGroup([
                Model('iic/gte_Qwen2-1.5B-instruct', 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'),
                Model('iic/gte_Qwen2-7B-instruct', 'Alibaba-NLP/gte-Qwen2-7B-instruct'),
            ]),
        ],
        SentenceTransformersLoader,
        template=TemplateType.dummy,
        architectures=['Qwen2ForCausalLM']))

register_model(
    ModelMeta(
        LLMModelType.mimo, [
            ModelGroup([
                Model('XiaomiMiMo/MiMo-7B-Base', 'XiaomiMiMo/MiMo-7B-Base'),
                Model('XiaomiMiMo/MiMo-7B-SFT', 'XiaomiMiMo/MiMo-7B-SFT'),
                Model('XiaomiMiMo/MiMo-7B-RL-Zero', 'XiaomiMiMo/MiMo-7B-RL-Zero'),
                Model('XiaomiMiMo/MiMo-7B-RL', 'XiaomiMiMo/MiMo-7B-RL'),
            ], TemplateType.qwen),
            ModelGroup([
                Model('XiaomiMiMo/MiMo-7B-RL-0530', 'XiaomiMiMo/MiMo-7B-RL-0530'),
            ], TemplateType.mimo_rl),
        ],
        model_arch=ModelArch.llama,
        architectures=['MiMoForCausalLM'],
        requires=['transformers>=4.37']))

register_model(
    ModelMeta(
        LLMModelType.dots1,
        [
            ModelGroup([
                Model('rednote-hilab/dots.llm1.base', 'rednote-hilab/dots.llm1.base'),
                Model('rednote-hilab/dots.llm1.inst', 'rednote-hilab/dots.llm1.inst'),
            ])
        ],
        template=TemplateType.dots1,
        architectures=['Dots1ForCausalLM'],
        requires=['transformers>=4.53'],
    ))

register_model(
    ModelMeta(
        LLMModelType.hunyuan,
        [ModelGroup([
            Model('Tencent-Hunyuan/Hunyuan-A13B-Instruct', 'tencent/Hunyuan-A13B-Instruct'),
        ])],
        template=TemplateType.hunyuan_moe,
        architectures=['HunYuanMoEV1ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.hunyuan_v1_dense,
        [
            ModelGroup([
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Instruct', 'tencent/Hunyuan-0.5B-Instruct'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Instruct', 'tencent/Hunyuan-1.8B-Instruct'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Instruct', 'tencent/Hunyuan-4B-Instruct'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Instruct', 'tencent/Hunyuan-7B-Instruct'),
                # pretrain
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Pretrain', 'tencent/Hunyuan-0.5B-Pretrain'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Pretrain', 'tencent/Hunyuan-1.8B-Pretrain'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Pretrain', 'tencent/Hunyuan-4B-Pretrain'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Pretrain', 'tencent/Hunyuan-7B-Pretrain'),
                # fp8
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-FP8', 'tencent/Hunyuan-0.5B-Instruct-FP8'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-FP8', 'tencent/Hunyuan-1.8B-Instruct-FP8'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Instruct-FP8', 'tencent/Hunyuan-4B-Instruct-FP8'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Instruct-FP8', 'tencent/Hunyuan-7B-Instruct-FP8'),
                # awq
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-AWQ-Int4', 'tencent/Hunyuan-0.5B-Instruct-AWQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-AWQ-Int4', 'tencent/Hunyuan-1.8B-Instruct-AWQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Instruct-AWQ-Int4', 'tencent/Hunyuan-4B-Instruct-AWQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Instruct-AWQ-Int4', 'tencent/Hunyuan-7B-Instruct-AWQ-Int4'),
                # gptq
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-0.5B-Instruct-GPTQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-1.8B-Instruct-GPTQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-4B-Instruct-GPTQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-7B-Instruct-GPTQ-Int4'),
            ])
        ],
        template=TemplateType.hunyuan,
        requires=['transformers>=4.55.0.dev0'],
        architectures=['HunYuanDenseV1ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.hy_v3,
        [
            ModelGroup([
                Model('Tencent-Hunyuan/Hy3-preview', 'tencent/Hy3-preview'),
                Model('Tencent-Hunyuan/Hy3-preview-Base', 'tencent/Hy3-preview-Base'),
            ],
                       template=TemplateType.hy_v3_preview),
            ModelGroup([
                Model('Tencent-Hunyuan/Hy3', 'tencent/Hy3'),
                Model('Tencent-Hunyuan/Hy3-FP8', 'tencent/Hy3-FP8'),
            ],
                       template=TemplateType.hy_v3),
        ],
        requires=['transformers>=5.6.0'],
        architectures=['HYV3ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gpt_oss, [
            ModelGroup([
                Model('openai-mirror/gpt-oss-20b', 'openai/gpt-oss-20b'),
                Model('openai-mirror/gpt-oss-120b', 'openai/gpt-oss-120b'),
            ])
        ],
        template=TemplateType.gpt_oss,
        ignore_patterns=['metal/', 'original/'],
        architectures=['GptOssForCausalLM'],
        requires=['transformers>=4.55']))

register_model(
    ModelMeta(
        LLMModelType.longchat,
        [
            ModelGroup([
                Model('meituan-longcat/LongCat-Flash-Chat', 'meituan-longcat/LongCat-Flash-Chat'),
                Model('meituan-longcat/LongCat-Flash-Chat-FP8', 'meituan-longcat/LongCat-Flash-Chat-FP8'),
            ])
        ],
        template=TemplateType.longchat,
        architectures=['LongcatFlashForCausalLM'],
        requires=['transformers>=4.54,<4.56'],
    ))

register_model(
    ModelMeta(
        LLMModelType.bailing_moe,
        [
            ModelGroup([
                Model('inclusionAI/Ling-mini-2.0', 'inclusionAI/Ling-mini-2.0'),
                Model('inclusionAI/Ling-mini-base-2.0', 'inclusionAI/Ling-mini-base-2.0'),
                Model('inclusionAI/Ling-1T', 'inclusionAI/Ling-1T'),
            ],
                       template=TemplateType.ling2),
            ModelGroup([
                Model('inclusionAI/Ring-mini-2.0', 'inclusionAI/Ring-mini-2.0'),
            ], template=TemplateType.ring2)
        ],
        architectures=['BailingMoeV2ForCausalLM'],
    ))


class BailingHybridLoader(ModelLoader):

    @staticmethod
    def _patch_ling3_packing(model, compat_error=None):
        """Validate collator metadata before the remote model discards its keep mask.

        Attention modules also validate boundaries locally: direct module calls
        and activation-checkpoint recomputation can bypass this outer hook.
        """
        import inspect
        import torch

        if hasattr(model, '_swift_ling3_packing_handle'):
            return

        signature = inspect.signature(model.forward)

        def prepare_packing(module, args, kwargs):
            call_kwargs = kwargs
            if args:
                bound = signature.bind_partial(*args, **kwargs)
                call_kwargs = dict(bound.arguments)
                call_kwargs.update(call_kwargs.pop('kwargs', {}))
            fields = ('cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k')
            position_ids = call_kwargs.get('position_ids')
            mask = call_kwargs.get('attention_mask')
            if not any(call_kwargs.get(key) is not None for key in fields + ('cu_seqlens', )):
                # Left-padding positions can repeat zero without representing packed documents.
                if (position_ids is not None and position_ids.ndim == 2 and position_ids.shape[0] == 1
                        and (mask is None or (mask.ndim == 2 and bool((mask == 1).all()))) and bool(
                            (position_ids[:, 1:] == 0).any())):
                    raise ValueError('Ling3 packed position_ids require explicit sequence metadata.')
                if (position_ids is not None and position_ids.ndim == 2 and position_ids.shape[0] == 1
                        and mask is not None and mask.ndim == 4 and bool(
                            ((position_ids[:, :-1] > 0) & (position_ids[:, 1:] == 0)).any())):
                    # Follow the remote decoder's last-query padding interpretation.
                    # Do not classify ordinary 4-D padding as packed documents.
                    keep = mask[..., -1, :] if mask.dtype == torch.bool else mask[..., -1, :] > -1e4
                    if bool(keep.all()):
                        raise ValueError('Ling3 packed position_ids require explicit sequence metadata.')
                return args, kwargs
            if any(call_kwargs.get(key) is None for key in fields):
                raise ValueError('Ling3 packing requires cu_seq_lens_q/k and max_length_q/k together.')
            if not getattr(getattr(module, 'model', module), '_supports_packing_metadata', False):
                raise ValueError(
                    f'Ling3 packing is unavailable: {compat_error or "the model has no packing adapters"}. '
                    'Use Transformers 4.57.1 or 5.14.1 and official FLA revision 35dceaee5408e69a555fec34cb215c93c375dabe.'
                )
            if module.config._attn_implementation != 'flash_attention_2':
                raise ValueError('Ling3 packing requires the native flash_attention_2 varlen backend.')
            use_cache = call_kwargs.get('use_cache')
            if use_cache is None:
                use_cache = module.config.use_cache
            if call_kwargs.get('past_key_values') is not None or use_cache:
                raise ValueError('Ling3 packing requires use_cache=False and no past_key_values.')
            if getattr(module.config, 'num_nextn_predict_layers', 0):
                raise ValueError('Ling3 packing with MTP is not supported.')
            if (position_ids is None or position_ids.ndim != 2 or position_ids.shape[0] != 1
                    or position_ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError('Ling3 packing requires integer position_ids with shape [1, total_tokens].')
            total = position_ids.shape[1]
            inputs = call_kwargs.get('input_ids')
            if inputs is None:
                inputs = call_kwargs.get('inputs_embeds')
            if inputs is None and args:
                inputs = args[0]
            if inputs is None or inputs.shape[:2] != (1, total):
                raise ValueError('Ling3 packing requires one flattened input batch matching position_ids.')
            if mask is not None:
                if mask.ndim != 2 or mask.shape != position_ids.shape:
                    raise ValueError('Ling3 packing accepts only a 2-D all-valid keep mask or attention_mask=None.')
                if not bool((mask == 1).all()):
                    raise ValueError('Ling3 padding-free packing does not support padding or non-binary keep masks.')
            cu = call_kwargs['cu_seq_lens_q']
            if (not isinstance(cu, torch.Tensor) or cu.ndim != 1 or cu.numel() < 2
                    or cu.dtype not in (torch.int32, torch.int64) or not cu.is_contiguous()
                    or cu.device != inputs.device or position_ids.device != inputs.device):
                raise ValueError('Ling3 cumulative lengths must be contiguous integer tensors on the input device.')
            if int(cu[0]) != 0 or int(cu[-1]) != total or not bool((cu.diff() > 0).all()):
                raise ValueError(
                    'Ling3 cumulative lengths must start at zero, end at total_tokens and increase strictly.')
            for key in ('cu_seq_lens_k', 'cu_seqlens'):
                other = call_kwargs.get(key)
                if other is not None and (not isinstance(other, torch.Tensor) or other.device != cu.device
                                          or other.dtype != cu.dtype or not torch.equal(other, cu)):
                    raise ValueError(f'Ling3 packing has conflicting {key} boundaries.')
            starts = (position_ids[0] == 0).nonzero().flatten()
            consecutive = (position_ids[:, 1:] == 0) | (position_ids.diff(dim=-1) == 1)
            if (not torch.equal(starts, cu[:-1].to(starts.dtype)) or bool((position_ids < 0).any())
                    or not bool(consecutive.all())):
                raise ValueError('Ling3 position_ids must be zero-based consecutive positions within every document.')
            maximum = cu.diff().max()
            for key in ('max_length_q', 'max_length_k'):
                value = call_kwargs[key]
                if isinstance(value, torch.Tensor) and (value.ndim != 0
                                                        or value.dtype not in (torch.int32, torch.int64)):
                    raise ValueError(f'Ling3 {key} must be an integer scalar.')
                if not isinstance(value, (int, torch.Tensor)) or int(value) != int(maximum):
                    raise ValueError(f'Ling3 {key} must match the largest document length.')
            return args, kwargs

        model._swift_ling3_packing_handle = model.register_forward_pre_hook(prepare_packing, with_kwargs=True)

    @staticmethod
    def _patch_dynamic_cache_auto_extend():
        """Patch DynamicCache.update to auto-extend layers for MTP/NextN layers."""
        from transformers.cache_utils import DynamicCache
        if getattr(DynamicCache, '_swift_auto_extend_patched', False):
            return

        _original_update = DynamicCache.update

        def auto_extend_update(self, key_states, value_states, layer_idx, *args, **kwargs):
            from transformers import DynamicLayer
            while len(self.layers) <= layer_idx:
                self.layers.append(DynamicLayer())
            return _original_update(self, key_states, value_states, layer_idx, *args, **kwargs)

        DynamicCache.update = auto_extend_update
        DynamicCache._swift_auto_extend_patched = True

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor, model_kwargs) -> Any:
        if 'BailingMoeV3ForCausalLM' in (getattr(config, 'architectures', None) or []):
            import transformers.utils.import_utils as import_utils
            if not hasattr(import_utils, 'is_torch_fx_available'):

                def is_torch_fx_available():
                    try:
                        import torch.fx  # noqa: F401
                        return True
                    except ImportError:
                        return False

                import_utils.is_torch_fx_available = is_torch_fx_available
            rope_scaling = getattr(config, 'rope_scaling', None)
            from transformers import __version__
            if __version__ == '4.57.1' and isinstance(rope_scaling, dict) and 'factor' not in rope_scaling:
                rope_scaling['factor'] = 1.0
                rope_scaling['rope_type'] = 'linear'
            if getattr(config, 'num_nextn_predict_layers', 0) > 0:
                self._patch_dynamic_cache_auto_extend()
        if 'BailingMoeV3ForCausalLM' in (getattr(config, 'architectures', None) or []):
            from .ling3_rope import ling3_rope_construction
            with ling3_rope_construction(model_dir, config, model_kwargs) as finalize:
                model = super().get_model(model_dir, config, processor, model_kwargs)
                if finalize is not None:
                    finalize(model)
        else:
            model = super().get_model(model_dir, config, processor, model_kwargs)
        if 'BailingMoeV3ForCausalLM' in (getattr(config, 'architectures', None) or []):
            from .ling3_compat import install_ling3_compat, is_ling3_tiny

            compat_error = None
            if is_ling3_tiny(config) and not getattr(config, 'num_nextn_predict_layers', 0):
                compat_error = install_ling3_compat(model)
                if compat_error is not None:
                    logger.warning(f'{compat_error}. Keeping upstream non-packing behavior; '
                                   'MLA causality, padding and short-sequence training are not certified on this path.')
            self._patch_ling3_packing(model, compat_error)
        return model


register_model(
    ModelMeta(
        LLMModelType.bailing_hybrid,
        [
            ModelGroup([
                Model('inclusionAI/Ling-2.5-1T', 'inclusionAI/Ling-2.5-1T'),
                Model('inclusionAI/Ling-2.6-1T', 'inclusionAI/Ling-2.6-1T'),
                Model('inclusionAI/Ling-2.6-flash', 'inclusionAI/Ling-2.6-flash'),
            ],
                       template=TemplateType.ling2),
            ModelGroup([
                Model('inclusionAI/Ling-3.0-tiny', 'inclusionAI/Ling-3.0-tiny'),
                Model('inclusionAI/Ling-3.0-flash', 'inclusionAI/Ling-3.0-flash'),
            ],
                       template=TemplateType.ling3),
            ModelGroup([
                Model('inclusionAI/Ring-2.5-1T', 'inclusionAI/Ring-2.5-1T'),
                Model('inclusionAI/Ring-2.6-1T', 'inclusionAI/Ring-2.6-1T'),
            ],
                       template=TemplateType.ring2_5),
        ],
        BailingHybridLoader,
        architectures=['BailingMoeV2_5ForCausalLM', 'BailingMoeV3ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.iquestcoder,
        [
            ModelGroup([
                Model('IQuestLab/IQuest-Coder-V1-40B-Base-Stage1', 'IQuestLab/IQuest-Coder-V1-40B-Base-Stage1'),
                Model('IQuestLab/IQuest-Coder-V1-40B-Base', 'IQuestLab/IQuest-Coder-V1-40B-Base'),
                Model('IQuestLab/IQuest-Coder-V1-40B-Instruct', 'IQuestLab/IQuest-Coder-V1-40B-Instruct'),
            ])
        ],
        template=TemplateType.iquestcoder,
        requires=['transformers==4.52.4'],
        architectures=['IQuestCoderForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.youtu_llm,
        [
            ModelGroup([
                Model('Tencent-YouTu-Research/Youtu-LLM-2B', 'tencent/Youtu-LLM-2B'),
                Model('Tencent-YouTu-Research/Youtu-LLM-2B-Base', 'tencent/Youtu-LLM-2B-Base'),
            ])
        ],
        template=TemplateType.youtu_llm,
        architectures=['YoutuForCausalLM'],
        requires=['transformers>=4.56'],
    ))

register_model(
    ModelMeta(
        LLMModelType.olmoe,
        [
            ModelGroup([
                Model('allenai/OLMoE-1B-7B-0125', 'allenai/OLMoE-1B-7B-0125'),
                Model('allenai/OLMoE-1B-7B-0125-Instruct', 'allenai/OLMoE-1B-7B-0125-Instruct'),
            ],
                       template=TemplateType.olmoe),
            ModelGroup([
                Model('allenai/OLMoE-1B-7B-0924', 'allenai/OLMoE-1B-7B-0924'),
                Model('allenai/OLMoE-1B-7B-0924-Instruct', 'allenai/OLMoE-1B-7B-0924-Instruct'),
                Model('allenai/OLMoE-1B-7B-0924-SFT', 'allenai/OLMoE-1B-7B-0924-SFT'),
            ],
                       template=TemplateType.olmoe_0924)
        ],
        architectures=['OlmoeForCausalLM'],
    ))
