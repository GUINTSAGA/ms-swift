# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU loader and adapter lifecycle contracts; spies do not run FLA kernels."""
import ast
import importlib.util
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_test(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'tests/models/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import shared CPU fixtures normally; do not recursively execute another suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import test_ling3_compat as base
finally:
    sys.path.pop(0)

mock_modules = base.mock_modules
compat = base.compat
apply_rotary_pos_emb_interleave = base.apply_rotary_pos_emb_interleave
chunk_kda = base.chunk_kda


class Shell(nn.Module):

    def forward(self, input_ids=None, position_ids=None, **kwargs):
        return input_ids


def shell():
    model = Shell()
    model.config = SimpleNamespace(
        architectures=['BailingMoeV3ForCausalLM'],
        num_nextn_predict_layers=0,
        hidden_size=1536,
        num_hidden_layers=24,
        q_lora_rank=256,
        rope_scaling=None,
        use_cache=False,
        _attn_implementation='flash_attention_2')
    kda, _, _ = base.kda()
    model.model = SimpleNamespace(layers=[
        SimpleNamespace(attention=kda, attention_layer_type='linear_attention'),
        SimpleNamespace(attention=base.MLA(), attention_layer_type='attention')
    ])

    return model


def dependencies(version='4.57.1', missing=False, bad_conv=False):
    names = ('swift', 'swift.model', 'swift.model.models', 'swift.model.patcher', 'transformers',
             'transformers.modeling_utils', 'transformers.utils', 'transformers.utils.import_utils', 'fla',
             'fla.modules', 'fla.modules.conv')
    modules = {name: ModuleType(name) for name in names}
    tree = ast.parse((ROOT / 'swift/model/patcher.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'patch_module_forward')
    ns = {'MethodType': MethodType}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<Swift patcher>', 'exec'), ns)
    modules['swift.model.patcher'].patch_module_forward = ns['patch_module_forward']
    modules['swift.model.models.ling3_compat'] = compat
    rope = ModuleType('swift.model.models.ling3_rope')
    actual = load_test('test_ling3_rope').rope
    rope.transformers_compatibility = actual.transformers_compatibility
    rope.ling3_rope_construction = lambda *args: nullcontext(None)
    modules[rope.__name__] = rope
    modules['transformers'].__version__ = version
    modules['transformers.modeling_utils'].ALL_ATTENTION_FUNCTIONS = {'sdpa': lambda *args, **kwargs: None}
    modules['transformers.utils.import_utils'].is_torch_fx_available = lambda: True
    if not missing:
        conv = ModuleType('fla.modules.conv.causal_conv1d')
        conv.causal_conv1d = (lambda x: x) if bad_conv else (lambda **kwargs: (kwargs['x'], None))
        modules[conv.__name__] = conv
    return modules


def loader(model):

    class ModelLoader:

        def get_model(self, *args):
            return model

    tree = ast.parse((ROOT / 'swift/model/models/llm.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BailingHybridLoader')
    messages = []
    ns = dict(
        ModelLoader=ModelLoader,
        PretrainedConfig=object,
        Processor=object,
        Any=object,
        logger=SimpleNamespace(warning=messages.append),
        __package__='swift.model.models')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual BailingHybridLoader>', 'exec'), ns)
    return ns['BailingHybridLoader'](), messages


def unmodified(model):
    attentions = [layer.attention for layer in model.model.layers]
    convs = [getattr(attentions[0], name) for name in ('q_conv1d', 'k_conv1d', 'v_conv1d')]
    return [(dict(module.__dict__), getattr(module, 'forward', None))
            for module in [model, model.model, *attentions, *convs]]


class LoaderAndAdapterTests(unittest.TestCase):

    def test_eager_future_mask_guard(self):
        x = torch.zeros(1, 3, 4)
        noncausal = torch.zeros(1, 1, 3, 3)
        with self.assertRaisesRegex(ValueError, 'future token'):
            compat._validate_eager_causal_mask(x, noncausal)
        causal = noncausal.masked_fill(torch.ones(3, 3, dtype=torch.bool).triu(1), float('-inf'))
        compat._validate_eager_causal_mask(x, causal)
        nan_mask = causal.clone()
        nan_mask[..., 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            compat._validate_eager_causal_mask(x, nan_mask)
        for mask in (torch.ones(1, 3), torch.ones(1, 1, 3, 3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                compat._validate_eager_causal_mask(x, mask)
        q = torch.ones(1, 1, 3, 2)
        for mask, leaks in ((noncausal, True), (causal, False)):
            v = torch.randn(1, 1, 3, 2, requires_grad=True)
            y = torch.nn.functional.scaled_dot_product_attention(q, q, v, attn_mask=mask)
            grad = torch.autograd.grad(y[:, :, 0].sum(), v)[0]
            self.assertEqual(bool(grad[:, :, 1:].abs().sum() > 0), leaks)

    def test_registry_failure_preserves_instance(self):
        modules = dependencies()
        del modules['transformers.modeling_utils'].ALL_ATTENTION_FUNCTIONS
        with mock_modules(modules):
            model = shell()
            before = unmodified(model)
            with self.assertRaisesRegex(RuntimeError, 'registry'):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))

    def test_patch_failure_rolls_back_all_instances(self):
        modules = dependencies()
        original = modules['swift.model.patcher'].patch_module_forward
        count = []

        def failing(module, forward):
            original(module, forward)
            count.append(module)
            if len(count) == 2:
                raise RuntimeError('injected patch failure')

        modules['swift.model.patcher'].patch_module_forward = failing
        with mock_modules(modules):
            model = shell()
            before = unmodified(model)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))

    def test_supported_loader_install_idempotence_and_parameter_identity(self):
        model = shell()
        parameters = [(name, id(p), p.shape) for layer in model.model.layers
                      for name, p in layer.attention.named_parameters()]
        with mock_modules(dependencies()):
            instance, messages = loader(model)
            self.assertIs(instance.get_model('.', model.config, None, {}), model)
            forward = model.model.layers[0].attention.forward
            compat.install_ling3_compat(model)
            self.assertIs(model.model.layers[0].attention.forward, forward)
        self.assertFalse(messages)
        self.assertTrue(model.model._supports_packing_metadata)
        self.assertEqual(parameters, [(name, id(p), p.shape) for layer in model.model.layers
                                      for name, p in layer.attention.named_parameters()])
        self.assertFalse(hasattr(model.model.layers[0].attention, '_swift_ling3_origin_forward'))

    def test_certified_versions_install_and_dispatch(self):
        # Contract migration: 5.14.1 is certified only with the required APIs.
        for version in ('4.57.1', '5.14.1'):
            with self.subTest(version=version), mock_modules(dependencies(version=version)):
                model = shell()
                attention = model.model.layers[0].attention
                original = attention.forward
                parameters = [id(p) for p in attention.parameters()]
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))
                self.assertIsNone(compat.install_ling3_compat(model))
                self.assertIsNot(attention.forward, original)
                self.assertTrue(model.model._supports_packing_metadata)
                x = torch.randn(1, 3, 4, requires_grad=True)
                output = attention(x, position_ids=torch.arange(3)[None])[0]
                output.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(attention.q_conv1d.calls)
                self.assertEqual(parameters, [id(p) for p in attention.parameters()])
                installed = attention.forward
                compat.install_ling3_compat(model)
                self.assertIs(attention.forward, installed)

    def test_required_adapter_failure_stops_tiny_loading(self):
        for options in ({'missing': True}, {'bad_conv': True}, {'version': '6.0.0'}):
            with self.subTest(options=options), mock_modules(dependencies(**options)):
                model = shell()
                before = unmodified(model)
                instance, messages = loader(model)
                with self.assertRaises(RuntimeError):
                    instance.get_model('.', model.config, None, {})
                self.assertEqual(before, unmodified(model))
                self.assertFalse(messages)

    def test_preflight_failure_has_no_partial_mutation(self):
        for options in ({'missing': True}, {'bad_conv': True}, {'version': '6.0.0'}):
            with self.subTest(options=options), mock_modules(dependencies(**options)):
                model = shell()
                before = unmodified(model)
                with self.assertRaises(RuntimeError):
                    compat.install_ling3_compat(model)
                self.assertEqual(before, unmodified(model))
        with mock_modules(dependencies()), patch.dict(globals(), {'chunk_kda': lambda q: q}):
            model = shell()
            before = unmodified(model)
            with self.assertRaises(RuntimeError):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))
        with mock_modules(dependencies()):
            model = shell()
            model.model.layers[0].attention.v_conv1d._patched = True
            before = unmodified(model)
            with self.assertRaisesRegex(ValueError, 'conflicting patch'):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))

    def test_other_models_and_configurations_are_not_adapted(self):
        for architecture, width in (('BailingMoeV2_5ForCausalLM', 1536), ('BailingMoeV2ForCausalLM', 1536),
                                    ('BailingMoeV3ForCausalLM', 2560)):
            with self.subTest(architecture=architecture), mock_modules(dependencies()):
                model = shell()
                model.config.architectures = [architecture]
                model.config.hidden_size = width
                instance, messages = loader(model)
                with patch(
                        'swift.model.models.ling3_rope.ling3_rope_construction',
                        side_effect=AssertionError('non-target entered tiny construction')):
                    instance.get_model('.', model.config, None, {})
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))
                self.assertFalse(hasattr(model, '_swift_ling3_packing_handle'))
                self.assertFalse(messages)

    def test_mtp_preserves_loading_without_claiming_compatibility(self):
        for version in ('4.57.1', '5.14.1'):
            with self.subTest(version=version), mock_modules(dependencies(version=version)):
                model = shell()
                model.config.num_nextn_predict_layers = 1
                instance, _ = loader(model)
                # Cache extension is unchanged; this fixture tests Loader routing.
                extensions = []
                instance._patch_dynamic_cache_auto_extend = lambda: extensions.append(True)
                self.assertIs(instance.get_model('.', model.config, None, {}), model)
                self.assertEqual(extensions, [True])
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))
                self.assertFalse(getattr(model.model, '_supports_packing_metadata', False))
                inputs = torch.ones(1, 4, dtype=torch.long)
                self.assertIs(model(input_ids=inputs, position_ids=torch.arange(4)[None]), inputs)
                with self.assertRaisesRegex(NotImplementedError, 'MTP'):
                    compat.install_ling3_compat(model)
                model.model._supports_packing_metadata = True
                positions = torch.tensor([[0, 1, 0, 1]])
                with mock_modules({'swift.utils': SimpleNamespace(get_packed_seq_params=base.helper)}):
                    with self.assertRaisesRegex(ValueError, 'MTP'):
                        model(input_ids=inputs, position_ids=positions, **base.helper(positions))

    def test_padding_cache_semantics_and_unpadded_eval_delegation(self):
        for config_cache, explicit in ((True, False), (False, None), (True, None), (False, True)):
            model, kernel, _ = base.kda()
            model.eval()
            model.config = SimpleNamespace(use_cache=config_cache)
            mask, positions = base.padded('right')
            kwargs = dict(
                attention_mask=mask, position_ids=positions, use_cache=explicit, original=None, chunk_kda=kernel)
            if config_cache if explicit is None else explicit:
                with self.assertRaisesRegex(ValueError, 'use_cache=False'):
                    compat._kda_forward(model, torch.randn(2, 5, 4), **kwargs)
            else:
                compat._kda_forward(model, torch.randn(2, 5, 4), **kwargs)
        with self.assertRaisesRegex(ValueError, 'historical cache'):
            compat._kda_forward(model, torch.randn(2, 5, 4), past_key_value=(), **dict(kwargs, use_cache=False))
        delegated = []
        compat._kda_forward(
            model,
            torch.randn(1, 5, 4),
            use_cache=False,
            original=lambda *args, **kw: delegated.append(kw),
            chunk_kda=kernel)
        self.assertEqual(len(delegated), 1)

    @unittest.skipUnless(os.environ.get('LING3_MODEL_SOURCE'), 'requires official source, no weights')
    def test_packing_module_isolation_in_shared_process(self):
        sentinel = ModuleType('swift.utils')
        with mock_modules({'swift.utils': sentinel}):
            module = load_test('test_ling3_packing')
            self.assertIs(sys.modules['swift.utils'], sentinel)
            module.setUpModule()
            try:
                self.assertIs(sys.modules['swift.utils'], module.utils)
                raise RuntimeError('expected fixture interruption')
            except RuntimeError:
                pass
            finally:
                module.tearDownModule()
            self.assertIs(sys.modules['swift.utils'], sentinel)


class TransformersCompatibilityTests(unittest.TestCase):

    def test_version_warning_policy(self):
        policy = load_test('test_ling3_rope').rope.transformers_compatibility
        tree = ast.parse((ROOT / 'swift/utils/logger.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'warning_once')
        namespace = {'warning_set': set()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<Swift warning_once>', 'exec'), namespace)
        messages = []
        logger = SimpleNamespace(warning=messages.append)
        logger.warning_once = MethodType(namespace['warning_once'], logger)
        utils = ModuleType('swift.utils')
        utils.get_logger = lambda: logger
        with mock_modules({'swift.utils': utils}):
            for version in ('4.57.1', '5.14.1'):
                policy(version, warn=True)
            self.assertEqual(messages, [])
            for _ in range(2):
                self.assertEqual(policy('5.20.0', warn=True), 'tf5')
            self.assertEqual(len(messages), 1)
            self.assertIn('unverified', messages[0])
            policy('5.21.0', warn=True)
            self.assertEqual(len(messages), 2)
            self.assertEqual(policy('4.58.0', warn=True), 'tf4')
            self.assertEqual(len(messages), 3)
            self.assertIn('4.58.0', messages[-1])
        namespace['warning_set'].clear()
        messages.clear()
        modules = dependencies(version='5.20.0')
        modules['swift.utils'] = utils
        with mock_modules(modules):
            for _ in range(2):
                model = shell()
                instance, _ = loader(model)
                instance.get_model('.', model.config, None, {})
        self.assertEqual(len(messages), 1)
        for version in ('4.56.0', '6.0.0'):
            with self.assertRaises(NotImplementedError):
                policy(version)

    def test_official_flash_config_isolation(self):
        # Loader-relevant fields from inclusionAI/Ling-3.0-flash at fcc60a52b34c69c6271d0ac8a8cb0815020b6a43.
        flash = dict(
            architectures=['BailingMoeV3ForCausalLM'],
            hidden_size=2560,
            num_hidden_layers=42,
            q_lora_rank=None,
            num_nextn_predict_layers=1,
            rope_scaling=None)
        # inclusionAI/Ling-3.0-tiny at 9a98e35fe1c9ee255f78dd64771c7ae15a799481.
        tiny = SimpleNamespace(hidden_size=1536, num_hidden_layers=24, q_lora_rank=256)
        self.assertTrue(compat.is_ling3_tiny(tiny))
        self.assertFalse(compat.is_ling3_tiny(SimpleNamespace(**flash)))
        for version in ('4.57.1', '5.14.1'):
            with self.subTest(version=version), mock_modules(dependencies(version=version)):
                model = shell()
                vars(model.config).update(flash)
                instance, _ = loader(model)
                with patch.object(instance, '_patch_dynamic_cache_auto_extend') as extend, patch.object(
                        compat, 'install_ling3_compat') as install, patch.object(compat, 'patch_ling3_packing') as pack:
                    self.assertIs(instance.get_model('.', model.config, None, {}), model)
                extend.assert_called_once_with()
                install.assert_not_called()
                pack.assert_not_called()
                self.assertIsNone(model.config.rope_scaling)

    def test_flash_retains_legacy_factor_and_mtp(self):
        for version in ('4.57.1', '5.14.1', '5.20.0'):
            with self.subTest(version=version), mock_modules(dependencies(version=version)):
                model = shell()
                model.config.hidden_size = 2560
                model.config.num_hidden_layers = 42
                model.config.q_lora_rank = None
                model.config.num_nextn_predict_layers = 1
                model.config.rope_scaling = {'rope_type': 'default'}
                instance, _ = loader(model)
                extensions = []
                instance._patch_dynamic_cache_auto_extend = lambda: extensions.append(True)
                with patch(
                        'swift.model.models.ling3_rope.ling3_rope_construction',
                        side_effect=AssertionError('Flash entered Tiny construction')):
                    self.assertIs(instance.get_model('.', model.config, None, {}), model)
                expected = {'rope_type': 'linear', 'factor': 1.0} if version == '4.57.1' else {'rope_type': 'default'}
                self.assertEqual(model.config.rope_scaling, expected)
                self.assertEqual(extensions, [True])
                self.assertFalse(hasattr(model, '_swift_ling3_packing_handle'))
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))

    def test_non_tiny_preserves_rope_scaling_outside_legacy_factor_case(self):
        for version, original in (
            ('5.14.1', {
                'rope_type': 'default'
            }),
            ('5.20.0', {
                'rope_type': 'default'
            }),
            ('4.57.1', {
                'rope_type': 'linear',
                'factor': 2.0
            }),
            ('5.14.1', {
                'rope_type': 'linear',
                'factor': 2.0
            }),
        ):
            with self.subTest(version=version, original=original), mock_modules(dependencies(version=version)):
                model = shell()
                model.config.hidden_size = 2560
                scaling = dict(original)
                model.config.rope_scaling = scaling
                instance, _ = loader(model)
                self.assertIs(instance.get_model('.', model.config, None, {}), model)
                self.assertIs(model.config.rope_scaling, scaling)
                self.assertEqual(scaling, original)

    def test_unknown_tf5_api_failure_remains_fatal(self):
        with mock_modules(dependencies(version='5.20.0', missing=True)):
            model = shell()
            before = unmodified(model)
            with self.assertRaises(RuntimeError):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))


class MLAWrapperContractTests(unittest.TestCase):

    def test_verified_kda_avoids_repeated_keep_mask_reduction(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        hidden = torch.randn(1, 4, 4)
        mask = torch.ones(1, 4)
        model, kernel, _ = base.kda()
        record = compat._PackingRecord(positions, hidden, base.helper(positions), mask)
        original_bool = torch.Tensor.__bool__
        calls = []

        def counted(value):
            calls.append(value)
            return original_bool(value)

        with patch.object(torch.Tensor, '__bool__', counted):
            actual = compat._kda_forward(
                model,
                hidden,
                mask,
                position_ids=positions,
                original=lambda *a, **kw: None,
                chunk_kda=kernel,
                _swift_ling3_packing_record=record)[0]
        self.assertEqual(calls, [])
        with mock_modules({'swift.utils': SimpleNamespace(get_packed_seq_params=base.helper)}):
            expected = compat._kda_forward(
                model, hidden, mask, position_ids=positions, original=lambda *a, **kw: None, chunk_kda=kernel)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_packed_kda_rejects_changed_or_mutated_keep_mask(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        hidden = torch.randn(1, 4, 4)
        model, kernel, _ = base.kda()
        for mutate in (False, True):
            mask = torch.ones(1, 4)
            record = compat._PackingRecord(positions, hidden, base.helper(positions), mask)
            leaf_mask = mask if mutate else mask.clone()
            leaf_mask[0, 0] = 0
            with self.assertRaisesRegex(ValueError, 'all-valid leaf attention mask'):
                compat._kda_forward(
                    model,
                    hidden,
                    leaf_mask,
                    position_ids=positions,
                    original=None,
                    chunk_kda=kernel,
                    _swift_ling3_packing_record=record)

    def test_forward_scoped_metadata_reaches_mla_varlen_dispatch(self):
        boundaries = []

        def varlen(module, q, k, v, mask, **kwargs):
            cu = kwargs.get('cu_seq_lens_q')
            cu = cu.tolist() if cu is not None else [0, q.shape[2]]
            boundaries.append(cu)
            output = torch.cat([
                torch.nn.functional.scaled_dot_product_attention(
                    q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], is_causal=True, scale=kwargs['scaling'])
                for a, b in zip(cu, cu[1:])
            ], 2)
            return output.transpose(1, 2), None

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module()])
                attention = base.MLA('flash_attention_2')
                attention.forward = MethodType(
                    compat._make_mla_forward(attention._old_forward, {'flash_attention_2': varlen}), attention)
                self.model.layers[0].attention = attention
                self.model._supports_packing_metadata = True
                self.config = SimpleNamespace(
                    num_nextn_predict_layers=0, _attn_implementation='flash_attention_2', use_cache=False)
                compat.patch_ling3_packing(self)

            def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
                # Deliberately drop kwargs, as the frozen remote call sites do.
                return self.model.layers[0].attention(
                    hidden_states=inputs_embeds,
                    position_embeddings=(None, None),
                    attention_mask=None,
                    position_ids=position_ids)[0]

        torch.manual_seed(57)
        model = Model()
        reference = Model()
        reference.load_state_dict(model.state_dict())
        positions = torch.tensor([[0, 1, 0, 1]])
        x = torch.randn(1, 4, 8, requires_grad=True)
        with mock_modules({'swift.utils': SimpleNamespace(get_packed_seq_params=base.helper)}):
            with patch.object(compat, '_validate_packed_params', wraps=compat._validate_packed_params) as full:
                output = model(inputs_embeds=x, position_ids=positions, **base.helper(positions))
                self.assertEqual(full.call_count, 1)
            self.assertEqual(boundaries[-1], [0, 2, 4])
            other = torch.cat([
                reference(inputs_embeds=x[:, a:b], position_ids=torch.arange(b - a)[None]) for a, b in ((0, 2), (2, 4))
            ], 1)
            torch.testing.assert_close(output, other, rtol=0, atol=0)
            grad = torch.autograd.grad(output[:, 2:].sum(), x)[0]
            self.assertEqual(float(grad[:, :2].abs().max()), 0.)
        self.assertFalse(model.model.layers[0].attention._forward_pre_hooks)

    def test_real_cache_and_legacy_alias(self):
        from transformers.cache_utils import DynamicCache

        torch.manual_seed(51)
        module = base.MLA().eval()
        module.layer_idx = 0
        x = torch.randn(1, 4, 8)
        full = module(x, position_ids=torch.arange(4)[None])[0]
        for key in ('past_key_values', 'past_key_value'):
            cache = DynamicCache()
            prefix = module(x[:, :3], position_ids=torch.arange(3)[None], **{key: cache})[0]
            last = module(
                x[:, 3:],
                torch.zeros(1, 1, 1, 4),
                position_ids=torch.tensor([[3]]),
                cache_position=torch.tensor([3]),
                **{key: cache})[0]
            torch.testing.assert_close(prefix, full[:, :3])
            torch.testing.assert_close(last, full[:, 3:])
            self.assertEqual(cache.get_seq_length(), 4)

    def test_unknown_dispatch_and_decorator_rejected_before_install(self):
        from functools import wraps
        from ling3_remote_forward_fixture import remote_forward

        def unsupported(self, hidden_states, **kwargs):
            return hidden_states

        with self.assertRaisesRegex(ValueError, 'dispatch'):
            compat._make_mla_forward(unsupported, {})

        @wraps(remote_forward)
        def unknown(*args, **kwargs):
            return remote_forward(*args, **kwargs)

        with self.assertRaisesRegex(ValueError, 'decorator'):
            compat._make_mla_forward(unknown, {})
        from inspect import unwrap
        from transformers.utils.deprecation import deprecate_kwarg

        changed = deprecate_kwarg('different_name', new_name='past_key_values', version='4.58')(unwrap(remote_forward))
        with self.assertRaisesRegex(ValueError, 'decorator arguments'):
            compat._make_mla_forward(changed, {})

    def test_wrapper_preserves_remote_globals_and_instance_state(self):
        from inspect import unwrap
        from ling3_remote_forward_fixture import remote_forward

        raw = unwrap(remote_forward)
        original = raw.__globals__['eager_attention_forward']
        first = base.MLA().eval()
        second = base.MLA().eval()
        second.load_state_dict(first.state_dict())
        before = {name: id(value) for name, value in first.named_parameters()}
        x = torch.randn(1, 4, 8, requires_grad=True)
        a = first(x, position_ids=torch.arange(4)[None])[0]
        b = second(x, position_ids=torch.arange(4)[None])[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.sum().backward()
        self.assertIs(raw.__globals__['eager_attention_forward'], original)
        self.assertEqual(before, {name: id(value) for name, value in first.named_parameters()})


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
