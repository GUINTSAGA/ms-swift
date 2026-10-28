# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-only maintainer fixes; stubs do not exercise real FLA kernels."""
import ast
import importlib.util
import io
import os
import sys
import unittest
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


base = load_test('test_ling3_compat')
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
    modules['transformers'].__version__ = version
    modules['transformers.modeling_utils'].ALL_ATTENTION_FUNCTIONS = {'sdpa': object()}
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


class MaintainerFixTests(unittest.TestCase):

    def test_supported_loader_install_idempotence_and_parameter_identity(self):
        model = shell()
        parameters = [(name, id(p), p.shape) for layer in model.model.layers
                      for name, p in layer.attention.named_parameters()]
        with patch.dict(sys.modules, dependencies()):
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

    def test_unsupported_lane_preserves_loading_and_rejects_packing(self):
        for options in ({'missing': True}, {'bad_conv': True}, {'version': '5.14.1'}):
            with self.subTest(options=options), patch.dict(sys.modules, dependencies(**options)):
                model = shell()
                before = [layer.attention.forward for layer in model.model.layers]
                instance, messages = loader(model)
                self.assertIs(instance.get_model('.', model.config, None, {}), model)
                self.assertEqual(before, [layer.attention.forward for layer in model.model.layers])
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))
                self.assertTrue(messages)
                self.assertIn('causality', messages[0])
                inputs = torch.tensor([[1, 2, 3, 4]])
                positions = torch.tensor([[0, 1, 0, 1]])
                self.assertIs(model(input_ids=inputs), inputs)
                kwargs = base.helper(positions)
                with self.assertRaisesRegex(ValueError, '4.57.1.*35dceaee'):
                    model(input_ids=inputs, position_ids=positions, **kwargs)

    def test_preflight_failure_has_no_partial_mutation(self):
        for options in ({'missing': True}, {'bad_conv': True}, {'version': '5.14.1'}):
            with self.subTest(options=options), patch.dict(sys.modules, dependencies(**options)):
                model = shell()
                before = unmodified(model)
                self.assertIsInstance(compat.install_ling3_compat(model), str)
                self.assertEqual(before, unmodified(model))
        with patch.dict(sys.modules, dependencies()), patch.dict(globals(), {'chunk_kda': lambda q: q}):
            model = shell()
            before = unmodified(model)
            self.assertIsInstance(compat.install_ling3_compat(model), str)
            self.assertEqual(before, unmodified(model))
        with patch.dict(sys.modules, dependencies()):
            model = shell()
            model.model.layers[0].attention.v_conv1d._patched = True
            before = unmodified(model)
            with self.assertRaisesRegex(ValueError, 'conflicting patch'):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))

    def test_other_models_and_configurations_are_not_adapted(self):
        for architecture, width in (('BailingMoeV2_5ForCausalLM', 1536), ('BailingMoeV2ForCausalLM', 1536),
                                    ('BailingMoeV3ForCausalLM', 2560)):
            with self.subTest(architecture=architecture), patch.dict(sys.modules, dependencies()):
                model = shell()
                model.config.architectures = [architecture]
                model.config.hidden_size = width
                instance, messages = loader(model)
                instance.get_model('.', model.config, None, {})
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))
                self.assertFalse(messages)

    def test_training_and_eval_padding_oracle(self):
        for training in (True, False):
            for side in ('left', 'right'):
                with self.subTest(training=training, side=side):
                    model, kernel, _ = base.kda()
                    model.train(training)
                    model.config = SimpleNamespace(use_cache=True)
                    mask, positions = base.padded(side)
                    x = torch.randn(2, 5, 4, requires_grad=training)
                    with torch.set_grad_enabled(training):
                        output = compat._kda_forward(
                            model,
                            x,
                            attention_mask=mask,
                            position_ids=positions,
                            use_cache=False,
                            original=None,
                            chunk_kda=kernel)[0]
                        refs = [
                            compat._kda_chunk_forward(model, x[i, mask[i].bool()][None], None, chunk_kda=kernel)[0][0]
                            for i in range(2)
                        ]
                        torch.testing.assert_close(output[mask.bool()], torch.cat(refs))
                        self.assertEqual(output[~mask.bool()].abs().sum().item(), 0)
                        if training:
                            output.sum().backward()
                            self.assertEqual(x.grad[~mask.bool()].abs().sum().item(), 0)

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
        with patch.dict(sys.modules, {'swift.utils': sentinel}):
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
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            with patch.dict(sys.modules, {module.__name__: module}):
                result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
            self.assertTrue(result.wasSuccessful(), result.errors or result.failures)
            self.assertEqual(result.testsRun, 15)
            self.assertIs(sys.modules['swift.utils'], sentinel)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
