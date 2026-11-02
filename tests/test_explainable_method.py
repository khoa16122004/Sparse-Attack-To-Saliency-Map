import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import explain_method
from util import get_explainable_method, get_torchvision_model


class ExplainableMethodTests(unittest.TestCase):
    def test_supported_methods_resolve_to_existing_implementations(self):
        methods = {
            "simple_gradient": explain_method.simple_gradient_map,
            "integrated_gradients": explain_method.integrated_gradients,
            "input_gradient": explain_method.input_gradient_map,
            "grad_cam": explain_method.grad_cam,
            "raw_attention": explain_method.raw_attention,
            "xgrad_cam": explain_method.xgrad_cam,
            "attention_grad": explain_method.attention_grad,
        }
        for name, implementation in methods.items():
            with self.subTest(method=name):
                self.assertIs(get_explainable_method(name), implementation)

    def test_unknown_method_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown explainable method: unknown"):
            get_explainable_method("unknown")

    def test_standard_models_and_swin_alias_resolve_without_downloads(self):
        for name in (
            "resnet18", "vgg16", "densenet121", "vit_b_32", "vit_b_16",
            "swin_v2_t", "swin_v2",
        ):
            resolved = "swin_v2_t" if name == "swin_v2" else name
            constructor = Mock()
            weights = object()
            with self.subTest(model=name), patch(
                f"util.tv_models.{resolved}", constructor,
            ), patch("util.get_model_weights", return_value=SimpleNamespace(DEFAULT=weights)) as get_weights, patch(
                "util.split_transform_from_weights", return_value=("spatial", "normalize"),
            ):
                model, spatial, normalize = get_torchvision_model(name)
                get_weights.assert_called_once_with(resolved)
                constructor.assert_called_once_with(weights=weights)
                self.assertIs(model, constructor.return_value)
                self.assertEqual((spatial, normalize), ("spatial", "normalize"))
            constructor.reset_mock()
            with self.subTest(model=name, pretrained=False), patch(
                f"util.tv_models.{resolved}", constructor,
            ):
                model, spatial, normalize = get_torchvision_model(
                    name, pretrained=False, num_classes=10,
                )
                constructor.assert_called_once_with(weights=None, num_classes=10)
                self.assertIs(model, constructor.return_value)
                self.assertIsNone(spatial)
                self.assertIsNone(normalize)


if __name__ == "__main__":
    unittest.main()
