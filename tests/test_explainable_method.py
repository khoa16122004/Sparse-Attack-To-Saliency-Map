import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import explain_method
from util import get_explainable_method


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


if __name__ == "__main__":
    unittest.main()
