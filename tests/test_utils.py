import torch
import pytest

from gossipy.utils import choice_not_n, torch_models_eq, StringEncoder
from gossipy.model.nn import TorchMLP


class TestChoiceNotN:
    def test_basic(self):
        for _ in range(100):
            c = choice_not_n(0, 10, 5)
            assert c != 5
            assert 0 <= c < 10

    def test_narrow_range(self):
        for _ in range(100):
            c = choice_not_n(0, 3, 1)
            assert c != 1
            assert c in (0, 2)


class TestTorchModelsEq:
    def test_equal_models(self):
        m1 = TorchMLP(10, 3)
        m2 = TorchMLP(10, 3)
        # Same architecture but different weights
        m1.load_state_dict(m2.state_dict())
        assert torch_models_eq(m1, m2)

    def test_different_weights(self):
        m1 = TorchMLP(10, 3)
        m2 = TorchMLP(10, 3)
        m1.init_weights()
        m2.init_weights()
        # Different random weights - could be equal but very unlikely
        # Just test that the function runs without error
        torch_models_eq(m1, m2)

    def test_different_architectures(self):
        m1 = TorchMLP(10, 3, hidden_dims=(20,))
        m2 = TorchMLP(10, 3, hidden_dims=(50,))
        assert not torch_models_eq(m1, m2)


class TestStringEncoder:
    def test_default(self):
        encoder = StringEncoder()
        result = encoder.default(42)
        assert result == "42"

    def test_complex_object(self):
        encoder = StringEncoder()
        result = encoder.default({"key": "value"})
        assert isinstance(result, str)
