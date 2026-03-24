import copy
import torch
import numpy as np
import pytest

from gossipy.model.nn import TorchPerceptron, TorchMLP, AdaLine, LogisticRegression
from gossipy.model.handler import (
    TorchModelHandler,
    AdaLineHandler,
    PegasosHandler,
    KMeansHandler,
)
from gossipy.core import CreateModelMode
from gossipy.utils import torch_models_eq


class TestTorchPerceptron:
    def test_creation(self):
        model = TorchPerceptron(10)
        assert model.input_dim == 10

    def test_forward(self):
        model = TorchPerceptron(10)
        model.init_weights()
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 1)

    def test_get_size(self):
        model = TorchPerceptron(10, bias=True)
        assert model.get_size() == 11  # 10 weights + 1 bias

    def test_str(self):
        model = TorchPerceptron(10)
        assert "TorchPerceptron" in str(model)


class TestTorchMLP:
    def test_creation(self):
        model = TorchMLP(10, 3, hidden_dims=(50, 20))
        assert model.get_size() > 0

    def test_forward(self):
        model = TorchMLP(10, 3, hidden_dims=(20,))
        model.init_weights()
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 3)

    def test_get_params_list(self):
        model = TorchMLP(10, 3)
        params = model.get_params_list()
        assert len(params) > 0

    def test_str(self):
        model = TorchMLP(10, 3)
        assert "TorchMLP" in str(model)


class TestAdaLine:
    def test_creation(self):
        model = AdaLine(10)
        assert model.input_dim == 10
        assert model.get_size() == 10

    def test_forward(self):
        model = AdaLine(10)
        model.init_weights()
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5,)


class TestLogisticRegression:
    def test_creation(self):
        model = LogisticRegression(10, 3)
        model.init_weights()
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 3)

    def test_str(self):
        model = LogisticRegression(10, 3)
        assert "LogisticRegression" in str(model)


class TestTorchModelHandler:
    def _make_handler(self, input_dim=10, output_dim=3):
        net = TorchMLP(input_dim, output_dim, hidden_dims=(20,))
        handler = TorchModelHandler(
            net=net,
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            criterion=torch.nn.CrossEntropyLoss(),
            local_epochs=1,
            batch_size=5,
            device="cpu"
        )
        handler.init()
        return handler

    def test_creation(self):
        handler = self._make_handler()
        assert handler.model is not None
        assert handler.device == torch.device("cpu")

    def test_update(self):
        handler = self._make_handler()
        x = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))
        handler._update((x, y))
        assert handler.n_updates == 1

    def test_evaluate(self):
        handler = self._make_handler()
        x = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))
        result = handler.evaluate((x, y))
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_merge(self):
        handler1 = self._make_handler()
        handler2 = self._make_handler()
        x = torch.randn(10, 10)
        y = torch.randint(0, 3, (10,))
        handler1._update((x, y))
        handler2._update((x, y))
        handler1._merge(handler2)

    def test_copy(self):
        handler = self._make_handler()
        handler_copy = handler.copy()
        assert handler_copy is not handler
        assert torch_models_eq(handler.model, handler_copy.model)

    def test_get_size(self):
        handler = self._make_handler()
        assert handler.get_size() > 0

    def test_caching(self):
        handler = self._make_handler()
        key = handler.caching(0)
        assert key is not None

    def test_str(self):
        handler = self._make_handler()
        assert "TorchModelHandler" in str(handler)

    def test_call_merge_update(self):
        handler = self._make_handler()
        other = self._make_handler()
        x = torch.randn(10, 10)
        y = torch.randint(0, 3, (10,))
        handler(other, (x, y))

    def test_binary_classification_auc(self):
        net = TorchMLP(10, 2, hidden_dims=(20,))
        handler = TorchModelHandler(
            net=net,
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            criterion=torch.nn.CrossEntropyLoss(),
            device="cpu"
        )
        handler.init()
        x = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        result = handler.evaluate((x, y))
        assert "auc" in result


class TestAdaLineHandler:
    def _make_handler(self, dim=10):
        net = AdaLine(dim)
        handler = AdaLineHandler(
            net=net,
            learning_rate=0.01,
            create_model_mode=CreateModelMode.UPDATE
        )
        handler.init()
        return handler

    def test_creation(self):
        handler = self._make_handler()
        assert handler.model is not None

    def test_update(self):
        handler = self._make_handler()
        x = torch.randn(5, 10)
        y = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        handler._update((x, y))
        assert handler.n_updates == 5

    def test_evaluate(self):
        handler = self._make_handler()
        x = torch.randn(20, 10)
        y = torch.tensor([1.0, -1.0] * 10)
        result = handler.evaluate((x, y))
        assert "accuracy" in result
        assert "auc" in result

    def test_merge(self):
        h1 = self._make_handler()
        h2 = self._make_handler()
        h1._merge(h2)


class TestPegasosHandler:
    def _make_handler(self, dim=10):
        net = AdaLine(dim)
        handler = PegasosHandler(
            net=net,
            learning_rate=0.01,
            create_model_mode=CreateModelMode.UPDATE
        )
        handler.init()
        return handler

    def test_update(self):
        handler = self._make_handler()
        x = torch.randn(5, 10)
        y = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        handler._update((x, y))
        assert handler.n_updates == 5


class TestKMeansHandler:
    def _make_handler(self, k=3, dim=10):
        handler = KMeansHandler(k=k, dim=dim)
        handler.init()
        return handler

    def test_creation(self):
        handler = self._make_handler()
        assert handler.model.shape == (3, 10)

    def test_update(self):
        handler = self._make_handler()
        x = torch.randn(5, 10)
        handler._update((x, None))

    def test_evaluate(self):
        handler = self._make_handler()
        x = torch.randn(30, 10)
        y = torch.randint(0, 3, (30,))
        result = handler.evaluate((x, y))
        assert "nmi" in result

    def test_merge_naive(self):
        h1 = self._make_handler()
        h2 = self._make_handler()
        h1._merge(h2)

    def test_merge_hungarian(self):
        h1 = KMeansHandler(k=3, dim=10, matching="hungarian")
        h1.init()
        h2 = KMeansHandler(k=3, dim=10, matching="hungarian")
        h2.init()
        h1._merge(h2)

    def test_get_size(self):
        handler = self._make_handler()
        assert handler.get_size() == 30

    def test_invalid_matching(self):
        with pytest.raises(AssertionError):
            KMeansHandler(k=3, dim=10, matching="invalid")
