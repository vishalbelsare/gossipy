import torch
import numpy as np
import pytest

from gossipy.data import DataDispatcher, load_classification_dataset
from gossipy.data.handler import ClassificationDataHandler, ClusteringDataHandler


class TestClassificationDataHandler:
    def _make_handler(self, n=100, d=10, c=3, test_size=0.2):
        X = torch.randn(n, d)
        y = torch.randint(0, c, (n,))
        return ClassificationDataHandler(X, y, test_size=test_size)

    def test_creation(self):
        handler = self._make_handler()
        assert handler.size() > 0
        assert handler.eval_size() > 0
        assert handler.n_classes == 3

    def test_getitem(self):
        handler = self._make_handler()
        x, y = handler[0]
        assert x.shape[-1] == 10

    def test_at_train(self):
        handler = self._make_handler()
        x, y = handler.at([0, 1])
        assert x.shape[0] == 2

    def test_at_eval(self):
        handler = self._make_handler()
        result = handler.at([0, 1], eval_set=True)
        assert result is not None

    def test_get_train_set(self):
        handler = self._make_handler()
        X, y = handler.get_train_set()
        assert X.shape[0] == y.shape[0]

    def test_get_eval_set(self):
        handler = self._make_handler()
        X, y = handler.get_eval_set()
        assert X.shape[0] == y.shape[0]

    def test_no_test_split(self):
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        handler = ClassificationDataHandler(X, y, test_size=0)
        assert handler.eval_size() == 0

    def test_with_provided_test_set(self):
        X_tr = torch.randn(80, 10)
        y_tr = torch.randint(0, 3, (80,))
        X_te = torch.randn(20, 10)
        y_te = torch.randint(0, 3, (20,))
        handler = ClassificationDataHandler(X_tr, y_tr, X_te, y_te)
        assert handler.size() == 80
        assert handler.eval_size() == 20

    def test_numpy_input(self):
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        handler = ClassificationDataHandler(X, y, test_size=0.2)
        assert handler.size() > 0

    def test_str(self):
        handler = self._make_handler()
        assert "ClassificationDataHandler" in str(handler)


class TestClusteringDataHandler:
    def test_creation(self):
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        handler = ClusteringDataHandler(X, y)
        assert handler.eval_size() == handler.size()

    def test_eval_set_is_train_set(self):
        X = torch.randn(50, 10)
        y = torch.randint(0, 3, (50,))
        handler = ClusteringDataHandler(X, y)
        X_tr, y_tr = handler.get_train_set()
        X_te, y_te = handler.get_eval_set()
        assert torch.equal(X_tr, X_te)


class TestDataDispatcher:
    def _make_dispatcher(self, n_clients=10, n_samples=100, d=10, c=3):
        X = torch.randn(n_samples, d)
        y = torch.randint(0, c, (n_samples,))
        handler = ClassificationDataHandler(X, y, test_size=0.2)
        return DataDispatcher(handler, n=n_clients)

    def test_creation(self):
        dispatcher = self._make_dispatcher()
        assert dispatcher.size() == 10

    def test_getitem(self):
        dispatcher = self._make_dispatcher()
        train_data, test_data = dispatcher[0]
        assert train_data is not None

    def test_has_test(self):
        dispatcher = self._make_dispatcher()
        assert dispatcher.has_test()

    def test_get_eval_set(self):
        dispatcher = self._make_dispatcher()
        X, y = dispatcher.get_eval_set()
        assert X is not None

    def test_out_of_range(self):
        dispatcher = self._make_dispatcher(n_clients=5)
        with pytest.raises(AssertionError):
            dispatcher[10]

    def test_str(self):
        dispatcher = self._make_dispatcher()
        assert "DataDispatcher" in str(dispatcher)


class TestLoadClassificationDataset:
    def test_load_iris(self):
        X, y = load_classification_dataset("iris")
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 150  # iris has 150 samples

    def test_load_breast(self):
        X, y = load_classification_dataset("breast")
        assert X.shape[0] == y.shape[0]

    def test_load_wine(self):
        X, y = load_classification_dataset("wine")
        assert X.shape[0] == y.shape[0]

    def test_load_as_numpy(self):
        X, y = load_classification_dataset("iris", as_tensor=False)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_load_no_normalize(self):
        X, y = load_classification_dataset("iris", normalize=False)
        assert X.shape[0] == 150
