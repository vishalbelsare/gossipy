import torch
import numpy as np
import pytest

from gossipy import set_seed
from gossipy.core import (
    AntiEntropyProtocol,
    CreateModelMode,
    MessageType,
    StaticP2PNetwork,
)
from gossipy.node import GossipNode
from gossipy.model.nn import TorchMLP
from gossipy.model.handler import TorchModelHandler
from gossipy.data import DataDispatcher
from gossipy.data.handler import ClassificationDataHandler


def _make_test_setup(n_nodes=5, input_dim=10, output_dim=3, n_samples=100):
    set_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    data_handler = ClassificationDataHandler(X, y, test_size=0.2)
    dispatcher = DataDispatcher(data_handler, n=n_nodes)
    p2p_net = StaticP2PNetwork(n_nodes)
    net = TorchMLP(input_dim, output_dim, hidden_dims=(20,))
    model_handler = TorchModelHandler(
        net=net,
        optimizer=torch.optim.SGD,
        optimizer_params={"lr": 0.01},
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu"
    )
    return dispatcher, p2p_net, model_handler


class TestGossipNode:
    def test_creation(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        assert node.idx == 0
        assert node.round_len == 10

    def test_init_model(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        node.init_model()
        assert node.model_handler.n_updates > 0

    def test_get_peer(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        peer = node.get_peer()
        assert peer != 0
        assert 0 <= peer < 5

    def test_timed_out_sync(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
            sync=True,
        )
        # Should time out exactly once per round
        timeouts = sum(1 for t in range(10) if node.timed_out(t))
        assert timeouts == 1

    def test_send_push(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        node.init_model()
        msg = node.send(0, 1, AntiEntropyProtocol.PUSH)
        assert msg.type == MessageType.PUSH
        assert msg.sender == 0
        assert msg.receiver == 1

    def test_send_pull(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        node.init_model()
        msg = node.send(0, 1, AntiEntropyProtocol.PULL)
        assert msg.type == MessageType.PULL
        assert msg.value is None

    def test_send_push_pull(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        node.init_model()
        msg = node.send(0, 1, AntiEntropyProtocol.PUSH_PULL)
        assert msg.type == MessageType.PUSH_PULL

    def test_has_test(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        assert node.has_test()

    def test_evaluate(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        node.init_model()
        result = node.evaluate()
        assert "accuracy" in result

    def test_str(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        node = GossipNode(
            idx=0,
            data=dispatcher[0],
            round_len=10,
            model_handler=model_handler.copy(),
            p2p_net=p2p_net,
        )
        assert "GossipNode" in str(node)

    def test_generate(self):
        dispatcher, p2p_net, model_handler = _make_test_setup()
        nodes = GossipNode.generate(
            data_dispatcher=dispatcher,
            p2p_net=p2p_net,
            model_proto=model_handler,
            round_len=10,
            sync=True,
        )
        assert len(nodes) == 5
        assert all(isinstance(n, GossipNode) for n in nodes.values())
