import torch
import numpy as np
import pytest

from gossipy import set_seed
from gossipy.core import AntiEntropyProtocol, ConstantDelay, StaticP2PNetwork
from gossipy.simul import GossipSimulator, SimulationReport
from gossipy.node import GossipNode
from gossipy.model.nn import TorchMLP
from gossipy.model.handler import TorchModelHandler
from gossipy.data import DataDispatcher
from gossipy.data.handler import ClassificationDataHandler


def _make_simulator(n_nodes=5, n_rounds=3):
    set_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    data_handler = ClassificationDataHandler(X, y, test_size=0.2)
    dispatcher = DataDispatcher(data_handler, n=n_nodes)
    p2p_net = StaticP2PNetwork(n_nodes)
    net = TorchMLP(10, 3, hidden_dims=(20,))
    model_handler = TorchModelHandler(
        net=net,
        optimizer=torch.optim.SGD,
        optimizer_params={"lr": 0.01},
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu"
    )
    nodes = GossipNode.generate(
        data_dispatcher=dispatcher,
        p2p_net=p2p_net,
        model_proto=model_handler,
        round_len=10,
        sync=True,
    )
    simulator = GossipSimulator(
        nodes=nodes,
        data_dispatcher=dispatcher,
        delta=10,
        protocol=AntiEntropyProtocol.PUSH,
        delay=ConstantDelay(0),
    )
    return simulator


class TestSimulationReport:
    def test_creation(self):
        report = SimulationReport()
        assert report._sent_messages == 0
        assert report._failed_messages == 0

    def test_update_message_sent(self):
        from gossipy.core import Message, MessageType
        report = SimulationReport()
        msg = Message(0, 0, 1, MessageType.PUSH, (1.0,))
        report.update_message(False, msg)
        assert report._sent_messages == 1

    def test_update_message_failed(self):
        report = SimulationReport()
        report.update_message(True)
        assert report._failed_messages == 1

    def test_clear(self):
        from gossipy.core import Message, MessageType
        report = SimulationReport()
        msg = Message(0, 0, 1, MessageType.PUSH, (1.0,))
        report.update_message(False, msg)
        report.clear()
        assert report._sent_messages == 0

    def test_get_evaluation_empty(self):
        report = SimulationReport()
        assert report.get_evaluation() == []
        assert report.get_evaluation(local=True) == []


class TestGossipSimulator:
    def test_creation(self):
        sim = _make_simulator()
        assert sim.n_nodes == 5
        assert sim.delta == 10

    def test_init_nodes(self):
        sim = _make_simulator()
        sim.init_nodes()
        assert sim.initialized

    def test_start(self):
        sim = _make_simulator()
        report = SimulationReport()
        sim.add_receiver(report)
        sim.init_nodes()
        sim.start(n_rounds=2)
        assert report._sent_messages > 0

    def test_start_with_stop_condition(self):
        sim = _make_simulator()
        report = SimulationReport()
        sim.add_receiver(report)
        sim.init_nodes()

        def stop_after_1_round(round_num, simulator):
            return round_num >= 1

        sim.start(n_rounds=10, stop_condition=stop_after_1_round)
        # Should have stopped early
        assert report._sent_messages > 0

    def test_str(self):
        sim = _make_simulator()
        s = str(sim)
        assert "GossipSimulator" in s

    def test_remove_receiver(self):
        sim = _make_simulator()
        report = SimulationReport()
        sim.add_receiver(report)
        sim.remove_receiver(report)
        # Should not raise when starting without receiver
        sim.init_nodes()
        sim.start(n_rounds=1)

    def test_evaluation_with_sampling(self):
        set_seed(42)
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        data_handler = ClassificationDataHandler(X, y, test_size=0.2)
        dispatcher = DataDispatcher(data_handler, n=5)
        p2p_net = StaticP2PNetwork(5)
        net = TorchMLP(10, 3, hidden_dims=(20,))
        model_handler = TorchModelHandler(
            net=net,
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            criterion=torch.nn.CrossEntropyLoss(),
            device="cpu"
        )
        nodes = GossipNode.generate(
            data_dispatcher=dispatcher,
            p2p_net=p2p_net,
            model_proto=model_handler,
            round_len=10,
            sync=True,
        )
        sim = GossipSimulator(
            nodes=nodes,
            data_dispatcher=dispatcher,
            delta=10,
            protocol=AntiEntropyProtocol.PUSH,
            sampling_eval=0.5,
        )
        report = SimulationReport()
        sim.add_receiver(report)
        sim.init_nodes()
        sim.start(n_rounds=2)
        assert len(report.get_evaluation(local=True)) > 0
