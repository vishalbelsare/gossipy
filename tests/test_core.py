import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gossipy.core import (
    CreateModelMode,
    AntiEntropyProtocol,
    MessageType,
    Message,
    ConstantDelay,
    UniformDelay,
    LinearDelay,
    P2PNetwork,
    StaticP2PNetwork,
)
from gossipy import CacheKey, Sizeable


class TestCreateModelMode:
    def test_values(self):
        assert CreateModelMode.UPDATE.value == 1
        assert CreateModelMode.MERGE_UPDATE.value == 2
        assert CreateModelMode.UPDATE_MERGE.value == 3
        assert CreateModelMode.PASS.value == 4


class TestAntiEntropyProtocol:
    def test_values(self):
        assert AntiEntropyProtocol.PUSH.value == 1
        assert AntiEntropyProtocol.PULL.value == 2
        assert AntiEntropyProtocol.PUSH_PULL.value == 3


class TestMessageType:
    def test_values(self):
        assert MessageType.PUSH.value == 1
        assert MessageType.PULL.value == 2
        assert MessageType.REPLY.value == 3
        assert MessageType.PUSH_PULL.value == 4


class TestMessage:
    def test_creation(self):
        msg = Message(0, 1, 2, MessageType.PUSH, (3.0, 4.0))
        assert msg.timestamp == 0
        assert msg.sender == 1
        assert msg.receiver == 2
        assert msg.type == MessageType.PUSH
        assert msg.value == (3.0, 4.0)

    def test_get_size_none(self):
        msg = Message(0, 1, 2, MessageType.PULL, None)
        assert msg.get_size() == 1

    def test_get_size_tuple_floats(self):
        msg = Message(0, 1, 2, MessageType.PUSH, (1.0, 2.0, 3.0))
        assert msg.get_size() == 3

    def test_get_size_single_int(self):
        msg = Message(0, 1, 2, MessageType.PUSH, 42)
        assert msg.get_size() == 1

    def test_repr(self):
        msg = Message(0, 1, 2, MessageType.PUSH, (1.0,))
        r = repr(msg)
        assert "1 -> 2" in r
        assert "PUSH" in r

    def test_repr_ack(self):
        msg = Message(0, 1, 2, MessageType.PULL, None)
        r = repr(msg)
        assert "ACK" in r


class TestConstantDelay:
    def test_get(self):
        delay = ConstantDelay(5)
        msg = Message(0, 0, 1, MessageType.PUSH, (1.0,))
        assert delay.get(msg) == 5

    def test_zero_delay(self):
        delay = ConstantDelay(0)
        msg = Message(0, 0, 1, MessageType.PUSH, None)
        assert delay.get(msg) == 0

    def test_negative_delay_raises(self):
        with pytest.raises(AssertionError):
            ConstantDelay(-1)

    def test_str(self):
        assert "ConstantDelay(5)" == str(ConstantDelay(5))


class TestUniformDelay:
    def test_get_in_range(self):
        delay = UniformDelay(1, 10)
        msg = Message(0, 0, 1, MessageType.PUSH, (1.0,))
        for _ in range(100):
            d = delay.get(msg)
            assert 1 <= d <= 10

    def test_invalid_range(self):
        with pytest.raises(AssertionError):
            UniformDelay(10, 5)

    def test_negative_min(self):
        with pytest.raises(AssertionError):
            UniformDelay(-1, 5)

    def test_str(self):
        assert "UniformDelay(1, 10)" == str(UniformDelay(1, 10))


class TestLinearDelay:
    def test_get(self):
        delay = LinearDelay(1.0, 2)
        msg = Message(0, 0, 1, MessageType.PUSH, (1.0, 2.0, 3.0))
        d = delay.get(msg)
        assert d == int(1.0 * 3) + 2

    def test_zero_timexunit(self):
        delay = LinearDelay(0, 5)
        msg = Message(0, 0, 1, MessageType.PUSH, (1.0,))
        assert delay.get(msg) == 5

    def test_str(self):
        s = str(LinearDelay(1.0, 2))
        assert "LinearDelay" in s


class TestStaticP2PNetwork:
    def test_fully_connected(self):
        net = StaticP2PNetwork(5)
        assert net.size() == 5
        for i in range(5):
            peers = net.get_peers(i)
            assert peers is None  # fully connected returns None

    def test_with_topology(self):
        topology = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        net = StaticP2PNetwork(3, topology)
        assert net.get_peers(0) == [1]
        assert sorted(net.get_peers(1)) == [0, 2]
        assert net.get_peers(2) == [1]

    def test_with_sparse_topology(self):
        dense = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        sparse = csr_matrix(dense)
        net = StaticP2PNetwork(3, sparse)
        assert net.get_peers(0) == [1]
        assert sorted(net.get_peers(1)) == [0, 2]

    def test_invalid_node_id(self):
        net = StaticP2PNetwork(3)
        with pytest.raises(AssertionError):
            net.get_peers(5)
