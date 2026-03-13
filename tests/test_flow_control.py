import pytest

from gossipy.flow_control import (
    PurelyProactiveTokenAccount,
    PurelyReactiveTokenAccount,
    SimpleTokenAccount,
    GeneralizedTokenAccount,
    RandomizedTokenAccount,
)


class TestPurelyProactiveTokenAccount:
    def test_proactive(self):
        ta = PurelyProactiveTokenAccount()
        assert ta.proactive() == 1

    def test_reactive(self):
        ta = PurelyProactiveTokenAccount()
        assert ta.reactive(1) == 0


class TestPurelyReactiveTokenAccount:
    def test_proactive(self):
        ta = PurelyReactiveTokenAccount()
        assert ta.proactive() == 0

    def test_reactive_useful(self):
        ta = PurelyReactiveTokenAccount(k=2)
        assert ta.reactive(1) == 2

    def test_reactive_not_useful(self):
        ta = PurelyReactiveTokenAccount(k=2)
        assert ta.reactive(0) == 0


class TestSimpleTokenAccount:
    def test_proactive_below_capacity(self):
        ta = SimpleTokenAccount(C=5)
        assert ta.proactive() == 0

    def test_proactive_at_capacity(self):
        ta = SimpleTokenAccount(C=5)
        ta.add(5)
        assert ta.proactive() == 1

    def test_reactive_with_tokens(self):
        ta = SimpleTokenAccount(C=5)
        ta.add(3)
        assert ta.reactive(1) == 1

    def test_reactive_no_tokens(self):
        ta = SimpleTokenAccount(C=5)
        assert ta.reactive(1) == 0

    def test_add_sub(self):
        ta = SimpleTokenAccount(C=5)
        ta.add(3)
        assert ta.n_tokens == 3
        ta.sub(2)
        assert ta.n_tokens == 1
        ta.sub(5)  # should not go below 0
        assert ta.n_tokens == 0

    def test_invalid_capacity(self):
        with pytest.raises(AssertionError):
            SimpleTokenAccount(C=0)


class TestGeneralizedTokenAccount:
    def test_reactive_useful(self):
        ta = GeneralizedTokenAccount(C=10, A=5)
        ta.add(10)
        r = ta.reactive(1)
        assert r > 0

    def test_reactive_not_useful(self):
        ta = GeneralizedTokenAccount(C=10, A=5)
        ta.add(10)
        r = ta.reactive(0)
        assert r >= 0

    def test_invalid_params(self):
        with pytest.raises(AssertionError):
            GeneralizedTokenAccount(C=1, A=5)  # A > C


class TestRandomizedTokenAccount:
    def test_proactive_zero_tokens(self):
        ta = RandomizedTokenAccount(C=10, A=5)
        assert ta.proactive() == 0

    def test_proactive_above_capacity(self):
        ta = RandomizedTokenAccount(C=10, A=5)
        ta.add(15)
        assert ta.proactive() == 1

    def test_proactive_in_range(self):
        ta = RandomizedTokenAccount(C=10, A=5)
        ta.add(7)
        p = ta.proactive()
        assert 0 <= p <= 1

    def test_reactive_useful(self):
        ta = RandomizedTokenAccount(C=10, A=5)
        ta.add(10)
        r = ta.reactive(1)
        assert r >= 0

    def test_reactive_not_useful(self):
        ta = RandomizedTokenAccount(C=10, A=5)
        ta.add(10)
        assert ta.reactive(0) == 0
