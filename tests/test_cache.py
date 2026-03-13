import pytest

from gossipy import Cache, CacheKey, CacheItem, set_seed


class TestCacheKey:
    def test_creation(self):
        key = CacheKey(1, 2)
        assert key.get() == (1, 2)

    def test_equality(self):
        k1 = CacheKey(1, 2)
        k2 = CacheKey(1, 2)
        assert k1 == k2

    def test_inequality(self):
        k1 = CacheKey(1, 2)
        k2 = CacheKey(1, 3)
        assert k1 != k2

    def test_inequality_with_non_cachekey(self):
        k = CacheKey(1, 2)
        assert k != "not a key"

    def test_hash(self):
        k1 = CacheKey(1, 2)
        k2 = CacheKey(1, 2)
        assert hash(k1) == hash(k2)

    def test_repr(self):
        k = CacheKey(1, 2)
        assert "(1, 2)" in repr(k)


class TestCacheItem:
    def test_creation(self):
        item = CacheItem(42)
        assert item.get() == 42
        assert item.is_referenced()

    def test_add_ref(self):
        item = CacheItem(42)
        item.add_ref()
        assert item._refs == 2

    def test_del_ref(self):
        item = CacheItem(42)
        val = item.del_ref()
        assert val == 42
        assert not item.is_referenced()

    def test_get_size_scalar(self):
        item = CacheItem(42)
        assert item.get_size() == 1

    def test_get_size_bool(self):
        item = CacheItem(True)
        assert item.get_size() == 1

    def test_str(self):
        item = CacheItem(42)
        assert "42" in str(item)


class TestCache:
    def test_push_and_pop(self):
        cache = Cache()
        cache.clear()
        key = CacheKey(1, 1)
        cache.push(key, "value")
        val = cache.pop(key)
        assert val == "value"

    def test_pop_removes_unreferenced(self):
        cache = Cache()
        cache.clear()
        key = CacheKey(2, 1)
        cache.push(key, "value")
        cache.pop(key)
        assert cache[key] is None

    def test_multiple_refs(self):
        cache = Cache()
        cache.clear()
        key = CacheKey(3, 1)
        cache.push(key, "value")
        cache.push(key, "value")  # adds a ref
        cache.pop(key)
        assert cache[key] == "value"  # still referenced
        cache.pop(key)
        assert cache[key] is None  # now removed

    def test_pop_nonexistent(self):
        cache = Cache()
        cache.clear()
        key = CacheKey(99, 99)
        assert cache.pop(key) is None

    def test_getitem_nonexistent(self):
        cache = Cache()
        cache.clear()
        key = CacheKey(99, 99)
        assert cache[key] is None

    def test_clear(self):
        cache = Cache()
        key = CacheKey(4, 1)
        cache.push(key, "val")
        cache.clear()
        assert cache[key] is None

    def test_load(self):
        cache = Cache()
        cache.clear()
        key = CacheKey(5, 1)
        item = CacheItem("loaded_value")
        cache.load({key: item})
        assert cache[key] == "loaded_value"

    def test_get_cache(self):
        cache = Cache()
        cache.clear()
        key = CacheKey(6, 1)
        cache.push(key, "val")
        c = cache.get_cache()
        assert key in c


class TestSetSeed:
    def test_set_seed_runs(self):
        set_seed(42)
        # Just ensure it runs without error
