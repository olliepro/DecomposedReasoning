"""Small executor-local LRU cache helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class LocalLruCache(Generic[KeyT, ValueT]):
    """Store a bounded number of recently used in-memory values.

    Args:
        max_entries: Maximum number of entries retained in the cache.

    Returns:
        Executor-local cache that evicts the least recently used key first.

    Example:
        >>> cache = LocalLruCache[str, int](max_entries=2)
        >>> _ = cache.set(key="a", value=1)
        >>> cache.get(key="a")
        1
    """

    def __init__(self, *, max_entries: int) -> None:
        assert max_entries > 0, "max_entries must be positive"
        self.max_entries = max_entries
        self._entries: OrderedDict[KeyT, ValueT] = OrderedDict()

    def get(self, *, key: KeyT) -> ValueT | None:
        """Return the cached value for `key`, or `None` when absent.

        Args:
            key: Cache lookup key.

        Returns:
            Cached value when present, otherwise `None`.
        """

        if key not in self._entries:
            return None
        value = self._entries.pop(key)
        self._entries[key] = value
        return value

    def set(self, *, key: KeyT, value: ValueT) -> ValueT:
        """Insert or refresh one cached value and return it.

        Args:
            key: Cache entry key.
            value: Cache entry value.

        Returns:
            The stored value.
        """

        if key in self._entries:
            self._entries.pop(key)
        self._entries[key] = value
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return value
