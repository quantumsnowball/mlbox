from collections import deque


class Replay:
    def __init__(self) -> None:
        self._memory = deque(maxlen=5)
        self._cache = deque()

    def remember(self, item):
        self._memory.append(item)

    def cache(self, item):
        self._cache.append(item)

    def flush(self):
        self._memory.extend(self._cache)
        self._cache.clear()


rp = Replay()

for i in range(7):
    rp.remember(i)

print(rp._memory)
print(rp._cache)
print()

for i in ['a', 'b', 'c', 'd', 'e', 'f']:
    rp.cache(i)

print(rp._memory)
print(rp._cache)
print()

rp._cache[-1] = 'F'

print(rp._memory)
print(rp._cache)
print()

rp.flush()

print(rp._memory)
print(rp._cache)
print()
