"""
Click management helpers for planar pose UI (Assignment 2)
"""
import numpy as np

class ClickStore:
    def __init__(self):
        self._pts = []
    def add(self, u: int, v: int) -> None:
        self._pts.append((u, v))
    def undo(self) -> None:
        if self._pts:
            self._pts.pop()
    def clear(self) -> None:
        self._pts.clear()
    def as_array(self, dtype=float) -> np.ndarray:
        return np.array(self._pts, dtype=dtype) if self._pts else np.empty((0,2), dtype=dtype)
    def __len__(self):
        return len(self._pts)
    def __iter__(self):
        return iter(self._pts)
    def __repr__(self):
        return f"ClickStore({self._pts})"

# Quick self-test
if __name__ == "__main__":
    cs = ClickStore()
    cs.add(10, 20)
    cs.add(30, 40)
    cs.add(50, 60)
    cs.undo()
    assert len(cs) == 2
    arr = cs.as_array()
    assert arr.shape == (2,2)
    print("ClickStore self-test passed.")
