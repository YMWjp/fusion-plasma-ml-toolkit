from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

ParamFunc = Callable[[Any, dict[str, Any]], Any]

class _Registry:
    def __init__(self) -> None:
        self.by_name: dict[str, ParamFunc] = {}
        self.meta: dict[str, dict[str, Any]] = {}

    def register(self, name: str, func: ParamFunc, *, deps: Iterable[str]=(), needs: Iterable[str]=(), doc: str=""):
        if name in self.by_name:
            raise ValueError(f"Parameter '{name}' already registered")
        self.by_name[name] = func
        self.meta[name] = {"deps": list(deps), "needs": list(needs), "doc": doc}

REG = _Registry()

def param(name: str, *, deps: Iterable[str]=(), needs: Iterable[str]=(), doc: str=""):
    def deco(fn: ParamFunc) -> ParamFunc:
        REG.register(name, fn, deps=deps, needs=needs, doc=doc)
        return fn
    return deco