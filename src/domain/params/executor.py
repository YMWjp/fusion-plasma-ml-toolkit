from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import REG
from .context import Context


def _deps_of(name: str) -> list[str]:
    return REG.meta.get(name, {}).get("deps", [])

def _expand(targets: Iterable[str]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    temp: set[str] = set()

    def dfs(x: str):
        if x in seen:
            return
        if x in temp:
            raise RuntimeError(f"Cyclic dependency at {x}")
        temp.add(x)
        if x in REG.by_name:
            for d in _deps_of(x):
                dfs(d)
        temp.remove(x)
        seen.add(x)
        order.append(x)

    for t in targets:
        dfs(t)
    return order

def compute(ctx: Context, requested: list[str], *, strict: bool=True) -> dict[str, Any]:
    order = _expand(requested)
    results: dict[str, Any] = {}
    for name in order:
        if name not in REG.by_name:
            raise KeyError(f"Unknown parameter: {name}")
        deps = {d: results[d] for d in _deps_of(name)}
        try:
            results[name] = REG.by_name[name](ctx, deps)
        except Exception:
            if strict:
                raise
            results[name] = None
    return {k: results.get(k) for k in requested}

def list_required_egs(requested: list[str]) -> tuple[list[str], dict[str, str]]:
    order = _expand(requested)
    need_keys: set[str] = set()
    for p in order:
        need_keys.update(REG.meta.get(p, {}).get("needs", []))

    required_egs = sorted(need_keys)
    data_sources = {}
    for eg in required_egs:
        data_sources[eg] = f"{{root}}/{eg}@{{shotNO}}.dat"
    return required_egs, data_sources