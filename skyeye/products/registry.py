# -*- coding: utf-8 -*-
"""Product-level aggregated strategy registry.

Fans out to individual product registries (dividend_low_vol, tx1, etc.)
so that evaluation tools like rolling_score can depend on a single
unified interface instead of hardcoding a specific product.
"""

from __future__ import annotations

from pathlib import Path


def _collect_registries():
    """Lazily collect available product registries."""
    registries = []
    try:
        from skyeye.products.dividend_low_vol import registry as dlv_registry
        registries.append(dlv_registry)
    except ImportError:
        pass
    try:
        from skyeye.products.tx1 import registry as tx1_registry
        registries.append(tx1_registry)
    except ImportError:
        pass
    return registries


def find_strategy_spec_by_file(strategy_file: str | Path):
    """Find a strategy spec by its entrypoint file path.

    Tries each product registry in order and returns the first match,
    or None if no product claims the file.
    """
    for registry in _collect_registries():
        spec = registry.find_strategy_spec_by_file(strategy_file)
        if spec is not None:
            return spec
    return None


def list_all_strategy_specs() -> list:
    """Return combined strategy specs from all product registries."""
    specs = []
    for registry in _collect_registries():
        specs.extend(registry.list_strategy_specs())
    return specs
