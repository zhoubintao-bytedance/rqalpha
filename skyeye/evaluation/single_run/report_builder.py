"""Build structured summaries for single-run RQAlpha backtests."""

from collections import OrderedDict


def build_single_run_summary(strategy_spec, rqalpha_summary, plot_path=None):
    """Normalize single-run outputs into a stable report shape."""
    spec = strategy_spec or {}
    summary = rqalpha_summary or {}
    return OrderedDict(
        [
            ("strategy_id", spec.get("strategy_id")),
            ("strategy_name", spec.get("name")),
            ("benchmark", spec.get("benchmark")),
            ("instrument", spec.get("instrument")),
            ("hypothesis", spec.get("hypothesis")),
            ("plot_path", plot_path),
            ("summary", summary),
        ]
    )
