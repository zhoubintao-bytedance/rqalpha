from skyeye.products.tx1 import run_live_advisor


def test_render_table_uses_readable_headers_and_interpretation(monkeypatch):
    """验证表格输出会补充结果解读、中文列名和易读格式。"""
    monkeypatch.setattr(
        run_live_advisor,
        "_resolve_stock_name_map",
        lambda order_book_ids: {"000001.XSHE": "平安银行"},
    )
    monkeypatch.setattr(run_live_advisor, "_supports_color_output", lambda: False)
    result = {
        "package_id": "tx1_demo",
        "gate_level": "canary_live",
        "score_date": "2026-03-31",
        "fit_end_date": "2026-03-03",
        "status": "ok",
        "recommendations": [
            {
                "order_book_id": "000001.XSHE",
                "rank": 1,
                "percentile": 1.0,
                "win_rate": 0.48,
                "mean_return": 0.0108,
                "median_return": -0.0032,
                "return_quantile_range": {"p25": -0.0473, "p75": 0.0506},
                "sample_count": 49952,
            }
        ],
    }

    output = run_live_advisor.render_table(result)

    assert "结果解读:" in output
    assert "不是高胜率信号" in output
    assert "右偏收益排序器" in output
    assert "指标说明:" in output
    assert "分位: 当日横截面分位" in output
    assert "代码" in output
    assert "股票名" in output
    assert "均值收益" in output
    assert "平安银行" in output
    assert "100.0%" in output
    assert "48.0%" in output
    assert "+1.08%" in output
    assert "-0.32%" in output
    assert "49,952" in output


def test_render_table_colors_positive_and_negative_returns(monkeypatch):
    """验证终端支持颜色时，正负收益会用不同颜色渲染。"""
    monkeypatch.setattr(run_live_advisor, "_resolve_stock_name_map", lambda order_book_ids: {})
    monkeypatch.setattr(run_live_advisor, "_supports_color_output", lambda: True)
    result = {
        "package_id": "tx1_demo",
        "gate_level": "canary_live",
        "score_date": "2026-03-31",
        "fit_end_date": "2026-03-03",
        "status": "ok",
        "recommendations": [
            {
                "order_book_id": "000001.XSHE",
                "rank": 1,
                "percentile": 1.0,
                "win_rate": 0.52,
                "mean_return": 0.0108,
                "median_return": -0.0032,
                "return_quantile_range": {"p25": -0.0473, "p75": 0.0506},
                "sample_count": 49952,
            }
        ],
    }

    output = run_live_advisor.render_table(result)

    assert "\x1b[32m" in output
    assert "\x1b[31m" in output


def test_render_table_emits_red_warning_for_stale_requested_trade_date(monkeypatch):
    """验证 stop-serve 时会用红色 warning 明确提示请求日和最新快照日不一致。"""
    monkeypatch.setattr(run_live_advisor, "_supports_color_output", lambda: True)
    result = {
        "package_id": "tx1_demo",
        "gate_level": "canary_live",
        "requested_trade_date": "2026-04-12",
        "latest_available_trade_date": "2026-03-31",
        "score_date": "2026-03-31",
        "fit_end_date": "2026-03-03",
        "status": "stopped",
        "reasons": ["freshness_requested_trade_date_exceeded"],
        "warnings": [
            {
                "level": "critical",
                "code": "requested_trade_date_stale",
                "message": "WARNING: requested_trade_date=2026-04-12 but latest_available_trade_date=2026-03-31; advisor stopped, old snapshot will not be used as today's advice.",
            }
        ],
        "recommendations": [],
    }

    output = run_live_advisor.render_table(result)

    assert "WARNING:" in output
    assert "2026-04-12" in output
    assert "2026-03-31" in output
    assert "\x1b[31m" in output
