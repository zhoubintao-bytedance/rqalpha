import json

def test_load_holdings_file_from_csv_normalizes_weights(tmp_path):
    """验证 CSV 持仓文件会被解析并归一化。"""
    from skyeye.products.tx1.live_advisor.holdings_io import load_holdings_file

    path = tmp_path / "holdings.csv"
    path.write_text(
        "order_book_id,weight\n000001.XSHE,60\n000002.XSHE,40\n",
        encoding="utf-8",
    )

    holdings = load_holdings_file(path)

    assert set(holdings) == {"000001.XSHE", "000002.XSHE"}
    assert abs(sum(holdings.values()) - 1.0) < 1e-9


def test_load_holdings_file_from_json_filters_invalid_rows(tmp_path):
    """验证 JSON 持仓文件会过滤非法代码和非正权重。"""
    from skyeye.products.tx1.live_advisor.holdings_io import load_holdings_file

    path = tmp_path / "holdings.json"
    path.write_text(
        json.dumps(
            {
                "000001.XSHE": 0.4,
                "000002.XSHE": 0.6,
                "": 0.2,
                "000004.XSHE": 0.0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    holdings = load_holdings_file(path)

    assert set(holdings) == {"000001.XSHE", "000002.XSHE"}
    assert abs(sum(holdings.values()) - 1.0) < 1e-9
