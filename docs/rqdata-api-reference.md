# RQData Python API Quick Reference (A-share Stock Module)

> Source: https://www.ricequant.com/doc/rqdata/python/stock-mod and generic-api
> Auto-extracted 2026-03-31. For full field lists see official docs.

## Initialization

```python
import rqdatac
rqdatac.init(username="license", password="<license_key>")
```

---

## 1. Instrument & Calendar APIs (generic-api)

### `all_instruments(type=None, date=None, market='cn')`
List all instruments. `type`: `'CS'`(stock), `'ETF'`, `'LOF'`, `'INDX'`(index), `'Future'`, `'Spot'`, `'Option'`, `'Convertible'`, `'Repo'`, `'REITs'`, `'FUND'`. Returns DataFrame.

### `instruments(order_book_ids, market='cn')`
Get detailed Instrument object(s). Fields: `order_book_id`, `symbol`, `listed_date`, `de_listed_date`, `type`, `exchange`, `board_type`(`MainBoard`/`GEM`/`SME`/`KSH`), `status`(`Active`/`Delisted`/`TemporarySuspended`/`PreIPO`), `special_type`(`Normal`/`ST`/`StarST`/`PT`), `market_tplus`.
Methods: `days_from_listed(date)`, `days_to_expire(date)`, `tick_size()`.

### `id_convert(order_book_ids, to=None)`
Convert between RQData codes and exchange codes. `to='normal'`: RQData->exchange; `to=None`: exchange->RQData.

### `get_trading_dates(start_date, end_date, market='cn')`
Returns `list[datetime.date]` of trading dates (inclusive).

### `get_previous_trading_date(date, n=1, market='cn')`
N-th trading date before `date`. Returns `datetime.date`.

### `get_next_trading_date(date, n=1, market='cn')`
N-th trading date after `date`. Returns `datetime.date`.

### `get_latest_trading_date(market='cn')`
Most recent trading date (today if trading day, else previous).

### `get_trading_periods(order_book_ids, start_date=None, end_date=None, frequency='1m', market='cn')`
Continuous auction trading hours. Replaces deprecated `get_trading_hours`.

---

## 2. Market Data APIs (generic-api)

### `get_price(order_book_ids, start_date=None, end_date=None, frequency='1d', fields=None, adjust_type='pre', skip_suspended=False, expect_df=True, time_slice=None, market='cn')`
Core price data API. Supports all instrument types.
- `frequency`: `'1d'`, `'1w'`, `'1m'`/`'5m'`/`'15m'`/`'60m'`, `'tick'`
- `adjust_type`: `'pre'`(forward), `'post'`(backward), `'none'`, `'pre_volume'`, `'post_volume'`
- `skip_suspended`: `False`(default, fills last price) / `True`
- `time_slice`: intraday time filter, e.g. `('23:55', '09:05')`. Fetches full data first then slices.
- Bar fields: `open`, `close`, `high`, `low`, `limit_up`, `limit_down`, `total_turnover`, `volume`, `num_trades`, `prev_close`, `settlement`(futures), `open_interest`(futures/options), `iopv`(ETF)
- Tick fields: `last`, `a1`-`a5`, `a1_v`-`a5_v`, `b1`-`b5`, `b1_v`-`b5_v`, `change_rate`, plus bar fields

### `get_price_change_rate(order_book_ids, start_date=None, end_date=None, expect_df=True, market='cn')`
Daily return rate based on backward-adjusted prices.

### `get_vwap(order_book_ids, start_date=None, end_date=None, frequency='1d')`
VWAP. Supports `'1d'`, `'1m'`/`'5m'` etc.

### `current_snapshot(order_book_ids, market='cn')`
Real-time L1 snapshot. Returns `Tick` object(s) with `asks`/`bids` lists, `trading_phase_code`, etc.

### `current_minute(order_book_ids, skip_suspended=False, fields=None, market='cn')`
Latest 1-min bar (real-time, no history).

### `get_ticks(order_book_id, start_date=None, end_date=None, expect_df=True, market='cn')`
Current day tick snapshot (trial). Single instrument only.

### `get_live_ticks(order_book_ids, start_dt=None, end_dt=None, fields=None, market='cn')`
Current day tick with intraday time slicing. Supports multiple instruments.

### `get_auction_info(order_book_ids, start_date=None, end_date=None, frequency='1d', fields=None, market='cn')`
After-hours fixed-price trading data (科创板/创业板).

### `get_open_auction_info(order_book_ids, start_date=None, end_date=None, fields=None, market='cn')`
Pre-market opening auction (集合竞价) snapshot.

### `get_yield_curve(start_date=None, end_date=None, tenor=None, market='cn')`
China gov bond yield curve (ChinaBond, 2002+). Tenors: `'0S'`, `'1M'`, `'1Y'`, `'10Y'`, etc.

### `LiveMarketDataClient` (WebSocket Push)
```python
client = LiveMarketDataClient()
client.subscribe('tick_000001.XSHE')      # tick
client.subscribe('bar_000001.XSHE')       # 1m bar
client.subscribe('bar_000001.XSHE_5m')    # 5m bar
for market in client.listen(): print(market)
```

---

## 3. Index Data APIs (indices-mod)

### `index_components(order_book_id, date=None, start_date=None, end_date=None, market='cn', return_create_tm=False)`
Get index constituent list. Returns `list[order_book_id]` or DataFrame (with date range).

### `index_weights(order_book_id, date=None)`
Get index constituent weights (monthly updated). Returns DataFrame indexed by order_book_id with weight values.

### `index_indicator(order_book_ids, start_date, end_date, fields=None)`
Index daily valuation: `pe_lyr`, `pe_ttm`, `pb_lf`, `pb_lyr`, `pb_ttm`, `total_market_value`, `circulation_market_value`, `free_circulation_market_value`, `dividend_yield_ttm`.

---

## 4. Financial Data APIs (stock-mod)

### `get_pit_financials_ex(order_book_ids, fields, start_quarter, end_quarter, date=None, statements='latest', market='cn')`
Point-in-time quarterly financial data (三大表). Quarter format: `'2015q2'`=H1, `'2015q4'`=annual.
- `statements`: `'latest'`(most recent record) / `'all'`(all revisions)
- Returns: DataFrame with `quarter`, `info_date`, `if_adjusted`(0=current, 1=restated), financial fields
- Available tables: Income Statement (利润表 ~80 fields), Balance Sheet (资产负债表 ~100 fields), Cash Flow Statement (现金流量表 ~60 fields)
- Key fields: `revenue`, `net_profit`, `total_assets`, `total_liabilities`, `total_equity`, `net_operate_cash_flow`, `basic_eps`

### `current_performance(order_book_ids, info_date=None, quarter=None, interval='1q', fields=None, market='cn')`
Financial express reports (快报). Key fields: `operating_revenue`, `np_parent_owners`, `net_profit_cut`, `total_assets`, `basic_eps`, `roe`, plus YoY changes.

### `performance_forecast(order_book_ids, info_date=None, end_date=None, fields=None, market='cn')`
Earnings forecast (业绩预告). Fields: `forecast_type`, `forecast_growth_rate_floor/ceiling`, `forecast_np_floor/ceiling`, `forecast_eps_floor/ceiling`.

### `get_audit_opinion(order_book_ids, start_quarter, end_quarter, date=None, type=None, opinion_types=None, market='cn')`
Audit opinions. `type`: `'financial_statements'` / `'internal_control'`. Returns `opinion_type`.

---

## 5. Factor Data APIs (stock-mod)

### `get_factor(order_book_ids, factor, start_date=None, end_date=None, universe=None, expect_df=True, market='cn')`
Get factor values. Use `get_all_factor_names()` for valid factor list.

### `get_all_factor_names(type=None, market='cn')`
List available factors by category:

| type | Category | Key Factors |
|------|----------|-------------|
| `'eod_indicator'` | Valuation | `pe_ratio_ttm`, `pb_ratio_lf`, `ps_ratio_ttm`, `market_cap`, `market_cap_2`(float), `market_cap_3`(total), `dividend_yield_ttm`, `ev_ratio` |
| `'operational_indicator'` | Operating | `eps_ttm`, `roe_ttm`, `roa_ttm`, `gross_profit_margin_ttm`, `net_profit_margin_ttm` |
| `'cash_flow_indicator'` | Cash Flow | `cash_flow_per_share_ttm`, `free_cash_flow_per_share_ttm` |
| `'financial_indicator'` | Financial | `debt_to_equity_ratio`, `current_ratio`, `quick_ratio`, `interest_bearing_debt` |
| `'growth_indicator'` | Growth | `revenue_growth_yoy`, `net_profit_growth_yoy`, `eps_growth_yoy` |
| `'income_statement'` | Income(raw) | Base fields + `_mrq_n`, `_ttm_n`, `_lyr_n` suffixes (n=0-11) |
| `'balance_sheet'` | Balance(raw) | Same suffix pattern |
| `'cash_flow_statement'` | CashFlow(raw) | Same suffix pattern |
| `'alpha101'` | Alpha101 | 101 quantitative alpha factors |
| `'moving_average_indicator'` | MA | `MA5`..`MA250`, `EMA12`, `EMA26`, `MACD`, `BOLL_UPPER/MIDDLE/LOWER` |
| `'obos_indicator'` | Overbought/sold | `KDJ_K/D/J`, `RSI6`, `RSI10`, `WR`, `CCI`, `ATR`, `ADX` |
| `'energy_indicator'` | Energy | `OBV`, `VR`, `CR`, `PSY`, `EMV` |

---

## 6. Share Capital & Ownership APIs (stock-mod)

### `get_shares(order_book_ids, start_date=None, end_date=None, fields=None, expect_df=True, market='cn')`
Share capital. Fields: `total`, `circulation_a`, `non_circulation_a`, `total_a`, `free_circulation`(from 2005), `preferred_shares`.

### `get_main_shareholder(order_book_ids, start_date=None, end_date=None, is_total=False, start_rank=None, end_rank=None, market='cn')`
Major shareholders (top-10). `is_total`: `False`=based on float shares, `True`=total shares.

### `get_holder_number(order_book_ids, start_date=None, end_date=None, market='cn')`
Shareholder count. Fields: `share_holders`, `a_share_holders`, `avg_share_holders`.

### `get_restricted_shares(order_book_ids, start_date=None, end_date=None, market='cn')`
Lock-up expiry details. Fields: `relieve_date`, `relieve_shares`, `reason`.

### `get_leader_shares_change(order_book_ids, start_date=None, end_date=None, market='cn')`
Executive shareholding changes. Fields: `leader_name`, `shares_change`, `price_change`, `change_reason`.

### `get_staff_count(order_book_ids, start_date=None, end_date=None, market='cn')`
Employee count.

---

## 7. Dividend & Split APIs (stock-mod)

### `get_dividend(order_book_ids, start_date=None, end_date=None, expect_df=True, market='cn')`
Cash dividend data. Fields: `dividend_cash_before_tax`, `ex_dividend_date`, `payable_date`, `quarter`.

### `get_dividend_info(order_book_ids, start_date=None, end_date=None, market='cn')`
All dividend types. `dividend_type`: `'transferred share'`, `'bonus share'`, `'cash'`, `'cash and share'`.

### `get_dividend_amount(order_book_ids, start_quarter=None, end_quarter=None, date=None, market='cn')`
Dividend total amount. `event_procedure`: `'预案'`/`'决案'`/`'方案实施'`.

### `get_split(order_book_ids, start_date=None, end_date=None, market='cn')`
Stock splits. Fields: `split_coefficient_from`, `split_coefficient_to`, `cum_factor`.

### `get_ex_factor(order_book_ids, start_date=None, end_date=None, market='cn')`
Adjustment factors. Fields: `ex_factor`(period), `ex_cum_factor`(cumulative).

---

## 8. Industry & Concept APIs (stock-mod)

### `get_instrument_industry(order_book_ids, source='citics_2019', level=1, date=None, market='cn')`
Get stock's industry classification. `source`: `'citics_2019'`(CITIC new), `'citics'`(old), `'gildata'`(聚源). `level`: 0(all), 1/2/3, or `'citics_sector'`(derived sector/style).

### `get_industry(industry, source='citics_2019', date=None, market='cn')`
Get stocks in a specific industry (by name, index code, or classification code).

### `get_industry_mapping(source='citics_2019', date=None, market='cn')`
Full industry classification overview with 3-level codes and names.

### `get_industry_change(industry, source='citics_2019', level=None, market='cn')`
Stock inclusion/removal dates for an industry.

### `sector(code, market='cn')`
MSCI GICS sector stocks. Codes: `Energy`, `Materials`, `ConsumerDiscretionary`, `ConsumerStaples`, `HealthCare`, `Financials`, `RealEstate`, `InformationTechnology`, `TelecommunicationServices`, `Utilities`, `Industrials`.

### `industry(code, market='cn')`
National economy industry codes (A01-S90).

### `get_concept_list(start_date=None, end_date=None, market='cn')`
List all concept/theme names.

### `get_concept(concepts, start_date=None, end_date=None, market='cn')`
Stocks belonging to given concept(s). Returns DataFrame with `concept`, `order_book_id`, `inclusion_date`.

### `get_stock_concept(order_book_ids, market='cn')`
Concepts for given stocks (reverse lookup).

---

## 9. Turnover, Suspension, ST APIs (stock-mod)

### `get_turnover_rate(order_book_ids, start_date=None, end_date=None, fields=None, expect_df=True, market='cn')`
Historical turnover rate. `fields`: `'today'`, `'week'`, `'month'`, `'year'`, `'current_year'`.

### `current_freefloat_turnover(order_book_ids)`
Real-time free-float turnover = cumulative daily turnover / free-float market cap.

### `is_suspended(order_book_ids, start_date=None, end_date=None, market='cn')`
Boolean DataFrame of suspension status.

### `is_st_stock(order_book_ids, start_date=None, end_date=None, market='cn')`
Boolean DataFrame of ST status (covers S*ST, *ST, ST, SST).

### `get_special_treatment_info(order_book_ids, market='cn')`
ST status change history.

### `get_symbol_change_info(order_book_ids, market='cn')`
Stock name change history.

---

## 10. Capital Flow APIs (stock-mod)

### `get_capital_flow(order_book_ids, start_date=None, end_date=None, frequency='1d', market='cn')`
Capital inflow/outflow. `frequency`: `'1d'`, `'1m'`, `'tick'`.
- Daily/Minute: `buy_volume`, `buy_value`, `sell_volume`, `sell_value`
- Tick: `direction`(1=active buy, -1=active sell), `volume`, `value`

### `current_capital_flow_minute(order_book_ids, market='cn')`
Latest minute capital flow (real-time, no history).

---

## 11. Margin Trading APIs (stock-mod, 融资融券)

### `get_securities_margin(order_book_ids, start_date=None, end_date=None, fields=None, expect_df=True, market='cn')`
Margin trading data (from 2010-03-31). Also accepts `'XSHG'`/`'XSHE'` for market-level.
Fields: `margin_balance`(融资余额), `buy_on_margin_value`(融资买入额), `short_balance`(融券余额), `short_sell_quantity`, `total_balance`.

### `get_margin_stocks(date=None, exchange=None, margin_type='stock', market='cn')`
Margin-eligible stock list. `margin_type`: `'stock'`(short selling) / `'cash'`(margin buying).

### `get_eligible_securities_margin(date=None, exchange=None, market='cn')`
Margin collateral eligible securities.

---

## 12. Stock Connect APIs (stock-mod, 沪深港通)

### `get_stock_connect(order_book_ids, start_date=None, end_date=None, fields=None, expect_df=True)`
Northbound/southbound holding data. `order_book_ids`: individual stock, `'shanghai_connect'`, `'shenzhen_connect'`, or `'all_connect'`.
Fields: `shares_holding`, `holding_ratio`, `adjusted_holding_ratio`.

### `current_stock_connect_quota(connect=None, fields=None)`
Real-time daily quota. `connect`: `'hk_to_sh'`, `'hk_to_sz'`, `'sh_to_hk'`, `'sz_to_hk'`.

### `get_stock_connect_quota(connect=None, start_date=None, end_date=None, fields=None)`
Historical daily quota.

---

## 13. Corporate Action APIs (stock-mod)

### `get_private_placement(order_book_ids, start_date=None, end_date=None, progress='complete', issue_type='private', market='cn')`
Private placement (定增). `progress`: `'complete'`/`'incomplete'`/`'all'`. `issue_type`: `'private'`/`'public'`/`'all'`.

### `get_allotment(order_book_ids, start_date=None, end_date=None, fields=None, market='cn')`
Rights issue (配股). Fields: `proportion`, `allotment_price`, `ex_right_date`.

### `get_buy_back(order_book_ids, start_date=None, end_date=None, fields=None, market='cn')`
Share buyback. Fields: `buy_back_volume`, `buy_back_value`, `buy_back_price`, `purpose`.

### `get_incentive_plan(order_book_ids, start_date=None, end_date=None, market='cn')`
Equity incentive plan. Fields: `shares_num`, `incentive_price`, `incentive_mode`.

### `get_block_trade(order_book_ids, start_date=None, end_date=None, market='cn')`
Block trade (大宗交易). Fields: `price`, `volume`, `total_turnover`, `buyer`, `seller`.

### `get_announcement(order_book_ids, start_date=None, end_date=None, fields=None, market='cn')`
Company announcements. Fields: `category`, `title`, `announcement_link`.

### `get_investor_ra(order_book_ids, start_date=None, end_date=None, market='cn')`
Investor relations activity (调研). Fields: `participant`, `institution`, `detail`.

### `get_forecast_report_date(order_book_ids, start_quarter, end_quarter, market='cn')`
Scheduled report disclosure dates.

### `get_share_transformation(predecessor=None, market='cn')`
Stock code change info (e.g. mergers, code migration).

---

## 14. Dragon & Tiger Board APIs (stock-mod, 龙虎榜)

### `get_abnormal_stocks(start_date=None, end_date=None, types=None, market='cn')`
Dragon & Tiger Board daily summary. `types`: anomaly codes (e.g. `'U01'`=daily rise deviation >= 7%).

### `get_abnormal_stocks_detail(order_book_ids, start_date=None, end_date=None, sides=None, types=None, market='cn')`
Institutional trade details. `sides`: `'buy'`, `'sell'`, `'cum'`. Fields: `agency`, `buy_value`, `sell_value`.

---

## Common Parameter Patterns

- **Date types**: `int`(20230101), `str`('2023-01-01'), `datetime.date`, `datetime.datetime`, `pd.Timestamp`
- **market**: `'cn'`(mainland China, default), `'hk'`(Hong Kong) - not all APIs support HK
- **order_book_id format**: `'000001.XSHE'`(SZSE), `'600000.XSHG'`(SSE), `'000300.XSHG'`(index)
- **Quarter format**: `'2015q1'`(Q1), `'2015q2'`(H1), `'2015q3'`(Q3), `'2015q4'`(annual)
- **expect_df**: `True`(default) returns DataFrame, `False` returns legacy structure
- Default date range for most APIs: last 3 months if omitted
- Returns `None` for stocks not listed or already delisted during query period

## API Count Summary

| Category | Count | Key APIs |
|----------|-------|----------|
| Instrument & Calendar | 8 | `all_instruments`, `instruments`, `id_convert`, `get_trading_dates`, `get_previous/next_trading_date` |
| Market Data | 11 | `get_price`, `current_snapshot`, `get_ticks`, `get_live_ticks`, `get_vwap`, `get_yield_curve` |
| Index Data | 3 | `index_components`, `index_weights`, `index_indicator` |
| Financial Data | 4 | `get_pit_financials_ex`, `current_performance`, `performance_forecast`, `get_audit_opinion` |
| Factor Data | 2 | `get_factor`, `get_all_factor_names` |
| Share Capital | 5 | `get_shares`, `get_main_shareholder`, `get_holder_number`, `get_restricted_shares`, `get_leader_shares_change` |
| Dividend & Split | 5 | `get_dividend`, `get_dividend_info`, `get_dividend_amount`, `get_split`, `get_ex_factor` |
| Industry & Concept | 9 | `get_instrument_industry`, `get_industry`, `get_concept`, `sector`, `industry` |
| Turnover/ST/Suspension | 6 | `get_turnover_rate`, `is_suspended`, `is_st_stock` |
| Capital Flow | 2 | `get_capital_flow`, `current_capital_flow_minute` |
| Margin Trading | 3 | `get_securities_margin`, `get_margin_stocks`, `get_eligible_securities_margin` |
| Stock Connect | 3 | `get_stock_connect`, `current/get_stock_connect_quota` |
| Corporate Actions | 9 | `get_private_placement`, `get_buy_back`, `get_block_trade`, `get_announcement` |
| Dragon & Tiger | 2 | `get_abnormal_stocks`, `get_abnormal_stocks_detail` |
| **Total** | **~72** | |
