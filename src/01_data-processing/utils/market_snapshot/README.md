# market_v1：离线行情数据快照

这个目录的代码把 Yahoo 日线、FRED 宏观序列和 SEC XBRL 财报数据下载下来，按快照截止日 2025-10-30 统一口径，写进一个 SQLite 文件，供离线的 `market` 智能体环境用工具查询。

- 数据库：`$FP_DATA_D2/market_v1/market_v1.sqlite`
- 原始响应：同目录 `raw/` 下每个来源一个 gzip JSONL 文件（`yahoo_chart`、`fred`、`fred_alfred`、`fred_meta`、`sec_companyfacts`、`misc`），每行是一次请求的 URL、状态码、时间和响应体。
- 构建日志：同目录 `build*.log`。

## 时间范围和截止日

所有价格、公司行为和 FRED 观测值的日期都在 2000-01-01 到 2025-10-30 之间，SEC 财报数据只保留 2025-10-30 及以前提交（`filed`）的申报。数据实际下载于 2026-10-31，因此代码把下载日已知、截止日还不知道的信息去掉了：

- **价格按截止日的拆股口径复权。** Yahoo 返回的开高低收和成交量已经按下载日以前的全部拆股复权；如果拆股发生在截止日之后（如 AMCR 2026 年的 5 股合 1 股），代码把价格乘回这些拆股的比例，使 `close` 等于截止日当时看到的拆股复权价，`securities.post_cutoff_split_factor` 记下这个乘数（多数证券为 1）。
- **`adj_close` 也按截止日重算。** Yahoo 的 `adjclose` 包含截止日之后的分红，代码把它除以截止日最后一个交易日的 `adjclose/close`，所以截止日最后一个交易日 `adj_close = close`。
- **FRED 同时存两个版本。** `fred.value` 是下载日的最新值（含之后的修订），`fred.value_vintage` 是 ALFRED 上截止日当天发布的值；截止日之后才首次发布的观测（如 2025 年三季度 GDP）`value_vintage` 为 NULL。按截止日出题应使用 `value_vintage`。

## 表结构

| 表 | 主键 | 内容 |
|---|---|---|
| `securities` | `symbol` | 每个 Yahoo 代码一行：名称、`kind`（equity、index、etf、future、fx、crypto）、交易所、币种、`cik`、`first_date`/`last_date`、时区、来源清单 `list`（sp500、index、etf、future、fx、crypto、extra）、标普 500 成分股的 GICS 行业、`sp500_date_added`、交易日数 `n_days`、`post_cutoff_split_factor` |
| `prices` | `(symbol, date)` | `open high low close volume`：截止日口径的拆股复权值（不含分红）；`adj_close`：拆股加分红复权；`raw_open raw_high raw_low raw_close raw_volume`：重建的不复权值 |
| `actions` | `(symbol, date, kind)` | `kind='split'`：`value` 是新股数/旧股数（1 拆 4 为 4.0，1 合 5 为 0.2），`ratio_text` 是 Yahoo 的原文（如 `4:1`），`split_type` 区分普通拆股 `split`、合股 `reverse_split` 和 Yahoo 记成拆股的分拆或股利调整 `adjustment`；`kind='dividend'`：`value` 是截止日拆股口径的每股分红，`value_raw` 是除息日当时的实际每股分红 |
| `fred_series` | `series_id` | 标题、单位、频率、季调方式、说明 `notes`、`copyright_notice`（说明里出现 copyright 时为 1）、日期范围和观测数，以及 ALFRED 截止日版本的最后日期和观测数 |
| `fred` | `(series_id, date)` | `value`（最新版本）、`value_vintage`（截止日版本） |
| `sec_companies` | `cik` | 主代码 `ticker`、同一 CIK 的全部代码 `tickers`（如 GOOGL,GOOG）、SEC 登记名称、事实行数 `n_facts`、`role`（`current` 或 `predecessor registrant of <代码>`） |
| `facts` | 无（索引 `(cik, tag, end)`） | 每个 XBRL 事实一行：`taxonomy`（us-gaap、ifrs-full 或 dei）、`tag`、`unit`、`val`、`fy`、`fp`、`form`、`start`、`end`、`filed`、`frame`、`accn` |
| `sec_tags` | `(taxonomy, tag)` | 科目的标签和定义 |
| `universe` | `symbol` | 构建用的代码清单；标普 500 成分股来自构建当天的维基百科表格，含 CIK 和 GICS 行业 |
| `fetch_log` | `(source, id)` | 每个代码、序列或公司的下载状态（ok、empty、not_found、fail）、行数和错误信息 |
| `meta` | `key` | 截止日、构建时间、来源 URL、各表行数、失败清单、校验结果 `validation` |

不复权价按"之后生效的拆股比例连乘"重建：某日的 `raw_close = close × ∏(该日之后、截止日及以前生效的拆股比例)`；拆股生效日（除权日）当天已是拆股后的价格。

## 来源

- **Yahoo chart API**（不需要 key）：`https://query1.finance.yahoo.com/v8/finance/chart/{SYM}?period1=..&period2=..&interval=1d&events=div|split&includeAdjustedClose=true`，失败时换 `query2`。User-Agent 用 `Mozilla/5.0`：完整的 Chrome UA 字符串会被返回 HTTP 429。限速每秒不超过 2 次。
- **FRED**：`https://fred.stlouisfed.org/graph/fredgraph.csv?id={ID}` 取最新值；`https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={ID}&vintage_date=2025-10-30` 取截止日版本；`https://fred.stlouisfed.org/graph/api/series/?id={ID}` 取标题、单位、频率、季调和说明。
- **SEC**：`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`，User-Agent 为 `fin-posttrain academic research`，限速每秒不超过 5 次；保留 us-gaap、ifrs-full（按 IFRS 申报 20-F 的外国公司，如 TSM）和 dei 三个分类、10-K、10-K/A、10-Q、10-Q/A、20-F、40-F 六种表格。`www.sec.gov/files/company_tickers.json` 对这个 User-Agent 返回 403，所以标普 500 成分股的 CIK 取自维基百科表格，其它代码用 EDGAR 实体搜索 `efts.sec.gov/LATEST/search-index` 精确匹配代码。
- **标普 500 成分股**：`https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`，Yahoo 代码把 `.` 换成 `-`（BRK.B → BRK-B）。

## 使用许可

- **Yahoo 数据只供本项目内部研究使用，不得再分发、发布或上传**，包括由它算出的题目答案以外的原始价格表。
- SEC 的 XBRL 数据是美国政府公开数据。
- FRED 上多数序列是公开数据，但有些序列版权属于原数据提供方，再分发前要查 FRED 页面上的说明。本快照中 `fred_series.copyright_notice = 1` 的序列需要注意（见下文"环境设计须知"）。

## 构建和增量更新

```bash
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
PY=$FP_VENVS/mkt/bin/python   # pandas numpy requests lxml
C=src/01_data-processing/utils/market_snapshot
N=$FP_DATA_D2/market_v1/finsearchcomp_needs
$PY -B $C/build.py --symbols-file $N/extra_symbols.txt --fred-file $N/extra_fred.txt --sec-file $N/extra_sec.txt
$PY -B $C/validate.py
```

- 代码清单文件每行一个代码，`#` 之后是注释；`--sec-file` 的每行可以是代码或 CIK。
- 重新运行只下载 `fetch_log` 里没有成功记录的代码、序列和公司，状态为 fail 的会重试，not_found 的不再重试；中断后直接重跑即可续上。
- `--symbols-file` 里新加的美国上市股票（Yahoo 类型 EQUITY、币种 USD）也会下载 SEC 财报数据。
- `--only-symbols`、`--only-fred`、`--only-sec` 只处理给定的代码，配合 `--db` 和 `--raw-dir` 可在测试库上试跑。

## 环境设计须知

**规模（2026-10-31 构建）：** 565 个证券、3,195,251 条日线、35,727 条公司行为（35,136 条分红、591 条拆股类记录）、68 个 FRED 序列共 208,110 个观测、516 个 SEC 登记主体共 11,928,620 条事实；数据库 2.9 GB，原始响应 gzip 后约 210 MB。`validate.py` 的 13 项核对全部通过，结果存在 `meta.validation`。

**价格**
- **标普 500 名单是构建当天（2026-10-31）的成分股，不是截止日的。** 截止日前已被剔除的公司不在快照里（有幸存者偏差）；FDXF、HONA、VYLR 是截止日之后才上市的分拆公司，没有任何价格，Q（Qnity）只有 2025-10-27 起 4 个交易日。
- **Yahoo 把分拆、股票股利一类的调整也记成拆股。** `actions.split_type` 区分 `split`、`reverse_split` 和 `adjustment`（如 HON 2025-10-30 的 `1061:1000` 是 Solstice 分拆，GOOG 2014-03-27 的 `2002:1000` 是 C 类股发放）；问"最近一次拆股"时应只看 `split` 和 `reverse_split`。`close` 等"拆股复权价"同样按这些调整做过复权，`raw_*` 则是当天的实际成交价。
- **截止日之后有 18 个证券发生了拆股或分拆**（如 NFLX、NOW、KLAC 1 拆 10 或 1 拆 5，AMCR 5 股合 1 股，DD、FDX、HON、SPGI 分拆），代码已把它们的价格换回截止日口径，例如 NFLX 2025-10-30 的 `close` 是 1,089.00。
- **`adj_close` 是 Yahoo 的乘法复权**（每次除息把之前的价格乘以 1 − 分红/前收盘），不是总回报指数；分红 `value` 已按截止日拆股口径换算，`value_raw` 是当时公告的每股金额。
- **期货是 Yahoo 的连续主力合约**（GC=F 等），换月时不做平滑，也没有单月合约；CL=F 2020-04-20 收盘为负值（真实值）。BZ=F 从 2007-07-30 开始。
- **外汇按伦敦时区记日期**，收盘价是 Yahoo 的快照价，不同于 FRED `DEX*` 序列（纽约中午买入价）；EURUSD=X 等多数货币对从 2003-12 开始，更早的汇率可用 FRED。TWD=X 有明显错误点（2011-10-25 为 1.80、2014-12-31 为 3.67），用前要过滤。成交量对外汇、多数指数为 0。
- **数据起点晚于 2000-01-01 的**：^STOXX50E 2007-03-30，HYG 2007-04-11，SLV 2006-04-28，AUDUSD=X 2006-05-16，GLD 2004-11-18，BTC-USD 2014-09-17，ETH-USD 2017-11-09；^STOXX50E 的币种字段为空（应为 EUR）。

**FRED**
- `value` 含截止日之后的修订和新发布的观测：月度、季度序列在 2025-10-30 以前日期上的最后几个观测（如 GDP 的 2025-07-01、2025-10-01，PAYEMS 的 2025-09-01、2025-10-01）截止日时还没发布，`value_vintage` 为 NULL。
- 少数序列有截止日版本里存在、最新版本里被删掉的观测（`value` 为 NULL、`value_vintage` 有值），例如 DCOILBRENTEU 有 311 个、DCOILWTICO 也有类似情况。
- DGS10 在 2023-10-19 是 4.98（H.15 数据），不是常被引用的 4.99（美国财政部收益率曲线）；两者来源不同，题目答案应以快照为准并留容差。
- `copyright_notice = 1` 的序列：VIXCLS（Cboe）、MORTGAGE30US（Freddie Mac）、JPNNGDPRPCPPPT、JPNRGDPEXP、NZLNGDPRPCPPPT、TUNNGDPRPCHPT；用于内部研究没有问题，再分发前要看原提供方的条款。RUSCPIALLMINMEI 在 2022-03 后停更，IOER 于 2021-07-28 停止并由 IORB 接续。

**SEC**
- **`fy`、`fp` 是申报文件所属的财年和期间，不是这个数值所属的期间。** 同一个数值会出现在多次申报里，例如 NVIDIA 截至 2022-01-30 的商誉 4,349 百万在 FY2022 的 10-K 里是 `fy=2022, fp=FY`，在 FY2023 各季度 10-Q 里又以 `fy=2023` 出现。按期间取数应看 `start`/`end`（时点科目 `start` 为空），要"某财年年报中的数"再加 `form='10-K'` 和 `fy`；`frame`（如 `CY2021Q4I`）是 SEC 去重后按日历期间对齐的那条。
- **公司财年不等于日历年**：NVIDIA 的财年在 1 月底结束（FY2022 截至 2022-01-30），Apple 在 9 月底，Microsoft 在 6 月底。
- 数值是原始单位（美元、股），不是百万；`val` 存为浮点数。XBRL 数据从 2009 年起才有，早于 2009 年的期间只出现在后来申报的对比数里；`end` 中有极少数明显错误的日期（最大到 2199-12-31），来自公司原始申报。
- **一个代码可能对应多个 CIK**：XOM、BLK、SKYD（原 Paramount Global）、BG、TPL、FERG 在改组或被收购后换了新的登记主体，快照同时收录了旧主体（`sec_companies.role` 为 `predecessor registrant of …`）；XOM 的新主体在截止日前没有申报，所以 `securities.cik` 指向旧主体 34088。
- TSM 按 IFRS 申报 20-F，数据在 `ifrs-full` 分类里；2330.TW 没有 CIK。`dei:EntityCommonStockSharesOutstanding`（封面上的流通股数，日期为封面日期）已收录，可用于算市值。
