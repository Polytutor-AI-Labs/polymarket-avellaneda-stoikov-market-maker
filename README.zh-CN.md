# Polybot — Polymarket 上的 Avellaneda-Stoikov 做市机器人

[English](./README.md) · **中文**

面向 **Polymarket BTC 二元期权** 的高频做市引擎：用 **Black-Scholes** 计算公允价值，用 **Avellaneda-Stoikov** 报价，并在 **Hyperliquid 永续合约** 上做 Delta 对冲。另有一套变体，报价策略改为在同一套回测数据上训练的 **PPO 强化学习智能体**。

> 这是一个从研究走到部署的项目，覆盖完整闭环：实盘连接器、微观结构特征、风险管理、仿真、回测，以及离线强化学习对比。

<p align="center">
  <img src="./assets/market-maker-strategy.jpg" width="100%" alt="Polymarket 做市策略：公允价值、AS-lite 报价、库存偏移与 Delta 对冲" />
</p>

---

## 仓库内容

### 策略（`strategy/`）
- **`avellaneda_stoikov.py`** — AS-lite 报价公式：
  `bid = FV - half_spread - skew(inventory)`，`ask = FV + half_spread - skew(inventory)`。
  库存偏移把持仓拉回目标，半价差补偿被不利订单流打到的风险。
- **`pricing.py`** — **二元期权** 的 Black-Scholes 公允价值：
  `FV = N(d2)`，其中 `d2 = (ln(S/K) + (r - σ²/2)·T) / (σ·√T)`。用作 AS 报价中心（不用 Polymarket 上噪声较大的中间价）。
- **`microstructure.py`** — 订单簿失衡、动量、VPIN、价差形态。用于调整 AS 半价差，并识别逆向选择状态。
- **`binary_delta.py`** — 二元期权 Delta：`Δ = e^(-rT) · φ(d2) / (σ · S · √T)`。决定在 Hyperliquid 上 BTC 对冲的规模。
- **`realized_vol.py`** — 经卡尔曼滤波平滑的已实现波动率（Deribit 隐含波动率不可用时的回退）。
- **`risk_manager.py`** — 库存上限、敞口上限、单日亏损熔断、波动率飙升暂停，以及临近到期阶段（WIND_DOWN / STOP / FAK）。
- **`state.py`**、**`fill_model.py`** — 回测用的共享状态与模拟成交。

### 连接器（`connectors/`）
- **`poly_ws.py`** — Polymarket CLOB WebSocket 消费端（订单簿、逐笔）。
- **`hyperliquid_client.py`** — Hyperliquid REST + WebSocket：BTC 永续市价对冲单、持仓轮询。
- **`deribit_ws.py`** — Deribit BTC 期权 WebSocket，用于获取 Black-Scholes 定价所需的实时隐含波动率。
- **`binance_ws.py`** — Binance BTC 现货 WebSocket，作为二元期权定价的标的价格。
- **`cf_bypass.py`** — 处理受 Cloudflare 保护的接口。

### 执行（`execution/`）
- **`trader.py`** — 主循环：拉取实时数据 → 计算公允价值 → AS 报价 → 通过 Polymarket CLOB 下单 → 轮询成交 → 触发对冲。
- **`quote_manager.py`** — 维护实时双边报价，价格偏离时撤单重挂，GTD 到期缓冲，边际超过阈值时在挂单与吃单之间切换。
- **`hedge_manager.py`** — 在 Hyperliquid 上做 Delta 对冲：汇总期权 Delta 得到目标 BTC 仓位，偏离超过容忍度时再平衡。

### 工具（`tools/`）
- **`backtest_AB.py`** / **`backtest_B_fulltick.py`** / **`backtest_comparison.py`** —
  逐笔回测引擎，在相同历史 CSV 上比较 AS-lite 启发式策略与 RL PPO 策略。
- **`rl_ppo_train.py`** — 在由实盘数据构建的自定义 Gym 环境上用 stable-baselines3 训练 PPO。动作 = 价差档位 + 偏移档位。
- **`rl_quick_test.py`** — 已训练模型的健全性检查与评估辅助。
- **`build_rl_dataset.py`** — 从原始行情 CSV 划分训练 / 评估集。
- **`poly_iv_analysis.py`** — Polymarket 数据上的隐含波动率曲面分析。
- **`param_optimizer.py`** — AS-lite 超参数网格搜索（半价差、最大库存、最大偏移）。

### `main.py`
生产入口：加载配置、启动连接器、拉起报价与对冲管理器，并处理优雅退出。

### `config.py`
全部交易参数集中在一处：库存上限、价差 / 偏移、对冲容忍度、RL 开关、动量过滤、熔断。密钥放在 `.env`（不提交），通过 `os.getenv` 读取。

---

## 方法

1. **用公允价值，不用中间价** — Polymarket 二元期权的中间价常常又宽又旧，因此 AS 报价中心是由实时 BTC 现货与 Deribit 隐含波动率算出的 Black-Scholes 公允价值，而不是订单簿中间价。
2. **用偏移把库存拉回** — 半价差决定挂单补偿，偏移把库存拉回零。两者都有明确上限，避免尾部事件把账簿打穿。
3. **通过 Hyperliquid 保持 Delta 中性** — 每个未平期权仓位贡献一份 Delta；对冲管理器把净 BTC 敞口维持在 Hyperliquid 永续的容忍区间内。
4. **按到期时间切换阶段** — 临近到期时，机器人从 `NORMAL → WIND_DOWN → STOP`，先撤销风险更高的一侧，若仍希望把仓位打平，则用 fill-and-kill 吃单收尾。
5. **用 RL 做对照基线** — 生产策略是 AS-lite 启发式规则。PPO 智能体在同一数据上训练，并在相同的样本外切片上比较，用来判断额外复杂度是否值得。

---

## 结果（回测，仅供参考）

历史数据、样本内，单笔报价约 5 USDC，每个市场最大库存 10：

| 策略 | 日均盈亏 | 平均夏普 | 说明 |
|----------|---------------:|-----------:|-------|
| AS-lite 启发式 | 约 +44 USDC | （样本内） | 生产基线策略 |
| PPO 强化学习 | 约 +54 USDC | （样本内） | 相对启发式约 +22%，样本内 |

这些是模拟的样本内数字。`tools/backtest_*` 脚本的目的，是在严格的样本外窗口上重复实验，并量化强化学习的优势（或确认没有优势）。合理预期是：样本外差距会收窄。

---

## 公开仓库中故意不包含的内容

- **`data/`** — 录制的行情与成交日志，其中的交易哈希可以在链上追溯到真实钱包。
- **`results/`** — 真实策略运行产生的回测输出。
- **`.env`** — Polymarket / Hyperliquid 私钥与交易所 API 密钥。
- **`*.pem`** — 生产虚拟机的 SSH 密钥。
- **`rl_models/`** — 训练好的模型权重。

若要复现：用 `data/recorder.py` 录制自己的行情，填写自己的 `.env`，再用 `tools/backtest_comparison.py` 运行回测。

---

## 技术栈

Python 3.11+ · py-clob-client · web3 / eth_account · websockets · curl_cffi ·
stable-baselines3 · gymnasium · numpy · pandas · scipy · matplotlib

---

## 状态

最初是一个关于「二元期权遇上加密货币」做市的个人研究项目。实盘版本曾在 2026 年初于 AWS 上运行数周；本仓库中的本地副本是回测与迭代用的沙箱。

---

## 许可证

MIT — 见 [LICENSE](LICENSE)。
