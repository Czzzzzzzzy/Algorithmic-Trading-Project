# BUSI70575 课程项目指南

## 1. 大方向：我们到底在做什么？

这个 coursework 给了我们一组 **primary trading signals**，也就是已有的基础交易信号。

primary signal 每天对每个 instrument 给出一个方向：

- `+1`：基础模型想做多。
- `-1`：基础模型想做空。
- `0`：不交易。

我们的任务 **不是从零开始预测市场涨跌**。

我们的任务是做一个 **metamodel**，也就是在 primary signal 上面再加一层判断：

> 在当前市场环境下，这个 primary signal 值不值得跟？

可以把它理解成一个过滤器。

简单类比：

- Primary signal = 有人告诉你“买”或“卖”。
- Metamodel = 再问一句：“现在这种情况，听这个信号靠谱吗？”

所以最终模型输出的不是“价格会涨还是跌”，而是：

```text
跟随这个 primary signal 的概率 / 置信度
```

最终需要提交的 CSV 格式是：

```text
date,instrument,prediction
```

其中 `prediction` 必须是 `[0, 1]` 之间的概率。

## 2. 一个训练样本是什么？

一个样本是一行：

```text
某一天 x 某个 instrument
```

如果 `primary_signal` 不是 0，这一行就是一个交易机会。

- `primary_signal = +1`：基础模型建议做多。
- `primary_signal = -1`：基础模型建议做空。
- `primary_signal = 0`：没有真正的交易机会。

在我们的建模问题里：

```text
X = 交易前已经知道的特征
y = triple-barrier meta-label
```

X 包括：

- `primary_signal`
- 滞后的 OHLCV 特征
- 收益率 / 动量特征
- 波动率 / 价格区间特征
- 技术指标
- 成交量 / 流动性 / open interest 特征
- 横截面排名特征
- GMM / PCA 特征
- signal-history 特征

y 的含义：

- `1` = 这个 primary signal 值得跟。
- `0` = 这个 primary signal 不值得跟。

注意：`primary_signal = 0` 的行不是真正的交易机会。如果最终提交文件要求所有 `date x instrument` 都有预测值，我们给这些 no-trade 行一个中性概率 `0.5`。

## 3. Triple-Barrier Labeling 用最简单的话解释

Triple-barrier labeling 是用来生成 `y` 标签的方法。

对于每一个非零 primary signal，我们往未来看固定天数，并设置三条“边界”：

1. Profit-taking barrier：止盈线。
2. Stop-loss barrier：止损线。
3. Vertical time barrier：时间到期线。

如果交易先碰到止盈线，`y = 1`。

如果交易先碰到止损线，`y = 0`。

如果时间到了但没有明确先碰到止盈线，在我们的二分类 meta-label 设置里，这不是一个明确成功的交易，所以作为 `0` 处理。

### 做多例子

假设：

- 入场价格 = `100`
- `primary_signal = +1`
- 滞后波动率 = `2%`
- multiplier = `1.5`

那么：

- barrier width = `3%`
- 止盈价格 = `103`
- 止损价格 = `97`

如果价格先到 `103`，说明跟随这个做多信号成功，`y = 1`。

如果价格先到 `97`，说明这个信号失败，`y = 0`。

### 做空例子

假设：

- 入场价格 = `100`
- `primary_signal = -1`
- barrier width = `3%`

对于做空来说，价格下跌才是盈利：

- 止盈价格 = `97`
- 止损价格 = `103`

如果价格先到 `97`，说明跟随这个做空信号成功，`y = 1`。

如果价格先到 `103`，说明这个信号失败，`y = 0`。

重点：

- 这个标签不是 raw future return。
- 它不是简单问“未来收益是不是正的”。
- 它问的是：“按照一套风险管理规则跟随 primary signal，结果是否成功？”

不要把它叫作 “true ground truth”。更准确的说法是：

```text
triple-barrier meta-label
```

或者：

```text
selected triple-barrier label specification
```

## 4. 为什么所有特征都要 lag？

因为我们必须避免 look-ahead bias。

模型在日期 `t` 做决定时，只能使用交易前已经知道的信息。

简单说：

```text
模型不能在今天做决定时偷看明天的价格。
```

所以：

- rolling volatility 要 lag；
- return / momentum 特征要 lag；
- RSI、MACD、Bollinger 等技术指标要 lag；
- signal-history 特征也只能使用已经完成的历史交易结果。

如果不 lag，模型可能会在测试中看起来很强，但那只是因为它偷偷用了未来信息，真实交易中无法复现。

## 5. 我们用了哪些 feature？

代码里把 feature 分成几个主要组。

### 1. Return 和 Momentum

用于捕捉最近价格方向。

它回答的问题类似：

> 最近这个 instrument 是在上涨、下跌，还是横盘？

### 2. Volatility 和 Range

用于捕捉市场是否波动很大、风险是否较高。

它回答的问题类似：

> 当前市场是很平静，还是价格波动异常大？

### 3. Technical Indicators

包括 RSI、MACD、Bollinger-style indicators、stochastic indicators 等。

它回答的问题类似：

> 价格是不是过热？是不是有反转迹象？趋势是否明显？

### 4. Volume、Liquidity 和 Open Interest

用于捕捉市场参与度和交易活跃程度。

它回答的问题类似：

> 这个价格变化有没有成交量支持？市场是不是太薄？

### 5. Cross-Sectional Ranks

把一个 instrument 和同一天其他 instruments 做比较。

它回答的问题类似：

> 今天这个 instrument 相对其他品种更强还是更弱？

### 6. GMM Regime Features

GMM 用来捕捉软性的市场 regime。

它回答的问题类似：

> 最近市场更像低波动 regime、高波动 regime，还是其他状态？

### 7. PCA Latent Features

PCA 把一批相关的特征压缩成少数几个 latent components。

它回答的问题类似：

> 能不能用更少的维度概括一组相关的 OHLCV 信息？

### 8. Signal-History Features

用于捕捉 primary signal 最近是否有效。

它回答的问题类似：

> 最近这个 instrument 上的 primary signal 靠谱吗？连续成功还是连续失败？

金融特征之间经常高度相关。比如几个 momentum 指标可能都在描述类似的价格趋势。

所以我们的报告里使用 **cluster-level feature importance**，而不是只看单个 feature 的 importance。

## 6. 我们比较了哪些模型？

Coursework 要求比较多个模型家族。

我们比较了：

1. Logistic Regression。
2. Tree-based models。
3. MLP neural network。

各自作用：

- Logistic Regression：稳定、可解释，适合作为基础 anchor。
- Tree-based models：可以捕捉非线性关系。
- MLP：可以捕捉更复杂的非线性交互，尤其是 signal-history 相关信息，但也更容易 overfit。

最终模型是：

```text
Calibrated 0.50 Logistic + 0.50 signal-history MLP probability blend
```

为什么用 blend？

- Logistic Regression 提供稳定基础。
- MLP 提供非线性的 signal-history 信息。
- Sigmoid calibration 改善概率质量。

## 7. 为什么选择这个最终模型？

最终模型不是单纯追求最高 public score 选出来的。

我们用了 guardrails，也就是一些防止过拟合和过度复杂化的规则。

最终 calibrated Logistic-MLP blend 相比简单 logistic model 有提升，同时仍然比较容易解释。

一些更复杂的 challenger 虽然测试过，但没有被提升为 final model。

重要结论：

- HMM extension 实现了，但保持关闭。
- Label-search challenger validation 很强，但 public 2022H1 sanity check 不够好，所以没有 promoted。
- Stacking、autoencoder 等 advanced challengers 测试过，但没有通过 guardrails。
- Primary-signal-only baseline 的 AUC 较弱，说明 market-context features 是有用的。

当前 frozen final prediction file：

```text
outputs/metamodel_predictions.csv
```

当前 frozen final prediction hash：

```text
c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369
```

## 8. 项目文件结构

清理后的结构是：

```text
coursework/
├── coursework_metamodel.py
├── README.md
├── report_summary.md
├── requirements.txt
├── notebooks/
│   ├── 01_pipeline_walkthrough.ipynb
│   ├── 02_label_search_appendix.ipynb
│   └── 03_robustness_experiments.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── validation.py
│   ├── features.py
│   ├── labeling.py
│   ├── models.py
│   ├── evaluation.py
│   ├── importance.py
│   ├── reporting.py
│   └── utils.py
└── outputs/
    ├── metamodel_predictions.csv
    ├── evaluation_summary.csv
    ├── model_comparison.csv
    ├── cluster_importance.csv
    ├── threshold_analysis.csv
    ├── baseline_comparison.csv
    ├── final_integration_audit.txt
    └── archive/
```

各部分作用：

- `coursework_metamodel.py`：主 runner，按 coursework pipeline 串起来。
- `src/`：可复用函数。
- `notebooks/`：解释流程和 appendix 实验。
- `outputs/metamodel_predictions.csv`：最终提交文件。
- `outputs/archive/`：旧 challenger、label search、advanced experiments，不是 final outputs。

## 9. 如何运行代码？

从项目根目录运行：

```bash
cd coursework
python3 coursework_metamodel.py
```

或者从上一层目录运行：

```bash
python3 coursework/coursework_metamodel.py
```

这个 runner 主要用于最终 integration check。

它会验证 frozen final prediction file，并写出：

```text
outputs/final_integration_audit.txt
```

它不应该重新调参，也不应该 promote challengers。

## 10. 哪些输出最重要？

最重要的输出是：

```text
outputs/metamodel_predictions.csv
```

这是最终提交文件。它必须只有三列：

```text
date,instrument,prediction
```

其他重要输出：

- `outputs/evaluation_summary.csv`：按 instrument 的 OOS 表现。
- `outputs/model_comparison.csv`：模型家族比较和 validation 结果。
- `outputs/feature_ablation_results.csv`：feature-set ablation 结果。
- `outputs/feature_ablation_summary.csv`：feature ablation 汇总。
- `outputs/model_ablation_summary.csv`：模型家族表现汇总。
- `outputs/cluster_importance.csv`：cluster-level feature importance。
- `outputs/threshold_analysis.csv`：threshold sensitivity。
- `outputs/baseline_comparison.csv`：和 blindly following primary signals 的对比。
- `outputs/final_submission_freeze_audit.txt`：final file freeze 审计。
- `outputs/final_integration_audit.txt`：最终整合审计。

Appendix / archive 输出：

- `outputs/archive/label_search/`
- `outputs/archive/challenger_experiments/`
- `outputs/archive/advanced_challengers/`
- `outputs/archive/debug_or_temporary/`

这些可以用来解释我们做过的稳健性检查，但它们不是最终提交文件。

## 11. 提交前不要改什么？

不要覆盖：

```text
outputs/metamodel_predictions.csv
```

不要 promote：

- label-search challenger；
- stacking challenger；
- autoencoder challenger；
- HMM extension；
- primary-signal-only baseline。

除非绝对必要，不要再跑新的 tuning experiments。

不要用 public 2022H1 去调新参数。

不要使用 hidden test data。

不要把最终模型描述成 baseline logistic。

不要把 triple-barrier labels 叫作 “true ground truth”。

应该叫：

```text
triple-barrier meta-labels
```

或者：

```text
selected triple-barrier label specification
```

提交前检查：

1. `outputs/metamodel_predictions.csv` 存在。
2. 它只有三列：`date,instrument,prediction`。
3. `prediction` 都在 `[0, 1]`。
4. hash 是：

```text
c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369
```

如果 hash 变了，先停下来检查，不要直接提交。

