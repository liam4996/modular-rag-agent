## 1. 投资要点（开篇 3-5 句话）

- 给出明确投资评级：买入/增持/中性/减持/卖出
- 给出核心逻辑（不超过 3 条）
- **必须给出目标价区间**，格式: `目标价区间: XX.XX - YY.YY 元`
- 区间下限对应保守估值（如 PE 估值法），上限对应乐观估值（如 PEG/行业对比法）
- 标注使用的估值方法

### 可用数据
- fixed_numbers 中的 target_price_low / target_price_high（计算参考值，主智能体综合判断后给出最终区间）
- fixed_numbers 中的 expected_return / peg_ratio / pe_vs_industry
- computed_results.metrics 中的财务指标
