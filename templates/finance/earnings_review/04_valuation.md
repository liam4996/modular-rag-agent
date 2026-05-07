## 4. 估值分析

- PE/PB 水平与历史区间、行业均值对比
- 判断当前估值是否合理（高估/合理/低估）
- 结合成长性判断 PEG

### 数字约束
- PE、PB 等指标必须来自 fixed_numbers，不要自己计算
- **目标价格式改为区间**: `目标价: XX.XX - YY.YY 元（保守-乐观）`
- 必须标注估值方法：PE估值法 / PEG估值法 / 行业对比法 / DCF估值法
- 所有百分比保留两位小数

### 可用数据
- fixed_numbers (target_price_low, target_price_high, peg_ratio, pe_vs_industry)
- 旧字段 target_price 已弃用，仅作参考
- computed_results.valuation
- industry_benchmark
