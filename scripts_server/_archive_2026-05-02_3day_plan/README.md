# 归档：2026-05-02 起草的 3 天 plan，未跑

这里的 8 个脚本是 `CCdocs/2026-05-02_v1-5-hidden_3day_plan.md` 起草时设计的 Day 1 / Day 2 launcher。

**为什么归档**：
- 2026-05-04 发现 Bug G（`hidden_layer_index` 在 RL 路径被默默忽略）→ 旧 plan 的层消融全部需要重做
- 2026-05-05 转向 2-day plan（见 `CCdocs/2026-05-05_2day_plan_revised.md`），Branch A/B/C 分支决策树替换为新的 Phase 0/1/2 路径
- 旧脚本只跑了 `2026-05-03_day0_M1_hpo_09_C1_ep3.sh` 和 `2026-05-03_day0_M2_hidden_combined_L-18.sh`，未跑的全在这里

**保留意图**：留作历史；新计划如果失败可回看 Branch A/B/C 的决策树设计思路。

不要直接复用这里的脚本——它们的层 / pooling / α 选择是 Phase A 跑完之前的版本，未对齐 Bug G 修复后的现实。
