# Project Truth

**项目**：AnyD4RT。完整方案见 `README.md`：AnyView（生成器）和 OpenD4RT（重建器）结构独立，只通过跨视角的 4D 约束交替训练。

## 选定方向（2026-09-25，/omp:ideate）

**主线 I6：单纯的双向提升（H3）**。按 README §2.1、§5、S6 交替更新两个模型。核心证据是换搭档实验：更新后的搭档应比原始搭档更有帮助，两个模型单独推理时都在原任务上提升。

**2026-09-25 审阅后修订**：I5（分段 advantage）和 I1（编辑训练）降为**后续可选模块**，第一版不实现。
- 第一版奖励：标量 r → 组内 advantage。
- 编辑分成两类：人为图像退化只用于 S1 诊断；只有标签能严格推导的变换或可控 3D 重新渲染，才可用于训练。
- 当前优先级：先做 S1，再做 S3，不再扩充方案。

**主要查新对象**：World-R1、CamVerse、VIGOR、MVTrack4Gen（2606.26087）、GenFusion（2503.21219）、Ouroboros3D（2406.03184）、Vivid4D（2504.11092）、Crome（2506.16507）。

**最大风险**：H2 不成立时，双向链条就断了。

详情：`.pipeline/docs/idea_board.json`、`.pipeline/docs/gap_matrix.md`、`.pipeline/docs/paper_digests.md`。
