# MetaTrajectory 工作流程

## 概述流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MetaTrajectory                                │
│                    代谢轨迹分析完整工作流程                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   输入数据    │
                              │  scRNA-seq   │
                              │ (AnnData/    │
                              │  Seurat)     │
                              └──────┬───────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: 代谢知识库加载                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MetabolicKnowledgeBase                                             │   │
│  │  ├── 77个KEGG代谢通路 (科学过滤)                                     │   │
│  │  ├── 1661个代谢基因                                                 │   │
│  │  ├── 41个限流酶 (HK1, PKM, IDH1...)                                 │   │
│  │  └── 5个代谢轴 (糖酵解/OXPHOS, 合成/分解...)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: 代谢状态空间构建                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PathwayScoringMethods                                              │   │
│  │  ├── AUCell (rank-based, 抗噪)                                      │   │
│  │  ├── VISION (z-score normalized)                                   │   │
│  │  ├── ssGSEA (enrichment-based)                                     │   │
│  │  ├── Weighted (限流酶加权)         ← 推荐用于代谢                    │   │
│  │  └── Combined (多方法融合)                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                      │
│                                     ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MetabolicStateSpace                                                │   │
│  │  ├── 通路得分矩阵 (cells × pathways)                                │   │
│  │  ├── 代谢轴得分 (糖酵解指数等)                                       │   │
│  │  └── PCA降维嵌入                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: 最优传输轨迹推断                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MetabolicNetworkDistance (网络距离)                                 │   │
│  │  ├── Network-weighted (通路连接加权)                                 │   │
│  │  ├── Diffusion (网络扩散距离)                                       │   │
│  │  ├── Flux-constrained (通量约束)                                    │   │
│  │  └── Combined (融合距离)          ← 生物学有意义                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                      │
│                                     ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  AdaptiveParameterSelector (自适应参数)                              │   │
│  │  ├── Stability-based (稳定性选择)                                   │   │
│  │  ├── Entropy-based (熵选择)                                         │   │
│  │  └── Cross-validation                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                      │
│                                     ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MetabolicOT (最优传输)                                              │   │
│  │  ├── Sinkhorn算法 → 传输计划                                        │   │
│  │  ├── 转移概率矩阵                                                   │   │
│  │  └── 代谢Wasserstein距离                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: 命运概率计算                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Absorbing Markov Chain (吸收马尔可夫链)                             │   │
│  │  ├── 识别终端状态 (吸收态)                                          │   │
│  │  ├── 构建Q矩阵 (瞬态间转移)                                         │   │
│  │  ├── 构建R矩阵 (瞬态→吸收态)                                        │   │
│  │  ├── 基本矩阵 N = (I - Q)⁻¹                                         │   │
│  │  └── 命运概率 P = N × R                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  输出: 每个细胞到达各终端状态的概率                                          │
│        Fate_A: 0.7, Fate_B: 0.2, Fate_C: 0.1                              │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: 驱动基因识别                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DriverAnalysis (严谨统计)                                           │   │
│  │  ├── 差异表达: Wilcoxon / t-test                                    │   │
│  │  ├── 效应量: Hedges' g (带95% CI)                                   │   │
│  │  ├── 多重校正: 全局FDR (BH/BY/Storey)                               │   │
│  │  ├── Bayes Factor (贝叶斯证据)                                      │   │
│  │  └── 功效分析 (样本量建议)                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  输出: 排序的驱动基因列表                                                    │
│        Gene    | Effect | CI          | FDR    | Bayes                     │
│        HK2     | 1.23   | [0.8, 1.6]  | 0.001  | 15.3                      │
│        LDHA    | 0.95   | [0.5, 1.4]  | 0.003  | 8.7                       │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: RNA Velocity整合 (可选)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VelocityIntegration                                                │   │
│  │  ├── 投射velocity到代谢空间                                         │   │
│  │  ├── 计算velocity-OT一致性                                          │   │
│  │  └── 不确定性传播                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: 验证与基准测试                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TrajectoryBenchmark                                                │   │
│  │  ├── vs CellRank (命运概率相关性)                                   │   │
│  │  ├── vs Palantir (pseudotime相关性)                                 │   │
│  │  └── vs 合成数据 (ground truth)                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SyntheticDataGenerator                                             │   │
│  │  ├── 分叉轨迹 (bifurcating)                                         │   │
│  │  ├── 线性轨迹 (linear)                                              │   │
│  │  └── 多分支轨迹 (multibranch)                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │       输出结果        │
                         ├──────────────────────┤
                         │ • 代谢轨迹图          │
                         │ • 命运概率分布        │
                         │ • 驱动基因排名        │
                         │ • 分叉点位置          │
                         │ • 通路动态热图        │
                         │ • 验证报告            │
                         └──────────────────────┘


## 简化流程 (一图概览)

```
    scRNA-seq数据
         │
         ▼
    ┌─────────┐     ┌─────────────┐
    │ 知识库   │────▶│ 77个代谢通路 │
    └─────────┘     │ 41个限流酶   │
         │         └─────────────┘
         ▼
    ┌─────────┐     ┌─────────────┐
    │ 状态空间 │────▶│ 通路得分矩阵 │  (AUCell/VISION/Weighted)
    └─────────┘     │ 代谢轴得分   │
         │         └─────────────┘
         ▼
    ┌─────────┐     ┌─────────────┐
    │ 最优传输 │────▶│ 转移概率矩阵 │  (网络距离 + Sinkhorn)
    └─────────┘     │ 传输计划     │
         │         └─────────────┘
         ▼
    ┌─────────┐     ┌─────────────┐
    │ 命运计算 │────▶│ P(cell→fate)│  (吸收马尔可夫链)
    └─────────┘     │ 分叉点       │
         │         └─────────────┘
         ▼
    ┌─────────┐     ┌─────────────┐
    │ 驱动分析 │────▶│ 驱动基因排名 │  (Hedges' g + FDR)
    └─────────┘     │ 通路富集     │
         │         └─────────────┘
         ▼
    ┌─────────┐
    │ 可视化   │────▶ 轨迹图/热图/命运图
    └─────────┘
```


## 数学公式流程

```
Input: X ∈ ℝⁿˣᵍ (n cells, g genes)
       Pathways P = {P₁, P₂, ..., P₇₇}

Step 1: Pathway Scoring
        S_ij = score(X_i, P_j)  →  S ∈ ℝⁿˣ⁷⁷

Step 2: Network Distance
        C_ij = d_network(S_i, S_j)  →  C ∈ ℝⁿˣⁿ

Step 3: Optimal Transport
        T* = argmin_T ⟨T, C⟩ - ε·H(T)
        subject to: T1 = a, T'1 = b

Step 4: Transition Matrix
        P = normalize(T* + T*')

Step 5: Fate Probability (Absorbing Markov)
        Q = P[transient, transient]
        R = P[transient, absorbing]
        N = (I - Q)⁻¹
        Fate = N × R

Step 6: Driver Analysis
        For each gene g:
            effect_g = Hedges'_g(expr_fate1, expr_fate2)
            p_g = Wilcoxon(expr_fate1, expr_fate2)
        FDR correction: p_adj = BH(p_values)

Output: Trajectory, Fate probabilities, Driver genes
```


## 代码流程示例

### Python
```python
import metatrajectory as mt

# Step 1: Load knowledge base
kb = mt.MetabolicKnowledgeBase()
kb.load_kegg_pathways()

# Step 2: Build state space
scorer = mt.PathwayScoringMethods(kb.pathways, kb.rate_limiting_enzymes)
scores = scorer.score_pathways(X, genes, method="weighted")

state_space = mt.MetabolicStateSpace(kb)
state_space.fit(adata)

# Step 3: Compute network distance
network = mt.build_pathway_network_from_kb(kb.pathways)
net_dist = mt.MetabolicNetworkDistance(network, scores)
cost = net_dist.get_optimal_transport_cost(distance_type="combined")

# Step 4: Optimal transport with adaptive parameters
selector = mt.AdaptiveParameterSelector(cost, source_w, target_w)
reg = selector.select_regularization(method="stability")

mot = mt.MetabolicOT(state_space, kb)
mot.fit(adata, reg=reg.optimal_value)

# Step 5: Fate probabilities
fates = mot.compute_fate_probabilities(terminal_states)

# Step 6: Driver analysis with rigorous statistics
da = mt.DriverAnalysis(mot, state_space, kb)
da.fit(adata)
drivers = da.get_top_drivers(fate_pair=("Fate_A", "Fate_B"))
drivers = mt.global_fdr_correction(drivers)

# Step 7: Visualize
mt.plot_metabolic_landscape(adata, mot)
mt.plot_fate_probabilities(adata, fates)
mt.plot_driver_genes(drivers)
```

### R
```r
library(MetaTrajectory)

# Step 1: Load knowledge base
kb <- build_metabolic_graph()

# Step 2: Build state space
scores <- compute_pathway_scores_advanced(seurat_obj, kb, method = "weighted")
state_space <- metabolic_state_space(seurat_obj, kb)

# Step 3-4: Optimal transport
mot <- optimal_transport(state_space)
trajectory <- trajectory_inference(mot)

# Step 5: Fate probabilities
fates <- compute_fate_probabilities(mot, terminal_states)

# Step 6: Driver analysis
drivers <- find_driver_genes(seurat_obj, fates, kb)
drivers <- global_fdr_correction(drivers)

# Step 7: Visualize
plot_metabolic_landscape(seurat_obj, mot)
plot_fate_probabilities(seurat_obj, fates)
plot_driver_genes(drivers)
```
