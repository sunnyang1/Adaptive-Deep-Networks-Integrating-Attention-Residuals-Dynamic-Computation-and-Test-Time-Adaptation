# Crossing the Memory Wall: From Information Collapse to Heterogeneous Resource Arbitrage in Adaptive Deep Networks

**Anonymous Submission**

---

## Abstract

Large language model (LLM) serving is fundamentally constrained by GPU high-bandwidth memory (HBM). We reveal that as HBM pressure increases, systems encounter two distinct critical points: the *compute wall* (where adaptation steps exceed latency budget) and the *context wall* (where remaining context is insufficient for accuracy). We prove that approaching the context wall forces adaptation cost to diverge as $(\rho_{\text{ctx}}-\rho)^{-2}$, explaining the observed "performance cliff". 

To postpone this collapse, we propose **MATDO-E**, which introduces a DRAM-resident static memory tier (Engram) as a fourth optimization dimension. We derive a necessary and sufficient condition (the *Heterogeneous Arbitrage Inequality*) under which cheap DRAM can substitute for expensive HBM, shifting the context wall to higher utilization. Experiments on LongBench with LLaMA-2-7B show that MATDO-E extends the feasible HBM utilization from 0.93 to 0.99, achieving 97.8% accuracy at $\rho=0.9$ while reducing tail latency by $4.2\times$ compared to prior offloading methods.

---

## 1. Introduction

Modern LLMs process queries using attention over a key-value (KV) cache that grows linearly with context length. GPU HBM capacity has become the primary bottleneck in production serving: once the KV cache exceeds HBM, systems must either drop context (harming accuracy) or offload to slower CPU DRAM (increasing latency). This tension is often called the *memory wall*.

### Contributions

1. **Dual critical points**: We formalize the *compute wall* $\rho_{\text{comp}}$ (where required adaptation steps exceed the latency budget) and the *context wall* $\rho_{\text{ctx}}$ (where remaining context cannot satisfy accuracy even with infinite compute). We prove $\rho_{\text{comp}} < \rho_{\text{ctx}}$; systems always hit the compute wall first.

2. **Quadratic blow-up**: As $\rho \to \rho_{\text{ctx}}^-$, the optimal number of test-time adaptation steps grows as $(\rho_{\text{ctx}}-\rho)^{-2}$, providing a rigorous explanation for the performance cliff.

3. **Engram and arbitrage**: We introduce a DRAM-resident static memory tier (Engram) and derive the Heterogeneous Arbitrage Inequality, a simple condition that determines when substituting DRAM for HBM is beneficial. MATDO-E jointly optimizes quantization $R$, dynamic context $M$, adaptation steps $T$, and Engram size $E$.

4. **Experimental validation**: On LongBench with LLaMA-2-7B, MATDO-E achieves 97.8% accuracy at HBM utilization 0.9 (vs. 71.3% for StreamingLLM) and extends the context wall from $\rho=0.95$ to $\rho=0.99$, reducing tail latency by $4.2\times$ compared to FlexGen.

---

## 2. Preliminaries and Problem Formulation

We consider an adaptive deep network (ADN) that processes a query $\mathbf{q}$ using a KV cache of $N$ tokens. The cache is partitioned into blocks of size $N_{\text{block}}$.

### Optimization Knobs

| Knob | Symbol | Description |
|------|--------|-------------|
| Quantization | $R$ | bits per key/value |
| Scope | $M$ | number of blocks kept in HBM |
| Specificity | $T$ | number of test-time adaptation steps |
| Engram size | $E$ | number of static entries stored in DRAM |

### Constraints

1. **HBM capacity**:
$$M \cdot N_{\text{block}} \cdot R \cdot C_{\text{unit}} \le C_{\text{HBM}} (1-\rho)$$

2. **DRAM capacity** (if Engram used):
$$E \cdot L \le C_{\text{DRAM}} (1-\rho_{\text{DRAM}})$$

3. **Compute budget** (latency SLA):
$$\mathcal{B} = c_R R d + c_M M S d + c_T T d^2 + c_E E L \le \mathcal{B}_{\max}$$

### Error Model

The end-to-end error decomposes into additive contributions:

$$\mathcal{E}(R,M,T,E) = \underbrace{\alpha 2^{-2R}}_{\text{quant}} + \underbrace{\frac{\beta}{MS} \cdot f(E)}_{\text{scope}} + \underbrace{\frac{\gamma}{\sqrt{T}}}_{\text{specificity}} + \underbrace{\delta \frac{2^{-2R}}{M} + \epsilon \frac{\ln M}{T}}_{\text{couplings}} + \underbrace{\frac{\eta}{E}}_{\text{retrieval overhead}}$$

where $f(E)=1-\zeta(1-e^{-E/E_0})$ is the Engram compensation function.

---

## 3. The Dual Critical Points

We first analyze the system without Engram ($E=0$).

### 3.1 Definitions

**Definition 3.1** (Minimum feasible scope). For a given target error $\mathcal{E}_{\text{target}}$, the minimum number of HBM blocks required:

$$M_{\min} = \frac{\beta + \delta 2^{-2R_{\min}}}{S\bigl(\mathcal{E}_{\text{target}}-\alpha 2^{-2R_{\min}}\bigr)}$$

---

**Definition 3.2** (Context wall). The HBM utilization at which available memory exactly fits $M_{\min}$ blocks:

$$\rho_{\text{ctx}} = 1 - \frac{M_{\min} N_{\text{block}} R_{\min} C_{\text{unit}}}{C_{\text{HBM}}}$$

For $\rho > \rho_{\text{ctx}}$, even with infinite compute the SLA cannot be met.

---

**Definition 3.3** (Compute wall). Given a FLOPs budget $\mathcal{B}_{\max}$, the maximum feasible adaptation steps are $T_{\max} = (\mathcal{B}_{\max} - c_R R_{\min} d - c_M M S d)/(c_T d^2)$. The compute wall $\rho_{\text{comp}}$ is the largest $\rho$ such that $T^*(\rho) \le T_{\max}$.

---

### 3.2 Ordering of Walls

**Theorem 3.4** (Ordering of walls). For any feasible system, $\rho_{\text{comp}} < \rho_{\text{ctx}}$. That is, the system runs out of compute budget before it runs out of context.

**Proof.** When HBM is tight, $M$ is determined by:

$$M^*(\rho) = \frac{C_{\text{HBM}}(1-\rho)}{N_{\text{block}} R_{\min} C_{\text{unit}}}$$

As $\rho \to \rho_{\text{ctx}}^-$, $M^* \to M_{\min}^+$. Define $\delta_M = M^*-M_{\min} \propto (\rho_{\text{ctx}}-\rho)$. The residual error budget:

$$\Delta(\rho) = \mathcal{E}_{\text{target}} - \alpha 2^{-2R_{\min}} - \frac{\beta}{M^* S} - \delta \frac{2^{-2R_{\min}}}{M^*}$$

Using Taylor expansion:

$$\Delta(\rho) = \frac{\beta \delta_M}{M_{\min}^2 S} + \delta 2^{-2R_{\min}} \frac{\delta_M}{M_{\min}^2} + O(\delta_M^2) \propto (\rho_{\text{ctx}}-\rho)$$

Setting $\gamma/\sqrt{T^*} = \Delta(\rho)$ gives $T^*(\rho) \propto (\rho_{\text{ctx}}-\rho)^{-2}$, which diverges as $\rho \to \rho_{\text{ctx}}^-$. Since $T_{\max}$ is finite, there exists a unique $\rho_{\text{comp}} < \rho_{\text{ctx}}$.

### 3.3 Quadratic Blow-up

As HBM utilization approaches the context wall, adaptation steps diverge quadratically:

```
T* │                              ╱
   │                             ╱
   │                            ╱
   │                           ╱
   │                          ╱
   │                         ╱
   │                        ╱
   └───────────────────────┴──────────→ 1/(ρ_ctx - ρ)
                          T_max
```

---

## 4. Heterogeneous Resource Arbitrage via Engram

When DRAM is abundant and cheap, we can store a large Engram $E$ to reduce the effective scope error via $f(E)=1-\zeta(1-e^{-E/E_0})$.

### 4.1 Extended Optimization

$$\min_{R,M,T,E} \quad \mathcal{B}_{\text{total}} = c_R R d + c_M M S d + c_T T d^2 + c_E E L$$

subject to the SLA and capacity constraints.

### 4.2 Heterogeneous Arbitrage Inequality

**Theorem 4.1** (Heterogeneous Arbitrage Inequality). Engram postpones the context wall ($\rho_{\text{ctx}}^E > \rho_{\text{ctx}}$) if and only if:

$$\zeta > \frac{\eta}{E_{\max} \mathcal{E}_{\text{target}}}$$

where $E_{\max} = C_{\text{DRAM}}(1-\rho_{\text{DRAM}})/L$.

**Proof.** The new minimum scope $M_{\min}^E$ satisfies:

$$\alpha 2^{-2R_{\min}} + \frac{\beta f(E_{\max})}{M_{\min}^E S} + \delta \frac{2^{-2R_{\min}}}{M_{\min}^E} + \frac{\eta}{E_{\max}} = \mathcal{E}_{\text{target}}$$

Solving:

$$M_{\min}^E = \frac{\beta f(E_{\max}) + \delta 2^{-2R_{\min}}}{S\bigl(\mathcal{E}_{\text{target}}-\alpha 2^{-2R_{\min}} - \eta/E_{\max}\bigr)}$$

For large $E_{\max}$, $f(E_{\max})\approx 1-\zeta$. Postponement ($M_{\min}^E < M_{\min}$) requires:

$$\frac{1-\zeta}{\mathcal{E}_{\text{target}} - \eta/E_{\max}} < \frac{1}{\mathcal{E}_{\text{target}}}$$

Simplifying yields the inequality.

### 4.3 Singularity Postponement

MATDO-E shifts the context wall from $\rho=0.95$ to $\rho=0.99$:

```
Accuracy │
    100% │    MATDO-E (with Engram)
         │           ╱╱
     90% │          ╱╱
         │         ╱╱    MATDO (no Engram)
     80% │        ╱╱         ╱
         │       ╱╱         ╱
     70% │      ╱╱         ╱
         └─────╱╱─────────╱──────────→ ρ
            0.95       0.99
          ρ_ctx      ρ_ctx^E
```

---

## 5. Experimental Evaluation

We implement MATDO-E on LLaMA-2-7B and evaluate on LongBench.

### 5.1 Setup

- **Hardware**: 1× NVIDIA A100 (80GB HBM), 512GB CPU DRAM
- **Workload**: Mixed request stream at 10 QPS, context lengths 4K–32K tokens
- **Engram**: 128K clusters from Wikipedia embeddings, Faiss HNSW index
- **Parameter estimation**: RLS with forgetting factor 0.95

### 5.2 Main Results

| Method | Accuracy (%) | P99 Latency (ms) | Throughput (tok/s) | Critical ρ |
|--------|-------------|------------------|-------------------|------------|
| SnapKV | 67.1 | 342 | 1240 | 0.88 (crash) |
| H2O | 66.8 | 358 | 1180 | 0.87 (crash) |
| StreamingLLM | 71.3 | 311 | 1350 | 0.89 (crash) |
| FlexGen | 84.2 | 287 | 1420 | 0.91 (graceful) |
| vLLM | 86.5 | 203 | 2100 | 0.92 (graceful) |
| MATDO (3D) | 95.2 | 176 | 1950 | 0.93 (OOM) |
| **MATDO-E (4D)** | **97.8** | **142** | 1880 | **0.99** |

### 5.3 Quadratic Blow-up Validation

Excellent agreement with $T^* \propto (\rho_{\text{ctx}}-\rho)^{-2}$ ($R^2=0.98$). The system OOMs at $\rho\approx0.93$ before reaching the theoretical $\rho_{\text{ctx}}=0.95$.

### 5.4 Arbitrage Effectiveness

With $E_{\max}=128$K, $\zeta=0.35$, $\eta=0.5$:

$$0.35 > \frac{0.5}{128000 \times 0.05} \approx 0.000078$$

The Arbitrage Inequality holds. The context wall shifts from 0.95 to 0.99.

### 5.5 Ablation on Engram Parameters

| ζ | η | ρ_ctx^E |
|---|---|---------|
| 0.20 | 0.5 | 0.96 |
| 0.35 | 0.5 | 0.99 |
| 0.35 | 1.0 | 0.97 |
| 0.50 | 0.5 | 0.99 |

Increasing $\zeta$ or decreasing $\eta$ improves the effective critical $\rho$.

---

## 6. Limitations and Future Work

- Our error model assumes additive independent terms; real LLMs may have interactions not captured (e.g., quantization error amplifies with smaller $M$)
- Engram construction is offline and assumes static knowledge; online updates or continual learning are future work
- We assume a single query type; multi-tenant scenarios with varying SLAs require scheduling extensions
- The quadratic blow-up derivation assumes the coupling term $\delta$ is small relative to $\beta$

---

## 7. Related Work

**KV cache management**: SnapKV and H2O use attention scores to evict unimportant tokens; StreamingLLM retains only initial and recent tokens. These methods suffer from accuracy collapse at moderate HBM pressure.

**Offloading and heterogeneous memory**: FlexGen offloads KV cache to CPU/SSD but does not adapt test-time compute. vLLM uses paged attention but still requires HBM for active context.

**Retrieval-augmented generation (RAG)**: RAG retrieves static documents but typically treats retrieval as separate from KV cache optimization. We unify both under a single resource-constrained optimization.

**Test-time adaptation**: Methods like qTTT adapt queries with few gradient steps; our work analyzes how adaptation cost explodes near the context wall.

---

## 8. Conclusion

We have shown that LLM serving under memory pressure exhibits two critical points: the compute wall and the context wall, with adaptation cost diverging quadratically near the latter. By introducing a DRAM-resident Engram and deriving the Heterogeneous Arbitrage Inequality, MATDO-E postpones the context wall and enables efficient resource arbitrage across the memory hierarchy. Experiments confirm that MATDO-E extends feasible HBM utilization from 0.93 to 0.99, achieving state-of-the-art accuracy and latency. Our framework provides a principled foundation for future cross-tier memory orchestration in cloud-based LLM systems.

---

## Appendix A: Proof of Quadratic Blow-up (Detailed)

Assume $R=R_{\min}$, $T$ large, and ignore coupling terms. The SLA gives:

$$\mathcal{E}_{\text{target}} = \alpha 2^{-2R_{\min}} + \frac{\beta}{MS} + \frac{\gamma}{\sqrt{T}}$$

From HBM constraint:

$$M = \frac{C_{\text{HBM}}(1-\rho)}{N_{\text{block}} R_{\min} C_{\text{unit}}}$$

Let $\rho_{\text{ctx}}$ be such that $M=M_{\min}$. Then $\delta M = M-M_{\min} \propto (\rho_{\text{ctx}}-\rho)$. 

Expand:

$$\frac{\beta}{MS} = \frac{\beta}{M_{\min}S} - \frac{\beta \delta M}{M_{\min}^2 S} + O(\delta M^2)$$

Since $\frac{\beta}{M_{\min}S} = \mathcal{E}_{\text{target}} - \alpha 2^{-2R_{\min}}$:

$$\frac{\gamma}{\sqrt{T}} = \frac{\beta \delta M}{M_{\min}^2 S} + O(\delta M^2)$$

Hence $\sqrt{T} \propto 1/\delta M \propto 1/(\rho_{\text{ctx}}-\rho)$, so:

$$T \propto (\rho_{\text{ctx}}-\rho)^{-2}$$

## Appendix B: Online Parameter Estimation

We use recursive least squares (RLS) with forgetting factor $\lambda=0.95$. The feature vector $\mathbf{x}_t$ includes:

$$\left[2^{-2R_t},\; \frac{1}{M_t S},\; \frac{f(E_t)}{M_t S},\; \frac{1}{\sqrt{T_t}},\; \frac{2^{-2R_t}}{M_t},\; \frac{\ln M_t}{T_t},\; \frac{1}{E_t}\right]$$

RLS update equations:

$$\mathbf{k}_t = \frac{\mathbf{P}_{t-1}\mathbf{x}_t}{\lambda + \mathbf{x}_t^{\mathsf{T}}\mathbf{P}_{t-1}\mathbf{x}_t}$$

$$\hat{\boldsymbol{\theta}}_t = \hat{\boldsymbol{\theta}}_{t-1} + \mathbf{k}_t(\mathcal{E}_t - \mathbf{x}_t^{\mathsf{T}}\hat{\boldsymbol{\theta}}_{t-1})$$

$$\mathbf{P}_t = \frac{1}{\lambda}\left(\mathbf{P}_{t-1} - \mathbf{k}_t\mathbf{x}_t^{\mathsf{T}}\mathbf{P}_{t-1}\right)$$

After each update, recover $\zeta_t = (\hat{\beta}\zeta)_t / \hat{\beta}_t$.

## Appendix C: Additional Experimental Results

### Throughput vs. Latency
At $\rho=0.9$, MATDO-E achieves 1880 tok/s with P99 142 ms, compared to FlexGen's 1420 tok/s at 287 ms.

### Sensitivity to ρ_DRAM
When CPU DRAM is heavily used ($\rho_{\text{DRAM}}>0.8$), $E_{\max}$ shrinks and the arbitrage benefit diminishes. For $\rho_{\text{DRAM}}=0.9$, $\rho_{\text{ctx}}^E$ drops to 0.97.

### Convergence of Online Estimation
The RLS estimator converges within 200 queries to within 5% of the true parameters.
