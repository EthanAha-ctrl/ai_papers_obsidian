---
source_pdf: Continual Vision–Language–Action Learning.pdf
paper_sha256: 658c6ad84df7b85e3a86f64dccdeeca09b275b3a2aef956a5c65c67d5811d95b
processed_at: '2026-08-03T17:18:48-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 CRL-VLA

---

## 1. Problem 是什么

想象你训了一个 robot VLA model, 它能听懂 "把黑碗放到盘子上"。现在你想让它持续学新技能: 先学 "把黑碗放盘子", 再学 "把橙汁放篮子", 再学 "把酒瓶放架子上"... 每次学新 task 时, model 只能看当前 task 的 data, 旧 task 的 data 和 reward 都拿不到了。

**核心痛点**: 学新 task 时, 旧 task 的能力会掉, 这叫 catastrophic forgetting。SL (sequential learning) 在 Table 1 里 FAR=0.00, 意思是学完三个 task 后, 整体成功率基本归零。

传统 solution 有几种, 但在 large-scale VLA 上都有问题:
- **Experience Replay**: 把旧 data 存下来混着训, 但旧 data 的 gradient 跟新 task conflict, off-policy error 大
- **EWC (Fisher information)**: 约束重要参数不变, 但 Fisher 在 multi-modal transformer 上算二阶信息太贵
- **Backbone freezing**: 冻结 backbone, 但 task shift 后 representation 就失效了
- **Global KL constraint**: 像 PPO 那样约束 policy 不要漂移太多, 但这会同时抑制学新 task 的能力

reference: https://arxiv.org/abs/1612.00796 (EWC), https://arxiv.org/abs/2509.22195 (VLM2VLA)

---

## 2. Key Insight: 换个角度看 forgetting

Paper 的核心 claim 是: forgetting 本质上由一个量决定, 叫 **advantage magnitude** $M_g$。

直觉是这样的: 假设你有一个 "老司机" policy $\pi^{old}$ (上一个 task 训完的 model), 和一个 "新手" policy $\pi^{new}$ (正在学新 task 的 model)。你在新 policy 访问的 state-action 上, 问老司机: "如果在这个 state 你来选 action, 你觉得 advantage (相对你的平均水平) 有多大?"

如果老司机觉得 "我 advantage 几乎是 0", 说明新手走的路老司机也认可, 没有遗忘。如果老司机觉得 "我 advantage 很大, 你这个 action 差远了", 说明新手在老司机熟悉的 state 上偏离了, 遗忘发生。

数学定义 (Definition 4.1):
$$M_g(\pi^\kappa) \triangleq \sup_{(s,a) \in \text{supp}(d_g^{\pi^\kappa})} |A^\pi(s,a)|$$

- $M_g(\pi^\kappa)$: 在 $\pi^\kappa$ (通常是 new policy) 访问的 state-action 分布下, anchored policy $\pi$ (通常是 old policy) 的 advantage 绝对值的上确界
- $\text{supp}(d_g^{\pi^\kappa})$: $\pi^\kappa$ 在 goal $g$ 下访问的 state-action 分布的 support
- $A^\pi(s,a)$: anchored policy 的 advantage function, 注意是 anchor 的, 不是 visiting 的

然后定义两个 metric:
- $M_{\text{old}} = \mathbb{E}_{g_{\text{old}} \sim p_{\text{old}}}[M_{g_{\text{old}}}(\pi^{new})]$: 在旧 task state 分布下, 老司机觉得 advantage 多大 → 衡量遗忘
- $M_{\text{new}} = \mathbb{E}_{g_{\text{new}} \sim p_{\text{new}}}[M_{g_{\text{new}}}(\pi^{new})]$: 在新 task state 分布下, 老司机觉得 advantage 多大 → 衡量学习潜力

---

## 3. Theorem 4.1: 统一的 bound

Paper 证明了一个 unified bound, 把 stability 和 plasticity 都跟 $M \cdot D$ 联系起来, 其中 $D$ 是 policy divergence (KL)。

**Stability bound** (旧 task 性能退化):
$$|J_{\text{old}}(\pi^{new}) - J_{\text{old}}(\pi^{old})| \le \frac{2\gamma}{(1-\gamma)^2} \cdot M_{\text{old}} \cdot D_{\text{old}}$$

**Plasticity bound** (新 task 性能提升):
$$J_{\text{new}}(\pi^{new}) - J_{\text{new}}(\pi^{old}) \le \frac{1}{1-\gamma} \cdot M_{\text{new}} \cdot D_{\text{new}}$$

变量解释:
- $J_{\text{old}}(\pi)$: policy $\pi$ 在旧 task 上的 expected return
- $J_{\text{new}}(\pi)$: policy $\pi$ 在新 task 上的 expected return
- $\gamma$: discount factor
- $D_{\text{old}} = \sqrt{2\mathbb{E}_{s \sim d_{\text{old}}^{\pi^{old}}}[D_{\text{KL}}(\pi^{new} \| \pi^{old})]}$: 在旧 task state 分布下 policy 的 KL divergence
- $D_{\text{new}} = \sqrt{2\mathbb{E}_{s \sim d_{\text{new}}^{\pi^{new}}}[D_{\text{KL}}(\pi^{new} \| \pi^{old})]}$: 在新 task state 分布下 policy 的 KL divergence

**直觉解读**:
- 想要遗忘少: 让 $M_{\text{old}} \cdot D_{\text{old}}$ 小
- 想要学得快: 让 $M_{\text{new}} \cdot D_{\text{new}}$ 大
- 传统方法 (global KL) 同时压 $D_{\text{old}}$ 和 $D_{\text{new}}$, 所以 conflict
- **Paper 的洞察**: 既然乘积 $M \cdot D$ 控制两端, 能否把 $M$ 这一项 decouple 出来单独控制?

**为什么 stability bound 系数更大** $\frac{2\gamma}{(1-\gamma)^2}$ vs $\frac{1}{1-\gamma}$: stability 多了一个 occupancy mismatch term, 因为 new policy 在旧 task 上访问的 state 分布跟 old policy 不一样, 这个 distribution drift 需要额外 bounded。Plasticity 没有这个问题, 因为新 policy 在新 task 上访问的 state 就是它自己的 state, 没有分布 mismatch。

推导细节参考 Kakade & Langford 2002 PDL: https://proceedings.mlr.press/v2/kakade02a.html  
Achiam et al. 2017 CPO 的 occupancy bound: http://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf

---

## 4. 最关键的一步: $M_{\text{old}}$ 和 $M_{\text{new}}$ 由不同 mechanism 决定

这是 paper 最妙的地方。它证明这两个量由完全不同的 factor 决定, 所以可以独立控制。

### 4.1 $M_{\text{old}}$ 由 critic approximation error 决定

**Corollary 4.1**: 假设 reward $|r| \le R_{\text{max}}$, critic approximation error $\varepsilon_V = \sup_{s,g_{\text{old}}}|V_\theta - V_{\text{old}}|$, 则:
$$M_{\text{old}} \le R_{\text{max}} + (1+\gamma)\varepsilon_V + (1+\gamma)\|V_{\text{old}}\|_\infty$$

- $R_{\text{max}}$: reward 的最大绝对值, environment 决定
- $\varepsilon_V$: critic 在旧 task state 上的 approximation error, **这个我们可控**
- $\|V_{\text{old}}\|_\infty = \sup_s |V_{\text{old}}(s,g)|$: true value function 的上确界, 固定量

证明思路 (Appendix A.4.1): advantage $\hat{A}(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}[V_\theta(s')] - V_\theta(s)$, 三角不等式:
$$|\hat{A}| \le |r| + \gamma|V_\theta(s')| + |V_\theta(s)| \le R_{\text{max}} + (1+\gamma)(\|V_{\text{old}}\|_\infty + \varepsilon_V)$$

**人话**: $M_{\text{old}}$ 跟 critic 在旧 data 上估得多准成正比。如果 critic 在旧 data 上完全准 ($\varepsilon_V = 0$), $M_{\text{old}}$ 就被压到最低。所以我们只要在旧 data 上保持 critic 输出不变, 就能控制遗忘。

### 4.2 $M_{\text{new}}$ 由 environment return range 决定

**Corollary 4.2**: 假设 MC return $G_t^g \in [G_{\text{min}}, G_{\text{max}}]$, $G_{\text{abs}} = \max(|G_{\text{min}}|, |G_{\text{max}}|)$, 则:
$$M_{\text{new}} \le 2(1+\gamma)G_{\text{abs}}$$

- $G_t^g$: 从 state $s_t$ 出发执行 goal $g$ 的 Monte Carlo return
- $G_{\text{abs}}$: return 的最大绝对值, **environment 内生决定, 我们管不着也不用管**

证明 (Appendix A.4.2): 因为 critic 用 MC 拟合, $|V(s)| \le G_{\text{abs}}$。从 $G_t = r_t + \gamma G_{t+1}$ 得 $r_t = G_t - \gamma G_{t+1}$, 所以 $|r_t| \le (1+\gamma)G_{\text{abs}}$。代入 advantage:
$$|\hat{A}| \le |r| + \gamma|V(s')| + |V(s)| \le (1+\gamma)G_{\text{abs}} + \gamma G_{\text{abs}} + G_{\text{abs}} = 2(1+\gamma)G_{\text{abs}}$$

**人话**: $M_{\text{new}}$ 天然被 environment 的 reward range bounded, 不需要 model 做任何事。你只要在新 task 上正常 fit MC return, $M_{\text{new}}$ 就自动在一个安全范围内。

### 4.3 Decoupling 的真正含义

这两个 bound 的 source 完全 orthogonal:
- $M_{\text{old}}$ 来自 **model 内部** (critic accuracy), 可以通过 functional regularization 控制
- $M_{\text{new}}$ 来自 **environment 外部** (return range), 自然 bounded

所以可以:
- 用 frozen critic 在 old data 上 minimize $\varepsilon_V$ → 控制 $M_{\text{old}}$ (stability)
- 用 trainable critic 在 new task 上 fit MC return → $M_{\text{new}}$ 自然 bounded (plasticity)

这就是 dual-critic architecture 的理论依据。跟传统 global KL constraint 完全不同: 传统方法同时压 $D_{\text{old}}$ 和 $D_{\text{new}}$, 两者 conflict; 这里通过 decouple $M$ 这一项, 让两个 objective 在 architecture level 分开。

---

## 5. Architecture: Dual-Critic + GCVF

### 5.1 Goal-Conditioned Value Formulation (GCVF)

传统 VLA 的 value head 只 input state representation (VLM backbone 的 hidden state), 不显式 condition on language goal $g$。问题: 同一个 state 在不同 goal 下 value 完全不同 ("拿碗" vs "推盘子"), value head 不 condition on language 就会 confuse。

GCVF 的做法 (Figure 1): 把 VLM backbone 的 language token embedding 和 state representation **concatenate** 再喂给 MLP:
$$V(s,g) = \text{MLP}([\phi(s); \text{emb}(g)])$$

- $\phi(s)$: VLM backbone 输出的 state representation
- $\text{emb}(g)$: language goal 的 embedding (从 VLM 的 language token 拿)
- $[\cdot;\cdot]$: concatenation

这看起来简单, 但效果显著 (Table 6 ablation): 加 GCV 后 forgetting 从 0.06 降到 0.02, FT 从 -0.72 改善到 -0.70。

这与 Pi-0.5 (Intelligence et al., 2025) 中发现一致: VLA 的 value head 普遍 language-following 能力差。reference: https://arxiv.org/abs/2504.16054

### 5.2 Dual-Critic Architecture

```
                Shared Backbone ϕ (VLM, LoRA fine-tuned)
                       ↓
              [state emb] + [language emb]
                       ↓
        ┌──────────────┼──────────────┐
        ↓               ↓               ↓
   GCV Critic        MC Critic       Action Head ω
   (θ_GCV, frozen)  (θ_MC, trainable)
        ↓               ↓
   V_old(s,g)        V_new(s,g)     π(a|s,g)
```

**Frozen GCV Critic $\theta_{\text{GCV}}$**:
- Task 转移时, 把 converged MC critic 复制为 GCV critic, **冻结**
- 提供 $V_{\text{old}}(s,g)$ 的 reference value
- Backbone $\phi$ 仍可训练, 但受 GCV loss 约束: 当 $\phi$ 更新导致 $V_{\phi,\theta_{\text{GCV}}}$ 偏离 $V_{\text{old}}$ 时, penalize

**Trainable MC Critic $\theta_{\text{MC}}$**:
- 在新 task trajectories 上 fit MC return $G_t^{g_{\text{new}}}$
- 不受 frozen anchor 约束, 自然 enjoy plasticity
- $M_{\text{new}}$ 由 return range 自然 bounded

**Action Head $\omega$**: 标准 PPO 训练, advantage 用 GAE (Eq. 16):
$$\hat{A}_t = \sum_{k=0}^\infty (\gamma\lambda)^k \delta_{t+k}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\gamma \in [0,1]$: discount factor
- $\lambda \in [0,1]$: GAE smoothing parameter (bias-variance trade-off)
- $\delta_t^V$: TD error
- 这里 $V$ 用 trainable MC critic 的输出

### 5.3 为什么不用 direct advantage optimization?

Paper 4.3 节 explicit 说明: 在 VLA 下, accurate advantage estimation 需要 expensive multi-step rollouts + bootstrapping, 难以 scale。原因:
- VLA backbone 一个 forward pass 涉及 vision encoder + language encoder + transformer + action decoding, 每步 rollout 成本巨大
- TD bootstrapping 需要 target network, large-scale multi-modal transformer 上内存开销极大
- 所以选 **MC estimation** 而非 TD: bias-free, 不需要 target network

参考 Wen et al. 2024 TinyVLA: https://ieeexplore.ieee.org/document/10803168

---

## 6. Loss Function 详解

Total loss (Eq. 8):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + \alpha \mathcal{L}_{\text{KL}} + \beta_V \mathcal{L}_{\text{GCV}}^V + \eta \mathcal{L}_{\text{MC}}^V$$

### 6.1 $\mathcal{L}_{\text{PPO}}$ - 控制 $D_{\text{new}}$

PPO clipped surrogate (Eq. 14):
$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{(s,a,g) \sim \mathcal{B}_{\text{new}}}\left[\min(r_\theta \hat{A}, \text{clip}(r_\theta, 1-\epsilon, 1+\epsilon) \hat{A})\right]$$

- $r_\theta(s,a,g) = \frac{\pi_\theta(a|s,g)}{\pi_{\text{old}}(a|s,g)}$: probability ratio, 当前 policy vs 采样 policy
- $\epsilon$: clipping range (Tab. 7 中是 0.2)
- $\hat{A}$: GAE-based advantage

加 early stopping KL (Eq. 15):
$$D_{\text{KL}}(\pi' \| \pi_\theta) \approx \mathbb{E}_{(s,g) \sim \mathcal{B}_{\text{new}}}\left[\log \frac{\pi'(a|s,g)}{\pi_\theta(a|s,g)}\right] \le d_{\text{targ}}$$

这个 KL 是 new policy vs its own previous iteration (PPO 标准 trust region), 控制 $D_{\text{new}}$ 在新 task state 分布下的 drift。reference: https://arxiv.org/abs/1707.06347

### 6.2 $\mathcal{L}_{\text{KL}}$ - 控制 $D_{\text{old}}$

$$\mathcal{L}_{\text{KL}}(\phi, \omega) = \mathbb{E}_{(s,g) \sim \mathcal{B}_{\text{old}}}\left[D_{\text{KL}}(\pi_{\text{old}}(\cdot|s,g) \| \pi(\cdot|s,g))\right]$$

- $\mathcal{B}_{\text{old}}$: old task replay buffer 中采样的
- $\pi_{\text{old}}$: 生成 $\mathcal{B}_{\text{old}}$ 的那个 old policy snapshot
- 注意是 reverse KL $\pi_{\text{old}} \| \pi$: 让 $\pi$ 在 $\pi_{\text{old}}$ 高概率的地方也保持高概率 (mode-seeking)

这本质是 behavior cloning, 在 old task states 上保持 policy 不漂移。

### 6.3 $\mathcal{L}_{\text{GCV}}^V$ - 控制 $M_{\text{old}}$

$$\mathcal{L}_{\text{GCV}}^V(\phi) = \beta_V \mathbb{E}_{(s,g) \sim \mathcal{B}_{\text{old}}}\left[\|V_{\phi,\theta_{\text{GCV}}}(s,g) - V_{\text{old}}(s,g)\|^2\right]$$

- $\beta_V$: Lagrange multiplier (Tab. 7 中 0.01)
- $V_{\phi,\theta_{\text{GCV}}}(s,g)$: 当前 backbone $\phi$ + frozen GCV head 输出的 value
- $V_{\text{old}}(s,g)$: 在 old task 上 recorded 的 reference value

**这个 loss 在干嘛?** Backbone $\phi$ 是 shared 的 (single VLM), 训 new task 时 $\phi$ 会更新。我们希望: 即使 $\phi$ 改变了, 当 $\phi$ 输出经过 frozen GCV head, 在 old data 上仍然 reconstruct 出 $V_{\text{old}}$。这等价于约束 $\phi$ 在 old state 上的 representation 不漂移到 GCV head 不能理解的区域。

**与 EWC / L2 regularization 的区别**: EWC 直接约束参数空间, GCV 约束 *functional output*。在 representation 空间上更鲁棒, 因为不同参数可能产生相同 value output, EWC 会过度约束。

### 6.4 $\mathcal{L}_{\text{MC}}^V$ - Realize Bounded $M_{\text{new}}$

$$\mathcal{L}_{\text{MC}}^V(\theta_{\text{MC}}, \phi) = \mathbb{E}_{(s,g) \sim \mathcal{B}_{\text{new}}}\left[\|V_{\phi,\theta_{\text{MC}}}(s,g) - G_t^{g_{\text{new}}}\|^2\right]$$

- $\eta$: coefficient (Tab. 7 中 1.0)
- $G_t^{g_{\text{new}}}$: 新 task 上的 MC return

让 trainable MC critic 拟合 MC return, 自然 enjoy Corollary 4.2 的 boundedness。注意这里没有 anchor, critic 可以自由学新 task 的 value structure。

### 6.5 Q-based Version (Appendix B.2)

对偶的 Q-based 版本:
$$\mathcal{L}_{\text{GCV}}^Q(\phi) = \beta_Q \mathbb{E}_{(s,a,g) \sim \mathcal{B}_{\text{old}}}\left[\|Q_{\phi,\theta_{\text{GCV}}}(s,a,g) - Q_{\text{old}}\|^2\right]$$
$$\mathcal{L}_{\text{MC}}^Q(\phi, \theta_{\text{MC}}) = \mathbb{E}_{(s,a,g) \sim \mathcal{B}_{\text{new}}}\left[\|Q_{\phi,\theta_{\text{MC}}}(s,a,g) - G_t^{g_{\text{new}}}\|^2\right]$$

Q-based 的 bound (Corollary A.4, A.5):
$$M_{g_{\text{old}}} \le 2(\|Q_{\text{old}}\|_\infty + \varepsilon_Q), \quad M_{g_{\text{new}}} \le 2 G_{\text{abs}}$$

注意 Q-based 的 bound 是 $2$ 而非 $2(1+\gamma)$, 因为 Q-based advantage $\hat{A} = Q(s,a) - \mathbb{E}_{a'}[Q(s,a')]$ 没有 reward 显式出现。

---

## 7. Algorithm 1 流程

```
Input: pretrained VLA π_θ0, task stream {T_1, ..., T_K}

for k = 1 to K:
    if k == 1:
        Train π_θ1 on T_1 using standard RL (PPO + MC critic)
        Freeze π_θ1 and its value as π_old, V_old
        Store replay buffer B_old
    else:
        for each training iteration:
            Collect new task trajectories τ ~ π_θ on T_k
            Sample B_new ~ τ, B_old ~ D_old
            Compute L_total = L_PPO + α L_KL + β_V L_GCV^V + η L_MC^V
            Update θ via gradient descent
        At task transition: copy converged MC critic → freeze as GCV critic
                            Initialize fresh MC critic
```

**关键操作顺序**: 在 task 边界 $k \to k+1$:
1. Copy $\theta_{\text{MC}}^{(k)} \to \theta_{\text{GCV}}^{(k+1)}$ (frozen)
2. Init 新的 $\theta_{\text{MC}}^{(k+1)}$ (random 或某种 init)
3. 把 $\pi_\theta$ 当前 snapshot 存为 $\pi_{\text{old}}$ (用于 $\mathcal{L}_{\text{KL}}$)
4. 更新 $\mathcal{B}_{\text{old}}$ (合并 prior tasks 的 buffer)

---

## 8. 实验解读

### 8.1 Setup
- **Benchmark**: LIBERO (Liu et al., 2023), 4 个 task subset (Task-1 到 Task-4)
- **Base model**: OpenVLA-OFT (Kim et al., 2025)
- **Training**: PPO + LoRA rank 32
- **Hardware**: A100-SXM4-80GB
- **Codebase**: 基于 RIPT-VLA (Tan et al., 2025)

reference: https://libero-project.github.io/, https://arxiv.org/abs/2502.19645, https://arxiv.org/abs/2505.17016

### 8.2 Single-task (Table 1)

| Method | FAR ↑ | BWT ↑ | FT ↑ | F ↓ |
|---|---|---|---|---|
| SL | 0.00 | -0.62 | -0.50 | 0.62 |
| MTL | 0.96 | 0.11 | -0.06 | 0.06 |
| ER | 0.60 | -0.51 | -0.06 | 0.53 |
| LWF | 0.67 | -0.50 | -0.06 | 0.50 |
| **CRL-VLA (V)** | 0.67 | -0.49 | -0.06 | 0.50 |
| **CRL-VLA (Q)** | **0.98** | -0.02 | -0.06 | **0.03** |

Metric 解释:
- **FAR** (Final Average Return) $= \frac{1}{T}\sum_{i=1}^T R_{T,i}$: 所有 task 最终平均 success rate
- **BWT** (Backward Transfer) $= \frac{1}{T-1}\sum_{i=1}^{T-1}(R_{T,i} - R_{i,i})$: 学新 task 对旧 task 影响, 负值表示遗忘
- **FT** (Forward Transfer) $= \frac{1}{T-1}\sum_{i=2}^T (R_{i-1,i} - b_i)$: 学之前 task 对新 task 初始化帮助
- **F** (Forgetting) $= \frac{1}{T-1}\sum_{i=1}^{T-1}(\max_{k\ge i} R_{k,i} - R_{T,i})$: peak performance 到 final 的 degradation

观察:
- SL 完全 fail (FAR=0, F=0.62), 经典 catastrophic forgetting
- MTL 是 oracle (能同时看所有 task), 0.96 FAR, 但 unrealistic
- CRL-VLA (Q) 0.98 FAR, F=0.03, **超过 MTL** → 暗示 GCV 约束反而帮助了 single-task 的 generalization (可能通过 value consistency 防止 overfit)
- CRL-VLA (V) 在 single-task 表现差 (0.67), 与 LWF 类似

### 8.3 Multi-task (Table 2)

| Method | FAR ↑ | BWT ↑ | FT ↑ |
|---|---|---|---|
| SL | 0.62 | 0.07 | 0.25 |
| MTL | 0.49 | 0.00 | 0.25 |
| ER | 0.62 | 0.05 | 0.00 |
| LWF | 0.63 | 0.10 | 0.00 |
| **CRL-VLA (V)** | **0.74** | **0.17** | 0.00 |
| CRL-VLA (Q) | 0.66 | -0.03 | 0.00 |

完全反转:
- **CRL-VLA (V) 最佳**, FAR=0.74, BWT=0.17 (正 BWT!)
- CRL-VLA (Q) 反而 BWT=-0.03, 不如 V 版本

**Paper 5.2 节解释**: 在 multi-task 中, state-goal distribution 多样化, 同一 state 在不同 task 下可能对应多个 effective action。
- **Q-based** 在 action-conditioned level 约束 → 隐式强制旧 action semantics, 可能与 new task 的多 action 实现 conflict
- **V-based** 在 state-goal level marginalize action → 对 action 多样性 robust, 仍能 preserve long-horizon value semantics

直觉: Q-based over-constrains, 它不仅约束 value 不变, 还约束特定 (s,a) pair 的 value 不变。V-based 只约束 state-level value, action 可以重新选择。

### 8.4 Ablation 关键发现

**$\mathcal{L}_{\text{GCV}}^V$ weight (Table 4)**:
- 0.001 最佳: FAR=0.98, BWT=0.00, F=0.00
- 0.00001 太小: 不能 suppress $M_{\text{old}}$, FAR=0.91
- 0.1 太大: over-constrain, FAR=0.93

**Policy KL weight α (Fig. 3 right)**:
- 0.00001 最佳: FAR=0.98
- 太大: narrow trust region, 抑制 $M_{\text{new}}$ 利用
- 太小: policy drift, 实际 $M_{\text{old}}$ 反而变大

**$\mathcal{L}_{\text{MC}}^V$ weight (Table 5)**:
- 1.0 最佳: FAR=0.95
- 太小: critic error 增大, $M_g$ 实际变大
- 太大: optimization 过 conservative, FAR 下降

**GCV ablation (Table 6)**:
- With GCV: FAR=0.31, F=0.02, FT=-0.70
- Without GCV: FAR=0.29, F=0.06, FT=-0.72

GCV 提升不算 dramatic, 但 FT 从 -0.72 → -0.70, forgetting 从 0.06 → 0.02, 说明 GCV 主要起 stability 作用。

---

## 9. 跟相关工作的比较

### 9.1 vs EWC (Kirkpatrick et al., 2017)
EWC 用 Fisher information 约束参数空间。问题: Fisher 在 large multi-modal transformer 上计算 expensive, 而且 Fisher 是 local approximation, 在 task shift 下不准确。CRL-VLA 用 functional constraint (value output), 不需要二阶信息。

reference: https://arxiv.org/abs/1612.00796

### 9.2 vs LwF (Li & Hoiem, 2017)
LwF 用 knowledge distillation: 让 new model 在 new data 上模仿 old model output。问题: old model 在 new data 上 output 本身就 misleading (out of distribution)。CRL-VLA 反过来: 在 old data 上让 new model 重现 old value, 这是 in-distribution 的。

reference: https://arxiv.org/abs/1606.09282

### 9.3 vs Experience Replay
ER 把 old data 混入新 task 训练。问题: old data 的 gradient 与 new task gradient 可能 conflict, 产生 off-policy error (Korbak et al., 2022)。CRL-VLA 用 old data 约束 value consistency, 是更 soft 的 constraint, 不直接产生 policy gradient。

reference: https://proceedings.mlr.press/v151/korbak22a.html

### 9.4 vs Backbone Freezing (VLM2VLA, Hancock et al., 2025)
VLM2VLA 用 LoRA + frozen backbone。问题: 假设 pretrained representation 在 task shift 后仍 aligned with new language goal, 但实际会失效。CRL-VLA 保留 backbone 可训练性, 用 GCV 约束其 drift。

reference: https://arxiv.org/abs/2509.22195

### 9.5 vs Shenfeld 2025 "RL's Razor"
Shenfeld 主张 online RL 比 supervised learning 更不容易遗忘, 因为 RL 的 update 是 on-policy 的。CRL-VLA extends 这个观点: 在 continual setting 下, 用 KL regularization + GCV consistency 替代 experience replay 的 off-policy gradient。

reference: https://arxiv.org/abs/2509.04259

### 9.6 vs Recent VLA continual work
- Stellar-VLA (Wu et al., 2025): skill-centric knowledge space, evolving task representation
- DMPEL (Lei et al., 2025): mixture of progressive PEFT experts
- ChatVLA (Zhou et al., 2025): decouple multi-modal understanding from action execution

这些都是 parameter-efficient 或 architecture-based, CRL-VLA 是 **regularization-based** with explicit theoretical bound, 不同 paradigm。

reference: https://arxiv.org/abs/2511.18085, https://arxiv.org/abs/2506.05985, https://aclanthology.org/2025.emnlp-main.305/

---

## 10. 最直觉的 take-away

让我用几句话总结最核心的 intuition:

1. **Forgetting = old policy 觉得 new policy 偏离的程度 × policy 实际漂移程度**: 不是单纯控制 policy 不变, 而是控制 "old policy 觉得 new policy 选的 action 有多差" 这个乘积。

2. **这个乘积可以解耦**: "old policy 觉得多差" 在旧 task 上由 critic 准不准决定 (可控), 在新 task 上由 environment reward range 决定 (天然 bounded)。两个 control mechanism 完全 orthogonal, 所以可以独立控制。

3. **Dual critic 是这个 decoupling 的 architecture 实现**: Frozen critic 锚定旧 value (stability), trainable critic 自由学新 value (plasticity), 两者通过 shared backbone 但不同 head 分开 optimize。

4. **GCVF 让 value 对 language 敏感**: 这是 task shift 下保持 value semantics 的关键, 没有 language conditioning 的 value head 在多 task 下会 collapse。

5. **Q vs V 的 trade-off**: Q 更 fine-grained (per-action value), 适合 single-task; V 更 coarse (state-level), 适合 multi-task 因为它对 action 多样性 robust。

6. **Asymmetric regulation**: 强约束 old task (用 frozen critic + KL + GCV consistency), 弱约束 new task (只有 PPO trust region)。这正是 "asymmetric" 的含义。

7. **Lagrangian relaxation**: 把 constrained optimization (max new return s.t. old return ≥ old - δ) 转化为 unconstrained, 三个 multiplier $\alpha, \beta_V, \eta$ 控制 trade-off。

---

## 11. 我觉得最 interesting 的点

1. **Theorem 4.1 的 unified bound**: 把 stability 和 plasticity 放在同一个 $M \cdot D$ 框架下, 然后 corollary 给出 decoupling mechanism。这种 "find a quantity that couples two objectives, then show it has independent control mechanisms" 的思路在 ML theory 中很 powerful, 跟 control theory 中的 passivity-based control 有 conceptual 类似。

2. **Frozen critic 是 value anchor, 不是 policy anchor**: 传统 distillation 锚定 policy output (LwF), CRL-VLA 锚定 value function output, 让 action space 自由。这更符合 RL 的本质, 因为 RL 中 value 才是真正的 objective, policy 只是实现 value 的手段。

3. **Q vs V 的 reversal 现象**: Single-task Q 好, multi-task V 好。这给出 value function granularity 的重要 insight: 多 task 共享 state 时, per-action level 的约束过强。这跟 multi-task RL 文献中 "shared state, diverse action" 的 observation 一致。

4. **GCVF 的简单性**: 就是 concatenate language embedding, 但效果显著。暗示 VLA 社区之前 under-explored value function 的 language conditioning, 都把重心放在 policy head 上。这跟 RLHF 中 value head 的重要性类似。

---

## 12. 可以延展的方向

如果我来 extend:

1. **World model 替代 value critic**: GCV 提供了 value-level anchor, 能否 extend 到 dynamics-level anchor? 比如用 latent world model 在 old data 上保持 latent transition consistency, 这会提供更强的 representation-level constraint。参考 DreamerV3: https://arxiv.org/abs/2301.04104

2. **Hierarchical goal decomposition**: 当前 goal 是 atomic language instruction。如果 task 是 hierarchical, GCV 需要在 hierarchy 每一层都 condition, 可能需要 hierarchical value function (like FuN, FeUdAL Networks)。

3. **Active replay buffer selection**: 当前 $\mathcal{B}_{\text{old}}$ 是 passive store。能否基于 GCV loss 的 gradient magnitude 主动选择哪些 old data 对当前更新最 critical? 这是 importance sampling for continual learning。

4. **Theoretical tightness via concentration bounds**: 当前 bound 是 worst-case (sup over states), 能否用 PAC-Bayes 或 information-theoretic bound 给出更紧的 average-case bound?

5. **Multi-agent extension**: 在 multi-robot 场景中, 每个 robot 学不同 task, GCV 能否作为 communication protocol, 让 robots share value anchors? 这跟 federated continual learning 有关。

6. **完全不用 replay buffer**: Replay buffer 在实际 robot learning 中 storage 成本巨大。能否用 generative model (VAE / diffusion) 在 latent space 生成 "pseudo old states", 让 GCV 在 pseudo states 上做 consistency? 这是 model-based continual learning 方向。

---

## 13. Limitations

1. **实验规模有限**: 只有 LIBERO, 没有真实机器人, 没有 long-horizon task (task chain 长度最大 3)。Single-task 只有 3 个 task, multi-task 最多 3 个 task。真正 lifelong 应该是 100+ task sequence。

2. **$\mathcal{B}_{\text{old}}$ 细节缺失**: 没说 buffer size, 是否 cumulative, 是否 prioritized sampling。在真实 long-horizon 场景中, buffer 管理是 scalability 关键。

3. **Theoretical bound 的 tightness**: Stability bound 用了 Hölder + Pinsker, 这些不等式都很 loose。实际遗忘通常远小于 bound, 所以 $\beta_V$ 才能取小值 (0.01-0.001)。没有给出 lower bound 或 tightness analysis。

4. **GCV 的累积问题**: 当任务序列很长时, GCV critic 数量增加 (每个 task transition 一个)? 还是只保留一个 cumulative? Paper 没说清楚。如果只保留一个, 是否有 representational drift 累积?

5. **Language shift 没测**: GCVF 假设 VLM 的 language embedding 已经语义 aligned。如果 task shift 涉及 distribution shift in language (e.g., 从 "pick and place" 切到 "pour and stir"), embedding 是否仍然 useful? Paper 没测这种 cross-domain language shift。

---

## 14. Final Thoughts

这篇 paper 在我看来是 RL theory + VLA practical engineering 的一个不错 marriage。理论部分虽然 bound 不紧, 但给出了 actionable insight (decouple $M$ via dual critic)。工程部分相对 simple (PPO + 两个 critic + KL + value consistency), 没有太多 bells and whistles, 容易复现。

最大的 limitation 是实验规模: 4 个 LIBERO task subset, 最多 3 个 task 的 sequence, 远不足以 claim "lifelong" learning。但作为 initial exploration, 这个 framework 提供 promising foundation。

我觉得最值得深挖的方向是 **GCV 作为 representation anchor 的角色**: 它表面上是 value function 的 constraint, 但实际效果是约束 backbone representation 不在 old state 分布上 drift。这与 PlaNet / Dreamer 的 latent dynamics model 有 conceptual 相似性。如果能把 GCV 升级为 latent dynamics + reward + value 的 joint anchor, 可能能获得更强的 stability guarantee。

另一个值得思考的方向: **能否完全不用 replay buffer?** Replay buffer 在实际 robot learning 中 storage 成本巨大 (high-dim observations)。GCV consistency 已经在 old data 上做了约束, 但仍需 $\mathcal{B}_{\text{old}}$。能否用 generative model (e.g., VAE 或 diffusion) 在 latent space 生成 "pseudo old states", 让 GCV 在 pseudo states 上做 consistency? 这是 model-based continual learning 的方向。

总之, paper 给出了 continual VLA 的一个 principled framework, 后续工作可以在 scalability、theory tightness、architecture (e.g., world model) 上 extend。

---

**主要 reference 链接**:
- LIBERO: https://libero-project.github.io/
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- RIPT-VLA: https://arxiv.org/abs/2505.17016
- PDL (Kakade & Langford 2002): https://proceedings.mlr.press/v2/kakade02a.html
- CPO (Achiam et al. 2017): http://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf
- PPO: https://arxiv.org/abs/1707.06347
- EWC: https://arxiv.org/abs/1612.00796
- LwF: https://arxiv.org/abs/1606.09282
- Korbak 2022: https://proceedings.mlr.press/v151/korbak22a.html
- Shenfeld "RL's Razor": https://arxiv.org/abs/2509.04259
- PI-0.5: https://arxiv.org/abs/2504.16054
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- VLM2VLA: https://arxiv.org/abs/2509.22195
- Stellar-VLA: https://arxiv.org/abs/2511.18085
- DMPEL: https://arxiv.org/abs/2506.05985
- ChatVLA: https://aclanthology.org/2025.emnlp-main.305/
- TinyVLA: https://ieeexplore.ieee.org/document/10803168
- DreamerV3: https://arxiv.org/abs/2301.04104

希望这个 walk-through 帮你 build 起对 CRL-VLA 的 intuition!核心 idea 一句话: **把 continual learning 重新 formulate 为 advantage magnitude 控制, 然后用 critic approximation error vs environment return range 这两个 independent source 解耦 stability 和 plasticity**, 这是 paper 的 essential contribution。

---

# CRL-VLA: Continual Vision–Language–Action Learning 深度解析

Hi Andrej, 这篇 paper 我觉得最亮的地方在于它**把 stability-plasticity dilemma 重新 formulate 为一个 advantage magnitude 的统一控制问题**, 然后 theoretically 证明这两个量在 old/new task 上由完全不同的 mechanism 决定, 所以可以 decouple。下面我尽量 build 起完整的 intuition chain。

---

## 1. Problem Setup: Continual Post-Training of VLA

Setting 是 goal-conditioned MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{G}, P, r, \gamma \rangle$, 其中 $g \in \mathcal{G}$ 是 natural language instruction (e.g., "pick up the black bowl from table center and place it on the plate")。

- $\pi_\theta(a|s,g)$: VLA policy, backbone 是 OpenVLA-OFT (Kim et al., 2025), 一个 Vision-Language-Action transformer
- 任务序列 $\mathcal{T} = \{\mathcal{T}_1, \dots, \mathcal{T}_K\}$, 在 stage $k$ 只能访问 $\mathcal{T}_k$ 的 data, 没有之前任务的 reward/gradients
- Value function $V^\pi(s,g) = \mathbb{E}_{\tau \sim \pi, g}[\sum_{t=0}^\infty \gamma^t r(s_t,a_t,g) | s_0=s]$
- Advantage $A^\pi(s,a,g) = Q^\pi(s,a,g) - V^\pi(s,g)$

**核心约束 (Eq. 1-2)**:
$$\max_{\pi^k} J_{g_k}(\pi^k; \mathcal{T}_k) \quad \text{s.t.} \quad J_{g_{k-1}}(\pi^k; \mathcal{T}_{k-1}) \ge J_{g_{k-1}}(\pi^{k-1}; \mathcal{T}_{k-1}) - \delta$$

变量解释:
- $J_{g_k}(\pi^k; \mathcal{T}_k)$: 在 task $k$ 上 policy $\pi^k$ 关于 goal $g_k$ 的 expected return
- $\delta$: 对旧 task 性能退化的 tolerance
- $g_k$: task $k$ 对应的 language goal

这是一个 constrained optimization, 但 $\delta$ 在实际中 intractable to measure directly, 因为需要在新 policy 下重新评估 old task 的 return, 这正是灾难性遗忘 (catastrophic forgetting) 的本质。

---

## 2. The Key Insight: Advantage Magnitude $M_g$

**Definition 4.1**: 对于 anchored policy $\pi$ (通常取 $\pi^{old}$) 和 visiting policy $\pi^\kappa$, 定义 advantage magnitude:
$$M_g(\pi^\kappa) \triangleq \sup_{(s,a) \in \text{supp}(d_g^{\pi^\kappa})} |A^\pi(s,a)|$$

变量解释:
- $M_g(\pi^\kappa)$: 标量, 在 $\pi^\kappa$ 访问的 state-action 分布下, anchored policy 的 advantage 绝对值上确界
- $d_g^{\pi^\kappa}(s)$: $\pi^\kappa$ 在执行 goal $g$ 时的 discounted state occupancy measure, $d_g^\pi(s) = (1-\gamma)\sum_{t=0}^\infty \gamma^t \Pr(s_t = s | g, \pi)$
- $\text{supp}(\cdot)$: distribution 的 support
- $A^\pi(s,a)$: anchored policy 的 advantage, 注意这里 advantage 是用 anchored policy $\pi^{old}$ 计算的, 而不是 visiting policy

$M_g \to 0$ 意味着 visiting policy 在 advantage 上接近 anchor; $M_g$ 大意味着存在 state-action 上 anchored policy 觉得有较大改进空间。

**两个 metric**:
$$M_{\text{old}} \triangleq \mathbb{E}_{g_{\text{old}} \sim p_{\text{old}}}[M_{g_{\text{old}}}(\pi^{\text{new}})] \quad \text{(stability)}$$
$$M_{\text{new}} \triangleq \mathbb{E}_{g_{\text{new}} \sim p_{\text{new}}}[M_{g_{\text{new}}}(\pi^{\text{new}})] \quad \text{(plasticity)}$$

- $p_{\text{old}}(g)$: 旧 task 的 goal distribution
- $p_{\text{new}}(g)$: 新 task 的 goal distribution
- $M_{\text{old}}$ 衡量新 policy 在旧 task state 分布下, 旧 policy 觉得 advantage 多大 → 这个值大说明新 policy 在 old state 上偏离了 anchor
- $M_{\text{new}}$ 衡量新 policy 在新 task state 分布下, 旧 policy 觉得 advantage 多大 → 这是 plasticity 的来源, 旧 policy 在新 task 上有改进空间

**Intuition**: 把 continual learning 想成 "how much does old policy think new policy's actions deviate from optimal?"。如果新 policy 在旧 task 的 states 上选择的 action 让 old policy 觉得 "I could do much better", 那就是遗忘。反之, 在新 task 上, 我们 *希望* old policy 觉得 advantage 大, 因为这意味着 new policy 正在探索 old policy 没学到的区域。

---

## 3. Theorem 4.1: Unified Stability-Plasticity Bounds

这是 paper 的理论核心。先定义 policy divergence:
$$D_{\text{old}} \triangleq \sqrt{2 \mathbb{E}_{s \sim d_{\text{old}}^{\pi^{\text{old}}}}[D_{\text{KL}}(\pi^{\text{new}}(\cdot|s) \| \pi^{\text{old}}(\cdot|s))]}$$
$$D_{\text{new}} \triangleq \sqrt{2 \mathbb{E}_{s \sim d_{\text{new}}^{\pi^{\text{new}}}}[D_{\text{KL}}(\pi^{\text{new}}(\cdot|s) \| \pi^{\text{old}}(\cdot|s))]}$$

变量解释:
- $D_{\text{old}}$: 在 old task 的 state occupancy 下, new vs old policy 的 KL divergence, scaled by $\sqrt{2}$ (Pinsker 不等式的标准形式)
- $d_{\text{old}}^{\pi^{\text{old}}}$: old policy 在 old task 上的 state occupancy
- $D_{\text{new}}$: 在 new task 下, new policy 的 state occupancy 上的 KL divergence

注意两个 KL 的 state distribution 不同: 一个是 old task 上 old policy 的 visitation, 一个是 new task 上 new policy 的 visitation。这是为什么后面可以 decouple 的关键。

**Bound 1 (Stability)**:
$$|J_{\text{old}}(\pi^{\text{new}}) - J_{\text{old}}(\pi^{\text{old}})| \le \frac{2\gamma}{(1-\gamma)^2} \cdot M_{\text{old}} \cdot D_{\text{old}}$$

**Bound 2 (Plasticity)**:
$$J_{\text{new}}(\pi^{\text{new}}) - J_{\text{new}}(\pi^{\text{old}}) \le \frac{1}{1-\gamma} \cdot M_{\text{new}} \cdot D_{\text{new}}$$

注意:
- Stability bound 前系数是 $\frac{2\gamma}{(1-\gamma)^2}$, 包含 $\gamma$ 在分子上, 还有 $(1-\gamma)^2$ 在分母上
- Plasticity bound 前系数只有 $\frac{1}{1-\gamma}$
- 这个系数差异很重要: stability 多了一个 occupancy mismatch 项, 因为 new policy 访问的 state 分布与 old policy 不同, 这个 drift 需要额外被 bounded
- 当 $\gamma \to 1$ (long-horizon), stability bound 增长得比 plasticity 快, 这意味着长 horizon 任务中 stability 更难维持, 与直觉一致

### 3.1 Stability Bound 推导的关键步骤

使用 **Performance Difference Lemma (PDL)** (Kakade & Langford, 2002):
$$J_g(\pi^{\text{new}}) - J_g(\pi^{\text{old}}) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d_g^{\pi^{\text{new}}}}\big[\mathbb{E}_{a \sim \pi^{\text{new}}(\cdot|s,g)}[A_g^{\pi^{\text{old}}}(s,a)]\big]$$

关键 trick 是把 $\mathbb{E}_{s \sim d_g^{\pi^{\text{new}}}}$ 分解:
$$\mathbb{E}_{s \sim d_g^{\pi^{\text{new}}}}[f_g(s)] = \underbrace{\mathbb{E}_{s \sim d_g^{\pi^{\text{old}}}}[f_g(s)]}_{\text{(I) Action Mismatch}} + \underbrace{\sum_s (d_g^{\pi^{\text{new}}}(s) - d_g^{\pi^{\text{old}}}(s)) f_g(s)}_{\text{(II) Occupancy Mismatch}}$$

其中 $f_g(s) \triangleq \mathbb{E}_{a \sim \pi^{\text{new}}}[A_g^{\pi^{\text{old}}}(s,a)]$。

**Term (I) - Action Mismatch**: $|f_g(s)| \le \sup_a |A_g^{\pi^{\text{old}}}(s,a)| \le M_g(\pi^{\text{new}})$, 因为 $(s,a) \in \text{supp}(d_g^{\pi^{\text{new}}})$ (a 是从 new policy 采样的)。

**Term (II) - Occupancy Mismatch**: 用 Hölder 不等式 + discounted occupancy bound (Achiam et al., 2017 - CPO):
$$\|d_g^{\pi^{\text{new}}} - d_g^{\pi^{\text{old}}}\|_1 \le \frac{2\gamma}{1-\gamma} \mathbb{E}_{s \sim d_g^{\pi^{\text{old}}}}[D_{\text{TV}}(\pi^{\text{new}} \| \pi^{\text{old}})]$$

这里 $D_{\text{TV}}$ 是 total variation distance。然后用 **Pinsker 不等式** $D_{\text{TV}} \le \sqrt{\frac{1}{2} D_{\text{KL}}}$:
$$|\text{Term (II)}| \le \frac{2\gamma}{1-\gamma} \mathbb{E}_{s \sim d_g^{\pi^{\text{old}}}}\left[\sqrt{\frac{1}{2} D_{\text{KL}}(\pi^{\text{new}} \| \pi^{\text{old}})}\right] \cdot M_g(\pi^{\text{new}})$$

再乘以 PDL 前面的 $\frac{1}{1-\gamma}$, 并对 $g$ 取期望 + Jensen 不等式 $\mathbb{E}[\sqrt{X}] \le \sqrt{\mathbb{E}[X]}$:
$$|J_{\text{old}}(\pi^{\text{new}}) - J_{\text{old}}(\pi^{\text{old}})| \le \frac{2\gamma}{(1-\gamma)^2} M_{\text{old}} D_{\text{old}}$$

### 3.2 Plasticity Bound 推导

对 new task 用 PDL:
$$J_g(\pi^{\text{new}}) - J_g(\pi^{\text{old}}) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d_g^{\pi^{\text{new}}}}[\mathbb{E}_{a \sim \pi^{\text{new}}}[A_g^{\pi^{\text{old}}}(s,a)]]$$

关键: $\mathbb{E}_{a \sim \pi^{\text{old}}}[A_g^{\pi^{\text{old}}}(s,a)] = 0$ (advantage by definition 在 own policy 下 zero-mean), 所以:
$$\mathbb{E}_{a \sim \pi^{\text{new}}}[A_g^{\pi^{\text{old}}}] = \sum_a (\pi^{\text{new}}(a|s) - \pi^{\text{old}}(a|s)) A_g^{\pi^{\text{old}}}(s,a)$$

用 Hölder: $|\mathbb{E}_{a \sim \pi^{\text{new}}}[A]| \le 2 D_{\text{TV}}(\pi^{\text{new}} \| \pi^{\text{old}}) \cdot \sup_a |A_g^{\pi^{\text{old}}}(s,a)|$, 对 new task state, supremum $\le M_{g_{\text{new}}}(\pi^{\text{new}})$。

Pinsker + Jensen:
$$J_{\text{new}}(\pi^{\text{new}}) - J_{\text{new}}(\pi^{\text{old}}) \le \frac{1}{1-\gamma} M_{\text{new}} D_{\text{new}}$$

**为什么没有 occupancy mismatch 项?** 因为 plasticity 是从 new policy 的 perspective 看 improvement, 而新 policy 在 new task 上自然访问 new task 的 states, 不存在 "distribution shift" 问题。而 stability 是 new policy 在 old task 上访问的 states 与 old policy 访问的 states 之间的 distribution shift。

### 3.3 这个 unified bound 的意义

- Stability 和 plasticity 都线性依赖于 $M \cdot D$ 的乘积
- 想要 plasticity 好就要让 $M_{\text{new}} \cdot D_{\text{new}}$ 大
- 想要 stability 好就要让 $M_{\text{old}} \cdot D_{\text{old}}$ 小
- 传统 CRL 方法用 global KL constraint 同时压 $D_{\text{old}}$ 和 $D_{\text{new}}$, 这就 conflict 了
- **新思路**: 既然乘积 $M \cdot D$ 控制两端, 能不能解耦 $M$ 这一项?

---

## 4. The Decoupling Insight: Why $M_{\text{old}}$ and $M_{\text{new}}$ are Independently Controllable

这是 paper 最巧妙的部分。

**Corollary 4.1 (Controllability of Stability)**: 假设 reward $|r(s,a,g_{\text{old}})| \le R_{\text{max}}$ 且 critic approximation error $\varepsilon_V \triangleq \sup_{s,g_{\text{old}}}|V_\theta - V_{\text{old}}|$, 则:
$$M_{\text{old}} \le R_{\text{max}} + (1+\gamma)\varepsilon_V + (1+\gamma)\|V_{\text{old}}\|_\infty$$

变量:
- $R_{\text{max}}$: reward 的最大绝对值
- $\varepsilon_V$: critic approximation error in sup norm
- $\|V_{\text{old}}\|_\infty = \sup_s |V_{\text{old}}(s,g)|$: true value function 的 infinity norm
- $(1+\gamma)$ 系数来自 $\gamma V(s')$ + $V(s)$ 这两项

证明思路: advantage $\hat{A}(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}[V_\theta(s')] - V_\theta(s)$, 三角不等式:
$$|\hat{A}| \le |r| + \gamma |V_\theta(s')| + |V_\theta(s)| \le R_{\text{max}} + (1+\gamma)(\|V_{\text{old}}\|_\infty + \varepsilon_V)$$

**关键 takeaway**: $M_{\text{old}}$ 由 $\varepsilon_V$ 控制, 而 $\varepsilon_V$ 可以通过在 old data $\mathcal{B}_{\text{old}}$ 上的 critic 一致性来 minimize。

**Corollary 4.2 (Natural Boundedness of Plasticity)**: 假设 MC return $G_t^g \in [G_{\text{min}}, G_{\text{max}}]$, 令 $G_{\text{abs}} = \max(|G_{\text{min}}|, |G_{\text{max}}|)$, 则:
$$M_{\text{new}} \le 2(1+\gamma) G_{\text{abs}}$$

变量:
- $G_t^g$: 从 state $s_t$ 出发, 执行 goal $g$ 的 Monte Carlo return
- $G_{\text{abs}}$: return range 的最大绝对值, 由 environment reward 结构决定

证明: 在新 task 上, $V$ 是用 MC 拟合的, 所以 $|V(s)| \le G_{\text{abs}}$。同时 $r_t = G_t - \gamma G_{t+1}$ (从 $G_t = r_t + \gamma G_{t+1}$ 重排), 所以 $|r_t| \le (1+\gamma) G_{\text{abs}}$。代入 advantage:
$$|\hat{A}| \le |r| + \gamma |V(s')| + |V(s)| \le (1+\gamma) G_{\text{abs}} + \gamma G_{\text{abs}} + G_{\text{abs}} = 2(1+\gamma) G_{\text{abs}}$$

**关键 takeaway**: $M_{\text{new}}$ 由 environment 的 return range 自然 bounded, 不需要任何额外约束。

### 4.1 Decoupling 的真正意义

这两个 bound 来源完全 orthogonal:
- $M_{\text{old}}$ bound 来自 critic approximation error → 可以通过 frozen critic + replay buffer 控制
- $M_{\text{new}}$ bound 来自 environment reward structure → 任务本身决定, 不需要 model 做什么

所以可以:
- 用一个 frozen critic 在 old data 上 minimize $\varepsilon_V$ → 控制 $M_{\text{old}}$ (stability)
- 用一个 trainable MC critic 在 new task 上 fit return → $M_{\text{new}}$ 自然 bounded (plasticity)

这就是 **dual-critic architecture** 的理论依据。这比用单一 KL constraint 同时控制 $D_{\text{old}}, D_{\text{new}}$ 优雅得多。

参考 Kakade & Langford 2002 原始 PDL paper: https://proceedings.mlr.press/v2/kakade02a.html  
Achiam et al. 2017 CPO: http://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf

---

## 5. The CRL-VLA Architecture

### 5.1 Goal-Conditioned Value Formulation (GCVF)

传统 VLA 的 value head 只取 state representation (即 VLM backbone 的 hidden state), 不显式 condition on language goal $g$。问题在于: 在 task shift 下, 同一个 state 在不同 goal 下的 value 完全不同, 比如 "拿起黑碗" 的 value head 看到 "推盘子到炉子前" 的 state, 如果 value head 不 condition on language, 就会 confuse。

GCVF 的做法: 将 VLM backbone 的 language token embedding 与 state representation **concatenate** 后再喂给 MLP:
$$V(s,g) = \text{MLP}([\phi(s); \text{emb}(g)])$$

- $\phi(s)$: VLM backbone 的 state representation
- $\text{emb}(g)$: language goal 的 embedding (从 VLM 的 language token 拿)
- $[\cdot;\cdot]$: concatenation

这样 value function 对 language 敏感, 不同 goal 下 value 自然区分。这与 Pi-0.5 (Intelligence et al., 2025) 中的发现一致: VLA 的 value head 普遍 language-following 能力差。

### 5.2 Dual-Critic Architecture

```
                Shared Backbone ϕ (VLM, LoRA fine-tuned)
                       ↓
              [state emb] + [language emb]
                       ↓
        ┌──────────────┼──────────────┐
        ↓               ↓               ↓
   GCV Critic        MC Critic       Action Head ω
   (θ_GCV, frozen)  (θ_MC, trainable)
        ↓               ↓
   V_old(s,g)        V_new(s,g)     π(a|s,g)
```

**Frozen GCV Critic $\theta_{\text{GCV}}$**:
- Task 转移时, 把 converged MC critic 复制为 GCV critic 并冻结
- 提供 $V_{\text{old}}(s,g)$ 的 reference
- Backbone $\phi$ 仍可训练, 但要受 GCV loss 约束: 当 $\phi$ 更新导致 $V_{\phi,\theta_{\text{GCV}}}(s,g)$ 偏离 $V_{\text{old}}(s,g)$ 时, penalize

**Trainable MC Critic $\theta_{\text{MC}}$**:
- 在新 task trajectories 上 fit MC return $G_t^{g_{\text{new}}}$
- 不受任何 frozen anchor 约束, 自然 enjoy plasticity
- $M_{\text{new}}$ 由 return range bounded

**Action Head $\omega$**: 标准 PPO 训练, 用 GAE 计算 advantage (Eq. 16 in appendix):
$$\hat{A}_t = \sum_{k=0}^\infty (\gamma\lambda)^k \delta_{t+k}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\gamma \in [0,1]$: discount factor
- $\lambda \in [0,1]$: GAE smoothing parameter (bias-variance trade-off)
- $\delta_t^V$: TD error

这里 $\hat{A}$ 用 trainable MC critic 的 $V$ 计算, 让 advantage 估计忠于新 task。

### 5.3 Why Not Direct Advantage Optimization?

Paper 在 4.3 节 explicit 说明: 在 VLA settings 下, accurate advantage estimation 需要 expensive multi-step rollouts + bootstrapping, 难以 scale。原因:
- VLA backbone 一个 forward pass 就涉及 vision encoder + language encoder + transformer + action decoding, 每步 rollout 成本巨大
- TD bootstrapping 需要 target network, 在 large-scale multi-modal transformer 上做这个内存开销极大
- 所以选择 **MC estimation** 而非 TD: bias-free, 不需要 target network

参考 Wen et al. 2024 TinyVLA 关于 VLA 推理成本的讨论: https://ieeexplore.ieee.org/document/10803168

---

## 6. Loss Functions 详解

Total loss (Eq. 8 / Eq. 13):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + \alpha \mathcal{L}_{\text{KL}} + \beta_V \mathcal{L}_{\text{GCV}}^V + \eta \mathcal{L}_{\text{MC}}^V$$

### 6.1 $\mathcal{L}_{\text{PPO}}$ - 控制 $D_{\text{new}}$

PPO clipped surrogate (Eq. 14):
$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{(s,a,g) \sim \mathcal{B}_{\text{new}}}\left[\min(r_\theta \hat{A}, \text{clip}(r_\theta, 1-\epsilon, 1+\epsilon) \hat{A})\right]$$

- $r_\theta(s,a,g) = \frac{\pi_\theta(a|s,g)}{\pi_{\text{old}}(a|s,g)}$: probability ratio
- $\epsilon$: clipping range (Tab. 7 中是 0.2)
- $\hat{A}$: GAE-based advantage estimate

加上 early stopping KL threshold (Eq. 15):
$$D_{\text{KL}}(\pi' \| \pi_\theta) \approx \mathbb{E}_{(s,g) \sim \mathcal{B}_{\text{new}}}\left[\log \frac{\pi'(a|s,g)}{\pi_\theta(a|s,g)}\right] \le d_{\text{targ}}$$

这个 KL 是 new policy vs its own previous iteration, 是 PPO 的 trust region, 控制 $D_{\text{new}}$ (在新 task state 分布下的 drift)。

### 6.2 $\mathcal{L}_{\text{KL}}$ - 控制 $D_{\text{old}}$

$$\mathcal{L}_{\text{KL}}(\phi, \omega) = \mathbb{E}_{(s,g) \sim \mathcal{B}_{\text{old}}}\left[D_{\text{KL}}(\pi_{\text{old}}(\cdot|s,g) \| \pi(\cdot|s,g))\right]$$

变量:
- $\mathcal{B}_{\text{old}}$: old task replay buffer 中采样的
- $\pi_{\text{old}}$: 生成 $\mathcal{B}_{\text{old}}$ 的那个 old policy
- 注意: 这里是 $\pi_{\text{old}} \| \pi$ (reverse KL), 让 $\pi$ 在 $\pi_{\text{old}}$ 高概率的地方也保持高概率

这是 behavior cloning 形式的约束, 在 old task states 上保持 policy 不漂移。

### 6.3 $\mathcal{L}_{\text{GCV}}^V$ - 控制 $M_{\text{old}}$

$$\mathcal{L}_{\text{GCV}}^V(\phi) = \beta_V \mathbb{E}_{(s,g) \sim \mathcal{B}_{\text{old}}}\left[\|V_{\phi,\theta_{\text{GCV}}}(s,g) - V_{\text{old}}(s,g)\|^2\right]$$

变量:
- $\beta_V$: Lagrange multiplier (Tab. 7 中是 0.01)
- $V_{\phi,\theta_{\text{GCV}}}(s,g)$: 当前 backbone $\phi$ + frozen GCV head 输出的 value
- $V_{\text{old}}(s,g)$: 在 old task 上 recorded 的 reference value

**这个 loss 在干嘛?** Backbone $\phi$ 是 shared 的 (single VLM), 训练 new task 时 $\phi$ 会更新。我们希望: 即使 $\phi$ 改变了, 当 $\phi$ 输出经过 frozen GCV head, 在 old data 上仍然 reconstruct 出 $V_{\text{old}}$。这等价于约束 $\phi$ 在 old state 上的 representation 不漂移到 GCV head 不能理解的区域。

**与 EWC / L2 regularization 的区别**: EWC 是直接约束参数, GCV 是约束 *functional output*, 在 representation 空间上更鲁棒, 因为不同参数可能产生相同 value。

### 6.4 $\mathcal{L}_{\text{MC}}^V$ - Realize Bounded $M_{\text{new}}$

$$\mathcal{L}_{\text{MC}}^V(\theta_{\text{MC}}, \phi) = \mathbb{E}_{(s,g) \sim \mathcal{B}_{\text{new}}}\left[\|V_{\phi,\theta_{\text{MC}}}(s,g) - G_t^{g_{\text{new}}}\|^2\right]$$

变量:
- $\eta$: coefficient (Tab. 7 中是 1.0)
- $G_t^{g_{\text{new}}}$: 新 task 上的 MC return

让 trainable MC critic 拟合 MC return, 自然 enjoy Corollary 4.2 的 boundedness。

### 6.5 Q-based Version (Appendix B.2)

Q-based 对偶版本:
$$\mathcal{L}_{\text{GCV}}^Q(\phi) = \beta_Q \mathbb{E}_{(s,a,g) \sim \mathcal{B}_{\text{old}}}\left[\|Q_{\phi,\theta_{\text{GCV}}}(s,a,g) - Q_{\text{old}}\|^2\right]$$
$$\mathcal{L}_{\text{MC}}^Q(\phi, \theta_{\text{MC}}) = \mathbb{E}_{(s,a,g) \sim \mathcal{B}_{\text{new}}}\left[\|Q_{\phi,\theta_{\text{MC}}}(s,a,g) - G_t^{g_{\text{new}}}\|^2\right]$$

Q-based 的 Corollary bounds (A.4, A.5):
$$M_{g_{\text{old}}} \le 2(\|Q_{\text{old}}\|_\infty + \varepsilon_Q), \quad M_{g_{\text{new}}} \le 2 G_{\text{abs}}$$

注意 Q-based 的 bound 是 $2$ 而非 $2(1+\gamma)$, 因为 Q-based advantage $\hat{A} = Q(s,a) - \mathbb{E}_{a'}[Q(s,a')]$, 没有 reward 显式出现。

---

## 7. Algorithm 1 流程

```
Input: pretrained VLA π_θ0, task stream {T_1, ..., T_K}

for k = 1 to K:
    if k == 1:
        Train π_θ1 on T_1 using standard RL (PPO + MC critic)
        Freeze π_θ1 and its value as π_old, V_old
        Store replay buffer B_old
    else:
        for each training iteration:
            Collect new task trajectories τ ~ π_θ on T_k
            Sample B_new ~ τ, B_old ~ D_old
            Compute L_total = L_PPO + α L_KL + β_V L_GCV^V + η L_MC^V
            Update θ via gradient descent
        At task transition: copy converged MC critic → freeze as GCV critic
                            Initialize fresh MC critic
```

**关键操作顺序**: 在任务边界 $k \to k+1$ 时:
1. Copy $\theta_{\text{MC}}^{(k)} \to \theta_{\text{GCV}}^{(k+1)}$ (frozen)
2. Initialize new $\theta_{\text{MC}}^{(k+1)}$ (random 或某种 init)
3. 把 $\pi_\theta$ 的当前 snapshot 存为 $\pi_{\text{old}}$ (用于 $\mathcal{L}_{\text{KL}}$)
4. 更新 $\mathcal{B}_{\text{old}}$ (合并之前的 buffer, 或仅保留 task $k$ 的 data)

注意 paper 没说清楚 $\mathcal{B}_{\text{old}}$ 是否 cumulative (即包含所有 prior tasks 的 data) 还是只包含上一个 task 的 data。从 algorithm 1 看是单数 buffer, 可能只存上一次的, 但实验中应该是 cumulative 更合理 (否则无法防御更早任务的 forgetting)。

---

## 8. Experiments 深度解读

### 8.1 Setup
- **Benchmark**: LIBERO (Liu et al., 2023), 4 个 task subset (Task-1 到 Task-4)
- **Base model**: OpenVLA-OFT (Kim et al., 2025)
- **Training**: PPO + LoRA rank 32
- **Hardware**: A100-SXM4-80GB
- **Codebase**: 基于 RIPT-VLA (Tan et al., 2025)

参考 LIBERO: https://libero-project.github.io/  
OpenVLA-OFT: https://arxiv.org/abs/2502.19645  
RIPT-VLA: https://arxiv.org/abs/2505.17016

### 8.2 Single-task 结果 (Table 1)

| Method | FAR ↑ | BWT ↑ | FT ↑ | F ↓ |
|---|---|---|---|---|
| SL | 0.00 | -0.62 | -0.50 | 0.62 |
| MTL | 0.96 | 0.11 | -0.06 | 0.06 |
| ER | 0.60 | -0.51 | -0.06 | 0.53 |
| LWF | 0.67 | -0.50 | -0.06 | 0.50 |
| **CRL-VLA (V)** | 0.67 | -0.49 | -0.06 | 0.50 |
| **CRL-VLA (Q)** | **0.98** | -0.02 | -0.06 | **0.03** |

Metric 解释:
- **FAR** (Final Average Return) $= \frac{1}{T}\sum_{i=1}^T R_{T,i}$: 所有 task 上的最终平均 success rate
- **BWT** (Backward Transfer) $= \frac{1}{T-1}\sum_{i=1}^{T-1}(R_{T,i} - R_{i,i})$: 学习新 task 对旧 task 性能的影响, 负值表示遗忘
- **FT** (Forward Transfer) $= \frac{1}{T-1}\sum_{i=2}^T (R_{i-1,i} - b_i)$: 学了之前 task 对新 task 的初始化帮助
- **F** (Forgetting) $= \frac{1}{T-1}\sum_{i=1}^{T-1} F_i$, 其中 $F_i = \max_{k \ge i} R_{k,i} - R_{T,i}$: peak performance 到 final 的 degradation

观察:
- SL 完全 fail (FAR=0, F=0.62), 经典 catastrophic forgetting
- MTL 是 oracle (能同时看所有 task), 0.96 FAR, 但 unrealistic
- CRL-VLA (Q) 0.98 FAR, F=0.03, 实际上**超过 MTL** → 这暗示 GCV 约束反而帮助了 single-task 学习的 generalization (可能是通过 value consistency 防止 overfit)
- CRL-VLA (V) 在 single-task 上表现差 (0.67), 与 LWF 类似

### 8.3 Multi-task 结果 (Table 2)

| Method | FAR ↑ | BWT ↑ | FT ↑ |
|---|---|---|---|
| SL | 0.62 | 0.07 | 0.25 |
| MTL | 0.49 | 0.00 | 0.25 |
| ER | 0.62 | 0.05 | 0.00 |
| LWF | 0.63 | 0.10 | 0.00 |
| **CRL-VLA (V)** | **0.74** | **0.17** | 0.00 |
| CRL-VLA (Q) | 0.66 | -0.03 | 0.00 |

完全反转:
- **CRL-VLA (V) 最佳**, FAR=0.74, BWT=0.17 (正 BWT!)
- CRL-VLA (Q) 反而 BWT=-0.03, 不如 V 版本

**Paper 给的解释** (5.2 节最后): 在 multi-task 中, state-goal distribution 多样化, 同一 state 在不同 task 下可能对应多个 effective action。
- **Q-based** 在 action-conditioned level 约束 → 隐式强制旧 action semantics, 可能与 new task 的多 action 实现 conflict
- **V-based** 在 state-goal level marginalize action → 对 action 多样性 robust, 仍能 preserve long-horizon value semantics

这个 intuition 我觉得很有趣, 本质上是 Q-based over-constrains: 它不仅约束 value 不变, 还约束特定 (s,a) pair 的 value 不变, 这在多 task 共享 state 的情况下太严格。V-based 只约束 state-level value, action 可以重新选择。

### 8.4 Ablation Studies

**$\mathcal{L}_{\text{GCV}}^V$ weight (Fig. 3 middle, Table 4)**:
- 0.001 最佳: FAR=0.98, BWT=0.00, F=0.00
- 0.00001 太小: 不能 suppress $M_{\text{old}}$, FAR=0.91
- 0.1 太大: over-constrain, FAR=0.93

**$\mathcal{L}_{\text{GCV}}^Q$ weight (Fig. 3 left)**:
- 0.01 最佳: FAR=0.98, BWT=0.02

**Policy KL weight α (Fig. 3 right)**:
- 0.00001 最佳: FAR=0.98
- 太大: narrow trust region, 抑制 $M_{\text{new}}$ 利用
- 太小: policy drift, 实际 $M_{\text{old}}$ 反而变大

**$\mathcal{L}_{\text{MC}}^V$ weight (Fig. 4 left, Table 5)**:
- 1.0 最佳: FAR=0.95
- 太小: critic error 增大, $M_g$ 实际变大
- 太大: optimization 过 conservative, FAR 下降

**GCV ablation (Table 6, Fig. 4 right)**:
- With GCV: FAR=0.31, F=0.02
- Without GCV: FAR=0.29, F=0.06, FT=-0.72

GCV 的提升不算 dramatic (0.29 → 0.31 FAR), 但 FT 从 -0.72 改善到 -0.70, 而且 forgetting 显著降低。这说明 GCV 主要起 stability 作用, 而非直接提升 performance。

### 8.5 Hyperparameter Analysis 的更深解读

Table 4 中有几个值得注意的现象:
- $\mathcal{L}_{\text{GCV}}^V = 0.001$ 给出 FAR=0.98 (最佳), 但 $\mathcal{L}_{\text{GCV}}^V = 0.1$ 仍然 FAR=0.93
- KL weight 的 sweet spot 很窄 (0.00001), 0.001 已经 drop 到 0.91
- 这暗示 policy KL 比 value consistency 更 sensitive, 因为它直接约束 action distribution

**为什么 $\mathcal{L}_{\text{MC}}$ 的最佳系数是 1?** (Table 5)
- 当 $\eta = 0.001$ 时 FAR=0.85, BWT=-0.08
- 当 $\eta = 10$ 时 FAR=0.93, BWT=0.00
- 1.0 是个 sweet spot
- 原因: MC critic 的准确性直接决定 advantage estimate 的 quality, 太小让 advantage 噪声大, 太大让 critic 过拟合 MC return 的 specific trajectory

---

## 9. 与相关工作的定位

### 9.1 vs EWC (Kirkpatrick et al., 2017)
EWC 用 Fisher information 约束参数空间。问题: Fisher 在 large multi-modal transformer 上计算 expensive, 而且 Fisher 是 local approximation, 在 task shift 下不准确。CRL-VLA 用 functional constraint (value output), 不需要二阶信息。

参考: https://arxiv.org/abs/1612.00796

### 9.2 vs LwF (Li & Hoiem, 2017)
LwF 用 knowledge distillation: 让 new model 在 new data 上模仿 old model 的 output。问题: old model 在 new data 上的 output 本身就是 misleading (out of distribution)。CRL-VLA 反过来: 在 old data 上让 new model 重现 old value, 这是 in-distribution 的。

参考: https://arxiv.org/abs/1606.09282

### 9.3 vs Experience Replay (ER)
ER 把 old data 混入新 task 训练。问题: old data 的 gradient 与 new task gradient 可能 conflict, 产生 off-policy error (Korbak et al., 2022 - RL with KL penalties is better viewed as Bayesian inference)。CRL-VLA 不是直接用 old data 更新 policy, 而是用它约束 value consistency, 是更 soft 的约束。

参考: https://proceedings.mlr.press/v151/korbak22a.html

### 9.4 vs Backbone Freezing (VLM2VLA, Hancock et al., 2025)
VLM2VLA 用 LoRA + frozen backbone。问题: 假设 pretrained representation 在 task shift 后仍 aligned with new language goal, 但实际 task shift 会让 representation 失效, value drift。CRL-VLA 保留 backbone 可训练性, 用 GCV 约束其 drift。

参考: https://arxiv.org/abs/2509.22195

### 9.5 vs Shenfeld et al. 2025 "RL's Razor"
Shenfeld 主张 online RL 比 supervised learning 更不容易遗忘, 因为 RL 的 update 是 on-policy 的, 不会强行 fit off-policy data。CRL-VLA extends 这个观点: 在 continual setting 下, 用 KL regularization + GCV consistency 替代 experience replay 的 off-policy gradient。

参考: https://arxiv.org/abs/2509.04259

### 9.6 vs Progress & Compress (Schwarz et al., 2018)
P&C 用 policy distillation: 学完一个 task 后, 把 policy 蒸馏到一个 "knowledge base" network。CRL-VLA 用 frozen critic 做 functional anchor, 类似思路但更轻量。

参考: https://proceedings.mlr.press/v80/schwarz18a.html

### 9.7 vs Recent VLA continual work (Stellar-VLA, DMPEL, ChatVLA)
- Stellar-VLA (Wu et al., 2025): skill-centric knowledge space, 通过 evolving task representation
- DMPEL (Lei et al., 2025): mixture of progressive PEFT experts
- ChatVLA (Zhou et al., 2025): decouple multi-modal understanding from action execution

这些 methods 都是 parameter-efficient 或 architecture-based, 而 CRL-VLA 是 **regularization-based** with explicit theoretical bound, 是不同的 paradigm。

参考:  
Stellar-VLA: https://arxiv.org/abs/2511.18085  
DMPEL: https://arxiv.org/abs/2506.05985  
ChatVLA: https://aclanthology.org/2025.emnlp-main.305/

---

## 10. Limitations and Open Questions

### 10.1 实验规模有限
- 只有 LIBERO, 没有真实机器人, 没有 long-horizon task (task chain 长度最大 3)
- Single-task benchmark 只有 3 个 task (Task-1), multi-task 最多 3 个 task (Task-2/3/4)
- 只测 LoRA rank 32, 没有 full fine-tuning 对比

### 10.2 $\mathcal{B}_{\text{old}}$ 的细节
- 没说 buffer size
- 没说是否 cumulative across all prior tasks
- 没说是否做 prioritized sampling
- 在真实 long-horizon 场景中, buffer 管理是 scalability 关键

### 10.3 Theoretical bound 的 tightness
- Stability bound $\frac{2\gamma}{(1-\gamma)^2} M_{\text{old}} D_{\text{old}}$ 用了 Hölder + Pinsker, 这些不等式都很 loose
- 实际遗忘通常远小于 bound, 所以 Lagrangian relaxation 中 $\beta_V$ 才能取小值 (0.01-0.001)
- 没有给出 lower bound 或 tightness analysis

### 10.4 GCV 的局限性
- Frozen GCV critic 假设旧 task value function 是 optimal anchor, 但实际旧 task value 也只是 approximate
- 当任务序列很长时, GCV critic 数量增加 (每个 task transition 一个)? 还是只保留一个 cumulative? Paper 没说清楚
- 如果只保留一个, 是不是会有 representational drift 累积?

### 10.5 与 model-based RL 的关系
GCV critic 本质上是 old task 的 value model, 可以视为一个简化的 world model。能否用 generative world model (DreamerV3 风格) 替代 value consistency 来做 continual learning? 这是一个 unexplored direction。

参考 DreamerV3: https://arxiv.org/abs/2301.04104

### 10.6 与 imitation learning 的关系
Paper 只考虑 RL post-training。在 SFT 阶段的 continual learning (e.g., 持续 fine-tune on new demonstrations) 没考虑。能否把 GCV 思路扩展到 supervised setting? 这会涉及 value function 替代为某种 supervised consistency。

### 10.7 Language conditioning 的更深问题
GCVF 通过 concatenate language embedding 实现 goal-conditioning, 但这假设 VLM 的 language embedding 已经语义 aligned。如果 task shift 涉及 distribution shift in language (e.g., 从 "pick and place" 切到 "pour and stir"), embedding 是否仍然 useful? Paper 没测这种 cross-domain language shift。

---

## 11. 关键直觉总结

让我把核心 intuition 提炼一下:

1. **Forgetfulness = advantage magnitude × policy divergence**: 不是单纯控制 policy 不变, 而是控制 "old policy 觉得 new policy 偏离的程度" 这个乘积。

2. **Decoupling 通过 control mechanism 不同**: $M_{\text{old}}$ 是 model 内生的 (critic accuracy), $M_{\text{new}}$ 是 environment 内生的 (return range), 所以用不同的 loss 控制。

3. **Frozen critic 是 value anchor, 不是 policy anchor**: 传统 distillation 锚定 policy output, CRL-VLA 锚定 value function output, 让 action space 自由。

4. **GCVF 让 value 对 language 敏感**: 这是 task shift 下保持 value semantics 的关键, 没有 language conditioning 的 value head 在多 task 下会 collapse。

5. **Q vs V 的 trade-off**: Q 更 fine-grained (per-action value), 适合 single-task; V 更 coarse (state-level), 适合 multi-task 因为它对 action 多样性 robust。

6. **Asymmetric regulation**: 不是 symmetric KL constraint, 而是 asymmetric (强约束 old, 弱约束 new)。这是论文标题中的 "asymmetric regulation"。

7. **Lagrangian relaxation**: 把约束 $\delta$ 通过 Lagrangian 转化为 weighted sum, 三个 multiplier $\alpha, \beta_V, \eta$ 控制 trade-off, 比 trust region 直接更灵活。

---

## 12. 我觉得 paper 最 interesting 的几个点

1. **Theorem 4.1 的 unified bound**: 把 stability 和 plasticity 放在同一个 $M \cdot D$ 框架下, 然后 corollary 给出 decoupling 的 mechanism。这种 "find a quantity that couples two objectives, then show it has independent control mechanisms" 的思路在 ML theory 中很 powerful。

2. **Frozen critic + trainable critic 的 dual 设计**: 看起来像 distillation, 但其实是把 "preserving old knowledge" 和 "learning new" 这两个 objective 在 architecture level 解耦。每个 critic 只 optimize 一个 objective, 没有 trade-off。

3. **Q vs V 的实验现象**: Single-task Q 好, multi-task V 好, 这个 reversal 给出关于 value function granularity 的重要 insight: 多 task 共享 state 时, per-action level 的约束过强。

4. **GCVF 的简单性**: 就是 concatenate language embedding, 但效果显著 (Table 6). 这暗示 VLA 社区之前可能 under-explored value function 的 language conditioning, 都把重心放在 policy head 上。

---

## 13. 一些可以延展的方向

如果我来 extend 这个工作, 会考虑:

1. **World model 替代 value critic**: GCV 提供了 value-level anchor, 能否 extend 到 dynamics-level anchor? 比如用 latent world model 在 old data 上保持 latent transition consistency。这会提供更强的 representation-level constraint。

2. **Hierarchical goal decomposition**: 当前 goal 是 atomic language instruction。如果 task 是 hierarchical (e.g., "make breakfast" = "fetch egg" + "crack egg" + "cook egg"), GCV 需要在 hierarchy 每一层都 condition。可能需要 hierarchical value function。

3. **Active replay buffer selection**: 当前 $\mathcal{B}_{\text{old}}$ 是 passive store。能否基于 GCV loss 的 gradient magnitude 主动选择哪些 old data 对当前更新最 critical? 这是 importance sampling for continual learning。

4. **Theoretical tightness via concentration bounds**: 当前的 bound 是 worst-case (sup over states), 能否用 PAC-Bayes 或 information-theoretic bound 给出更紧的 average-case bound?

5. **Multi-agent extension**: 在 multi-robot 场景中, 每个 robot 学不同 task, GCV 能否作为 communication protocol, 让 robots share value anchors?

参考一些可以深挖的相关工作:
- PI-0.5 (https://arxiv.org/abs/2504.16054) - open-world VLA generalization
- SimpleVLA-RL (https://arxiv.org/abs/2509.09674) - scaling VLA RL
- ARFM (https://arxiv.org/abs/2509.04063) - adaptive offline RL for VLA
- RLinf-VLA (https://arxiv.org/abs/2510.06710) - unified VLA+RL framework
- IRL-VLA (https://arxiv.org/abs/2508.06571) - closed-loop RL for autonomous driving VLA
- Robust finetuning via parameter merging (https://arxiv.org/abs/2512.08333)

---

## 14. Final Thoughts

这篇 paper 在我看来是 RL theory + VLA practical engineering 的一个不错的 marriage。理论部分虽然 bound 不紧, 但给出了 actionable insight (decouple $M$ via dual critic)。工程部分相对 simple (PPO + 两个 critic + KL + value consistency), 没有太多 bells and whistles, 容易复现。

最大的 limitation 是实验规模: 4 个 LIBERO task subset, 最多 3 个 task 的 sequence, 这远不足以 claim "lifelong" learning。真正 lifelong 应该是 100+ task sequence。但作为 initial exploration, 这个 framework 提供 promising foundation。

我觉得最值得深挖的方向是 **GCV 作为 representation anchor 的角色**: 它表面上是 value function 的 constraint, 但实际效果是约束 backbone representation 不在 old state 分布上 drift。这与 PlaNet / Dreamer 的 latent dynamics model 有 conceptual 相似性。如果能把 GCV 升级为 latent dynamics + reward + value 的 joint anchor, 可能能获得更强的 stability guarantee。

另一个值得思考的方向: **能否完全不用 replay buffer?** Replay buffer 在实际 robot learning 中 storage 成本巨大 (high-dim observations)。GCV consistency 已经在 old data 上做了约束, 但仍需 $\mathcal{B}_{\text{old}}$。能否用 generative model (e.g., VAE 或 diffusion) 在 latent space 生成 "pseudo old states", 让 GCV 在 pseudo states 上做 consistency? 这是 model-based continual learning 的方向。

总之, paper 给出了 continual VLA 的一个 principled framework, 后续工作可以在 scalability、theory tightness、architecture (e.g., world model) 上 extend。

---

**主要 reference 链接汇总**:
- Paper 本身 (假设 ArXiv): https://arxiv.org/abs/Continual-VLA
- LIBERO benchmark: https://libero-project.github.io/
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- RIPT-VLA (codebase base): https://arxiv.org/abs/2505.17016
- PDL (Kakade & Langford 2002): https://proceedings.mlr.press/v2/kakade02a.html
- CPO (Achiam et al. 2017): http://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf
- PPO (Schulman et al. 2017): https://arxiv.org/abs/1707.06347
- EWC: https://arxiv.org/abs/1612.00796
- LwF: https://arxiv.org/abs/1606.09282
- Korbak 2022 (KL as Bayesian): https://proceedings.mlr.press/v151/korbak22a.html
- Shenfeld "RL's Razor": https://arxiv.org/abs/2509.04259
- PI-0.5: https://arxiv.org/abs/2504.16054
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- VLM2VLA: https://arxiv.org/abs/2509.22195
- Stellar-VLA: https://arxiv.org/abs/2511.18085
- DMPEL: https://arxiv.org/abs/2506.05985
- ChatVLA: https://aclanthology.org/2025.emnlp-main.305/
- TinyVLA: https://ieeexplore.ieee.org/document/10803168
- DreamerV3 (for extension ideas): https://arxiv.org/abs/2301.04104

希望这个 walk-through 帮你 build 起对 CRL-VLA 的 intuition!核心 take-away: 把 continual learning 重新 formulate 为 advantage magnitude 控制, 然后用 critic approximation error vs environment return range 这两个 independent source 解耦 stability 和 plasticity, 这是 paper 的 essential contribution。
