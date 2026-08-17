---
source_pdf: πRL.pdf
paper_sha256: 484e9f8be118be4f7adfd18470abc330e29b1b4cf457d316dc85a63e65299d36
processed_at: '2026-08-13T06:56:17-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# πRL 人话版

## 一句话先 build intuition

π0 这类 flow-based VLA 想用 RL fine-tune,卡在两件事上:**算不出 action 的对数概率**,而且**采样过程是确定性的没法 explore**。πRL 给出两招:第一招在每一步 denoising 上偷偷塞个 Gaussian,让 K 步 denoising 变成一个能算概率的 Markov chain;第二招干脆把 ODE 换成 SDE,保持 marginal 分布不变但自带随机性。两招都把"不可算的 flow likelihood"转成"K 个能解析写出公式的 Gaussian 转移概率",PPO 直接就能跑了。

---

## 为什么 flow-based VLA 上 RL 这么难

先 recall 一下 PPO 的核心公式:

$$\nabla_\theta \mathcal{J} = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)\right]$$

这里 $\log \pi_\theta(a|s)$ 是"给定 observation $s$,网络输出这个 action $a$ 的对数概率"。对 autoregressive VLA(OpenVLA 那类),action 是离散 token,softmax 一下直接得到。对 Gaussian policy,也是 closed form。**对 flow matching,这个东西是个 monster**。

Flow matching 的 action 生成过程:从 $\epsilon \sim \mathcal{N}(0, I)$ 采噪声,沿学到的 vector field $v_\theta$ 走 $K$ 步 Euler:

$$A^{\tau+\delta} = A^\tau + v_\theta(A^\tau, o) \cdot \delta, \quad \delta = 1/K$$

最终得到 action chunk $A^1 = [a_0, ..., a_{H-1}]$。问:输出这个特定 $A^1$ 的概率是多少?

理论上要用 instantaneous change of variables:

$$\log q_1(A^1) = \log q_0(A^0) - \int_0^1 \nabla \cdot v_\theta(A^\tau, o)\, d\tau$$

中间那个 $\nabla \cdot v_\theta$ 是 vector field 对输入的 divergence(Jacobian 的 trace),要算向量函数对向量的散度。理论上 Hutchinson trace estimator(https://doi.org/10.1080/03610918908812851)能估,加 Neural ODE 的 adjoint method(https://arxiv.org/abs/1806.07366)能 backprop。**但实际当 $K$ 很小(π0 默认 $K=4$)时,这个估计的方差大到你根本收敛不了**。

更糟的是,flow 推理是 deterministic 的 —— 同样的 $o$ 和同样的 $\epsilon$,永远出同一个 $A^1$。即使你能算 likelihood,policy 退化成 delta distribution,没有 entropy,PPO ratio 永远是 1,没法 explore。

πRL 同时解决这两个 obstacle。

---

## 第一招:Flow-Noise —— 给每一步 Euler 加 learnable Gaussian

### 直觉

既然端到端的 $\log \pi(A|o)$ 算不了,那就**把整个 denoising 轨迹显式展开成 $K$ 个 Gaussian 转移**,每一步的概率 closed form,整条链的概率就是连乘。

### Mechanism

把 Euler update:

$$A^{\tau+\delta} = A^\tau + v^\tau \cdot \delta$$

改写成 Gaussian transition:

$$p(A^{\tau+\delta} | A^\tau) \sim \mathcal{N}(\mu_\tau, \Sigma_\tau)$$

其中:

$$\mu_\tau = A^\tau + v^\tau \cdot \delta, \quad \Sigma_\tau = \mathrm{diag}(\sigma_{\theta'}^2)$$

变量解释:
- $v^\tau = v_\theta(A^\tau, o)$ 是 flow matching 学到的 velocity(原 π0 的 action expert 输出)
- $\sigma_{\theta'}$ 是**一个新加的小网络**(noise network),输入 $(A^\tau, o)$,输出每个 action 维度的标准差
- $\delta = 1/K$ 是离散化步长
- $\mathrm{diag}$ 表示各向同性对角协方差

也就是说,每步 denoising 既前进一点(mean 沿 $v_\theta$ 走),又抖动一点(variance 由新网络控制)。这个 noise network 学的是"在哪些维度、什么状态下应该多探索"。**训完就丢掉,推理时回到 deterministic policy**。

灵感来自 ReinFlow(https://arxiv.org/abs/2505.22094),那是 diffusion policy 的 RL fine-tune 工作。πRL 把它移植到 flow matching VLA 上,加了几个 VLA-specific 适配(chunk-level macro-step、shared actor-critic 等)。

### Log-likelihood 怎么算

把推理离散成 $K$ 个时间点 $\tau_0=0, \tau_1=\delta, ..., \tau_K=1$。整条 denoising 序列 $\mathcal{A} = (A^0, A^\delta, ..., A^1)$。联合 log probability:

$$\log \pi(A | o) = \log \pi(A^0 | o) + \sum_{k=0}^{K-1} \log \pi(A^{\tau_{k+1}} | A^{\tau_k}, o)$$

- $\log \pi(A^0 | o)$:$A^0 \sim \mathcal{N}(0, I)$,是 prior,与 $\theta$ 无关,梯度为 0,但保留以保证 probability measure 完整
- 每个 $\log \pi(A^{\tau_{k+1}} | A^{\tau_k}, o)$:就是上面那个 Gaussian 的 log density,解析可写

$$\log \mathcal{N}(A^{\tau_{k+1}}; \mu_{\tau_k}, \Sigma_{\tau_k}) = -\frac{1}{2}\sum_i \left[\log(2\pi\sigma_{\theta',i}^2) + \frac{(A^{\tau_{k+1}}_i - \mu_{\tau_k,i})^2}{\sigma_{\theta',i}^2}\right]$$

下标 $i$ 跑过 action 维度。**这就是 tractable 的精确 log-likelihood**,没有 Hutchinson、没有 adjoint、没有 variance 问题。

### 为什么这是一个 one-layer MDP

denoising 过程被视作 $K$ 步 discrete MDP:state 是 $(o, A^\tau)$,action 是 $A^{\tau+\delta}$,transition 是 Gaussian。直接套 PPO,policy ratio 就是两个 Gaussian density 的比值,clip 一下就完事。

优雅之处:**flow matching 的 $K$ 步 inference 本来就要跑,每步都有 $v_\theta$ 输出,加一个 $\sigma_{\theta'}$ head 几乎零额外开销**。

---

## 第二招:Flow-SDE —— 把 ODE 直接转成 SDE

### 直觉

Flow-Noise 加了额外 noise network,有点 hacky。Flow-SDE 想利用 score-based generative modeling 里的经典结论:**对于任意 marginal distribution $q_\tau$,存在一个 ODE 和一个 SDE,它们的 marginal 在所有 $\tau$ 上完全一致**(Song et al., https://arxiv.org/abs/2011.13456)。所以把 ODE 换成 SDE 不改变 action 分布形状,但自带 stochasticity 用于 exploration。

灵感来自 Flow-GRPO(https://arxiv.org/abs/2505.05470)。

### ODE → SDE 的转换

原始 Euler ODE:

$$dA^\tau = v^\tau\, d\tau$$

对应的等价 SDE:

$$dA^\tau = \underbrace{\left[v^\tau - \frac{1}{2}g^2(\tau)\nabla \log q_\tau(A^\tau)\right]d\tau}_{\text{drift}} + \underbrace{g(\tau)\, dw}_{\text{diffusion}}$$

变量:
- $g(\tau)$:scalar noise schedule,控制扰动幅度
- $\nabla \log q_\tau(A^\tau)$:marginal distribution $q_\tau$ 的 score function
- $dw$:Wiener process,增量 $dw \sim \mathcal{N}(0, d\tau)$

drift 里多出来的 $-\frac{1}{2}g^2 \nabla \log q$ 项是 **Fokker-Planck 修正项**,抵消 diffusion 引入的"额外扩散",保证 marginal 不变。这是 score SDE 论文的核心 trick。

### Score 用 velocity 表达

要让 SDE 实用,得把 $\nabla \log q_\tau$ 替换成能算的东西。对 rectified flow(π0 用的就是,https://arxiv.org/abs/2209.03003),有经典关系:

$$\nabla \log q_\tau(A^\tau) = -\frac{A^\tau}{\tau} - \frac{1-\tau}{\tau} v^\tau$$

这里 $\tau$ 是 flow matching 时间,$\tau \to 0$ 时退化为噪声分布,$\tau = 1$ 时是目标。这个关系把 score 用 $v_\theta$ 显式表达 —— 关键 trick,因为 $v_\theta$ 我们本来就有。

### 选 noise schedule

作者选 $g(\tau) = \sigma_\tau = a \cdot \sqrt{\tau/(1-\tau)}$,$a$ 是 hyperparameter(实验里 $a=0.5$ 最好)。注意这个 schedule 在 $\tau \to 1$ 时发散 —— 越接近最终 action 越不能加太多 noise,但 $\sqrt{\tau/(1-\tau)}$ 在 $\tau=1$ blow up。实际做法是 SDE 跑到接近 1 但不到 1,或者用 hybrid 策略(后面讲)。

代入化简得最终 SDE:

$$dA^\tau = \left[v^\tau + \frac{\sigma_\tau^2}{2\tau}(A^\tau + (1-\tau)v^\tau)\right]d\tau + \sigma_\tau\, dw_\tau$$

离散化:

$$\mu_\tau = A^\tau + \left[v^\tau + \frac{\sigma_\tau^2}{2\tau}(A^\tau + (1-\tau)v^\tau)\right] \cdot \delta$$

$$\Sigma_\tau = \sigma_\tau^2 \delta \cdot I$$

注意 $\Sigma_\tau$ 是 isotropic,方差由 noise schedule 决定,**不需要额外网络**。这是 Flow-SDE 相对 Flow-Noise 的简化。

### Two-layer MDP:把 denoising 和环境交互缝起来

上面只是单步 denoising stochastic 化了,但 RL 还要和环境交互。作者引入 DPPO(https://arxiv.org/abs/2409.00588)的 two-layer MDP 思路:

- **Outer loop(环境层)**:标准 MDP,时间 $t$,state $o_t$,action $A_t^1$(去噪完的 action chunk),transition $P_{\text{ENV}}$
- **Inner loop(denoising 层)**:在每个 outer step 内,$\tau$ 从 0 走到 1,state 是 $(o_t, A_t^\tau)$,action 是 $A_t^{\tau+\delta}$

形式化:

| 组件 | $\tau < 1$(inner) | $\tau = 1$(outer) |
|------|------|------|
| State $\bar{s}_t^\tau$ | $(o_t, A_t^\tau)$ | $(o_t, A_t^1)$ |
| Action $\bar{a}_t^\tau$ | $A_t^{\tau+\delta}$ | $A_t^1$ |
| Next state | $(o_t, A_t^{\tau+\delta})$ | $(o_{t+1}, A_{t+1}^0 \sim \mathcal{N}(0,I))$ |
| Reward | 0 | $R_{\text{ENV}}(o_t, A_t^1)$ |

reward 只在 denoising 完成那一步给 —— **稀疏但物理正确**。两层 MDP 让 denoising step 也参与 advantage 传播(虽然 reward=0,但 transition probability 进 ratio)。

### Hybrid ODE-SDE 加速

Two-layer MDP 把 horizon 拉长 $K$ 倍,训练贵且难收敛。借鉴 MixGRPO(https://arxiv.org/abs/2507.21802)和 TempFlow-GRPO(https://arxiv.org/abs/2508.04324)的 trick:

**每个 outer step $t$,随机选一个 denoising time $\tau_t$ 做 SDE 探索,其他 $K-1$ 步用 deterministic ODE**。

从 state $(o_t, A_t^{\tau_t})$ 出发,policy 在这一步输出 stochastic action,然后 environment wrapper 把剩下的 ODE 跑完,再走环境 transition,输出 $(o_{t+1}, A_{t+1}^{\tau_{t+1}})$,$\tau_{t+1}$ 重新随机采。

公式上等价于原 two-layer MDP(marginal 等价),但实际 horizon 从 $T \times K$ 缩到 $T$,**2× wall-clock speedup**。

---

## Policy optimization 细节

### Chunk-level macro-step

π 系列输出 action chunk $A_t = [a_{t,0}, ..., a_{t,H-1}]$,长度 $H$(π0 是 50,π0.5 是 10)。作者把整个 chunk 视作**一个 macro-step**,reward 是 $H$ 步环境 reward 之和:

$$R_t = \sum_{j=0}^{H-1} r_{t,j}$$

PPO 在 outer MDP 上 horizon 是 $T = \text{episode\_length} / H'$,$H'$ 是 replan horizon(实验里大多 $H'=5$,即每 5 步重新观测一次)。这个 chunk-level 思路来自 RLinf-VLA(https://arxiv.org/abs/2510.06710)。

### GAE + PPO clip

$$\hat{A}_t = \sum_{k=0}^{T-t} (\gamma\lambda)^k \mathcal{T}_{t+k}$$

其中 TD-error $\mathcal{T}_t = R_t + \gamma V(s_{t+1}) - V(s_t)$。

- $\gamma$:discount factor,0.99
- $\lambda$:GAE bias-variance tradeoff,$\lambda=0$ 是 1-step TD(高 bias 低 variance),$\lambda=1$ 是 Monte Carlo(低 bias 高 variance),实验 0.95

PPO clip 目标:

$$\mathcal{J}(\pi_\theta) = \mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

ratio $\rho_t = \pi_{\theta_\text{new}}(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$。对 Flow-Noise 是整条 denoising 序列的联合概率;对 Flow-SDE 是 two-layer MDP 里的 $\bar{a}_t^\tau | \bar{s}_t^\tau$。两种都因 Gaussian transition 解析可算。clip $\epsilon = 0.2$ 标准。

### Critic 设计 —— 这块有意思

共享 actor-critic 架构省显存,但 π0 和 π0.5 的 state 输入位置不同:

**π0.5**:state(robot proprioception)**不**输入模型,merge 到 VLM 的 prompt embedding 里。critic 干脆接在 VLM 输出后:

$$V_{\text{vlm}}(o_t)$$

直接从 (image, language) 出 value,不接 state。简单粗暴但 ablation 显示反而效果更好。

**π0**:state 进 action expert,action expert 同时接收 state 和 noisy action $A_t^\tau$。critic 接在 action expert 后,但输入得带 noisy action —— 推理时 $A_t^\tau$ 在变。作者的做法是对整条 denoising trajectory 的 value 平均:

$$V_{\text{expert}}(o_t) \approx \mathbb{E}_{\tau \sim U[0,1]}[V_{\text{expert}}(o_t, A_t^\tau)]$$

直觉:value 是 state value,不该依赖 denoising step 的具体噪声实现;但 action expert 内部把 state 和 noisy action 耦合了,所以对 $\tau$ 平均掉噪声影响。

**Ablation 5.3.1 显示 $V_{\text{vlm}}$ 反而比 $V_{\text{expert}}$ 略好**,作者分析原因是 $V_{\text{expert}}$ 输入里 state 和 noisy action 耦合,优化困难。但作者为了"value function 应该接 state 信息"的概念正确性,π0 还保留 $V_{\text{expert}}$ 架构。

另外 critic 结构:4-layer MLP 比 1-layer MLP 显著好(explained variance 更高,eval 性能更好)。

---

## 实验结果

### In-distribution 主结果(Table 1)

| 模型 | 方法 | LIBERO | ManiSkill | MetaWorld | CALVIN | Avg. | $\Delta$ |
|------|------|--------|-----------|-----------|--------|------|---|
| π0 | SFT (few-shot) | 57.6 | 38.4 | 50.8 | 57.5 | 51.1 | - |
| π0 | Flow-SDE | 96.1 | 78.8 | 78.1 | 61.7 | 78.7 | +27.6 |
| π0 | Flow-Noise | 97.6 | 77.8 | 85.8 | 59.9 | 80.3 | +29.2 |
| π0.5 | SFT (few-shot) | 77.1 | 40.1 | 43.8 | 61.3 | 55.6 | - |
| π0.5 | Flow-SDE | 97.9 | 90.9 | 70.7 | 87.0 | 86.6 | +31.0 |
| π0.5 | Flow-Noise | 98.3 | 89.7 | 66.1 | 84.5 | 84.7 | +29.1 |

三个 striking 点:

1. **Few-shot SFT + RL 反超 full-dataset SFT**:LIBERO 上 π0.5 用 1 条轨迹 SFT + RL = 98.3%,full-dataset π0.5 SFT = 96.9%。RL 在 VLA 上"few-shot 起步 + RL 收尾"范式可行 —— **意味着 RL 有可能替代大规模数据采集**。

2. **CALVIN 长 horizon 上 RL 增益随序列长度放大**(Table 5):π0.5 Len-1 提升 +7.0%,Len-5 提升 +25.7%。SFT 的 compounding error 在长序列上更严重,RL 直接优化整个轨迹 return,从根上解决 error accumulation。

3. **ManiSkill 4352 种 pick-and-place 组合**:π0.5 从 40.1% → 90.9% (+50.8%),massive combinatorial 任务上 RL 的 generalization 能力很 striking。

### OOD 泛化(Table 4 + Figure 5)

| 场景 | 结果 |
|------|------|
| ManiSkill Vision OOD | ID 增益 +126.7% 时 OOD 增益 +73.9%,能 transfer |
| ManiSkill Semantics OOD | 提升明显但绝对值仍低(MultiCarrot 16.7% → 38.2%) |
| ManiSkill Execution OOD | π0 不如 π0.5(π0 接 state 易过拟合 control) |
| CALVIN ABC→D | 79.1% vs SFT 61.3%,视觉 shift 可 transfer |
| MetaWorld ML45 | 振荡,无 consistent 提升 |

重要结论:**RL 主要 enhance action-level refinement,对低层视觉/执行扰动 robust,对跨任务高层泛化帮助有限**。但 RL 至少避免了 SFT 的 catastrophic forgetting,OOD 上保留了 SFT 学到的东西。这跟 SimpleVLA-RL(https://arxiv.org/abs/2509.09674)的发现一致。

### Hyperparameter Ablation(Table 2,很重要)

LIBERO-Spatial 上跑:

**Noise level $a$**:
- $a=0.2$:train 59.5%,eval 73.1%,clip fraction 高(不稳定)
- $a=0.5$:train 93.5%,eval 94.5%(甜蜜点)
- $a=0.8$:train 95.3%,eval 98.1%(最好)

直觉:noise 小 → ratio 大 → gradient 大 → 不稳定;noise 大 → 探索足但 flow trajectory 被扭曲。低 noise 反而需要更小 lr。

**Denoise step $K$**:
- $K=1$:train 9.4%,灾难性失败(ODE→SDE 离散化误差 dominant)
- $K=2$:train 28.3%,eval 63.8%(仍不够)
- $K=4$:train 56.1%,eval 94.5%
- $K=8$:train 62.6%,eval 86.7%($K$ 大反而 eval 略降,因为 horizon 长,advantage 估计 noisy)

**Action chunk**:
- 5:train 93.5%,eval 94.5%
- 10:train 93.3%,eval 95.5%
- 20:train 87.5%,eval 89.2%(chunk 大 advantage credit assignment 难,explained variance 降低)

直觉:chunk 大 → SFT baseline 高(执行平滑),但 RL 上限被压低(reward credit 难分配到具体子动作)。Long-horizon 任务可以用大 chunk。

### VLM fine-tune 与否(Appendix F.2)

VLM frozen vs LoRA($r=32$, $\alpha=32$):LoRA 没有明显收益,且需要更保守 lr 才稳定。作者推测 LIBERO scene 变化少,预训练 VLM 已经够用。**这暗示 RL 阶段 action expert 才是 bottleneck,VLM 改动风险大于收益**。

### PPO vs GRPO(Table 9)

PPO 全面优于 GRPO(π0.5 LIBERO:PPO 97.9% vs GRPO 91.5%)。直觉上 GRPO 用 group baseline 代替 critic,但 chunk-level + 长 horizon 上 critic 提供的 low-variance advantage 信号更重要。

---

## 工程细节

### 训练设置

- 8×H100 80GB
- 共置策略:environment + rollout model + actor model 同一张 GPU 上 serial 执行(来自 RLinf codebase, https://arxiv.org/abs/2509.15965)
- VLM frozen,只 fine-tune 300M action expert(GPU memory + RL4VLA 发现 RL 主要 boost action generalization)

### GR00T N1.5 上的延伸(Appendix H)

证明方法 architecture-agnostic:SFT 52.5% → PPO + Flow-SDE 89.9% (+37.4%)。一个关键工程 trick:**把 action expert 里的 dropout 全部换成 identity layer**。

理由 elegant:PPO ratio $\rho_t = \pi_\text{new}(a|s) / \pi_\text{old}(a|s)$ 假设两次 forward 的 stochasticity 只来自 action sampling。如果网络里有 dropout,每次 forward 都换 mask,ratio 实际是:

$$\rho_t = \frac{\pi_{\alpha_\text{new}}(a|s)}{\pi_{\theta_\text{old}}(a|s)}$$

$\alpha_\text{new}$ 是 dropout mask 扰动后的 policy。这个 structural stochasticity 叠加 per-step update,会让训练根本不收敛。**这是 RL 训练里被低估的坑**。

### Real2Sim2Real 演示

用 ManiSkill(rigid body) + 3D Gaussian Splatting(photorealistic rendering, https://arxiv.org/abs/2308.14737)建 simulator,缩小 sim-to-real visual gap。20 条 expert trajectory 做 few-shot SFT,100 RL steps 后 zero-shot deploy 到 Franka Panda 真机,40% 成功率(SFT baseline 完全失败)。

参考 GSWorld(https://arxiv.org/abs/2510.20813)和 TwinAligner(https://arxiv.org/abs/2512.19390)的 Real2Sim2Real 思路。

### Temporal efficiency(Appendix G)

RL 训练后 episode length 收敛到 expert motion planner 的水平。两个原因:
1. RL 增强 error correction,能从执行失败中恢复
2. Partial reset + discounted reward 让 agent "完成得快才能触发更多 reset 攒更多 reward"

### Critic warmup 现象

ManiSkill 上 eval 曲线早期 dip 再上升 —— critic 早期 value estimate 不准,advantage 信号差,policy 先变差再修复。作者建议 early stage 不要 panic。

---

## 局限与未来方向

1. **ODE→SDE 转换的精度损失**:Flow-CPS(https://arxiv.org/abs/2509.05952)提出 coefficients-preserving sampling 来减误差,但作者实测 RL 增益有限 —— 问题更深,可能需要新的 noise injection 形式。

2. **Hybrid ODE-SDE 太简单**:只随机选 1 步做 SDE,其他全 ODE。MixGRPO 和 TempFlow-GRPO 有更复杂 schedule,未来可借鉴图像生成加速的进展(如 Flow policy optimization https://arxiv.org/abs/2507.21053,BranchGRPO https://arxiv.org/abs/2509.06040)。

3. **跨任务泛化仍是难题**:MetaWorld ML45 上 RL 帮不上,需要更 diverse 的 task distribution + 更丰富 language instruction 训练。

4. **依赖 sim-to-real**:online RL sample efficiency 太低,真机 RL 还不可行。未来需要 offline-to-online、world model 等 sample efficient 方法。

---

## 几点延伸思考

### Flow-Noise vs Flow-SDE 的本质差异

两个方法性能差不到 2%,但机制差异挺大:
- **Flow-Noise**:加额外 noise network,one-layer MDP,收敛快但 $K$ 大时 wall-clock 不省
- **Flow-SDE**:不加额外网络,two-layer MDP,hybrid 加速 2×,但训练稍慢起步

实际推荐 Flow-SDE(计算便宜且 ablation 多)。Flow-Noise 适合需要精细 control noise magnitude 的场景。

### Chunk size 的 RL 友好性悖论

SFT 上 chunk 大更好(执行平滑,temporal consistency),但 RL 上 chunk 大反而 cap 了 ceiling —— 因为 advantage credit assignment 难。这暗示一个 **decoupled chunk size** 设计:SFT 阶段用大 chunk,RL 阶段动态切小。π_fast(https://arxiv.org/abs/2501.09747)的 action tokenization 思路可能能帮上忙。

### 与 LLM RLHF 的统一视角

πRL 让我想到 LLM RLHF 早期:pre-train → SFT → PPO。VLA 现在重走这条路,加了两个 LLM 没有的 twist:
- Action 是连续 + chunk,不是 token sequence
- Flow matching decoder 比 autoregressive decoder 难处理 log-likelihood

但解决问题的 framework(policy gradient + clip + GAE)是一样的。**LLM RL 的所有积累(GRPO、DPO、RLOO、ReST 等)原则上都能搬到 VLA**,只要解决 likelihood tractability 这一个核心 obstacle。πRL 给的就是这个 obstacle 的两种解法。

未来值得探索:**flow-based VLA + DPO + preference from human feedback** 是否可行。FPO(https://arxiv.org/abs/2507.21053)已经在尝试用 advantage-weighted CFM loss 重塑 policy optimization,可能是另一条路。

### Sim-to-real 的真正瓶颈

πRL 用 3D GS 做 rendering 缩小 visual gap,但 40% 真机成功率仍不高。瓶颈可能在:
- Dynamics gap(rigid body simulator 与真机摩擦/接触差异)
- Action chunk 执行的 open-loop 问题(环境偏离预期时,预先规划的 $H$ 步 action 不再最优)
- Latency(VLM 推理慢,real-time control 难)

π0.5 的 flow matching expert 比较轻量(300M),VLM inference 是大头。未来用 speculative decoding 或 VLM cache 可能缓解。

### 为什么"few-shot SFT + RL"这个范式重要

机器人领域长期被数据采集成本卡脖子。一个 teleop session 一小时可能就产几十条轨迹,要 covering 各种场景成本巨大。πRL 展示了:用极少数据 SFT 起步(1 条!),让 RL 从环境 reward 自己探索到 expert-level 性能。**这跟 LLM 领域"少量 SFT + 大量 RL"的 scaling 思路完全对齐**。

如果这个范式真的 work,机器人领域的瓶颈会从"采集更多 demo"变成"建更好的 simulator + 设计更好的 reward"。这是个 paradigm shift 级别的事情。

---

## 参考 link 汇总

**核心 paper**:
- πRL code: https://github.com/RLinf/RLinf
- RLinf codebase: https://arxiv.org/abs/2509.15965
- RLinf-VLA: https://arxiv.org/abs/2510.06710

**Base models**:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- GR00T N1: https://arxiv.org/abs/2503.14734
- Octo: https://arxiv.org/abs/2405.12213
- TinyVLA / SmolVLA: https://arxiv.org/abs/2506.01844
- π_fast: https://arxiv.org/abs/2501.09747

**Flow matching / Diffusion theory**:
- Flow matching: https://arxiv.org/abs/2210.02747
- Rectified flow: https://arxiv.org/abs/2209.03003
- Score SDE: https://arxiv.org/abs/2011.13456
- Neural ODE: https://arxiv.org/abs/1806.07366
- Hutchinson trace estimator: https://doi.org/10.1080/03610918908812851

**RL methods**:
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- GRPO: https://arxiv.org/abs/2402.03300
- DPO: https://arxiv.org/abs/2305.18290
- DPPO: https://arxiv.org/abs/2409.00588
- ReinFlow: https://arxiv.org/abs/2505.22094
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- MixGRPO: https://arxiv.org/abs/2507.21802
- TempFlow-GRPO: https://arxiv.org/abs/2508.04324
- BranchGRPO: https://arxiv.org/abs/2509.06040
- FPO: https://arxiv.org/abs/2507.21053
- Flow-CPS: https://arxiv.org/abs/2509.05952
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- RL4VLA: https://arxiv.org/abs/2505.19789

**Benchmarks**:
- LIBERO: https://arxiv.org/abs/2306.03310
- ManiSkill3: https://arxiv.org/abs/2410.00425
- MetaWorld+: https://arxiv.org/abs/2505.11289
- CALVIN: https://arxiv.org/abs/2112.03227
- SIMPLER: https://arxiv.org/abs/2405.05941
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

**Rendering / Sim**:
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- GSWorld: https://arxiv.org/abs/2510.20813
- TwinAligner: https://arxiv.org/abs/2512.19390

**Misc**:
- LoRA: https://arxiv.org/abs/2106.09685
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- MPLib: https://github.com/haosulab/MPlib

---

## TL;DR for intuition

πRL 把 PPO 套上 flow-based VLA 的关键是**让 action 的 log-likelihood 变得可算**,有两个等价路径:
1. **Flow-Noise**:denoising 每一步加一个 learnable Gaussian,把 $K$ 步视作 discrete MDP,联合概率就是 $K$ 个 Gaussian 的连乘
2. **Flow-SDE**:把 rectified flow 的 ODE 转成 marginal-preserving SDE,自然引入 stochasticity,再用 two-layer MDP 把 denoising 和环境交互缝起来

两条路殊途同归 —— 都把"不可算的 flow likelihood"转成"$K$ 个 tractable Gaussian transitions"。前者精确但慢,后者便宜但需要 hybrid 加速。最终 few-shot SFT + 100 步 RL 能反超 full-dataset SFT,证明 RL 在 VLA scaling 上的潜力远大于继续采集 demonstration 数据。

人话版总结就是:**给原本确定性的 flow 采样过程加点可学/可调的随机性,让每一步的概率都能用 Gaussian 公式写出来,PPO 就能跑了**。剩下的就是工程细节:chunk 怎么切、critic 放哪、noise 多大、$K$ 选几、怎么加速。最终结果是机器人领域第一次把 RLHF 那套 LLM 上的范式成功搬到 flow-based VLA 上,而且效果显著。

---

# πRL: Flow-based VLA 的 Online RL Fine-tuning 框架深度解析

## 0. 一句话定位

πRL 解决的是「把 PPO 这类 policy gradient 算法套到 π0/π0.5 这类 flow matching VLA 上」的根本障碍 —— **flow matching 的 action log-likelihood 不可计算 + deterministic ODE 无法 explore**。作者用两条路径打通这件事：Flow-Noise 在每一步 Euler update 上加 learnable Gaussian，把 denoising 视作 discrete MDP；Flow-SDE 直接把 probability flow ODE 转 SDE，保留 marginal distribution，再套一个 two-layer MDP 把 denoising 和 environment interaction 缝起来。最终 few-shot SFT + RL 反超 full-dataset SFT。

代码: https://github.com/RLinf/RLinf  
模型: https://huggingface.co/RLinf

---

## 1. 背景与动机

### 1.1 VLA 的两条技术路线

VLA (Vision-Language-Action) 的训练范式是 **pre-train VLM → SFT on demonstration → (optional) RL**。但 action decoder 上有两派：

1. **Autoregressive / Discrete decoder**: OpenVLA, OpenVLA-OFT, GR00T (部分)。action 离散化成 token，next-token softmax 直接给 log π(a|s)。RL 友好，已有 SimpleVLA-RL / RL4VLA / RLinf-VLA 等工作。
   - OpenVLA: https://arxiv.org/abs/2406.09246
   - OpenVLA-OFT: https://arxiv.org/abs/2502.19645

2. **Flow-based decoder**: π0, π0.5, GR00T N1.5。action 通过 **flow matching** 迭代精化得到 —— 从 N(0, I) 采样噪声 ε，沿学到的 vector field v_θ 走 K 步 Euler，得到 action chunk A_t = [a_{t,0}, ..., a_{t,H-1}]。
   - π0: https://arxiv.org/abs/2410.24164
   - π0.5: https://arxiv.org/abs/2504.16054
   - GR00T N1: https://arxiv.org/abs/2503.14734
   - Flow matching: https://arxiv.org/abs/2210.02747

π 系列的优势是高频 + dexterous + chunk 生成，但代价是 RL 接入难。

### 1.2 为什么 flow-based VLA 上 RL 这么难

两个互锁的障碍：

**障碍一：log-likelihood 不可计算**。Policy gradient 的核心公式（REINFORCE）：

$$\nabla_\theta \mathcal{J}(\pi_\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t,a_t)\right]$$

这里需要 log π(a|s)，即"给定 observation，最终输出这个 action chunk 的对数概率"。Flow matching 把 ε → A_t 的过程定义成一个 ODE 初值问题。从概率密度角度看，需要解 **instantaneous change of variables** 公式：

$$\log q_1(A_t^1) = \log q_0(A_t^0) - \int_0^1 \nabla \cdot v_\theta(A_t^\tau, o_t)\, d\tau$$

中间那个 divergence 项要算 vector field 对输入的 Jacobian trace。理论上用 **Hutchinson trace estimator** (https://doi.org/10.1080/03610918908812851) 加上 Neural ODE 的 adjoint method 可以估，但当 K 很小（π0 默认 K=4）时方差巨大，根本不收敛。这是 Hutchinson 1989 那篇老 paper 留下的 classical 工具，在 diffusion/flow 里也是 widely known 的痛点。

**障碍二：deterministic ODE 没法探索**。RL 需要 stochastic policy 来 explore，但 flow 的推理是 deterministic 的 —— 同样的 o_t + 同样的 ε，永远出同样的 action。即使你能算 likelihood，policy 也是 degenerate 的 delta distribution，没有 entropy，PPO ratio 全是 1，没法做 meaningful update。

πRL 就是同时解决这两件事。

---

## 2. Flow-Noise: 在 Euler step 上加可学噪声

### 2.1 核心思想

灵感来自 **ReinFlow** (https://arxiv.org/abs/2505.22094)。直觉是：既然 end-to-end 的 log π(A|o) 不可计算，那就**把整个 denoising 轨迹显式分解成 K 个 Gaussian 转移**，每一步转移的概率是 tractable 的，整条链的概率就是连乘。

### 2.2 Stochasticity injection

把 Euler update：

$$A_t^{\tau+\delta} = A_t^\tau + v_\theta(A_t^\tau, o_t) \cdot \delta$$

改写成一个 Gaussian transition：

$$p(A^{\tau+\delta} | A^\tau) \sim \mathcal{N}(\mu_\tau, \Sigma_\tau)$$

$$\mu_\tau = A^\tau + v^\tau \cdot \delta, \quad \Sigma_\tau = \mathrm{diag}(\sigma_{\theta'}^2)$$

这里：
- **v^τ = v_θ(A^τ, o)** 是 flow matching 学到的 velocity field（原 π0 的 action expert 输出）
- **σ_{θ'}** 是一个**新的小网络**（noise network），输入是 (A^τ, o)，输出每个 action 维度的标准差。和 v_θ 联合训练，但 fine-tuning 完就丢掉，推理时回到 deterministic policy
- **δ = 1/K** 是离散化步长
- diag 表示各向同性对角协方差

也就是说，每一步 denoising 既前进一点（mean 沿 v_θ 走），又抖动一点（variance 由新网络控制）。noise network 学到的是"在哪些维度、什么状态下应该多探索"。

### 2.3 Log-likelihood 的精确计算

把推理离散成 K 个时间点 τ_0=0 < τ_1=δ < ... < τ_K=1。整条 denoising 序列记作 𝒜 = (A^0, A^δ, ..., A^1)。联合 log probability：

$$\log \pi(A | o) = \log \pi(A^0 | o) + \sum_{k=0}^{K-1} \log \pi(A^{\tau_{k+1}} | A^{\tau_k}, o)$$

每一项：
- **log π(A^0 | o)**：A^0 ~ N(0, I)，是先验，与 θ 无关，是个常数（梯度为 0），但保留以保证 probability measure 完整
- **log π(A^{τ_{k+1}} | A^{τ_k}, o)**：是上面那个 Gaussian 的 log density，解析可写

$$\log \mathcal{N}(A^{\tau_{k+1}}; \mu_{\tau_k}, \Sigma_{\tau_k}) = -\frac{1}{2}\sum_i \left[\log(2\pi\sigma_{\theta',i}^2) + \frac{(A^{\tau_{k+1}}_i - \mu_{\tau_k,i})^2}{\sigma_{\theta',i}^2}\right]$$

下标 i 跑过 action 维度。**这就是 tractable 的精确 log-likelihood**，没有 Hutchinson、没有 adjoint、没有 variance 问题。

### 2.4 One-layer MDP 视角

现在 denoising 过程被视作 K 步的 discrete-time MDP：state 是 (o, A^τ)，action 是 A^{τ+δ}，转移是 Gaussian。把它直接套到标准 PPO 里，policy ratio 就是两个 Gaussian density 的比值，clip 一下就完事。

这里的优雅之处：**flow matching 的 K 步 inference 本来就要跑，每步都已经有 v_θ 的输出，加一个 σ_θ' 的 head 几乎零额外开销**。

---

## 3. Flow-SDE: ODE → SDE + Two-layer MDP

### 3.1 核心思想

灵感来自 **Flow-GRPO** (https://arxiv.org/abs/2505.05470)。和 Flow-Noise 加额外 noise network 不同，Flow-SDE 直接利用 score-based generative modeling 里著名的 **probability flow ODE ↔ SDE 等价性** (Song et al., https://arxiv.org/abs/2011.13456)：对于任意一个 marginal distribution q_τ，存在一个 ODE 和一个 SDE，它们的 marginal 分布在所有 τ 上完全一致。所以用 SDE 采样不会改变最终 action 的分布形状，但能注入随机性用于 exploration。

### 3.2 从 ODE 到 SDE 的转换

原始 Euler ODE：

$$dA^\tau = v^\tau\, d\tau$$

对应的等价 SDE：

$$dA^\tau = \underbrace{\left[v^\tau - \frac{1}{2}g^2(\tau)\nabla \log q_\tau(A^\tau)\right]d\tau}_{\text{drift}} + \underbrace{g(\tau)\, dw}_{\text{diffusion}}$$

变量解释：
- **g(τ)**：scalar noise schedule，控制扰动幅度
- **∇ log q_τ(A^τ)**：marginal distribution q_τ 的 score function
- **dw**：Wiener process（标准布朗运动），增量 dw ~ N(0, dτ)

drift term 里多出来的 -½ g² ∇log q_项 是 **Fokker-Planck 修正项**，用来抵消 diffusion 引入的"额外扩散"，保证 marginal 不变。这是 score SDE 论文里反复强调的点。

### 3.3 Score 与 velocity 的桥梁

要让这个 SDE 实用，得把 ∇log q_τ 替换成可计算的东西。对 rectified flow（π0 用的就是 rectified flow 系，https://arxiv.org/abs/2209.03003），有经典关系：

$$\nabla \log q_\tau(A^\tau) = -\frac{A^\tau}{\tau} - \frac{1-\tau}{\tau} v^\tau$$

这里 τ 是 flow matching 时间，τ→0 时退化为噪声分布，τ=1 时是目标分布。这个关系把 score 用 v_θ 显式表达 —— 关键 trick，因为 v_θ 我们本来就有。

### 3.4 选择 noise schedule

作者选 g(τ) = σ_τ = a · √(τ/(1-τ))，其中 a 是 hyperparameter（实验里 a=0.5 比较好）。这个 schedule 在 τ→1 时发散 —— 直觉上，越接近最终 action 越不能加太多 noise（会破坏 determinism），但 √(τ/(1-τ)) 在 τ=1 时 blow up。作者用的做法是 SDE 跑到 τ 接近 1 但不到 1，剩下用 ODE；或者作为 hybrid 的一部分（见 3.6）。

代入后化简得最终 SDE：

$$dA^\tau = \left[v^\tau + \frac{\sigma_\tau^2}{2\tau}(A^\tau + (1-\tau)v^\tau)\right]d\tau + \sigma_\tau\, dw_\tau$$

离散化：

$$\mu_\tau = A^\tau + \left[v^\tau + \frac{\sigma_\tau^2}{2\tau}(A^\tau + (1-\tau)v^\tau)\right]\cdot \delta$$

$$\Sigma_\tau = \sigma_\tau^2 \delta \cdot I$$

注意 Σ_τ 是 isotropic（各向同性）的，方差由 noise schedule 决定，不需要额外网络。这是 Flow-SDE 相对 Flow-Noise 的简化点。

### 3.5 Two-layer MDP

到这里事情还没完 —— 上面只是把单步 denoising stochastic 化了，但 RL 还要和环境交互。作者引入 **DPPO** (Diffusion Policy Policy Optimization, https://arxiv.org/abs/2409.00588) 的 two-layer MDP 思路：

- **Outer loop (环境层)**: 标准 MDP，时间 t，state o_t，action A_t^1（最终去噪完的 action chunk），transition P_ENV
- **Inner loop (denoising 层)**: 在每个 outer step 内，τ 从 0 走到 1，state 是 (o_t, A_t^τ)，action 是 A_t^{τ+δ}

形式化：

| 组件 | τ < 1 (inner) | τ = 1 (outer) |
|------|---------------|---------------|
| State s̄_t^τ | (o_t, A_t^τ) | (o_t, A_t^1) |
| Action ā_t^τ | A_t^{τ+δ} | A_t^1 |
| Next state | (o_t, A_t^{τ+δ}) | (o_{t+1}, A_{t+1}^0 ~ N(0,I)) |
| Reward | 0 | R_ENV(o_t, A_t^1) |

reward 只在 denoising 完成的那一步给 —— **稀疏但物理上正确**。两层 MDP 的好处是 denoising step 也参与 PPO 的 advantage 传播（虽然 reward=0，但 transition probability 进 ratio）。

### 3.6 Hybrid ODE-SDE sampling（加速 trick）

Two-layer MDP 把 horizon 拉长 K 倍（每 outer step 内有 K 个 inner step），训练贵且难收敛。作者借鉴 **MixGRPO** (https://arxiv.org/abs/2507.21802) 和 **TempFlow-GRPO** (https://arxiv.org/abs/2508.04324) 的 trick：

> 每个 outer step t，随机选**一个** denoising time τ_t 做 SDE 探索，其他 K-1 步用 deterministic ODE。

具体：从 state (o_t, A_t^{τ_t}) 出发，policy 在这一步输出 stochastic action，然后一个 environment wrapper 把剩下 1-τ_t 的 ODE 跑完，再走环境 transition，输出 (o_{t+1}, A_{t+1}^{τ_{t+1}})，τ_{t+1} 重新随机采。

这个公式上还等价于原 two-layer MDP（因为是 marginal 等价的），但实际 horizon 从 T×K 缩到 T，**2× wall-clock speedup**。在 LIBERO-Goal 的 ablation 里能看到 hybrid 跟 full SDE 性能相当但快一倍。

---

## 4. Policy Optimization 细节

### 4.1 Chunk-level formulation

π 系列输出 action chunk A_t = [a_{t,0}, ..., a_{t,H-1}]，长度 H（π0 是 50，π0.5 是 10）。作者把整个 chunk 视作一个 **macro-step**，reward 是 H 步环境 reward 之和：

$$R_t = \sum_{j=0}^{H-1} r_{t,j}$$

这样 PPO 在 outer MDP 上的 horizon 是 T = episode_length / H'，其中 H' 是 replan horizon（实验里大多 H'=5，即每 5 步重新观测一次）。这个 chunk-level 思路来自 **RLinf-VLA** (https://arxiv.org/abs/2510.06710)。

### 4.2 GAE + PPO clip

$$\hat{A}_t = \sum_{k=0}^{T-t} (\gamma\lambda)^k \mathcal{T}_{t+k}$$

其中 TD-error 𝒯_t = R_t + γV(s_{t+1}) - V(s_t)。

- **γ**: discount factor，控制对未来 reward 的折扣（实验 0.99）
- **λ**: GAE 的 bias-variance tradeoff 参数，λ=0 是 1-step TD（高 bias 低 variance），λ=1 是 Monte Carlo（低 bias 高 variance），实验 0.95

PPO clip 目标：

$$\mathcal{J}(\pi_\theta) = \mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

ratio：

$$\rho_t(\theta) = \frac{\pi_{\theta_\text{new}}(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

对 Flow-Noise，a_t|s_t 就是整条 denoising 序列的联合概率；对 Flow-SDE，是 two-layer MDP 里的 ā_t^τ | s̄_t^τ。两种形式都因为 Gaussian transition 而解析可算。clip ε=0.2 是标准值。

### 4.3 Critic 设计 —— 这块有意思

共享 actor-critic 架构省显存，但 π0 和 π0.5 的 state 输入位置不同，critic 要分别处理：

**π0.5**：state (robot proprioception) **不**输入模型，直接被 merge 到 VLM 的 prompt embedding 里。所以 critic 干脆接在 VLM 输出后：

$$V_\text{vlm}(o_t)$$

直接从 (image, language) 出 value，不接 state。简单粗暴但 ablation 显示反而效果更好（因为不耦合 noisy action）。

**π0**：state 进 action expert，action expert 同时接收 state 和 noisy action A_t^τ。critic 接在 action expert 后，但问题是 critic 输入也得带 noisy action —— 而推理时 A_t^τ 是变化的。作者的做法是对整条 denoising trajectory 上的 value 平均：

$$V_\text{expert}(o_t) \approx \mathbb{E}_{\tau \sim U[0,1]}[V_\text{expert}(o_t, A_t^\tau)]$$

直觉上：value 是 state value，不该依赖 denoising step 的具体噪声实现；但 action expert 内部把 state 和 noisy action 耦合了，所以最简单的近似是对 τ 平均掉噪声影响。

**Ablation 5.3.1 显示 V_vlm 反而比 V_expert 略好**，作者分析原因是 V_expert 的输入里 state 和 noisy action 耦合，优化困难。但作者为了"value function 应该接 state 信息"的概念正确性，π0 还保留 V_expert 架构。

另外 critic 结构：4-layer MLP 比 1-layer MLP 显著好（explained variance 更高，eval 性能更好）。

---

## 5. 实验结果

### 5.1 In-distribution 主结果

Table 1 关键数字：

| 模型 | 方法 | LIBERO | ManiSkill | MetaWorld | CALVIN | Avg. | Δ |
|------|------|--------|-----------|-----------|--------|------|---|
| π0 | SFT (few-shot) | 57.6 | 38.4 | 50.8 | 57.5 | 51.1 | - |
| π0 | Flow-SDE | 96.1 | 78.8 | 78.1 | 61.7 | 78.7 | +27.6 |
| π0 | Flow-Noise | 97.6 | 77.8 | 85.8 | 59.9 | 80.3 | +29.2 |
| π0.5 | SFT (few-shot) | 77.1 | 40.1 | 43.8 | 61.3 | 55.6 | - |
| π0.5 | Flow-SDE | 97.9 | 90.9 | 70.7 | 87.0 | 86.6 | +31.0 |
| π0.5 | Flow-Noise | 98.3 | 89.7 | 66.1 | 84.5 | 84.7 | +29.1 |

几个 striking 的点：

1. **Few-shot SFT + RL 反超 full-dataset SFT**: LIBERO 上 π0.5 用 1 条轨迹 SFT + RL = 98.3%，full-dataset π0.5 SFT = 96.9%。这是 RL 在 VLA 上"few-shot 起步 + RL 收尾"范式可行性的最强证据 —— **意味着 RL 有可能在未来替代大规模数据采集**。

2. **CALVIN 长 horizon 上 RL 增益随序列长度增加而放大**（Table 5）：π0.5 Len-1 提升 +7.0%，Len-5 提升 +25.7%。这符合直觉 —— SFT 的 compounding error 在长序列上更严重，RL 直接优化整个轨迹的 return，从根上解决 error accumulation。

3. **ManiSkill 4352 种 pick-and-place 组合**: π0.5 从 40.1% → 90.9% (+50.8%)，这种 massive combinatorial 任务上 RL 的 generalization 能力很 striking。

### 5.2 OOD 泛化

Table 4 + Figure 5:

| 场景 | 结果 |
|------|------|
| ManiSkill Vision OOD | ID 增益 +126.7% 时 OOD 增益 +73.9%，能 transfer |
| ManiSkill Semantics OOD | 提升明显但绝对值仍低（MultiCarrot 16.7% → 38.2%） |
| ManiSkill Execution OOD | π0 不如 π0.5（π0 接 state 易过拟合 control） |
| CALVIN ABC→D | 79.1% vs SFT 61.3%，视觉 shift 可 transfer |
| MetaWorld ML45 | 振荡，无 consistent 提升 |

重要结论：**RL 主要 enhance action-level refinement，对低层视觉/执行扰动 robust，对跨任务高层泛化帮助有限**。但 RL 至少避免了 SFT 的 catastrophic forgetting，OOD 上保留了 SFT 学到的东西。这跟 SimpleVLA-RL (https://arxiv.org/abs/2509.09674) 的发现一致。

### 5.3 Hyperparameter Ablation（Table 2，很重要）

LIBERO-Spatial 上跑：

**Noise level a**:
- a=0.2: train 59.5%，eval 73.1%，clip fraction 高（不稳定）
- a=0.5: train 93.5%，eval 94.5%（甜蜜点）
- a=0.8: train 95.3%，eval 98.1%（最好）

直觉：noise 小 → ratio 大 → gradient 大 → 不稳定；noise 大 → 探索足但 flow trajectory 被扭曲。低 noise 反而需要更小 lr。

**Denoise step K**:
- K=1: train 9.4%，灾难性失败（ODE→SDE 离散化误差 dominant）
- K=2: train 28.3%，eval 63.8%（仍不够）
- K=4: train 56.1%，eval 94.5%
- K=8: train 62.6%，eval 86.7%（K 大反而 eval 略降，因为 horizon 长，advantage 估计 noisy）

**Action chunk**:
- 5: train 93.5%，eval 94.5%
- 10: train 93.3%，eval 95.5%
- 20: train 87.5%，eval 89.2%（chunk 大 advantage credit assignment 难，explained variance 降低）

直觉：chunk 大 → SFT baseline 高（执行更平滑），但 RL 上限被压低（reward credit 难分配到具体子动作）。Long-horizon 任务可以用大 chunk。

### 5.4 VLM fine-tune 与否（Appendix F.2）

VLM frozen vs LoRA (r=32, α=32)：LoRA 没有明显收益，且需要更保守 lr 才稳定。作者推测 LIBERO scene 变化少，预训练 VLM 已经够用。**这暗示 RL 阶段 action expert 才是 bottleneck，VLM 改动风险大于收益**。

### 5.5 PPO vs GRPO（Table 9）

PPO 全面优于 GRPO（π0.5 LIBERO：PPO 97.9% vs GRPO 91.5%）。直觉上 GRPO 用 group baseline 代替 critic，但 chunk-level + 长 horizon 上 critic 提供的 low-variance advantage 信号更重要。

---

## 6. 工程细节与 Insights

### 6.1 训练设置

- 8×H100 80GB
- 共置策略：environment + rollout model + actor model 同一张 GPU 上 serial 执行（来自 RLinf codebase https://arxiv.org/abs/2509.15965）
- VLM frozen，只 fine-tune 300M action expert（GPU memory + RL4VLA 发现 RL 主要 boost action generalization）

### 6.2 GR00T N1.5 上的延伸（Appendix H）

证明方法的 architecture-agnostic：SFT 52.5% → PPO + Flow-SDE 89.9% (+37.4%)。一个关键工程 trick：**把 action expert 里的 dropout 全部换成 identity layer**。

理由很 elegant：PPO 的 ratio ρ_t = π_new(a|s) / π_old(a|s) 假设两次 forward 的 stochasticity 只来自 action sampling。如果网络里有 dropout，每次 forward 都换 mask，那 ratio 实际是：

$$\rho_t = \frac{\pi_{\alpha_\text{new}}(a|s)}{\pi_{\theta_\text{old}}(a|s)}$$

α_new 是 dropout mask 扰动后的 policy。这个 structural stochasticity 叠加 per-step update，会让训练根本不收敛。这是 RL 训练里被低估的坑。

### 6.3 Real2Sim2Real 演示

用 ManiSkill (rigid body) + 3D Gaussian Splatting (photorealistic rendering, https://arxiv.org/abs/2308.14737) 建 simulator，缩小 sim-to-real visual gap。20 条 expert trajectory 做 few-shot SFT，100 RL steps 后 zero-shot deploy 到 Franka Panda 真机，40% 成功率（SFT baseline 完全失败）。

参考 GSWorld (https://arxiv.org/abs/2510.20813) 和 TwinAligner (https://arxiv.org/abs/2512.19390) 的 Real2Sim2Real 思路。

### 6.4 Temporal efficiency（Appendix G）

RL 训练后 episode length 收敛到 expert motion planner 的水平。两个原因：
1. RL 增强 error correction，能从执行失败中恢复
2. Partial reset + discounted reward 让 agent "完成得快才能触发更多 reset 攒更多 reward"

### 6.5 Critic warmup 现象

ManiSkill 上 eval 曲线早期 dip 再上升 —— critic 早期 value estimate 不准，advantage 信号差，policy 先变差再修复。作者建议 early stage 不要 panic。

---

## 7. 局限与未来方向

1. **ODE→SDE 转换的精度损失**：Flow-CPS (https://arxiv.org/abs/2509.05952) 提出 coefficients-preserving sampling 来减误差，但作者实测 RL 增益有限 —— 说明问题更深，可能需要新的 noise injection 形式。

2. **Hybrid ODE-SDE 太简单**：只随机选 1 步做 SDE，其他全 ODE。MixGRPO 和 TempFlow-GRPO 有更复杂的 schedule，未来可借鉴图像生成加速的进展（如 Flow policy optimization https://arxiv.org/abs/2507.21053，BranchGRPO https://arxiv.org/abs/2509.06040）。

3. **跨任务泛化仍是难题**：MetaWorld ML45 上 RL 帮不上，需要更 diverse 的 task distribution + 更丰富的 language instruction 训练。

4. **依赖 sim-to-real**：online RL sample efficiency 太低，真机 RL 还不可行。未来需要 offline-to-online、world model 等 sample efficient 方法。

---

## 8. 我的几点延伸思考

### 8.1 与 ReinFlow 的关系

Flow-Noise 几乎是 ReinFlow (https://arxiv.org/abs/2505.22094) 在 VLA 上的直接应用，但加了几个 VLA-specific 适配：
- chunk-level macro-step（不是 per-action）
- critic 放在 VLM 后或 action expert 后的两种 placement
- shared actor-critic 节省显存（VLA 模型太大，分开存 critic 不现实）

ReinFlow 原文做的是 diffusion policy 的 RL fine-tuning，π0 是 flow matching —— 两者数学上是 cousin（flow matching 是 diffusion 的一般化），所以 ReinFlow 的 trick 直接可用。

### 8.2 与 DPPO 的关系

DPPO (https://arxiv.org/abs/2409.00588) 是 diffusion policy 的 PPO，πRL 的 Flow-SDE 的 two-layer MDP 思路直接来自 DPPO。差别：
- DPPO 用的是 DDPM/DDIM 的 SDE，πRL 用的是 rectified flow 的 SDE
- DPPO 的 inner step 是 DDPM 反向，πRL 是 flow 的 forward Euler
- πRL 加了 hybrid ODE-SDE 加速

### 8.3 为什么 Flow-Noise 和 Flow-SDE 性能接近

两个方法性能差不到 2%，但机制差异挺大：
- **Flow-Noise**: 加额外 noise network，one-layer MDP，收敛快但 K 大时 wall-clock 不省
- **Flow-SDE**: 不加额外网络，two-layer MDP，hybrid 加速 2×，但训练稍慢起步

实际推荐 Flow-SDE（计算便宜且 ablation 多）。Flow-Noise 适合需要精细 control noise magnitude 的场景。

### 8.4 Chunk size 的 RL 友好性悖论

SFT 上 chunk 大更好（执行平滑，temporal consistency），但 RL 上 chunk 大反而 cap 了 ceiling —— 因为 advantage credit assignment 难。这暗示一个 **decoupled chunk size** 的设计：SFT 阶段用大 chunk，RL 阶段动态切小。π_fast (https://arxiv.org/abs/2501.09747) 的 action tokenization 思路可能能帮上忙。

### 8.5 与 RLHF/DPO 的关系

RL4VLA 测过 PPO vs GRPO vs DPO (https://arxiv.org/abs/2305.18290)。DPO 不需要 reward model 和 log-likelihood 显式计算（用 preference pair），看起来适合 flow-based VLA。但 DPO 需要 preference data，机器人领域采集成本高。πRL 走 PPO 路线，保留了 dense online reward 的优势。

未来值得探索：**flow-based VLA + DPO + preference from human feedback** 是否可行。FPO (https://arxiv.org/abs/2507.21053) 已经在尝试用 advantage-weighted CFM loss 重塑 policy optimization，可能是另一条路。

### 8.6 Sim-to-real 的真正瓶颈

πRL 用 3D GS 做 rendering 缩小 visual gap，但 40% 真机成功率仍不高。瓶颈可能在：
- Dynamics gap（rigid body simulator 与真机摩擦/接触的差异）
- Action chunk 执行的 open-loop 问题（环境偏离预期时，预先规划的 H 步 action 不再最优）
- Latency（VLM 推理慢，real-time control 难）

π0.5 的 flow matching expert 比较轻量（300M），VLM inference 是大头。未来用 speculative decoding 或 VLM cache 可能缓解。

### 8.7 与 LLM RLHF 的统一视角

πRL 让我想到 LLM RLHF 的早期：pre-train → SFT → PPO。VLA 现在重走这条路，但加了两个 LLM 没有的 twist：
- Action 是连续 + chunk，不是 token sequence
- Flow matching decoder 比 autoregressive decoder 难处理 log-likelihood

但解决问题的 framework（policy gradient + clip + GAE）是一样的。**LLM RL 的所有积累（GRPO、DPO、RLOO、ReST 等）原则上都能搬到 VLA**，只要解决 likelihood tractability 这一个核心 obstacle。πRL 给的就是这个 obstacle 的两种解法。

---

## 9. 参考 link 汇总

**核心 paper**:
- πRL code: https://github.com/RLinf/RLinf
- RLinf codebase: https://arxiv.org/abs/2509.15965
- RLinf-VLA: https://arxiv.org/abs/2510.06710

**Base models**:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- GR00T N1: https://arxiv.org/abs/2503.14734
- Octo: https://arxiv.org/abs/2405.12213
- TinyVLA / SmolVLA: https://arxiv.org/abs/2506.01844
- π_fast: https://arxiv.org/abs/2501.09747

**Flow matching / Diffusion theory**:
- Flow matching: https://arxiv.org/abs/2210.02747
- Rectified flow: https://arxiv.org/abs/2209.03003
- Score SDE: https://arxiv.org/abs/2011.13456
- Neural ODE: https://arxiv.org/abs/1806.07366
- Hutchinson trace estimator: https://doi.org/10.1080/03610918908812851

**RL methods**:
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- GRPO: https://arxiv.org/abs/2402.03300
- DPO: https://arxiv.org/abs/2305.18290
- DPPO: https://arxiv.org/abs/2409.00588
- ReinFlow: https://arxiv.org/abs/2505.22094
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- MixGRPO: https://arxiv.org/abs/2507.21802
- TempFlow-GRPO: https://arxiv.org/abs/2508.04324
- BranchGRPO: https://arxiv.org/abs/2509.06040
- FPO: https://arxiv.org/abs/2507.21053
- Flow-CPS: https://arxiv.org/abs/2509.05952
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- RL4VLA: https://arxiv.org/abs/2505.19789

**Benchmarks**:
- LIBERO: https://arxiv.org/abs/2306.03310
- ManiSkill3: https://arxiv.org/abs/2410.00425
- MetaWorld+: https://arxiv.org/abs/2505.11289
- CALVIN: https://arxiv.org/abs/2112.03227
- SIMPLER: https://arxiv.org/abs/2405.05941
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

**Rendering / Sim**:
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- GSWorld: https://arxiv.org/abs/2510.20813
- TwinAligner: https://arxiv.org/abs/2512.19390

**Misc**:
- LoRA: https://arxiv.org/abs/2106.09685
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- MPLib: https://github.com/haosulab/MPlib

---

## TL;DR for intuition

πRL 把 PPO 套上 flow-based VLA 的关键是**让 action 的 log-likelihood 变得可算**，有两个等价路径：
1. **Flow-Noise**：denoising 每一步加一个 learnable Gaussian，把 K 步视作 discrete MDP，联合概率就是 K 个 Gaussian 的连乘
2. **Flow-SDE**：把 rectified flow 的 ODE 转成 marginal-preserving SDE，自然引入 stochasticity，再用 two-layer MDP 把 denoising 和环境交互缝起来

两条路殊途同归 —— 都把"intractable flow likelihood"转成"K 个 tractable Gaussian transitions"。前者精确但慢，后者便宜但需要 hybrid 加速。最终 few-shot SFT + 100 步 RL 能反超 full-dataset SFT，证明 RL 在 VLA scaling 上的潜力远大于继续采集 demonstration 数据。
