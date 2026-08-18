---
source_pdf: CFDLLMBench A Benchmark Suite for Evaluating Large Language Models in
  Computational Fluid Dynamics.pdf
paper_sha256: 8960af8691f9f2cbe853f8be8f3343e3f8b5209bcb030940e9ab7c16d6e559ff
processed_at: '2026-08-18T03:12:47-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 CFDLLMBench

## 一句话先给结论

这帮人给LLM出了一套CFD的"考研+工程师上岗"三级考试卷子, 结果发现: GPT这种顶级模型在"背概念"这关能考90分, 但是到了"自己写代码解PDE"这关直接掉到14分, 再到"配一套完整OpenFOAM工程case跑出物理正确的结果"这关最好也才34分. **LLM现在是CFD课的学霸, 但是CFD实验室的废物**.

## 为什么要搞这么个benchmark

你看现在的AI4Science圈子有点bubble. 大家都在喊"LLM要颠覆科学计算", 但是真拿LLM去跑一个CFD simulation, 基本上全炸. 之前的benchmark要么太toy (SciCode只测1D Burgers这种本科生作业题), 要么太封闭 (FEABench用COMSOL, 一年license几千美金). 这帮RPI+NREL+PNNL的人就想: 咱们用OpenFOAM吧, open source, 工业标准, 工作流又臭又长, 正好拿来折磨LLM. 

OpenFOAM为什么折磨? 你做一个case要写6-7个config文件, 分三个文件夹 `0/` (initial/boundary conditions), `constant/` (物理属性, turbulence model), `system/` (solver控制, mesh, schemes, solution算法). 每个文件里有一堆OpenFOAM特有的DSL keyword, 文件之间互相引用 (e.g., `system/controlDict`里写的solver名字必须对应`constant/`里的物理属性配置). 一个case 300-600行代码, 改错一个keyword整个simulation就跑不起来. 一个熟练的CFD工程师配一个case可能要半天到一天, 那让LLM来试试?

## 三级考试的设计

### 第一级: CFDQuery — 考概念

90道multiple choice, 都是graduate level的CFD题. 涉及finite difference的modified equation分析, WENO格式的dispersion/dissipation, curve grid上的geometric conservation law, discontinuous Galerkin这些博士资格考试级别的东西. 一道典型题: 给你1D advection方程 $\partial_t u + a \partial_x u = 0$, 问用二阶中心差分离散空间、RK3离散时间之后, modified equation里多出来的误差项长啥样. 正确答案是空间离散产生一个三阶导数项 $\frac{a\Delta x^2}{6}\partial_x^3 u$, 时间离散RK3产生另一个三阶导数项 $-\frac{a^3\Delta t^2}{6}\partial_x^3 u$, 两项符号要搞清楚, 时间项还涉及 $a^3$ 因为RK3的truncation error里有 $a$ 的三次方耦合.

结果: o3-mini考92分, Sonnet 3.5和Gemini 2.5 Flash都80+, Gemma-2-9B只有60. **模型们都能把CFD教材背下来**, 这部分不奇怪.

### 第二级: CFDCodeBench — 考数值reasoning

24道题, 给你一个PDE + boundary condition + initial condition + domain, 要你写Python把解算出来存成`.npy`. 题目从1D Burgers, 2D Laplace这种简单的, 到shock tube Euler, 2D Navier-Stokes lid-driven cavity, Rayleigh-Bénard convection, KdV-Burgers, pipe flow线性稳定性特征值问题这种硬核的都有. 平均70行Python.

这关用的metric很有讲究. 他们不用"代码像不像reference"来打分, 用三个东西:

1. 代码跑不跑得通 (binary)
2. 跑出来的solution和reference比对, normalized mean squared error在10%以内打1分, 10-30%打0.5分, 30%以上打0分
3. 把网格和时间步长refine之后误差是不是单调下降 (收敛性)

三个都得满分, 这题才算过. 10%和30%这两个threshold不是拍的, 他们做了sensitivity analysis: threshold从1%扫到15%, 发现10%那里有个拐点, 再放宽边际收益骤降; upper bound从25%扫到45%, 发现30%以上就会把"瞎猜但运气好"的case算对. 这是工程CFD validation的常识.

结果: 最好的Sonnet 3.5也只14%. 代码大部分能跑 ($M_{\mathrm{exec}}$ 0.6-0.8), 但是跑出来的数物理上不对 ($M_{\mathrm{NMSE}}$ 普遍0.2以下). **LLM能写出"看起来对"的CFD代码, 但是数值上不对**. 比如对1D Burgers, Sonnet 3.5能在shock附近拟对, 但是boundary condition搞错; 对2D convection, 出现数值不稳定. 这些都是"会背教材但是不会调schemes"的典型症状.

### 第三级: FoamBench — 考工程workflow

126个OpenFOAM case. 110个basic来自11个tutorial (cavity, cylinder, forwardStep, pitzDaily, damBreak这些), 每个tutorial衍生10个variation改inlet velocity, viscosity这些. 16个advanced是专家手写的, 不在tutorial里, 考三种extrapolation: turbulence model切换, geometry修改, 全新geometry (e.g., 两个方形障碍物, 45度旋转的菱形).

这关有5个metric: 跑得通 + folder结构对 + 文件内容相似度 + 解的NMSE + Success Rate. 他们还测了两个agent framework: MetaOpenFOAM和Foam-Agent, 都是用RAG (retrieve类似的tutorial case作为exemplar) + Reviewer (一个LLM当"审稿人"检查输出, 允许iterative debug) 的多agent结构.

结果:

| 配置 | Basic Success Rate | Advanced Success Rate |
|---|---|---|
| Sonnet 3.5 zero-shot | 4.5% | 0.7% |
| Sonnet 3.5 + RAG only | 20% | 12.5% |
| Sonnet 3.5 + Reviewer only | 24.5% | 12.5% |
| Sonnet 3.5 + RAG + Reviewer (Foam-Agent) | 33.6% | 25% |

Zero-shot基本归零, 说明LLM参数里根本没"记住"OpenFOAM的DSL, 全靠外部exemplar和iterative debug. RAG和Reviewer各自贡献~10%, 组合有协同效应.

成本分析: Sonnet 3.5 + Foam-Agent平均一个case $6.56 (主要花在RAG的token上, 平均prompt 378k token, 因为要把tutorial的case file全塞进去), GPT-4o $0.42, Gemini 2.5 Flash $0.01. 工业deployment算每美元成功率, Sonnet其实性价比最高.

## 几个有意思的failure mode

1. **Gemini 2.5 Flash在Advanced上直接0%**. 这模型在CFDQuery上83分, 但是在Advanced case上连executable都做不到 ($M_{\mathrm{exec}}$ 0.062). 这是典型的"长上下文+多文件依赖"下instruction-following崩溃, Google自己官方blog都承认过Gemini Flash在复杂agent task上会退化.

2. **o3-mini在CFDQuery上最高92%, 但是在FoamBench上反而被Sonnet 3.5吊打**. reasoning model擅长"选A/B/C/D", 但是在"组装6个config文件让OpenFOAM跑通"这种工程artifact任务上, Claude的工程训练碾压reasoning能力. 这是一个很重要的发现: **reasoning ≠ engineering**.

3. **Spatial reasoning是硬伤**. Figure 6那个doubleSquare case最直观: prompt清清楚楚说"在(2,2)放一个1×1的方框", 但是Foam-Agent生成的blockMeshDict完全不对. 问题在于: LLM在token层面理解"在(2,2)放square"是trivial的, 但是要把它翻译成blockMeshDict里 `hex (8 12 13 9 16 20 21 17)` 这种vertex index assignment, 需要的是几何坐标系下的mental rotation + vertex enumeration, 而这恰恰是autoregressive LLM最弱的部分. 这堵墙在vision-language model里被讨论很多, 但是在纯text LLM上做geometry generation基本是个open problem.

4. **Prompt engineering在这任务上几乎没用**. 附录A.4.4做了个实验: 用o3-mini去做prompt optimization, 生成5个prompt variant, 测下来Success Rate只从0.7%升到1.2%. 这说明在OpenFOAM这种高度structured的DSL任务上, prompt的措辞远不如framework的architecture重要. 这是给所有"prompt engineering网红"的一记耳光.

5. **RAG主要消解"配置缺失"类错误** (undefined keyword, missing file, wrong physical property), **Reviewer主要消解"一致性"类错误** (boundary condition patch name在mesh里找不到, file间引用断裂). 两者互补, 一起把"完全跑不起来"变成"能跑且物理对".

## 我的intuition

读完这篇paper我最强的intuition是: **LLM在CFD上的瓶颈不是知识, 是reasoning over structured artifacts**. 这三层能力 (recall → numerical reasoning → workflow implementation) 之间的gap远大于社区预期.

你看从CFDQuery 92%到CFDCodeBench 14%, 掉了78个绝对百分点. 这中间发生了什么? LLM能"讲清楚"modified equation长啥样, 但是不能"动手"把RK3的Butcher tableau写对, 不能"判断"什么样的CFL条件对当前网格够, 不能"调"discretization scheme让数值稳定. 这是典型的 declarative knowledge → procedural knowledge 的鸿沟, 人类教育里也是这样: 你能考过流体力学笔试, 不代表你能独立调通一个LES simulation.

然后从CFDCodeBench 14%到FoamBench 34%, 反而升了20个百分点 (用Foam-Agent). 为什么? 因为Foam-Agent引入了RAG和Reviewer, 等于给LLM配了"参考手册"和"审稿人". 这说明LLM的"内生能力"其实很差, 但是加上外部工具就能补上不少. 这是一个好消息也是一个坏消息: 好消息是agentic framework能放大LLM的能力, 坏消息是LLM本身不够用, 需要大量engineering effort去搭framework.

再看FoamBench Basic 34% vs Advanced 25%, 只掉了9个百分点. 这说明Advanced的"几何外推"和"turbulence model切换"虽然更难, 但是相对于"完全没RAG可比对的case"的loss, 不如"basic里tutorial相似度低"的loss大. 换句话说, RAG的覆盖度比task的内在难度更决定Success Rate. 这也解释了为什么Gemini 2.5 Flash在Advanced归零 (它RAG能力弱, 一旦没相似tutorial可retrieve就崩).

## 这篇paper没说清楚的事

1. **没human baseline**. 他们只在附录A.1里定性说"CFD PhD能轻松做CFDCodeBench, CFD工程师能轻松做FoamBench Basic", 但没quantitative study. 14%和34%到底离human差多少? 10倍还是100倍? 不清楚. 我估计一个CFD PhD做CFDCodeBench能到90%+, 一个OpenFOAM熟练工做FoamBench Basic能到80%+, 那LLM现在大概差5-6倍. 但是没测就是没测.

2. **Reviewer的rubric不透明**. Reviewer是另一个LLM call, 它怎么判断一个case file"对不对"? 如果只是检查syntactic consistency (patch name匹配, file引用正确), 那它本质上是个linter, 不解决physical correctness. 如果它要做physical reasoning, 那它自己也可能错. paper没说清楚Reviewer的内部机制, 这是一个黑盒.

3. **NMSE对field comparison过于严格**. CFD solution是空间field, 逐点NMSE对相位误差极度敏感. 比如涡街里vortex shedding频率偏1%, 整个field的NMSE会爆掉, 但是physical intuition是"涡街基本对了, 只是频率稍微shift". 工业CFD validation通常用drag coefficient, lift coefficient, Nusselt number这些integral quantity做对比, 比逐点field comparison robust得多. 这是paper一个明显的metric局限.

4. **几何complexity天花板太低**. 所有case都用blockMesh生成的hex mesh, 真实工业CFD 90%的effort在CAD import + snappyHexMesh处理复杂几何上. 这部分完全没覆盖, 这是v1的合理取舍但是明显scope limit.

5. **求解器覆盖偏窄**. 11个tutorial主要覆盖icoFoam, simpleFoam, pimpleFoam, rhoCentralFoam, reactingFoam, interFoam. 没测overSet (Chimera grid), foamRun (multi-region conjugate heat transfer), LDES (large-eddy simulation). 这些是工业CFD的"真实战场", 但是都没进benchmark.

## 我觉得下一步该往哪走

**(a) Symbolic-numeric hybrid**: LLM负责high-level workflow orchestration, 把geometry generation offload到OpenSCAD/FreeCAD Python API, 把discretization scheme选择offload到sympy的PDE module, 把convergence check offload到专门的uncertainty quantification tool. 这是"LLM as conductor"而不是"LLM as everything". Foam-Agent 2.0已经往这方向走, 但是还很初步.

**(b) Specialized CFD LLM**: 现在所有model都是general pretrain, 在CFD corpus上exposure严重不足. 一个合理路径: 用OpenFOAM tutorial (几千个公开case) + CFD textbook + forum post做continued pretrain, 再用FoamBench做RLHF. 这条路在medical (Med-PaLM) 和code (Code Llama) 都被验证过. 我估计1B-7B的specialized model能在FoamBench上beat Sonnet 3.5, 因为domain-specific knowledge比general reasoning更重要.

**(c) Hierarchical benchmark**: 现在3层是平的, 但是真实CFD workflow是深的: mesh generation → mesh quality check → solver selection → BC/IC setup → discretization scheme → time step control → run → post-process. 把这7层每一层都做一个细粒度benchmark, 才能定位LLM在哪一层最弱. 现在的14%和34%是composite number, 解剖不出来瓶颈在哪.

**(d) Verification-aware metric**: 把 $M_{\mathrm{NMSE}}$ 升级到QoI-based (drag, lift, Nusselt) + POD-based + adjoint-based error estimation. 这样能区分"solution物理对但是数值噪声大"和"solution物理错".

## 总结一句人话

这帮人给LLM出了一套从"CFD考研"到"OpenFOAM上岗"的三级考试, 发现LLM考研能考90分, 上岗考试只能做34%的题. 这中间的鸿沟不是知识不够, 是把知识"装配"成可执行工程artifact的能力不够. 当前的fix是搭agent framework (RAG + Reviewer) 给LLM配"参考手册"和"审稿人", 但是这只是补丁, 根本问题是autoregressive LLM在long-context multi-file dependency + spatial reasoning + numerical stability reasoning这些能力上有结构性缺陷. 想让LLM真正做CFD, 要么走specialized pretrain + RLHF的路, 要么走symbolic-numeric hybrid把geometry/discretization offload到专门工具, 大概率两条路都得走.

论文链接: https://openreview.net/forum?id=kTcH1MnkjY
代码: https://github.com/NLR-Theseus/cfdllmbench/
Foam-Agent: https://arxiv.org/abs/2509.18178
MetaOpenFOAM: https://arxiv.org/abs/2407.21320

---

# CFDLLMBench 深度解读:把LLM按到CFD的workbench上量一量

## 1. 为什么这篇paper值得认真读

这篇paper来自RPI的Shaowu Pan组联合NREL/PNNL, 对应OpenReview链接 https://openreview.net/forum?id=kTcH1MnkjY , code在 https://github.com/NLR-Theseus/cfdllmbench/ . 它的核心命题非常清楚: LLM在general NLP上已经登峰造极, 在scientific computing尤其是CFD这种"工程实践堆出来的craft"上到底能走多远? 现有的SciCode (Tian et al., 2024) 只测1D heat + 1D Burgers, 离真实CFD十万八千里; FEABench (Li et al., 2025) 锁在COMSOL的商业license里. 这篇paper选择OpenFOAM作为target, 既是open-source又是工业界事实标准, 同时workflow极其痛苦(mesh + boundary condition + solver + discretization scheme的多文件协同), 所以它是一个完美的"LLM-as-scientist"的试金石.

paper把"做CFD"这件事拆成了三层能力, 用三个子benchmark递进式测试:

| Benchmark | 任务形态 | 样本量 | 测的能力 | 最佳Success Rate |
|---|---|---|---|---|
| CFDQuery | graduate-level MCQ | 90 | conceptual recall | 92% (o3-mini) |
| CFDCodeBench | 从PDE描述生成Python数值求解器 | 24 | numerical & physical reasoning | 14% (Sonnet 3.5) |
| FoamBench | 自然语言→OpenFOAM完整case文件并跑出物理正确结果 | 110 basic + 16 advanced | context-dependent workflow implementation | 34% basic / 25% advanced (Sonnet 3.5 via Foam-Agent) |

这个从92%→14%→34%的drop本身就是paper最有冲击力的findings, 它说明: **LLM能"说"CFD, 但是不能"做"CFD**. 概念性的回忆和真正写出能跑、跑出来物理正确、数值收敛的code之间, 隔着一道巨大的reasoning鸿沟.

## 2. 三个benchmark的设计哲学

### 2.1 CFDQuery: 知识recall的baseline

90道MCQ, 由三位CFD专家curate, 覆盖linear algebra, numerical methods, fluid dynamics, 还包括modified equation analysis, dispersion/dissipation of high-order stencils (fourth-order central, compact schemes, WENO), curvilinear grids, geometric conservation law, discontinuous Galerkin这些真graduate level的内容. 一个典型例子是paper附录A.5.1里给出的题: 问2阶中心差分 + RK3离散1D advection方程 $\partial_t u + a \partial_x u = 0$ 的modified equation是哪一个, 正确答案是

$$\frac{\partial u}{\partial t} + a\frac{\partial u}{\partial x} = \frac{a \Delta x^2}{6}\frac{\partial^3 u}{\partial x^3} - \frac{a^3 \Delta t^2}{6}\frac{\partial^3 u}{\partial x^3} + \mathcal{O}(\Delta x^3)$$

这里 $a$ 是advection速度, $\Delta x$ 是空间步长, $\Delta t$ 是时间步长, $\partial_x^3 u$ 是三阶空间导数项, 两个系数分别对应空间离散的dispersion error和RK3时间离散的dispersion error, 合起来是经典的modified wavenumber analysis结果. 这种题考的是对 numerical scheme的truncation error结构的精确记忆, o3-mini在这里能到92%, 说明LLM确实"读过"CFD的教材. Gemini 2.5 Flash和Sonnet 3.5都答对了Option 4, 而GPT-4o答错选Option 3 (漏掉了 $-a^3\Delta t^2/6$ 的时间离散项, 把空间项的符号也搞反了), Haiku和Gemma选Option 1 (完全错误的dissipation形式). 这种错误pattern很能说明问题: 强模型做modified equation靠的是真的推导, 弱模型靠的是pattern matching.

### 2.2 CFDCodeBench: physical reasoning的第一道墙

24道题, 17道来自Barba的"CFD Python: 12 Steps to Navier-Stokes"和ENGR 491, 7道更难的来自Dedalus project (Burns et al., 2020, https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023068) 用spectral method. 覆盖1D Burgers, 2D Burgers, shock tube Euler, 2D Laplace/Poisson, lid-driven cavity, channel flow, Rayleigh-Bénard, KdV-Burgers, pipe flow linear stability eigenvalue problem等. 平均每题70行Python.

prompt设计很有讲究, 用JSON-to-natural-language pipeline, JSON字段包括: `equation` (LaTeX PDE), `boundary conditions`, `initial conditions`, `domain`, `save_values`, 可选的`numerical method`. 然后通过`generate_prompt()`函数拼成结构化user prompt. 比如对1D Burgers方程, JSON里写:

```
equation: \partial_t u + u \partial_x u = \nu \partial_x^2 u, \nu=0.07
boundary: u(0)=u(2\pi)  (periodic)
initial: u = -(2\nu/\phi) \partial_x \phi + 4, \phi = \exp(-x^2/(4\nu)) + \exp(-(x-2\pi)^2/(4\nu))
domain: x\in[0,2\pi], t\in[0,0.14\pi]
save: u
method: finite difference
```

system prompt是固定的: "You are a highly skilled assistant capable of generating Python code to solve CFD problems using appropriate numerical methods." temperature=0.0, 60秒sandbox timeout, 把最终time step的solution存成`.npy`和reference比对.

### 2.3 FoamBench: 真正的工程workflow

这是paper的核心. 126个OpenFOAM case, basic 110个来自11个tutorial: BernardCells, Cavity, counterFlowFlame2D, Cylinder, damBreakWithObstacle, forwardStep, obliqueShock, pitzDaily, shallowWaterWithSquareBump, squareBend, wedge. 每个tutorial再衍生10个variation (改inlet velocity, viscosity, boundary temperature等). Advanced 16个是CFD专家手写的, 不在tutorial里, 考三种extrapolation: turbulence model切换 (k-ε → SA → LES, 要同步改`momentumTransport`和`0/`文件夹下的初始场), geometric modification (改domain size), unseen geometry (e.g., doubleSquare: 两个方形障碍物, diamond: 45度旋转方形).

一个typical Advanced prompt像这样:

> Perform an incompressible turbulent flow simulation over a 2D diamond obstacle using the k-epsilon RANS turbulence model and pimpleFoam solver. The computational domain spans 0 to 15 in x, 0 to 5 in y, -0.5 to 0.5 in z. The diamond obstacle is a square rotated by 45 degrees with diagonal length of 1 unit centered at 2.5×2.5×0.0. Use one cell in z direction making the geometry effectively 2D. Refine the mesh near diamond. Use sufficient grid points to discretize the domain and don't use more than 10000 cells. The left boundary is the inlet with uniform velocity (1,0,0) m/s. The right boundary is the outlet with zero gradient pressure. Top and bottom are fixed walls with no-slip. Front and back faces are empty. Kinematic viscosity is 2e-6 m²/s. Use deltaT of 0.5 s and run till final time 5 s. Write results every 0.5 s. Max Courant number 1.0.

这要求LLM同时理解: 几何坐标变换, blockMeshDict的hex block构造, boundary patch命名 (inlet/outlet/wall/empty), turbulence model对`0/k`和`0/epsilon`的初始化要求, CFL约束下deltaT的合理性. 每个case ~300-600行代码散落在6-7个config文件: `0/{U,p,T,k,epsilon,...}`, `constant/physicalProperties`, `constant/momentumTransport`, `system/controlDict`, `system/blockMeshDict`, `system/fvSchemes`, `system/fvSolution`. 这种long-context的多文件依赖是LLM最不擅长的.

## 3. 评估指标的设计细节

### 3.1 CFDCodeBench的三元metric

这是paper我最喜欢的设计, 它拒绝用code similarity, 改用results-oriented metric:

**Metric 1: Executability $M_{\mathrm{exec}}\in\{0,1\}$** — 类似pass@1, 代码跑通就是1.

**Metric 2: Normalized Mean Squared Error $M_{\mathrm{NMSE}}$** — 把LLM生成的solution $\hat{y}_i$ 和reference $y_i$ 在final time step比对:

$$\mathrm{NMSE}\% = \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} y_i^2} \times 100$$

这里 $N$ 是空间网格点数, $y_i$ 是reference solution在第 $i$ 个网格点的值, $\hat{y}_i$ 是LLM生成code在同一点的解, 分母用 $\sum y_i^2$ 是为了normalize到能量范数. 然后离散化:

$$M_{\mathrm{NMSE}} = \begin{cases} 1, & \mathrm{NMSE} \le 10\% \\ 0.5, & 10\% < \mathrm{NMSE} \le 30\% \\ 0, & \mathrm{NMSE} > 30\% \end{cases}$$

10%和30%的阈值不是拍脑袋. 附录A.2做了sensitivity analysis: lower bound从1%扫到15%, upper bound固定30%, 看mean NMSE score和true success rate的拐点. 在lower=10%时mean NMSE score=0.4273, true success rate=33.6%, 再放宽到15%只升到34.5%, 边际收益骤降. upper bound从0.25扫到0.45, 0.30之后mean NMSE从0.4273跳到0.4955再到0.5045, 0.40以上明显过于宽松会把"乱写但运气好"的case算对. 所以最终选 (10%, 30%) 作为engineering practice的标准bracket.

**Metric 3: Numerical convergence $M_{\mathrm{conv}}\in\{0,1\}$** — 同时细化空间步长 $\Delta x$ 和时间步长 $\Delta t$, 检查relative error是否单调下降. 这是从CFD validation practice直接借来的Richardson extrapolation思想, 强制LLM不能只"瞎猜一个能跑的数", 而是要真的把discretization scheme做对.

**Success Rate** — 三元AND:

$$M_{\mathrm{success}}^{(i)} = \begin{cases} 1, & M_{\mathrm{exec}}^{(i)} = 1 \land M_{\mathrm{NMSE}}^{(i)} = 1 \land M_{\mathrm{conv}}^{(i)} = 1 \\ 0, & \text{otherwise} \end{cases}$$

这是非常严苛的, 三个指标都要"全垒打". 所以即便 $M_{\mathrm{exec}}$ 高达0.8, $M_{\mathrm{NMSE}}$ 通常会collapse到0.2-0.4, AND之后Success Rate掉到14%.

paper明确说: "we cannot rely on code similarity with respect to a reference solution, as numerical simulation code can vary significantly in implementation while yielding identical or equivalent solutions." 这是对所有LLM-for-code benchmark的一个根本性提醒. HumanEval用unit test是对的, 但scientific code连unit test都不好写, 因为correctness是分布的不是点状的.

### 3.2 FoamBench的五元metric

FoamBench扩展到5个metric:

1. $M_{\mathrm{exec}}$: OpenFOAM能否成功执行(case不崩)
2. $M_{\mathrm{struct}}$: 用ROUGE (Lin, 2004, https://aclanthology.org/W04-1013/) 比较folder structure, 因为OpenFOAM对`0/`, `constant/`, `system/`三层的依赖极其严格
3. $M_{\mathrm{file}}$: 同样用ROUGE比较每个file的内容
4. $M_{\mathrm{NMSE}}$: 同CFDCodeBench, 比对final time step的field
5. Success Rate: $M_{\mathrm{exec}}=1 \land M_{\mathrm{NMSE}}=1$ 的case比例 (注意这里没要求 $M_{\mathrm{struct}}$ 和 $M_{\mathrm{file}}$ 都1, 因为ROUGE高不代表物理对, physical accuracy才是终极判据)

ROUGE的引入是为了量化"结构性正确度", 但它本身有局限: 如果LLM换了一个等价但不同写法的`blockMeshDict`, ROUGE会偏低. 不过paper把这个metric作为辅助, 主判据还是 $M_{\mathrm{NMSE}}$, 这是合理的.

## 4. 实验结果的关键findings

### 4.1 跨model的total view (Figure 2)

- **CFDQuery**: o3-mini 92% > Sonnet 3.5 ~85% > Gemini 2.5 Flash ~83% > GPT-4o ~80% > Haiku 3.5 ~78% > Gemma-2-9B-IT 60%. Gemma明显掉队, 说明9B级别开源模型在graduate-level CFD知识上还撑不住.
- **CFDCodeBench**: Sonnet 3.5 14% 最佳, 其余基本个位数. $M_{\mathrm{exec}}$ 普遍60-80%, 但是 $M_{\mathrm{NMSE}}$ 普遍<20%.
- **FoamBench Basic**: Sonnet 3.5 + Foam-Agent (RAG+Reviewer) 34%, o3-mini 26.4%, GPT-4o 28.2%, Gemini 2.5 Flash 13.6%, Haiku 19.1%, Gemma 0%.
- **FoamBench Advanced**: Sonnet 3.5 25%, GPT-4o 25%, o3-mini 18.7%, Haiku 18.7%, Gemini 0%, Gemma 0%.

最反直觉的: **Gemini 2.5 Flash在Advanced上直接归零**. 这和它在Query上83%的表现形成巨大反差, paper没深挖原因, 但从Table 7能看出来: 它在Advanced上 $M_{\mathrm{exec}}$ 只有0.062, 几乎完全无法生成可执行的case file, 这暗示它在长上下文、多文件依赖任务上存在严重的instruction-following退化. 这和Gemini系列在long-context retrieval上reported的某些failure模式吻合 (参见 https://developers.googleblog.com/en/start-building-with-gemini-25-flash/ 的官方讨论).

另一个反直觉: **o3-mini在CFDQuery上92%最高, 但是在FoamBench上反而不及Sonnet 3.5**. 这说明reasoning model擅长"做一题选一个选项", 但在"组装多文件工程artifact"上, Claude的工程训练占了上风. 这是一个非常重要的发现: reasoning能力 ≠ 工程artifact能力.

### 4.2 Agentic framework的对比 (Table 2)

paper对比了两个framework: MetaOpenFOAM (Chen et al., 2024a, https://arxiv.org/abs/2407.21320) 和Foam-Agent (Yue et al., 2025c, https://arxiv.org/abs/2509.18178). 两者都是RAG + Reviewer + file generator的多agent结构, 区别在Foam-Agent是end-to-end composable, pipeline更优雅, retrieve能力更强.

Sonnet 3.5 + RAG + Reviewer:

| | MetaOpenFOAM | Foam-Agent |
|---|---|---|
| Basic $M_{\mathrm{exec}}$ | 0.555 | 0.836 |
| Basic $M_{\mathrm{struct}}$ | 0.883 | 0.879 |
| Basic $M_{\mathrm{file}}$ | 0.763 | 0.778 |
| Basic $M_{\mathrm{NMSE}}$ | 0.173 | 0.427 |
| Basic Success Rate | 0.136 | 0.336 |
| Adv $M_{\mathrm{exec}}$ | 0.125 | 0.625 |
| Adv $M_{\mathrm{NMSE}}$ | 0.125 | 0.406 |
| Adv Success Rate | 0.125 | 0.250 |

Foam-Agent在Basic上把Success Rate从MetaOpenFOAM的13.6%提升到33.6%, 在Advanced上从12.5%到25%, 直接翻倍. 关键在 $M_{\mathrm{NMSE}}$: MetaOpenFOAM只有0.173, Foam-Agent到0.427, 说明Foam-Agent在物理准确性上有结构性优势.

### 4.3 RAG和Reviewer的ablation (Table 2)

固定Sonnet 3.5 + Foam-Agent:

| Configuration | Basic Success Rate | Adv Success Rate |
|---|---|---|
| RAG + Reviewer | 0.336 | 0.250 |
| RAG only | 0.200 | 0.125 |
| Reviewer only | 0.245 | 0.125 |
| Zero-shot | 0.045 | 0.007 |

两个组件各自贡献大约10%的Success Rate, 组合起来有协同效应. Zero-shot基本接近0, 这说明OpenFOAM的文件生成几乎完全依赖exemplar-based learning (RAG)和iterative debugging (Reviewer), LLM自身参数里"记不住"OpenFOAM的DSL.

附录A.4.2的failure analysis很关键 (Figure 10, 11). 它把failure拆成5类:
- Inconsistent patch/patch field: boundary condition文件里定义的patch在mesh里不存在
- File not found: blockMeshDict/controlDict缺失
- Undefined keyword: flux scheme或参数没正确指定
- Numerical instability: CFL violation或scheme发散
- Geometry/mesh error: non-orthogonal cell, skewed element

RAG主要消解前3类 (配置/模板缺失), Reviewer主要消解第1和第4类 (一致性检查 + 数值稳定性反思). 这是符合直觉的: RAG给"事实grounding", Reviewer给"逻辑闭环".

### 4.4 成本分析 (Table 8)

Sonnet 3.5 + Foam-Agent (RAG+Reviewer): 平均每个case $6.56, 平均2个loop, prompt token 378k (大量来自RAG retrieve的tutorial文件). 对比GPT-4o: $0.42, 9个loop, prompt 147k. Sonnet贵15倍但Success Rate高6%, 对工业deployment算下来性价比其实更高 (每美元成功率更高). Gemini 2.5 Flash $0.01但Success Rate只有13.6%, 适合做大规模快速迭代的研究原型.

### 4.5 Prompt engineering的天花板 (A.4.4)

附录做了一个有意思的ablation: 用o3-mini做prompt optimizer, 对FoamBench Advanced的human-authored prompt做5个variant, 然后Sonnet 3.5 zero-shot测试. Success Rate从0.007升到0.012. 增量几乎可忽略. 这说明在OpenFOAM这种高度structured的DSL任务上, prompt的措辞远不如framework的architecture重要. 这是一个非常反"prompt engineering网红"的结论.

## 5. Spatial reasoning的failure mode

Figure 6展示的doubleSquare case是最让人不安的figure. Prompt清楚定义了两个方形障碍物的坐标和尺寸, 但是Foam-Agent生成的geometry和mesh完全偏离reference. LLM理解"在(2,2)放一个1×1的square"在token level上是trivial的, 但是把这个语义翻译成`blockMeshDict`里的hex (8 12 13 9 16 20 21 17) 这种vertex index assignment, 需要的是一种几何坐标系下的mental rotation + vertex enumeration能力, 这恰恰是autoregressive LLM最弱的部分.

这个问题在vision-language model里被讨论很多 (e.g., SpatialBench, SeeScan), 但是在纯text LLM上做geometry generation基本是个open problem. 这篇paper间接把这个问题顶到了CFD automation的瓶颈位置. 任何想让LLM做"给CAD画几何"的工作, 都会撞到这堵墙. 一个可能的fix是让LLM调用一个geometry DSL (e.g., OpenSCAD, FreeCAD Python API), 把geometry reasoning offload到symbolic engine上 — 这正是Cherian et al. (2024, https://arxiv.org/abs/2411.08027) 的LLMPhy思路, 把physical reasoning委托给world model. CFDLLMBench v2应该会引入这个方向.

## 6. 与相关工作的positioning

paper在Section 2里很谨慎地把自己和几个相邻benchmark区分开:

- **SciCode** (Tian et al., 2024, NeurIPS): 80个scientific coding题, 但CFD只覆盖1D heat和1D Burgers, 不足以represent CFD的算法/物理/几何complexity. CFDLLMBench的24题覆盖2D Navier-Stokes, shock tube Euler, Rayleigh-Bénard, KdV-Burgers, pipe flow eigenvalue, 量级完全不同.
- **FEABench** (Li et al., 2025, https://arxiv.org/abs/2503.06680): 用COMSOL做FEA, 商业license限制 + 范围窄. CFDLLMBench完全open-source.
- **MetaOpenFOAM / OpenFOAMGPT / Foam-Agent**: 这三个是LLM agent系统, 不是benchmark. CFDLLMBench把Foam-Agent作为reference framework来evaluate, 是一种"用最先进的agentic framework跑benchmark"的设计.
- **PaperBench** (Starace et al., 2025, https://arxiv.org/abs/2504.01848): AI复现AI paper, 工作流不同.
- **ScienceAgentBench** (Chen et al., 2024b, https://arxiv.org/abs/2410.05080): data-driven discovery workflow, 不涉及PDE数值求解.

paper的positioning是清楚的: 这是第一个holistic的CFD benchmark, 三层递进, results-oriented metric, 同时open-source可复现. 我认为它真正的贡献是**把"LLM能不能做CFD"从"它能不能讲CFD"剥离出来**, 这两件事过去被混为一谈.

## 7. 局限性

paper Section 6很诚实地承认了几个limitation, 我再加几个我看到的:

1. **Human baseline缺失**: 附录A.1只是定性说"CFD-trained graduate student能轻松解CFDCodeBench, CFD engineer能轻松解FoamBench Basic", 但是没有quantitative human study. 这是一个明显的future work, 应该招募3-5个PhD student + 2-3个industrial engineer, 测他们的Success Rate和time-on-task. 没有human baseline, 14%这个数字到底意味着"LLM离human还差10倍"还是"差100倍", 是不清楚的.

2. **几何complexity天花板低**: 所有FoamBench case的geometry都能用blockMesh生成, 即"hexahedral mesh of analytically describable shapes". 真实工业CFD 90%的effort在CAD import + snappyHexMesh/CFMesh上, 涉及STL surface mesh, boundary layer inflation, 这部分完全没覆盖. 这是合理的v1取舍, 但也是明显的scope limit.

3. **求解器覆盖偏窄**: 11个tutorial主要覆盖icoFoam, simpleFoam, pimpleFoam, rhoCentralFoam, reactingFoam, interFoam, shallowWaterFoam. 缺少overSet ( Chimera grid), foamRun ( multi-region conjugate heat transfer), LDES ( large-eddy simulation). 不过这跟paper scope匹配.

4. **Reviewer的设计不透明**: paper说Reviewer允许"trial and error", 但是Reviewer本身是另一个LLM call, 它的rubric是什么? 它怎么知道一个case file"对不对"? 如果Reviewer太松, 等于没有; 如果太严, 会reject掉本可执行的case. 附录A.4.2给出Reviewer主要消解"mismatched BC", 说明它在做lightweight syntactic consistency check, 但没做physical consistency check (e.g., "你的inlet velocity和turbulence intensity匹配吗"). 这是Foam-Agent framework的细节问题, 不完全是paper的问题.

5. **NMSE在field comparison上的局限**: CFD solution是空间field, 但是paper用逐点NMSE, 这对相位误差非常敏感 (e.g., 涡街的vortex shedding frequency偏1%, 整个field的NMSE会爆掉). 工业CFD validation通常用drag coefficient, lift coefficient, Nusselt number这些integral quantity, 或者用proper orthogonal decomposition (POD)的mode距离. 一个future direction是引入physical quantity of interest (QoI) based metric.

6. **Convergence test的cost**: $M_{\mathrm{conv}}$要跑3-4个refinement level, 算力cost不小, paper没报这个的wallclock.

## 8. 对未来方向的speculation

这篇paper最让我兴奋的不是结果, 是它暴露的问题. 我列几个值得追的方向:

**(a) Symbolic-numeric hybrid**: LLM负责high-level workflow orchestration, 把geometry generation offload到OpenSCAD/FreeCAD Python API, 把discretization scheme选择offload到sympy的PDE module, 把convergence check offload到专门的uncertainty quantification tool. 这是"LLM as conductor"而不是"LLM as everything". Foam-Agent 2.0 (https://arxiv.org/abs/2509.18178) 已经开始这个方向, 但还很初步.

**(b) Specialized CFD LLM**: 现有所有model都是general pretrain, 在CFD corpus上的exposure严重不足. 一个合理的方向是用OpenFOAM tutorial (大概几千个公开case) + CFD textbook + forum post做continued pretrain, 再用FoamBench做RLHF. 这条路在medical (Med-PaLM, https://www.nature.com/articles/s41591-024-03458-6) 和code (Code Llama) 已经被验证. 我估计1B-7B的specialized model能在FoamBench上beat Sonnet 3.5.

**(c) Verification-aware metric**: 把 $M_{\mathrm{NMSE}}$ 升级到QoI-based + POD-based + adjoint-based error estimation. 这样能区分"solution物理对但是数值噪声大"和"solution物理错".

**(d) Hierarchical benchmark**: 现在3层是平的, 但是真实CFD workflow是深的: mesh generation → mesh quality check → solver selection → BC/IC setup → discretization scheme → time step control → run → post-process. 把这7层每一层都做一个细粒度benchmark, 才能定位LLM在哪一层最弱. 现在的14%和34%是composite number, 解剖不出来瓶颈在哪.

**(e) Active learning of failure modes**: paper的Figure 10/11是failure mode histogram, 但是没有"哪种prompt特征 → 哪种failure"的可解释mapping. 一个有趣的research question: 用Foam-Agent跑1000个case, 用LLM-as-judge把failure自动分类, 再用contrastive learning学一个prompt embedding, 看failure mode在embedding space里的分布, 能不能预测新case的failure mode?

**(f) 直接对标AlphaFold 3的"domain-specific foundation model"**: CFD不像protein有central dogma那么clean的data来源, 但是OpenFOAM tutorial + GitHub repo + ANSYS Fluent case file (脱敏后) 加起来可能是几十万到百万级的labeled (problem → case file) pair, 足够训练一个CFD-specialized transformer. 这条路的关键是找一个"OpenFOAM的attention pattern" — 因为case file之间强依赖 (controlDict引用solver, solver引用schemes), 用graph transformer或set transformer可能比causal LM更合适.

## 9. 一句话总结

CFDLLMBench是一个milestone benchmark, 它用三层递进结构 + results-oriented metric把LLM在CFD上的能力逼到了真实工程的边界, 暴露的核心问题是: **conceptual recall能力 ≠ numerical reasoning能力 ≠ workflow implementation能力**, 这三层之间的gap远远大于社区预期. 14%和34%的Success Rate不是终点, 是起点, 它告诉所有想做AI4Science的researcher: 在CFD这个domain上, 现有LLM还只是一个"会念教材的实习生", 离"能独立搭case的工程师"差一个数量级.

主要references:
- Paper: https://openreview.net/forum?id=kTcH1MnkjY
- Code: https://github.com/NLR-Theseus/cfdllmbench/
- Foam-Agent: https://arxiv.org/abs/2509.18178
- MetaOpenFOAM: https://arxiv.org/abs/2407.21320
- OpenFOAMGPT: https://pubs.aip.org/aip/pfl/article/37/3/035108/3313135
- SciCode: https://proceedings.neurips.cc/paper_files/paper/2024/hash/cfdllmbench-equivalent
- Dedalus: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023068
- CFD Python 12 steps: https://github.com/barbagroup/CFDPython
- OpenFOAM v10: https://openfoam.org/version/10/
- ROUGE: https://aclanthology.org/W04-1013/
- FEABench: https://arxiv.org/abs/2503.06680
- ScienceAgentBench: https://arxiv.org/abs/2410.05080
- LLMPhy: https://arxiv.org/abs/2411.08027
- PaperBench: https://arxiv.org/abs/2504.01848
