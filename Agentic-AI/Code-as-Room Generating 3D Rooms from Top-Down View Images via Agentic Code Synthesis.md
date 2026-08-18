---
source_pdf: Code-as-Room Generating 3D Rooms from Top-Down View Images via Agentic
  Code Synthesis.pdf
paper_sha256: 3cf9bfda35ba662bee2c9b18ea774ba84ac44a20c40813ed88129312d88889c4
processed_at: '2026-08-18T03:36:02-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇Paper

## 一句话总结

给一张俯视图（就像房产中介给你看的户型图那种），自动用Blender代码"盖"出一个完整的3D房间。

---

## 之前的问题出在哪

想象你是个室内设计师，客户给你一张俯视平面图，让你照着摆家具。你脑子里会自然干几件事：

1. 先看图里有啥——床、桌子、柜子、窗户在哪
2. 在脑子里定个大位置——床靠墙，桌子在中间
3. 再想细节——床啥材质，桌子啥颜色
4. 最后搞灯光和氛围

但如果你让AI直接一口气干完所有事，它大概率会崩。原因是：

- **给文字描述**：你说"放张床"，AI不知道放哪、放多大、朝哪边。信息太少。
- **给图片让agent自己搞**：像VIGA那种方法，让AI自己反复"生成-检查-修改"，听起来很美，但实际跑起来AI容易陷入死循环，改了A坏了B，修了B又坏了A，最后卡死或者跑出一个垃圾结果。

这就像让一个人一次性把整栋楼盖完，不给中间检查的机会，大概率会出乱子。

---

## 他们怎么做的

核心思路特别朴素：**把大任务拆成小任务，每步干一件事，干完了记下来，下一步接着干。**

具体分了几步：

**第一步：看图说话**
AI先看俯视图，把能看到的东西分成三类：
- 大件家具（床、沙发、柜子）
- 小物件（杯子、书、台灯）
- 房间结构（墙、门、窗）

**第二步：搭骨架**
先不管长啥样，只管"放哪"。用简单的方块代表每件家具，摆个大位置。然后渲染一下看看跟原图像不像，不像就让AI自己批评自己，改一改，最多改5次。

这一步特别关键——**先定位置再管长相**，避免了一锅炖的混乱。

**第三步：填细节**
位置定好了，开始想每件家具长啥样——什么材质、什么颜色、什么功能。然后用Blender代码把方块替换成有形状的东西。

小物件（杯子、书这种）代码不好画，就从asset library里找现成的顶上。

**第四步：搞装修**
最后上材质、贴图、灯光。地板加木纹，墙壁刷颜色，窗户开光，整氛围。

---

## 为什么这样做好

三个关键设计：

### 1. 分阶段，别一锅炖

每阶段只干一件事，输出一个明确的artifact（比如layout code、object profile），下一阶段拿这个接着干。这就像工地施工——先打地基，再立框架，再砌墙，最后装修。你不会让工人们同时干所有事。

### 2. 有个"记忆本子"

所有阶段的东西都记在一个共享memory里，后面要用前面就翻本子，不用让AI重新猜。这解决了一个很实际的问题：long workflow里AI容易忘事。比如第五步想用第二步的信息，但如果不记下来，AI可能自己编一个不一样的。

Paper里做了ablation——去掉memory，layout IoU从73.2%掉到58.0%，说明这玩意确实管用。

### 3. 视觉反馈循环，但别太多

layout阶段会渲染出来给AI看，让它自己对比原图找问题。但paper发现迭代次数不是越多越好：0次很差，5次最好，10次反而开始变差——改太多了会"跑偏"。

这个发现挺直觉的：改代码改多了，AI会自己加戏，越改越离谱。

---

## 实验结果说啥

几个takeaway：

1. **Gemini配上这个harness最稳**：Gemini3.1-Pro + CaR 在几乎所有指标上最好，similarity 9.0，usability 8.0。
2. **GPT-5.5单独跑还行，配上harness反而没那么好**：因为GPT在multi-stage workflow里completion rate只有71.1%，经常中途崩。这说明harness设计对base model有依赖性。
3. **VIGA被全面碾压**：特别是layout保留和usability上差一大截。VIGA的lighting还行（8.0），但similarity只有5.5，说明它基本是在自己瞎编场景，没真正follow input image。

---

## 最直觉的总结

把这件事类比成做饭：

- 之前的text方法：你跟厨师说"做顿饭"，他做啥你吃啥，没法控制。
- 之前的image agent方法：你给厨师看张图，说"照着做"，然后厨师自己闷头做，做坏了你也不知道，最后端上来一坨。
- CaR的方法：你给厨师看图，但规定他必须按菜谱来——先备料，再切菜，再炒，再调味。每步做完记下来，下一步照着上一步的来。厨师不会忘事，也不会跑偏。

本质上就是用工程结构来约束AI的自由度，让它干该干的事，别乱来。

---

如果你想深入某个particular stage的公式或者具体prompt设计，可以继续问。

---

# Code-as-Room: 深度技术解析

## 1. Paper 核心问题与动机

这篇paper tackle的是一个相当实际的problem：给定一张top-down view的房间image（俯视图），自动生成一个完整的、可编辑的、可在Blender中执行的3D room scene code。这个problem的难点在于需要同时infer room structure、object layout、object geometry、material、lighting，而这些信息从单张top-down image中是高度under-determined的。

让我先build your intuition关于为什么这是一个hard problem：

- **Text-to-3D scene** 的limitation：text description无法精确specify object counts、precise locations、detailed orientations。比如"在卧室中央放一张床"这样的描述太vague。
- **Image-conditioned agents** 的limitation：像VIGA这样的agent在处理perspective image时表现不错，但当naively extend到top-down view时，会陷入infinite loop，generation不稳定。

CaR的核心insight是：**通过structured execution harness来organize MLLM agent的工作流程**，将一个monolithic的generation task decompose成coarse-to-fine的multi-stage pipeline，并通过cross-stage memory来避免long-horizon agent workflow中的context forgetting。

Reference: 
- Project page: https://code-as-room.github.io/
- VIGA paper: https://arxiv.org/abs/2601.11109
- Holodeck (CVPR 2024): https://arxiv.org/abs/2312.09067
- ProcTHOR (NeurIPS 2022): https://arxiv.org/abs/2206.06994

---

##
