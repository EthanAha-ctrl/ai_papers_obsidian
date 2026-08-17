---
source_pdf: FastUMI A Scalable and Hardware-Independent.pdf
paper_sha256: b357875997090c208da9dbacba641e143385e88b898c46696c9201e3b04351ea
processed_at: '2026-08-04T07:50:39-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 FastUMI

## 一句话概括

Stanford 那帮人做了一个叫 UMI 的东西，让你拿着一个带夹子的手持设备去录数据，然后把数据扔给机械臂让它学。想法很好，但用起来特别麻烦——要标定相机、调 SLAM 参数、还要买特定的夹子。上海 AI Lab 这帮人说：**这玩意儿能不能做得像乐高一样，拼上就能用？** FastUMI 就是这个"乐高版 UMI"。

---

## UMI 原版的问题在哪

先说 UMI 在干嘛。你想象一下：你想教机械臂"把杯子从桌上拿起来放到架子上"。传统做法是你拿个遥操作杆控制机械臂做一遍，机械臂记下来。但这个方法慢、贵、反直觉——你操作杆推一下机械臂动一下，跟你自己拿杯子完全是两码事。

UMI 的思路是：给你一个**手持夹子**，上面装着跟机械臂一样的相机。你就用这个手持夹子去拿杯子，该怎么做怎么做，自然得跟平时拿东西一样。相机把你看到的画面和你的手部运动都录下来。然后你把这些数据喂给一个神经网络，神经网络学会了以后，把这个模型放到真正的机械臂上，机械臂就能模仿你的动作。

这个思路特别优雅。但 UMI 原版有几个工程上的坑：

**第一个坑：硬件绑死了。** UMI 上面用了一个叫 Weiss WSG-50 的夹子，这个夹子跟整个系统深度耦合。你要换个机械臂、换个夹子？那你要重新设计机械结构、重新标定传感器、改一堆代码参数。这就像你买了个手机只能用某个牌子的充电器，换了就不行。

**第二个坑：软件太复杂。** UMI 用 GoPro 相机录视频，然后跑一个叫 VIO（Visual-Inertial Odometry）的算法来估算手在三维空间里的位置。VIO 这东西对参数特别敏感，标定起来很痛苦。而且一旦相机被挡住——比如你打开抽屉的时候手伸进去——VIO 就容易跟丢，数据就废了。

FastUMI 的作者说：这两个坑我们填了。

---

## FastUMI 怎么填坑的

### 硬件填坑：拆开、标准化、插上就能用

FastUMI 的硬件思路就三个字：**解耦**。把原来绑在一起的东西拆开，每一块都能独立替换。

**夹子的问题怎么解决？** FastUMI 设计了一种"插拔式指尖"。你想象一下，不管机械臂用的是什么夹子——xArm 的、Robotiq 的、Franka 的——FastUMI 都做了一个适配器，这个适配器**外面长一样**（所以相机看到的画面是一样的），**里面根据不同夹子定制**。你像换手机壳一样把适配器套到不同夹子上就行。他们做了五种适配器，覆盖了主流数据集里 90% 以上的夹子类型。

**相机角度怎么保持一致？** 这是最聪明的设计。UMI 的核心是"手持设备录的画面"和"机械臂上的画面"要一致，policy 才能迁移。FastUMI 定了一条特别简单的规则：**GoPro 鱼眼镜头画面的底部，对齐夹子指尖的底部**。就这么一条。不管夹子多大多小，你通过可调节的支架把相机位置调到满足这条规则就行。这就像拍照时说"脚要踩到画面底边"一样，简单但所有人都照着做就能保证一致性。

**机械臂接口怎么办？** 用了一个叫 ISO 9409 的工业标准法兰盘。这个标准在机械臂领域就像 USB 接口在电子产品领域一样通用——UR、xArm、Kuka 都支持。你把法兰盘一拧，机械臂就装好了。

### 软件填坑：把 VIO 换成黑盒传感器

这是 FastUMI 最关键的工程决策。

UMI 原版用 GoPro 录视频，然后跑 VIO 算法（ORB-SLAM3 之类的）来估算手的位置。这就像你自己拿个摄像头，后面跟一台电脑跑 SLAM，特别折腾。

FastUMI 直接换成了一个叫 **RealSense T265** 的小模块。这个模块长这样：一个香烟盒大小的小东西，里面装了两个鱼眼相机加一个 IMU，而且**VIO 算法跑在模块里面的专用芯片上**。你不用管 VIO 怎么调参，T265 直接吐给你手在三维空间里的位置和姿态，200 Hz，即插即用。

这就好比：原来你自己搭一套音响系统，要调功放、调均衡器、摆音箱位置。现在你买了个智能音箱，插上电就能放歌，音质差点但够用。

**GoPro 还在吗？** 在。但 GoPro 现在只负责录视频画面，不再负责追踪位置。FastUMI 把"录画面"和"追踪位置"这两个职能分开了——GoPro 录画面（宽视角、高分辨率），T265 追踪位置（自带 VIO、不怕参数调不好）。这样 GoPro 可以随便装，只要画面好看就行；T265 装在一个比较安全的位置，不容易被挡住。

**T265 会漂移怎么办？** 任何 VIO 系统用久了都会漂移，T265 也不例外。FastUMI 用了两招：第一招是"重启大法"——如果漂得严重了，把 T265 放到一个固定位置重启一下，内部状态归零。第二招更巧妙——在桌上放一个蓝色的 3D 打印小凹槽，作为视觉地标。T265 回到这个位置时会"认出"这个地标，自动把轨迹拉回对齐。这就像你走了很远的路，回到一个熟悉的地方会重新确认自己的方位。

---

## 数据怎么从手持变成机械臂能用的

你用手持设备录了一堆数据，里面有相机画面、有 T265 的位置。但机械臂需要的不是"手在哪儿"，是"我的关节角度应该是多少"。这中间有个转换过程。

FastUMI 输出三种数据格式，让不同的算法各取所需：

**第一种：绝对 TCP 轨迹。** TCP 就是 Tool Center Point，工具中心点，你可以理解为夹子的中心。这个格式直接告诉你"每一帧，夹子中心在机械臂坐标系下的位置和朝向"。公式大概是这样：

$$
\mathbf{p}_{\text{ee}}^{(i)} = \mathbf{p}_{b2g} + \mathbf{p}_i - \mathbf{R}_{b2g}\Delta_{c2g} + \mathbf{R}_{\text{cam}}^{(i)}\Delta_{c2g}
$$

变量意思：
- $\mathbf{p}_i$：T265 报的"相对于起始点的位移"；
- $\mathbf{p}_{b2g}$：夹子中心在机械臂基坐标系里的位置（标定一次的常数）；
- $\Delta_{c2g}$：T265 中心到夹子中心的偏移（标定一次的常数）；
- $\mathbf{R}_{b2g}$：基坐标系到夹子坐标系的旋转；
- $\mathbf{R}_{\text{cam}}^{(i)}$：当前帧相机姿态。

说白了就是把"T265 在自己坐标系里的运动"翻译成"夹子在机械臂坐标系里的位置"。翻译完了，机械臂就知道每一帧夹子应该在哪儿。

**第二种：相对 TCP 轨迹。** 不记录绝对位置，只记录"从上一帧到这一帧，夹子移动了多少"。公式：

$$
\mathbf{p}_{\text{rel}}^{(i)} = \mathbf{p}_{\text{ee}}^{(i+1)} - \mathbf{p}_{\text{ee}}^{(i)}
$$

好处是：机械臂起始位置变了也不怕，因为记录的都是相对变化。坏处是：误差会累积。

**第三种：关节轨迹。** 把 TCP 轨迹通过逆运动学（IK）转成"每个关节角度是多少"。这跟 ACT 算法的输入格式对齐。

**夹子开口宽度怎么算？** 这有个巧妙的办法。夹子两个爪上各贴一个 ArUco marker（就是那种黑白方块二维码）。相机看到这两个 marker 之间的像素距离，就知道夹子张开了多少。公式：

$$
W = \frac{d - d_{\min}}{d_{\max} - d_{\min}} \times G_{\max}
$$

变量意思：
- $d$：当前帧两个 marker 之间的像素距离；
- $d_{\min}$、$d_{\max}$：标定时记录的最小和最大像素距离；
- $G_{\max}$：夹子物理上的最大开口；
- $W$：当前物理开口宽度。

这个设计的好处是：不管夹子是圆的方的长的扁的，只要 marker 贴上，软件都能算。**软件跟硬件彻底解耦了。**

---

## 那个非平行夹子的补偿算法

这里有个很现实的问题。你用 handheld gripper 录数据的时候，夹子是平行运动的——两个爪子平着合拢。但很多机械臂的夹子不是这样的，比如 xArm 的夹子，闭合的时候爪子会往前伸大约 1 厘米。这就像你用筷子夹菜，筷子越夹越紧的时候筷子尖会往前移一点。

如果你不补偿这个差异，机械臂按照你录的轨迹执行的时候，夹子会偏离目标位置 1 厘米。拿杯子可能没事，按电饭煲按钮就按不到了。

FastUMI 的补偿算法很简单。夹子闭合时，沿着夹子指向的方向（Z 轴）往后退一点，抵消前移。公式：

$$
\mathbf{p}_{\text{ee}}^{\prime(i)} = \mathbf{p}_{\text{ee}}^{(i)} - d(i)\,\mathbf{z}_{\text{axis}}^{(i)}
$$

变量意思：
- $\mathbf{p}_{\text{ee}}^{(i)}$：原始命令位置；
- $d(i)$：当前帧的补偿距离（夹子越紧，$d(i)$ 越大）；
- $\mathbf{z}_{\text{axis}}^{(i)}$：当前帧夹子指向的方向；
- $\mathbf{p}_{\text{ee}}^{\prime(i)}$：修正后的位置——沿夹子指向**后退** $d(i)$。

这是从物理直觉出发的 closed-form 方案。不学数据、不要训练，就是根据夹子宽度算一个补偿量，沿着 Z 轴平移。

---

## 算法层面做了什么

FastUMI 的数据有个特点：相机装在夹子旁边，是第一人称视角。这跟传统的第三人称视角（相机架在旁边看整个机械臂）很不一样。好处是你能看到操作细节，坏处是机械臂大部分在画面外，神经网络不知道机械臂当前姿态，容易给出违反物理约束的关节角度。

作者针对两个主流算法做了改造。

### ACT 的两个变体

ACT 是 Stanford Aloha 那篇 paper 的算法，用 Transformer 预测关节角度。在 FastUMI 的第一人称视角下，ACT 经常预测出荒谬的关节构型——比如夹子完全翻过来的姿态。

**变体一：Smooth-ACT。** 在 Transformer decoder 后面接一个 GRU。Transformer 擅长抓全局模式，但帧间可能跳变。GRU 强制隐藏状态跨帧传播，相当于在动作序列上加了低通滤波。损失函数：

$$
\mathcal{L} = \|\hat{a} - a\|_1 + \|\hat{a}_{\text{GRU}} - a\|_1 + \lambda\,\text{KL}(\mu, \log\sigma^2)
$$

变量意思：
- $\hat{a}$：Transformer 直接输出的动作；
- $\hat{a}_{\text{GRU}}$：经过 GRU 平滑后的动作；
- $a$：真实动作；
- $\lambda$：KL 正则项权重；
- $\mu, \sigma$：VAE 隐变量的均值和标准差。

两个输出都跟 ground truth 算 L1 loss，KL 项是标准 VAE 正则。这个 trick 从语音识别借来的——GPT 系列生成 token 是离散的可以跳变，但连续控制不能跳。

**变体二：PoseACT。** 把预测目标从关节角度换成 TCP 位姿。关节角度跟具体机械臂绑死了，TCP 位姿跟机械臂无关，换机械臂时 policy 可以直接迁移。而且可以选绝对 TCP 或相对 TCP。实验发现：绝对 TCP 在"Pick Bear"这种需要高度估计的任务上更好，相对 TCP 在"Sweep Trash"这种长轨迹任务上更好。

### Diffusion Policy 加深度

Diffusion Policy 在原版 UMI 上效果很好，但 FastUMI 发现它在需要精确深度估计的任务上会崩——比如按电饭煲按钮，Z 轴差 1 厘米就按不到。

解决方案很轻量：用 **Depth Anything V2** 给每帧 RGB 图像补一个深度图。Depth Anything V2 是用几百万张图像预训练的单目深度估计模型，不需要真的深度传感器。工程上几个细节：

1. 鱼眼图像有黑边，Depth Anything V2 在黑边上会输出垃圾深度，所以先把图像裁剪到内接矩形再 resize 到 448×448；
2. 单通道深度图扩成三通道伪彩色，跟 RGB 一起送进 CLIP ViT encoder，两路 embedding concat 起来；
3. 推理时用 RTX 4090 跑 Depth Anything V2 大模型，能到 20 Hz，跟控制频率对齐。

实验结果很惊人。在"按电饭煲按钮"这个任务上，原始 DP 成功率只有 20%，加了深度后飙升到 93.33%。提升了 73 个百分点。这暗示：Diffusion Policy 本身没问题，瓶颈在感知——网络从单张 RGB 图像猜不准深度，给它深度信息就好了。

---

## 实验数据说了什么

### 追踪精度

T265 在低遮挡场景下平均误差 10.5 毫米，高遮挡场景下退化到 22 毫米。作为对比，RoboBaton MINI 在低遮挡下 15 毫米、中遮挡下 11 毫米。T265 在理想条件下精度更高，但遮挡下更脆弱。这是因为 T265 靠视觉特征做 VIO，近距离操作时夹子本身会挡住视觉特征。

### Baseline 对比

12 个任务上 ACT 和 DP 各有擅长：

- ACT 在铰链类任务（开柜子、开抽屉）和按钮按压类任务上更好——这些任务需要绝对位置精度；
- DP 在抓取放置类任务上更好——这些任务需要处理多模态动作分布（同一个杯子可以有多种抓法）；
- 两者在"Sweep Trash"这种多步长时序任务上都很差——这是 IL 领域公认的难题。

### 鱼眼视角的价值

实验里有个有意思的对比：单鱼眼第一人称视角的效果接近多视角配置。D435i 第一人称视角在"Open Container"上成功率 0%，GoPro 鱼眼第一人称视角达到 100%。鱼眼 155° 视场把末端、桌面、远场上下文全收进一帧，信息密度极高。

### 数据量 scaling

从 200 条到 800 条数据，"Pick Cup"成功率从 20% 涨到 53.33%。符合 IL 里常见的 power law scaling——数据翻 4 倍，成功率翻约 2.67 倍。这也是为什么 FastUMI 要开源 10000 条数据：让大家能在这个 power law 曲线上推得更远。

---

## 开源了什么

- 10000+ 条真实世界 demonstration，覆盖 22 个日常任务；
- HDF5 格式存储，也提供 Zarr 转换脚本；
- 硬件设计文件（3D 打印 STL）；
- 软件收集和处理 pipeline；
- ACT 和 DP 的训练和推理代码。

Project page: https://fastumi.com/

---

## 我的几点直觉判断

**第一，这篇 paper 的核心 contribution 是基础设施。** 算法部分（Smooth-ACT、PoseACT、Depth-DP）都是 incremental patch。真正有价值的是：让 IL data collection 的边际成本从"需要机械臂+SLAM 调试+数小时"降到"handheld gripper+开箱即用"。当边际成本足够低，data scaling 才有可能。

**第二，T265 停产是个隐患。** Intel 2023 年底已经停产 RealSense 产品线。作者自己也意识到，所以对比了 RoboBaton MINI 作为备选。但 MINI 输出频率只有 20 Hz，跟 T265 的 200 Hz 差了一个量级。供应链风险是实打实的。

**第三，Depth-Enhanced DP 的 +73% 提升非常 informative。** 这说明 IL 的瓶颈很多时候在 perception 而不在 policy。network 看不懂深度，你给它一个 depth model 做 prior，它就学会了。这指向一个方向：与其把 policy 做大，不如把 perception prior 做强。

**第四，没有 language grounding 是个遗憾。** 10000 条 domestic scene 数据，如果加上语言标注，可以直接训练 language-conditioned policy。拿 OpenVLA 或者 π0 做个 LoRA fine-tune baseline 会让 paper 更 forward-looking。

**第五，这个工作的定位跟 DROID、Open X-Embodiment 互补。** DROID 是 teleop-based，data quality 高但收集成本高；FastUMI 是 handheld-based，收集成本低但精度稍差。两者数据格式可以对齐，未来混合训练是个自然方向。

---

## 参考链接

- FastUMI project: https://fastumi.com/
- UMI 原版: https://arxiv.org/abs/2402.10329
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ACT / Aloha: https://tonyzhaozh.github.io/aloha/
- Depth Anything V2: https://depth-anything-v2.github.io/
- RealSense T265: https://www.intelrealsense.com/tracking-camera-t265/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.github.io/
- OpenVLA: https://openvla.github.io/
- LeRobot: https://github.com/huggingface/lerobot

---

# FastUMI 深度解读

## 1. Paper 定位与背景直觉

这篇 paper 由 Shanghai AI Lab 联合 SJTU、Bristol、Fudan、HKU 等机构完成（project page: https://fastumi.com/）。它的核心 motivation 在我看来是：把 Stanford Shuran Song 组的 **UMI**（arXiv:2402.10329, https://arxiv.org/abs/2402.10329）从"研究原型"变成"scalable + hardware-agnostic + 工程上 plug-and-play"的 ecosystem。

UMI 的核心 insight 是"human demonstration 直接用 handheld gripper 收集，再 zero-shot 迁移到 robot"，从而绕开 teleoperation 慢、expensive、counter-intuitive 的问题。但 UMI 原版有两个工程痛点：**硬件紧耦合**（Weiss WSG-50 + WSG50 内置的 SLAM-ish 标定）和 **软件 VIO pipeline 脆弱**（GoPro + OpenVINS / ORB-SLAM3）。

FastUMI 的策略是：在保留 UMI "handheld → robot transfer"思想的同时，**把感知与机械结构彻底 decouple**。这正好对应了机器人学习当下最重要的趋势——data scaling 取代 algorithm tuning——参见 Bridgedata V2（https://github.com/rail-berkeley/bridge_data_raw）、Open X-Embodiment（https://robotics-transformer-x.github.io/）、DROID（https://droid-dataset.github.io/）。

---

## 2. 硬件设计：从耦合到解耦

### 2.1 Decoupling 三个维度

FastUMI 在 hardware-centric 那一节明确写了三层 decoupling：

1. **Physical decoupling**：standardized ISO flange + plug-in fingertip，适配异质机械臂；
2. **Visual consistency**：handheld 与 robot-mounted 的 GoPro 视角必须对齐到同一规则；
3. **Operational independence**：tracking 模块自包含，不依赖外部 motion capture 或 SLAM 服务器。

### 2.2 Handheld device 的关键改动

原版 UMI 的 GoPro 既要做"observation capture"，又承担"VIO source"。FastUMI 把这两个职能拆分：

| 模块 | 原版 UMI | FastUMI |
|---|---|---|
| Observation camera | GoPro + fisheye | GoPro + fisheye（保留 155° FOV） |
| Tracking source | GoPro + VIO + SLAM | **RealSense T265**（VIO-on-chip） |
| Top cover | 出现在 GoPro 视野中（造成 hardware coupling） | 缩小并移出 fisheye 视场 |
| Fingertip marker | ArUco（粗略测量 aperture） | ArUco + 优化 marker 位置以抗畸变 |

**直觉**：T265 把 VIO 算法（来自 Intel Movidius VPU 上的 proprietary SLAM）固化为一个"black-box sensor"，所以用户无需标定相机 intrinsics、IMU-Camera extrinsics、VIO 参数；这部分原本在 UMI 中需要 30 分钟到数小时。这是工程上"模块化"对"算法端-到-端"的胜利。

参考 T265 spec：https://www.intelrealsense.com/tracking-camera-t265/

### 2.3 Robot-mounted device 设计

- **ISO flange plate ⑨**：ISO 9409-1-50-4-M6 是工业机械臂最通用末端接口，兼容 UR、xArm、Kuka IIWA 等。
- **Plug-in fingertip ⑩**：5 种 customized 内部 contour 覆盖 Open X-Embodiment 中 >90% 的 gripper（xArm、Robotiq 2F-85、Franka、Robotis 等）。外部交互点统一，policy transfer 时"看不到"机械差异。
- **Adjustable camera mount ⑥⑦⑧**：gopro mount + 两段 extension arm，blue arrow = lateral，red arrow = vertical；通过标准 male-female 接口串接（最多 3 段），实现 handheld 与 robot 视角对齐。

### 2.4 Visual Alignment 规则

> "Bottom of GoPro's fisheye lens image aligns with bottom of gripper's fingertips."

这是 FastUMI 的"反工程美学"决定——不追求 optical axis 与 TCP 完全重合，而是定义一个**几何上极简单但全局可复现的对齐 rule**。Intuition：handheld gripper 和 robot gripper 形态差异极大，硬要对齐 optical-TCP offset 反而引入新误差。把"fingertip bottom = image bottom"作为 anchor，再靠 adjustable mount 微调，这是工程取舍。

### 2.5 Dynamic Error-Compensation Algorithm（公式解析）

问题：xArm 这种 gripper 是 **non-parallel jaw**——闭合时 effective length 减少约 1 cm。直接把 handheld demo 的 TCP 复制到 robot，会引起 z 轴方向偏移。

**Stage 1**：Compensation distance

$$
d(i) = d_{\text{close}} - \frac{d_{\text{close}} - d_{\text{open}}}{W_{\max}} W(i)
$$

- $i$：frame index；
- $W(i)$：第 $i$ 帧的 gripper width（通过 ArUco 测量）；
- $W_{\max}$：gripper 物理最大开口；
- $d_{\text{close}}$：完全闭合时的最大补偿距离；
- $d_{\text{open}}$：完全打开时的最小补偿距离；
- $d(i)$：当前帧应补偿的距离，与 $W(i)$ 线性负相关——夹得越紧，补偿越多。

**Stage 2**：Pose correction

$$
\mathbf{z}_{\text{axis}}^{(i)} = \mathbf{R}_{\text{ee}}^{(i)} \hat{\mathbf{e}}_z, \quad \hat{\mathbf{e}}_z = [0,0,1]^\top
$$

- $\mathbf{R}_{\text{ee}}^{(i)}$：第 $i$ 帧 TCP 的旋转矩阵（base frame 下）；
- $\mathbf{z}_{\text{axis}}^{(i)}$：TCP local Z-axis 在 base frame 中的方向向量（即 gripper pointing direction）。

$$
\mathbf{p}_{\text{ee}}^{\prime(i)} = \mathbf{p}_{\text{ee}}^{(i)} - d(i)\, \mathbf{z}_{\text{axis}}^{(i)}
$$

- $\mathbf{p}_{\text{ee}}^{(i)}$：原始 commanded TCP 位置；
- 减号代表沿 TCP 指向**后退** $d(i)$，抵消 jaw 闭合造成的 forward 位移。

最终通过 IK 求关节角：

$$
\boldsymbol{\theta}^{(i)} = \text{IK}\left(\mathbf{p}_{\text{ee}}^{\prime(i)}, \mathbf{R}_{\text{ee}}^{(i)}\right)
$$

**Intuition**：这是一个 closed-form 的 kinematic hack——只补偿 1 个 DoF（沿 TCP Z 的平移），却解决了 80% 的 gripper 闭合偏移问题。比训练一个数据驱动的 gripper compensator 简单可控得多。

---

## 3. Software Pipeline 细节

### 3.1 三 ROS node 架构

- `camera_node`：GoPro 1920×1080@60 fps；
- `tracking_node`：T265 200 Hz 输出 $(x, y, z, q_x, q_y, q_z, q_w)$（translation + quaternion）；
- `storage_node`：HDF5 sync 写入。

### 3.2 Multi-sensor Synchronization

T265 是 200 Hz，GoPro 是 60 fps，GCD = 20 Hz。策略：

- Unified ROS clock 统一打时间戳；
- 每 sensor 独立 thread-safe queue 缓冲；
- Subsample：保留每 3 帧 GoPro 中的 1 帧，配对 temporally nearest T265 pose；
- Sub-millisecond offset（小于 T265 1/200 s = 5 ms 的一半）。

这种"GCD subsample + nearest-neighbor pair"在 robotics 数据收集中常见（参 DROID 的同步策略 https://droid-dataset.github.io/）。

### 3.3 T265 Drift 处理

两条策略：

1. **Reinitialization**：让 T265 静止在 pre-defined reference pose，重启内部 state machine；
2. **Loop closure**：在桌面放一个蓝色 3D-printed groove 作 visual anchor，T265 回到此区域时重新对齐 trajectory 到初始 reference（RVIZ 中可视化）。

注意：loop closure 在原版 VIO SLAM 中是自动的（ORB-SLAM3 的 atlas 机制，https://github.com/UZ-SLAMLab/ORB_SLAM3），FastUMI 改用 T265 后失去了这一特性，所以需要"人工 loop closure"——这是 trade-off：易用性换灵活性。

### 3.4 Data Quality Gate

T265 输出 4 个离散 confidence：Failed / Low / Medium / High。FastUMI 强制：

- 环境验证：≥95% 的 sample poses 必须 High confidence；
- 录制时：低 confidence pose 被 drop 并 interpolate；
- 用户定义 velocity / acceleration / orientation 阈值，检测 abrupt transition。

**直觉**：这是 paper 中比较"软"的部分——作者承认"no existing work fully quantifies what constitutes ideal data quality"，只能用 proxy metric。这里其实是 Imitation Learning 的一个 open problem，参 Data Quality in Imitation Learning（NeurIPS 2023, https://proceedings.neurips.cc/paper_files/paper/2023/file/fe692980c5d9732cf153ce27947653a7-Paper-Conference.pdf）。

### 3.5 三种数据表示

FastUMI 输出 3 种 trajectory 让下游 IL 算法选择：

#### Absolute TCP trajectory

$$
\mathbf{p}_{\text{cam}}^{(i)} = \mathbf{p}_{b2g} + \mathbf{p}_i - \mathbf{R}_{b2g} \Delta_{c2g}
$$

$$
\mathbf{R}_{\text{cam}}^{(i)} = \mathbf{R}_{\text{base}} \cdot \mathbf{R}_i
$$

- $\mathbf{p}_i, \mathbf{R}_i$：T265 输出的相对初始 pose 的 transform；
- $\mathbf{p}_{b2g}, \mathbf{R}_{b2g}$：gripper center 在 base frame 的 pose（标定常数）；
- $\Delta_{c2g}$：camera-to-gripper offset（标定常数）；
- $\mathbf{R}_{\text{base}}$：T265 初始 frame 到 base frame 的旋转（标定一次）。

然后加入 camera-to-gripper offset 得 TCP pose：

$$
\mathbf{p}_{\text{ee}}^{(i)} = \mathbf{p}_{\text{cam}}^{(i)} + \mathbf{R}_{\text{cam}}^{(i)} \Delta_{c2g}, \quad \mathbf{R}_{\text{ee}}^{(i)} = \mathbf{R}_{\text{cam}}^{(i)}
$$

#### Relative TCP trajectory

$$
\mathbf{p}_{\text{rel}}^{(i)} = \mathbf{p}_{\text{ee}}^{(i+1)} - \mathbf{p}_{\text{ee}}^{(i)}
$$

$$
\mathbf{R}_{\text{rel}}^{(i)} = \left(\mathbf{R}_{\text{ee}}^{(i)}\right)^{-1} \cdot \mathbf{R}_{\text{ee}}^{(i+1)}
$$

- 帧间相对 transform，去掉全局 reference 依赖。
- Intuition：当 base pose 变化时，relative 仍然有效，便于 cross-environment 泛化。

#### Absolute Joint Trajectory

对每个 absolute TCP pose 跑 IK，用上一帧 solution 作 initial guess 保证 continuity。如果 URDF 只到 flange，需要补 flange-to-gripper offset。

#### Continuous Gripper Width

$$
W = \frac{d - d_{\min}}{d_{\max} - d_{\min}} \times G_{\max}
$$

- $d$：当前帧 ArUco markers 的像素距离；
- $d_{\min}, d_{\max}$：标定的最小/最大像素距离；
- $G_{\max}$：夹爪物理最大开口；
- $W$：当前物理宽度。

如果只检测到一个 marker，按 gripper 中心镜像；如果都没检测到，impute。

---

## 4. 算法层 Adaptations

### 4.1 FastUMI 数据的特殊性

| 特性 | 后果 |
|---|---|
| Close-up FPV | 机械臂大部分不可见，policy 难以推断 kinematic feasibility |
| Variable geometry | 跨 arm 配置 + base frame 不同 → heterogeneous distribution |
| Limited depth | 单视角 fisheye 缺深度，pick/insertion 任务容易失败 |

### 4.2 Smooth-ACT

原版 ACT（arXiv:2304.13705, https://twitter-act.github.io/）是 CVAE + Transformer，输出 absolute joint action chunk。问题：FPV 下预测出"非法 joint config"——例如 Pick Bear 任务中预测出 fully inverted gripper posture。

Smooth-ACT 在 Transformer decoder 后面挂一个 GRU 层做 local temporal smoothing：

$$
\mathcal{L} = \|\hat{a} - a\|_1 + \|\hat{a}_{\text{GRU}} - a\|_1 + \lambda\,\text{KL}(\mu, \log\sigma^2)
$$

- $\hat{a}$：Transformer decoder 输出；
- $\hat{a}_{\text{GRU}}$：GRU 输出（接在 decoder 之后）；
- $a$：ground truth action chunk；
- $\mu, \sigma$：CVAE latent 的 mean 和 std；
- $\lambda$：KL 权重；
- $\text{KL}(\mu, \log\sigma^2)$：标准 VAE 正则项，鼓励 latent 接近 $\mathcal{N}(0, I)$。

**Intuition**：Transformer 学全局 attention pattern，但 FPV 下 attention 容易被局部 visual cue 误导，导致帧间跳变；GRU 强制 hidden state 跨帧传播，相当于在 action space 上加了一个 low-pass filter。这是从 language modeling 借来的 trick（GPT + GRU 在 speech 识别中类似组合）。

### 4.3 PoseACT

把 action 表示从 joint angle 换成 TCP pose（支持 absolute 和 relative 两种）。

- **Platform independence**：TCP pose 不依赖具体 arm kinematics，跨平台迁移更容易；
- **Numerical stability**：relative trajectory 数值范围小，gradient 更友好。

Inference 时用机器人 URDF 跑 IK 把 TCP 映射回 joint angle。

**Intuition**：这与 Diffusion Policy 在原 UMI 中的设计哲学一致——预测 TCP 而非 joint。但 ACT 原本设计是 joint-space，FastUMI 把它改成 TCP-space 是为了和 UMI ecosystem 对齐。

### 4.4 Depth-Enhanced Diffusion Policy

Diffusion Policy（arXiv:2210.02911, https://diffusion-policy.cs.columbia.edu/）在原 UMI 中用 relative TCP + latency matching 已经很强。FastUMI 发现它在 depth-sensitive 任务（Open Drawer、Pick Lid、Open Ricecooker）上崩——button press 类任务对 z 轴 1 cm 误差都敏感。

**Solution**：用 **Depth Anything V2**（arXiv:2406.09414, https://depth-anything-v2.github.io/）做 monocular depth estimation 作 post-processing。

工程细节：
1. Fisheye image 中有大块 black margin，Depth Anything V2 在 black 区域产生 garbage depth → crop 到 inscribed rectangle 再 resize 到 448×448；
2. RGB 也 crop + resize 到 448×448 保持对齐；
3. 单通道 depth 扩展到 3 通道 pseudo-color，与 RGB 并行通过 ViT-Base/16 CLIP encoder（arXiv:2103.00020, https://arxiv.org/abs/2103.00020）；
4. Embedding concat 后喂给 DP；
5. Inference：RTX 4090 上 Depth Anything V2 Large 模型跑到 20 Hz，与 control frequency 对齐。

**Intuition**：这是用 large pre-trained monocular depth model 做"zero-shot sensor augmentation"。比装 RealSense D435（窄 FOV）或 ZED（重 + 贵）便宜得多。代价：depth 是 estimated 不是 measured，绝对值不准；但 DP 用的是 relative depth cue，所以能用。

### 4.5 Algorithm Enhancement 实验结果

| Task | Original DP | Depth-Enhanced DP | 提升 |
|---|---|---|---|
| Pick Lid | 53.33% | 80.00% | +26.67% |
| Open Ricecooker | 20.00% | 93.33% | +73.33% |

| Task | ACT | Smooth-ACT | PoseACT(abs) | PoseACT(rel) |
|---|---|---|---|---|
| Pick Bear | 20% | 60% | 80% | 73.33% |
| Sweep Trash | 6.67% | 26.67% | 53.33% | 60% |

Open Ricecooker +73.33% 这个数字非常震撼——本质上 depth cue 把一个本来无法学的任务变成可学。这暗示：**IL bottleneck 在 perception 而非 policy**。

---

## 5. 实验数据深度解读

### 5.1 Pose Tracking 精度（Table I）

| Task | T265 平均 (mm) | MINI 平均 (mm) |
|---|---|---|
| Pick Cup（低遮挡） | 10.5 | 15.2 |
| Open Container（中遮挡） | 17.7 | 11.2 |
| Rearrange Coke（高遮挡） | 22.7 | – |

**Pattern**：T265 在低遮挡下精度高（接近其 spec 的 1% trajectory length），高遮挡下退化严重。MINI 反而更稳。这是因为 T265 的 VIO 在 close-range 时容易被 gripper 挡住 optical feature，IMU drift 短期无法纠正。

**Intuition**：UMI 原版用 GoPro + OpenVINS 也是这个问题，但 GoPro FOV 更大，部分遮挡影响小。FastUMI 改 T265 后 FOV 变小（T265 是 163° 但 fisheye 畸变更大），更易遮挡。

Fig. 11 的曲线很直观：Pick Cup 任务的 VIO error 在中间有两个 peak，对应 gripper 接近桌面遮挡视觉特征；末尾回原位后 loop closure 让 error 回到初始水平。

### 5.2 ACT vs DP Baseline（Table II）

12 tasks 的 success rate：

| 类型 | DP 平均 | ACT 平均 |
|---|---|---|
| Hinged (4 task) | 66.67% | 85.00% |
| Pick-Place (6 task) | 74.44% | 57.78% |
| Pick-Push (1) | 46.67% | 6.67% |
| Button Press (1) | 20.00% | 80.00% |

**Pattern**：
- ACT 强在 hinged 和 button press——这些任务需要绝对位置精度（IK 起点稳定）；
- DP 强在 pick-place 和 pick-push——这些任务需要 multimodal action distribution（多个 grasp 候选）；
- Sweep Trash 双方都差——multi-step long-horizon 任务是 IL 的硬骨头。

### 5.3 Camera Setup 影响（Table V）

| Setup | Pick Bear | Open Container |
|---|---|---|
| D435i FPV only | 0% | 0% |
| GoPro flat lens FPV | 6.67% | 93.33% |
| D435i FPV+TPV | 86.67% | 100% |
| GoPro fisheye FPV | 80% | 100% |

**惊人结果**：单 fisheye FPV 接近 multi-view 水平！Intuition：fisheye 155° FOV 把"末端 + 桌面 + 远场上下文"全收入一帧，对 single-image policy 来说信息密度极高。这是 UMI 设计核心 insight 之一，FastUMI 实验再次验证。

### 5.4 Data Scaling（Table VI）

| Data size | Pick Cup SR |
|---|---|
| 200 | 20% |
| 400 | 26.67% |
| 800 | 53.33% |

从 200 到 800 翻 4 倍，success rate 翻 2.67 倍——符合 IL 中"近似 power law scaling"的常见经验（参 Scaling Laws for Imitation Learning, https://arxiv.org/abs/2307.00090 之类的近期工作）。FastUMI 开源 10000+ trajectories 的 dataset，目的就是让社区能在这个 power law 曲线上推得更远。

---

## 6. Dataset 生态

### 6.1 规模

- 10,000+ demonstrations；
- 22 tasks（domestic scene）；
- 19 object categories；
- 12 manipulation skills（pick, open, sweep, press, etc.）；
- 单 demonstration 长 6–12 秒，多数 9 秒；
- 5 个 operators × 3 个 devices 收集，保证 user diversity。

### 6.2 Storage 格式

HDF5 schema：

```
episode_<idx>.hdf5
├── observations/
│   ├── images/
│   │   └── <camera_name_1>  # uint8 (num_frames, 1920, 1080, 3)
│   └── qpos                 # (num_timesteps, 7) = [x, y, z, qx, qy, qz, qw]
├── action                   # mirrors qpos
└── attributes/
    └── sim = False          # 标识 real-world
```

也提供 HDF5 → Zarr 转换 script（https://zarr.readthedocs.io/），Zarr 在 chunked、compressed、parallel access 上更友好，参 LeRobot dataset 格式（https://github.com/huggingface/lerobot）。

---

## 7. Limitations 与 Open Questions

paper 自述 limitation：

1. **Limited sensing modalities**——只有 RGB + pose，无 tactile / force。这对 fragile object 任务是硬伤；
2. **Restricted robot compatibility**——单臂/双臂支持，移动 manipulation + whole-body control 未覆盖；
3. **Wired data transfer**——限制现场部署。

我个人额外想到的几个 limitation（build your intuition）：

- **T265 已停产**：Intel 2023 年底宣布 discontinued RealSense 产品线（https://www.intelrealsense.com/）。FastUMI 自己也意识到这点，所以 section X-A 对照 RoboBaton MINI 作 backup。这是 supply chain risk。
- **Depth Anything V2 是 zero-shot estimated depth**：不是 metric depth，对 insertion 类任务仍可能不够。可考虑用 Metric3D v2（https://iglue-ai.github.io/Metric3Dv2/）或 ZoeDepth（https://github.com/isl-org/ZoeDepth）替代。
- **No language grounding**：FastUMI dataset 没有语言条件标注，无法直接训练 language-conditioned policy（参 RT-2, https://robotics-transformer2.github.io/；Octo, https://octo-models.github.io/）。把 VLM caption 接到 GoPro frame 上是个低成本扩展。
- **Single arm only**：UMI 原版支持 bimanual（两个 handheld gripper）。FastUMI paper 没明确写 bimanual 验证，但 hardware design 应该可以扩展——值得社区复现。
- **No active compliance**：所有 trajectory 都是 open-loop playback，对环境 perturbation 鲁棒性差。可结合 Diffusion Policy 的 force-aware variant（参 ETH 的 Force-DP, https://arxiv.org/abs/2405.07287）。

---

## 8. 与相关工作的定位

### 8.1 与原 UMI 对比

| 维度 | UMI | FastUMI |
|---|---|---|
| Tracking | GoPro + OpenVINS | T265 (VIO on-chip) |
| Gripper | WSG-50 强耦合 | 5 种 plug-in fingertip |
| Deployment time | 数小时标定 | plug-and-play 分钟级 |
| Dataset | 论文未开源大规模 | 10k+ trajectories open-source |
| Algorithms | DP | DP + ACT + 变体 |

### 8.2 与 DROID / Open-X 对比

DROID（https://droid-dataset.github.io/）是 teleop-based，76k trajectories，多平台多机臂。FastUMI 与之互补：
- DROID 数据更"真实 teleop"，但收集成本高；
- FastUMI 收集成本低（handheld），可大规模 crowd-source；
- 两者数据格式可对齐（都用 HDF5 + Zarr）。

Open X-Embodiment（https://robotics-transformer-x.github.io/）是 meta-dataset。FastUMI 的 gripper fingertip 设计直接覆盖 Open X 中 90% 的 gripper，意味着 FastUMI 收集的 data 可作为 Open X 的"高 density 补充"。

### 8.3 与 Aloha / Mobile Aloha 对比

Aloha（https://tonyzhaozh.github.io/aloha/）和 Mobile Aloha（https://mobile-aloha.github.io/）用 leader-follower teleop，data quality 极高但成本高（4 个机械臂 + SLAM）。
- Aloha 优势：精确的 bimanual 同步；
- FastUMI 优势：scalable、低门槛，适合 crowd-source collection。

### 8.4 与 Bunny-VisionPro / ARCap 对比

Bunny-VisionPro（arXiv:2407.03162, https://apple-vision-pro-teleop.github.io/）和 ARCap（arXiv:2410.08464）用 Apple Vision Pro 做 hand tracking teleop，dexterous 任务强但 cost 高。FastUMI 走另一极端——硬件最简、靠 data scale 弥补精度。

---

## 9. 我对这个工作的直觉评估

从你（Karpathy）的视角看，FastUMI 的 contribution 主要是 **data infrastructure** 而非 algorithm。它的核心价值在于：

1. **降低了 IL data collection 的"activation energy"**——从"需要机械臂 + SLAM 调试 + 数小时"降到"handheld gripper + 开箱即用"；
2. **Dataset 是真正的 asset**——10000+ trajectories 在 domestic scene 上是当下最 diverse 的 UMI-style dataset；
3. **算法 adaptations 是工程性 patch**——Smooth-ACT、PoseACT、Depth-Enhanced DP 都是 incremental，但它们暴露了一个重要 signal：**IL 失败更多是 perception bottleneck 而非 policy bottleneck**。这指向未来工作的方向——把更多 perception prior（depth, tactile, language）inject 到 IL pipeline。

**为什么不彻底 end-to-end？** paper 没讨论这点。但 my intuition 是：end-to-end VLA（如 RT-2, OpenVLA, https://openvla.github.io/）需要 100k+ data 才能收敛，而 FastUMI 在 200–800 demonstration 区间内显示 policy 仍能学。这是"specialized small policy"vs"generalist large policy"的 trade-off。在 data scale 还没到 VLA 临界点时，FastUMI 的 modular 设计是合理的工程选择。

如果让我给 paper 提一条建议：**加一个 zero-shot VLM fine-tune baseline**——用 OpenVLA 或 π0（https://www.physicalintelligence.company/blog/pi0）在 FastUMI 10000 trajectories 上 LoRA fine-tune，看是否超过 ACT + DP baseline。这会让 paper 的 algorithm section 更 forward-looking。

---

## 10. Web Reference 汇总

- **FastUMI project page**: https://fastumi.com/
- **UMI (原版)**: https://arxiv.org/abs/2402.10329, https://umi-gripper.com/
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **ACT / Aloha**: https://tonyzhaozh.github.io/aloha/
- **Depth Anything V2**: https://depth-anything-v2.github.io/
- **RealSense T265 (停产后文档)**: https://www.intelrealsense.com/tracking-camera-t265/
- **RoboBaton MINI**: https://www.robobaton.com/（替代 T265 的方案）
- **Open X-Embodiment**: https://robotics-transformer-x.github.io/
- **DROID**: https://droid-dataset.github.io/
- **Bridgedata V2**: https://github.com/rail-berkeley/bridge_data_raw
- **Mobile Aloha**: https://mobile-aloha.github.io/
- **Bunny-VisionPro**: https://apple-vision-pro-teleop.github.io/
- **OpenVLA**: https://openvla.github.io/
- **LeRobot (Zarr/HDF5 reference)**: https://github.com/huggingface/lerobot
- **ORB-SLAM3 (原 UMI 用过)**: https://github.com/UZ-SLAMLab/ORB_SLAM3
- **evo (轨迹误差评估工具)**: https://github.com/MichaelGrupp/evo
- **CLIP ViT encoder**: https://arxiv.org/abs/2103.00020
- **Data Quality in IL (NeurIPS 2023)**: https://proceedings.neurips.cc/paper_files/paper/2023/file/fe692980c5d9732cf153ce27947653a7-Paper-Conference.pdf

---

## 11. 一句话总结

FastUMI 把 UMI 从"研究 demo"重构为"scalable data factory"：硬件上用 T265 + plug-in fingertip 实现跨平台 decoupling，软件上用 modular pipeline + 算法 patch（Smooth-ACT / PoseACT / Depth-DP）适配 FPV 数据特征，开源 10k+ demonstration 作为社区基础设施。它的真实价值在于让 IL data collection 的 marginal cost 降到可 crowd-source 的水平——这是机器人学习迈向"data-driven scaling era"的关键一步。
