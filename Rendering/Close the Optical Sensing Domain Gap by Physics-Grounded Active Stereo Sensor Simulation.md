---
source_pdf: Close the Optical Sensing Domain Gap by Physics-Grounded Active Stereo
  Sensor Simulation.pdf
paper_sha256: d1de36a9d9871864e2421d652cdb617168aa517ee2b46a4cc100642104d550f0
processed_at: '2026-08-03T15:54:04-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话版本

你训练robot的perception model，如果在sim里用的depth map太"干净"太"完美"，到real world就废了——因为real depth sensor会在transparent、metallic这些东西上拍出一堆乱七八糟的noise和hole。这篇paper说：**别搞那些花里胡哨的GAN domain adaptation了，直接从physics出发，把real depth sensor从头到尾的工作机制在sim里复刻一遍**，出来的simulated depth就自带正确的noise pattern，训出来的model直接能用在real world。

---

## 为什么这是个真问题

想象你拿Intel RealSense D415拍一个透明玻璃杯。你会得到什么？一堆garbage depth——杯子中间全是hole，边缘全是noise，根本看不到杯子的shape。

为什么？D415是个**active stereovision sensor**。它的工作原理是：先往场景里投射一个IR dot pattern（就像撒一把小点），然后用两个IR camera从不同角度拍，再像人眼stereo vision一样做stereo matching算depth。

问题在于：玻璃杯是transparent的，IR dot pattern打上去直接refract穿过去了，根本不reflect回camera。所以camera看到的IR image里，杯子位置是空的——stereo matching就fail了，depth就出hole。

这是个**material-dependent**的error pattern。Diffuse material（比如木头）depth很准，glossy metal会specular reflection把dot弹飞，transparent material直接让dot穿透。

---

## 之前的approach为什么不行

### Approach 1: Clean depth（最naive）

直接用renderer的depth buffer，那depth干净得像CAD model的z-coordinate。

问题：real sensor的depth有hole有noise，sim里完全没有。你train的model从来没见过hole，到real world一看到hole就懵了。

### Approach 2: Rasterization + stereo matching（DepthSynth那类）

用rasterization渲染两个view的image，做stereo matching得到depth。

问题：**rasterization根本render不出refraction和specular reflection**。Rasterization的数学本质是projection——把三角形project到屏幕上，谁挡谁就画谁。它不知道光会折射、会镜面反射。所以你用rasterization render一个玻璃杯的IR image，杯子后面该有的东西都被杯子挡住了（因为rasterizer不知道光会穿过去），这跟real sensor看到的完全不一样。

### Approach 3: GAN-based domain adaptation（PixelDA那类）

用GAN把sim image"翻译"成real-looking image。

问题：GAN是在**pixel distribution层面**做matching，它不知道physics。它可能让杯子区域的pixel看起来像real sensor的hole pattern，但这个hole的**位置和shape跟实际optical physics决定的hole完全无关**。更糟的是，GAN可能introduce geometric distortion——本来sharp的边缘变模糊了，这会corrupt掉geometry信息。PVN3D这种用PointNet++提取geometry的算法直接崩了（Table IV里PixelDA只有0.00%）。

### Approach 4: Domain randomization（LearnAug那类）

往sim data上加各种random noise（Gaussian、salt & pepper等），希望model学到robustness。

问题：simple noise组合不出complex material-dependent pattern。你加Gaussian noise，那整个image均匀加噪；real sensor的noise是structured的——杯子中间是hole，金属表面是specular flare，这些你用simple augmentation根本拼不出来。

### Approach 5: Differentiable depth simulation（DDS那类）

把整个rendering + stereo matching pipeline做成differentiable，用real depth当supervision来optimize rendering参数。

问题：differentiable ray tracing现在很慢，而且他们用的那个differentiable renderer不支持transparent material（透明的都render不了，那还sim个啥）。另外differentiable stereo matching为了可微分牺牲了capability——没法用real sensor里常用的directional cost calculation。

---

## 这篇paper的key insight

**别绕了，直接用physics。**

Real depth sensor的工作流程是：
1. IR projector投射dot pattern
2. IR dot打到物体表面，根据material发生reflection/refraction/absorption
3. 两个IR camera拍到stereo image pair
4. Stereo matching算depth

那就在sim里一模一样地走一遍：
1. Simulate一个textured spot light（texture就是IR dot pattern）
2. 用**ray tracing**模拟IR dot跟material的interaction（ray tracing能正确处理refraction、specular reflection）
3. 用simulated noise model corrupt image
4. 用GPU加速的stereo matching算depth

这样出来的depth map，**天然就带正确的material-dependent error pattern**——因为它是physics决定的，不是你hardcode或learn出来的。

---

## 怎么做到的——拆解每个component

### Component 1: Material Acquisition（怎么知道物体是什么material）

你有CAD model和base color（texture），但不知道roughness、metallic、transmission这些PBR参数。怎么办？

**Key trick: 用multispectral loss同时match RGB和IR image。**

你在real world拍一组multispectral image（RGB + IR），然后在sim里render同样的view，optimize material参数让rendered image跟real image match。

Loss长这样：

$$L = L_{\text{RGB}} + \lambda L_{\text{IR}}$$

$$L_{\text{RGB}} = L_2(I_{\text{RGB}}^{\text{sim}}, I_{\text{RGB}}^{\text{real}}) + L_{\text{percept}}(I_{\text{RGB}}^{\text{sim}}, I_{\text{RGB}}^{\text{real}})$$

变量意思：
- $I_{\text{RGB}}^{\text{sim}}$：sim里render的RGB image
- $I_{\text{RGB}}^{\text{real}}$：real camera拍的RGB image
- $L_2$：pixel-wise的L2距离
- $L_{\text{percept}}$：perceptual loss，提取AlexNet feature之后算L2距离
- $\lambda$：balance RGB和IR的weight

为什么同时用RGB和IR？因为RGB主要constrain visible spectrum的material表现，IR constrain infrared spectrum的表现。一个material可能RGB看起来match了但IR不对——simulated depth sensor依赖的是IR，所以IR必须match。

为什么加perceptual loss？Plain L2对brightness、exposure太敏感。你real拍的image可能exposure偏高，sim里render的exposure是另一个值——plain L2会因为这种mismatch给出错误gradient。Perceptual loss在AlexNet feature space计算，high-level feature对brightness/exposure shift更robust。

实际操作是grid search：先coarse（每个参数10个grid）再fine（在best附近再10个grid）。一个object大概1小时搞定。

### Component 2: Kuafu Renderer（ray tracing + textured light）

这是工程上最难的部分。现有ray tracer（Blender Cycles、PBRT）要么太慢，要么不支持textured light。

**Textured light是什么？** 就是light source本身带一个texture map——从light发出的不同方向的光，强度不一样。IR projector投射的dot pattern就是这样的：dot位置光强，dot之间光弱。

如果没有textured light support，你只能模拟uniform illumination的light，那render出来的IR image是均匀照亮的，根本没有dot pattern。没有dot pattern，stereo matching就没东西可以match，depth就全废了。

Kuafu在shader里实现textured light：trace shadow ray时，算ray穿过light texture的哪个pixel，用那个pixel的值attenuate light intensity。

**Denoising也很关键。** Monte Carlo ray tracing的noise在under-illuminated场景特别严重——IR image正好就是这种场景（只有weak IR projector + weak ambient）。低SPP（sample per pixel）会有大量noise。他们集成NVIDIA OptiX Denoiser和Intel Open Image Denoise，用deep learning去denoise。关键是**share GPU memory**——Kuafu和OptiX的CUDA buffer共享memory，avoid user-space memory copy，性能损失最小。

性能数字（Table I, RTX 4090, 960×540）：

| SPP | w/ Denoiser FPS |
|-----|------------------|
| 2 | 140.63 |
| 8 | 100.47 |
| 32 | 43.70 |
| 128 | 24.79 |

SPP=2 with denoiser就有140 FPS，足够real-time。

### Component 3: Noise Simulation

Real sensor的IR image有两个noise source：

$$I_{\text{noisy}} = \gamma \cdot I_{\text{clean}} + n$$

- $\gamma$：multiplicative noise，model laser speckle（激光散斑）
- $n$：additive noise，model camera thermal noise（热噪声）

**Laser speckle为什么是multiplicative？** Laser是coherent light，打在粗糙表面会产生speckle pattern——不同pixel接收到的power有随机fluctuation，这个fluctuation是proportional to signal强度的，所以是multiplicative。

他们用**gamma distribution**拟合 $\gamma$：

$$f_\gamma(x; k, \theta) = \frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-\frac{x}{\theta}}$$

- $x$：随机变量（noise multiplier，> 0）
- $k$：shape parameter，控制分布形状
- $\theta$：scale parameter，控制分布scale
- $\Gamma(k)$：gamma function，$\Gamma(k) = \int_0^\infty t^{k-1} e^{-t} dt$

为什么用gamma distribution？因为根据Goodman的speckle理论，一个pixel接收到的total power可以model为多个exponentially distributed random variable的和，这个和恰好是gamma distributed。

Thermal noise用Gaussian $\mathcal{N}(\mu, \sigma^2)$，这是标准物理模型。

参数怎么得到？拍33组static scene的IR image，每组100帧。Average 100帧得到noise-free approximation，然后用MAP estimate拟合参数。RealSense D415的参数：$\mu=-0.231, \sigma=0.83, k=3.98, \theta=0.254$。

### Component 4: SimSense（GPU stereo matching）

Real depth sensor里的stereo matching是real-time的，但你用OpenCV的CPU SGBM只有0.7 FPS（Table II）。这在sim里没法用——你render一组image要等1.4秒才能得到depth，太慢了。

他们写了CUDA-accelerated的SimSense，pipeline是：
1. Stereo rectification
2. Center-Symmetric Census Transform (CSCT)——local transform让matching对brightness variation robust
3. 4-path Semi-Global Matching (SGM)——搜索best disparity
4. Uniqueness test
5. Sub-pixel disparity（quadratic fitting）
6. Left-right consistency check
7. Median filter
8. Optional depth registration

性能（Table II, 960×540 input）：

| Max Disparity | SGM (Ours) FPS | SGBM (OpenCV) FPS |
|---------------|-----------------|---------------------|
| 64 | 414.51 | 0.73 |
| 128 | 281.26 | 0.68 |

400倍加速，从<1 FPS到400+ FPS。

---

## 效果如何

### 6D Pose Estimation（最核心的实验）

三个algorithm：PVN3D（keypoint voting）、Frustum PointNets（frustum-based）、SegICP（segmentation + ICP）。

在real objects（包含transparent、specular等challenging material）上的结果（10°, 10mm threshold）：

| Training Data | PVN3D | Frustum | SegICP |
|----------------|--------|---------|--------|
| Clean depth | 9.59% | 1.55% | 2.48% |
| DepthSynth | 30.49% | 34.33% | 39.12% |
| PixelDA | 0.00% | 12.42% | 5.04% |
| LearnAug | 0.35% | 11.27% | 19.37% |
| DDS | 23.11% | 21.21% | 22.16% |
| **Ours** | **42.60%** | **38.53%** | **50.56%** |

从9.59%到42.60%，接近5倍提升。这就是physics-grounded simulation的威力——你把material-dependent error pattern faithfully reproduce了，model在sim里就学到了怎么处理这些pattern，到real world直接能用。

### Robot Grasping

Real world grasping success rate：
- Clean depth: 49/80 (61%)
- DepthSynth: 55/80 (69%)
- **Ours: 77/80 (96%)**

在10个unseen optically challenging objects上更是100% success。

### Algorithm Ranking Correlation

用sim data评估algorithm ranking，看跟real world ranking有多correlate：

| Test Data | Correlation (10°, 10mm) |
|------------|--------------------------|
| Clean depth | -0.562（负相关！） |
| DepthSynth | 0.973 |
| **Ours** | **0.982** |

Clean depth是**负相关**——这意味着如果你用clean depth做benchmark选algorithm，选出来的可能是real world里最差的那个。这是为什么不能naive地用clean depth做evaluation。

---

## 最深刻的几个insight

### Insight 1: Active sensing是sim-to-real的cheat code

人眼是passive sensing——依赖环境光照，环境光照千变万化，simulate起来极其复杂。

Active sensor主动emit已知signal（IR dot pattern），environment light可以忽略（IR band能量很弱，被filter掉了）。你只需要model这个已知signal怎么跟场景interact，复杂度大大降低。

这是robot相比human的unique advantage——**robot可以customize自己的sensing system来simplify perception problem**。这篇paper exploit了这个advantage。

### Insight 2: Depth simulation > Depth completion

你可以train一个depth completion network去fill real depth的hole（ClearGrasp、TransCG），但：
- 增加inference time cost
- 对unseen object/scene泛化不好
- 可能introduce artifacts

而depth sensor simulation是在**training time**让model见到realistic的depth noise pattern，**inference time零额外cost**。Table V证明depth completion反而让pose estimation变差。

### Insight 3: Perceptual loss对misalignment robust

做material acquisition时，sim和real的exposure/lighting不可能完全align。Plain L2会因为这种mismatch给出错误gradient。Perceptual loss在high-level feature space match，对low-level variation robust。这是一个small but important的design choice。

### Insight 4: SPP对depth transferability影响小

Ablation study（Table IX）显示SPP从2到128，pose estimation accuracy几乎没差。因为两rendered IR image被denoiser和SPP equally affect，stereo matching时这些effect cancel out。

但**高SPP对RGB/IR image fidelity重要**——低SPP会把metallic ball上的IR dot pattern erase掉（Fig. 13）。所以你的application如果只要depth，可以用low SPP + denoiser跑得飞快；如果也要高保真RGB，需要high SPP。

### Insight 5: Physics-grounded方法的generalizability

GAN-based或domain randomization方法在training distribution内可能work，但遇到新material、新shape就容易fail。Physics-grounded方法是**principled**的——只要material参数对、light transport方程对，任何object都能correctly simulate。这就是为什么他们能在10个unseen objects上达到100% grasping success。

---

## 我的take

这篇paper的哲学跟我个人很喜欢的approach一致：**别用learning去approximate你其实能从first principles算出来的东西**。

Domain adaptation、domain randomization这些方法本质上是用learning去"猜"domain gap长什么样。但depth sensor的domain gap是physics决定的——refraction、specular reflection、laser speckle、thermal noise——这些都是有well-established物理模型的。

当然physics-grounded方法的cost是**engineering complexity**：你要写ray tracer、支持textured light、GPU stereo matching、material acquisition pipeline...这比"加个GAN"复杂得多。但回报是fidelity和generalizability——你sim出来的data是真的接近real，不是"看起来像"。

从更大视角看，这篇paper指向一个有意思的方向：**robot系统的设计应该考虑simulation友好的sensing modality**。Active IR stereo是simulation友好的（environment light可忽略，signal已知controllable）。如果你设计一个新sensor，可以考虑"这个东西好不好simulate"作为一个design criterion。Simulation-friendly的sensor会让整个sim-to-real pipeline更efficient。

---

## References

- Paper: https://arxiv.org/abs/2210.15164
- SAPIEN Simulator: https://sapien.ucsd.edu/
- Disney PBR material (Burley 2012): https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
- GGX Microfacet Model (Walter et al. 2007): https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
- SGM (Hirschmuller 2007): https://ieeexplore.ieee.org/document/4359315
- Census Transform (Zabih & Woodfill 1994): https://link.springer.com/chapter/10.1007/3-540-57942-9_14
- Goodman Speckle Theory: https://www.spie.org/Publications/PM100
- NVIDIA OptiX: https://developer.nvidia.com/optix
- Intel Open Image Denoise: https://www.openimagedenoise.org/
- PVN3D: https://github.com/cvlab-columbia/pvn3d
- ClearGrasp: https://github.com/Shreeyak/cleargrasp
- TransCG: https://github.com/galaxies/depth_completion_dataset
- Intel RealSense D415: https://www.intelrealsense.com/depth-camera-d415/
- Vulkan Ray Tracing: https://www.khronos.org/blog/ray-tracing-in-vulkan
- AlexNet Perceptual Loss (Johnson et al. 2016): https://arxiv.org/abs/1603.08155
- SAC (Soft Actor-Critic): https://arxiv.org/abs/1801.01290
- ManiSkill: https://github.com/haosulab/ManiSkill
- DDS (Differentiable Depth Sensor): https://openaccess.thecvf.com/content/ICCV2021/papers/Planche_Physics-Based_Differentiable_Depth_Sensor_Simulation_ICCV_2021_paper.pdf
- DepthSynth: https://ieeexplore.ieee.org/document/8237300
- PixelDA (Bousmalis et al. 2017): https://arxiv.org/abs/1612.05424
- LearnAug (Pashevich et al. 2019): https://arxiv.org/abs/1907.08120

---

# Close the Optical Sensing Domain Gap by Physics-Grounded Active Stereo Sensor Simulation

这篇paper来自UCSD Hao Su组和清华大学Jing Xu组，核心思想非常elegant：**通过physics-grounded的方式模拟active stereovision depth sensor的完整pipeline，来缩小sim-to-real的optical sensing domain gap**。

---

## 1. 核心动机与设计哲学

### 1.1 为什么选择active stereovision depth sensor？

作者列出了5个reasons，我觉得最深刻的是第4和第5点：

**Reason 4: Active sensing simplifies simulation**

Active stereovision sensor工作在IR spectrum（通常是near-IR, ~850nm）。在indoor environment中，IR band的能量非常limited，且IR camera会filter out其他spectra的光。这意味着**passive environment light可以忽略**，从而避免了simulating RGB camera时最头疼的complex environment illumination问题。

这是一个非常深刻的insight：**robot相比human的优势在于可以customize sensing system，主动emit controllable signals来facilitate sensing**。当sensor主动发射已知pattern时，simulation的复杂度大幅降低——你只需要model这个已知的pattern如何与场景interaction，而不需要model未知的environment light如何照亮场景。

**Reason 5: Real-time ray tracing技术成熟**

最近几年hardware acceleration（NVIDIA RTX）+ learning-based denoising让real-time ray tracing成为可能。这刚好契合了active stereovision的需求——需要模拟IR pattern的transport，包括specular reflection、refraction等complex light effects。

### 1.2 核心challenge：Material-dependent depth errors

Fig. 1展示了关键问题：transparent/translucent物体上，projected IR pattern无法well reflected，导致depth measurement incomplete/noisy。要缩小domain gap，simulation必须reproduce这种material-dependent error patterns。

传统的rasterization rendering根本无法模拟这种现象，因为rasterization本质上只处理直接可见的geometry，无法正确处理refraction、specular reflection等indirect lighting effects。

---

## 2. Method Overview

### 2.1 形式化定义

Depth sensor simulator定义为：

$$I_{\text{depth}} = D_\psi(S_{\text{objects}}, S_{\text{lighting}})$$

变量解释：
- $I_{\text{depth}}$: output depth map
- $\psi$: depth sensor simulator的参数（包括noise model参数、stereo matching参数等）
- $S_{\text{objects}}$: object status，包括geometry、PBR materials（base color, metallic, specular, roughness, transmission, IOR, emission）和poses
- $S_{\text{lighting}}$: lighting status，包括light source positions和intensities

假设geometry和base color来自textured CAD models，需要acquire的是其他PBR material参数和light intensities。

### 2.2 Pipeline概览

完整pipeline包含5个stage：

1. **Material Acquisition** (Section V): 获取object的PBR material参数
2. **IR Pattern Projection**: 模拟projector投射IR pattern
3. **IR Light Transport** (Kuafu renderer): ray tracing模拟IR光的transport
4. **Sensor Noise Simulation**: 模拟camera和projector的noise
5. **Stereo Matching** (SimSense): 从stereo IR images生成depth map

---

## 3. Material Acquisition

这是paper的第一个关键技术贡献。

### 3.1 Pixel-wise Alignment

传统方法用markers（如QR code）来track camera pose，但markers会留在captured images中。作者的trick是**反向对齐**：先生成rendered images，然后把real scene layout align到simulation，而不是反过来。

具体步骤：
1. Camera intrinsic matrices从firmware获取
2. 用OpenCV的`solvePnP`计算camera-table transformation for all viewpoints
3. 用hand-eye calibration计算depth sensor到robot end effector的transformation
4. 实时overlay rendered image在captured image上作为feedback，手动调整object poses

精度：< 2mm, 1.5°（用motion capture system验证）

### 3.2 Multispectral Matching Loss

这是关键创新。给定一对aligned的multispectral images，loss定义为：

$$L = L_{\text{RGB}} + \lambda L_{\text{IR}}$$

$$L_{\text{RGB}} = L_2(I_{\text{RGB}}^{\text{sim}}, I_{\text{RGB}}^{\text{real}}) + L_{\text{percept}}(I_{\text{RGB}}^{\text{sim}}, I_{\text{RGB}}^{\text{real}})$$

$$L_{\text{IR}} = L_2(I_{\text{IR}}^{\text{sim}}, I_{\text{IR}}^{\text{real}}) + L_{\text{percept}}(I_{\text{IR}}^{\text{sim}}, I_{\text{IR}}^{\text{real}})$$

变量解释：
- $I_{\text{RGB}}^{\text{sim}}, I_{\text{IR}}^{\text{sim}}$: simulated RGB和IR images
- $I_{\text{RGB}}^{\text{real}}, I_{\text{IR}}^{\text{real}}$: real captured RGB和IR images
- $L_2$: pixel-wise L2 distance
- $L_{\text{percept}}$: perceptual loss，定义为AlexNet features的L2 difference [Johnson et al., 2016]
- $\lambda$: weighting factor balancing RGB和IR terms

**为什么需要perceptual loss？** 作者empirically发现plain L2会因为brightness和color的mismatch给出sub-optimal结果，因为plain L2对exposure和lighting condition的misalignment非常敏感。Perceptual loss在high-level feature space计算，对low-level的brightness/color variation更robust，从而允许在无需严格color/exposure/lighting alignment的情况下获得better material acquisition。

### 3.3 Adaptive Grid Search

搜索参数集 $\mathcal{P} = \{\text{roughness, metallic, specular, transmission}\}$

两阶段：
1. **Coarse**: 每个parameter 10 grid samples，找到 $\mathcal{P}^{\text{coarse}}$
2. **Fine**: 在 $\mathcal{P}^{\text{coarse}}$ 邻域再10 grid samples，获得 $\mathcal{P}^{\text{fine}}$

整个process < 1 hour per object，其中real capture约2分钟。

---

## 4. Active Stereovision Depth Sensor Simulation

### 4.1 Kuafu Renderer

这是作者新开发的ray tracer，基于Vulkan ray-tracing API，集成到SAPIEN simulator中。

**为什么不用Blender或3dsMax？** 这些商业PBR solutions太heavy，runtime overhead大。自己实现可以：
1. Optimize out unnecessary steps
2. 实现特定优化（如GPU memory sharing between Kuafu和stereo matcher）
3. 构建unified pipeline到robotics simulator

#### BSDF Model

最终的BSDF是diffuse和specular的mix：

$$f(\mathbf{i}, \mathbf{o}, \mathbf{m}) = w_d f_d(\mathbf{i}, \mathbf{o}, \mathbf{m}) + (1 - w_d) f_s(\mathbf{i}, \mathbf{o}, \mathbf{m})$$

$$w_d = (1 - \mu)(1 - \alpha)$$

变量解释：
- $\mathbf{i}$: incoming light direction（单位向量，从光源指向surface）
- $\mathbf{o}$: outcoming light direction（单位向量，从surface指向camera）
- $\mathbf{m}$: local surface normal（单位向量）
- $f_d$: diffuse term，使用Lambertian或Oren-Nayar model [Wolff et al., 1998]
- $f_s$: specular term，使用GGX microfacet model [Walter et al., 2007]
- $\mu$: material metallic parameter ∈ [0, 1]
- $\alpha$: object transmission parameter ∈ [0, 1]
- $w_d$: diffuse weight，当metallic=1或transmission=1时 $w_d = 0$，diffuse项消失

这个公式很重要——它解释了为什么metallic和transparent物体的depth measurement会出问题：当 $\alpha = 1$（完全transparent），$w_d = 0$，光完全通过refraction传播，projector的IR pattern不会被reflect回camera，导致depth hole。

#### Textured Light Support

这是**关键创新**，existing ray tracers通常不支持textured lights。

实现方法：在shader中，当tracing shadow ray时，首先确定ray passes through light texture的哪个pixel，然后根据light texture attenuate light intensity。

这个feature的importance：active stereovision sensor的IR pattern projector本质上就是一个textured spot light——它投射的是random dot pattern，不是uniform light。没有textured light support，根本无法正确模拟stereo matching所需的IR image。

#### Denoising

Monte Carlo ray tracing的noise在under-illuminated场景或small SPP时特别严重——IR image rendering正好两者都占。

集成两个denoisers：
- **NVIDIA OptiX Denoiser** [Parker et al., 2010]
- **Intel Open Image Denoise**

通过explicit GPU memory sharing between Kuafu和OptiX CUDA buffer，eliminate user-space memory copying。

### 4.2 IR Image Rendering

假设所有environment lights在IR spectrum也有输出，intensity = $a \cdot l$，其中 $a = 0.05$ 是attenuating factor，$l$ 是visible spectrum的light intensity。

还假设一个weak ambient light value模拟环境radiance（如sunlight）。

渲染后取denoised image的**R channel**作为IR result（因为monochrome IR image的所有RGB channels理论上应该相同）。

### 4.3 Sensor Noise Simulation

$$I_{\text{noisy}} = \gamma * I_{\text{clean}} + n$$

变量解释：
- $I_{\text{noisy}}$: 加噪后的IR image
- $I_{\text{clean}}$: ray tracing渲染的clean IR image（light intensity值）
- $\gamma$: multiplicative noise，modeling laser speckle
- $n$: additive noise，modeling camera thermal noise
- $*$, $+$: element-wise operations

**Multiplicative noise (laser speckle)** 用gamma distribution：

$$f_\gamma(x; k, \theta) = \frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-\frac{x}{\theta}}$$

变量解释：
- $x$: random variable（noise multiplier值，> 0）
- $k$: shape parameter（决定distribution形状）
- $\theta$: scale parameter（决定distribution尺度）
- $\Gamma(k)$: gamma function，$\Gamma(k) = \int_0^\infty t^{k-1} e^{-t} dt$
- $e$: natural base

为什么用gamma distribution？根据 [Goodman, 2007]，total power received by a pixel可以model为exponentially distributed power（Rayleigh voltage）random variables的和，这恰好是gamma distribution的物理基础。

**Additive noise (thermal noise)** 用Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$ [Perepelitsa, 2006]。

**Parameter estimation**: 捕获33组IR images，每组100帧static scene。用frame averaging获得noise-free approximation，然后用MAP estimate拟合noise model参数。

RealSense D415的estimated parameters：
- $\mu = -0.231$ (Gaussian mean)
- $\sigma = 0.83$ (Gaussian std)
- $k = 3.98$ (gamma shape)
- $\theta = 0.254$ (gamma scale)

### 4.4 Depth Generation (SimSense)

这是GPU-accelerated的stereo matching模块，pipeline：

1. **Stereo rectification**: 将images project到common image plane
2. **Center-Symmetric Census Transform (CSCT)** [Zabih & Woodfill, 1994]: local transform for robust matching
3. **4-path Semi-Global Matching (SGM)** [Hirschmuller, 2007]: 搜索best disparity candidates，hamming distance作为cost function
4. **Uniqueness test**: filter out disparities not better than second best by threshold
5. **Sub-pixel disparity**: quadratic curve fitting
6. **Left-right consistency check**
7. **Median filtering**
8. **Depth registration** (optional): align depth map到RGB camera frame

加速技巧 [Hernandez-Juarez et al., 2016]：
- CSCT和matching cost用shared memory优化data reuse
- Cost aggregation用warp-based optimization
- 也支持SGBM (Semi-Global Block Matching)，matching cost是local regions的hamming distance

---

## 5. Performance Analysis

### 5.1 Rendering Performance

Table I (RTX 4090, 960×540):

| SPP | w/o Denoiser (FPS) | w/ Denoiser (FPS) |
|-----|--------------------|--------------------|
| 2 | 352.80 | 140.63 |
| 8 | 146.51 | 100.47 |
| 32 | 57.36 | 43.70 |
| 128 | 27.93 | 24.79 |

观察：SPP=2 with denoiser就有140 FPS，足够real-time。

### 5.2 Stereo Matching Performance

Table II (960×540 input, 1920×1080 output):

| Max Disparity | SGM (Ours) | SGBM (Ours) | SGBM (OpenCV) |
|---------------|------------|-------------|----------------|
| 64 | 414.51 | 342.24 | 0.73 |
| 96 | 326.79 | 268.98 | 0.71 |
| 128 | 281.26 | 232.49 | 0.68 |
| 256 | 147.13 | 128.56 | 0.60 |

CPU OpenCV实现 < 1 FPS，GPU SimSense实现200+ FPS，加速约400倍。

---

## 6. Experimental Results

### 6.1 Object Detection (Table III)

mAP@0.5 on real data:

| Data | Depth | RGB | RGBD |
|------|-------|-----|------|
| Clean depth | 0.730 | - | - |
| Rasterization | 0.942 | 0.839 | 0.875 |
| Ours | **0.977** | **0.941** | **0.943** |

Key insight: depth map transferability最好（0.977），因为depth主要受geometry影响，而RGB还受texture、material、illumination影响。

### 6.2 6D Pose Estimation (Table IV)

这是最重要的实验。3个算法 × 6个training data sources：

**PVN3D** (10°, 10mm on real objects):
- Clean: 9.59%
- DepthSynth: 30.49%
- PixelDA: 0.00%
- LearnAug: 0.35%
- DDS: 23.11%
- **Ours: 42.60%**

**Frustum** (10°, 10mm on real objects):
- Clean: 1.55%
- DepthSynth: 34.33%
- **Ours: 38.53%**

**SegICP** (10°, 10mm on real objects):
- Clean: 2.48%
- DepthSynth: 39.12%
- **Ours: 50.56%**

为什么PixelDA和LearnAug表现差？作者解释：PVN3D用PointNet++提取geometry信息，对geometry distortion敏感。PixelDA和LearnAug引入additional unrealistic geometric distortions [Shen et al., 2022也提到这一点]。

为什么DDS sub-optimal？DDS用differentiable stereo matching module，capability limited compared with SGBM。而且DDS用的differentiable rendering不支持transparent materials。

### 6.3 Comparison with Depth Completion (Table V)

对比ClearGrasp和TransCG：

**PVN3D**:
- Clean → Real: 13.73%
- Clean → ClearGrasp: 1.67%
- Clean → TransCG: 0.04%
- **Ours → Real: 63.37%**

Depth completion反而worse！原因：learning-based depth completion对unseen objects/scene generalizability有限，可能introduce more artifacts。

这是一个重要发现：**与其在inference time做depth completion，不如在training time做realistic depth simulation**。前者增加inference cost，后者不增加任何inference burden。

### 6.4 Robot Grasping (Table VI)

Real-world grasping success rate:

| Data | Success |
|------|---------|
| Clean | 49/80 (61.25%) |
| DepthSynth | 55/80 (68.75%) |
| **Ours** | **77/80 (96.25%)** |

在10个unseen optically challenging objects上，成功率达到100%！

### 6.5 Algorithm Ranking (Table VII)

Correlation coefficients with real evaluation:

| Test Depth | Overall (10°,10mm) | Real (10°,10mm) |
|------------|---------------------|------------------|
| Clean | -0.562 | -0.689 |
| DepthSynth | 0.973 | 0.925 |
| **Ours** | **0.982** | **0.959** |

注意Clean是negative correlation！这意味着用clean depth评估algorithm ranking会得到完全错误的结论。Ours的correlation最高，可以作为reliable benchmark。

---

## 7. Ablation Studies

### 7.1 Material Acquisition (Table VIII)

对比default material、3个random material、和Ours（with/without perceptual loss）：

**SegICP** (10°, 10mm on real objects):
- Default: 39.13%
- Random1: 36.45%
- Random2: 44.94%
- Random3: 42.86%
- Ours w/o $L_{\text{percept}}$: 36.28%
- **Ours: 46.93%**

验证了accurate material parameters对optically challenging objects至关重要。

### 7.2 Rendering Settings (Table IX)

测试SPP ∈ {2, 8, 32, 128} 和denoiser on/off：

Key finding: **SPP对depth transferability影响不大**！原因是两rendered IR images被denoiser和SPP equally affect，stereo matching时这些effects cancel out。

但要注意：低SPP + denoiser会把metallic ball上的IR pattern erase掉（Fig. 13），所以high-fidelity RGB/IR需要high SPP，depth simulation可以用low SPP + denoiser。

### 7.3 Noise Scale (Table X)

测试noise scale ∈ {0, 0.1, 0.3, 1.0, 3.0, 10.0}：

最佳在scale=1.0（identified scale）。过aggressive noise会让algorithm变robust但performance下降。

---

## 8. 关键insights总结

1. **Active sensing是sim-to-real的key advantage**：robot可以主动emit controllable signals，简化simulation。这是human sensing没有的advantage。

2. **Material-dependent depth errors是核心challenge**：rasterization无法处理，必须用ray tracing + PBR materials。

3. **Textured light support是技术enabler**：没有这个，根本无法模拟IR pattern projector。

4. **Depth simulation > Depth completion**：前者不增加inference cost，后者增加且可能introduce artifacts。

5. **Real-time可行性**：通过GPU加速ray tracing + denoising + GPU stereo matching，整个pipeline可以达到real-time。

6. **Multispectral loss (RGB + IR)**：同时optimize两个spectrum的matching，比单spectrum更robust。

7. **Perceptual loss的重要性**：在material acquisition中，perceptual loss对exposure/lighting misalignment更robust。

---

## 9. 与相关工作的对比

| Method | Type | Pros | Cons |
|--------|------|------|------|
| DepthSynth [33] | Mechanism simulation (rasterization) | 支持stereo matching | 无法处理specular/transmission |
| PixelDA [25] | GAN-based domain adaptation | Unsupervised | Introduce geometric distortion |
| LearnAug [37] | Domain randomization (MCTS) | 灵活 | 无法imitate complex material errors |
| DDS [38] | Differentiable rendering | End-to-end optimizable | Slow, no transparent support, limited stereo matching |
| **Ours** | **Physics-grounded ray tracing** | **Full PBR, real-time, accurate** | **Requires material acquisition** |

---

## 10. Limitations & Future Work

1. **Motion blur和rolling shutter**未modeling
2. **Deformable objects**的material acquisition困难（需要geometry alignment）
3. **Unpaired sim-real images**的material acquisition（current需要pixel-wise aligned）
4. **Material properties change during interaction**未account
5. 当前只support active stereovision，其他sensor type（ToF, structured light）未集成

---

## References

- SAPIEN: https://sapien.ucsd.edu/
- Disney PBR material: https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
- Blender Principled BSDF: https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html
- NVIDIA OptiX: https://developer.nvidia.com/optix
- Intel Open Image Denoise: https://www.openimagedenoise.org/
- Intel RealSense D415: https://www.intelrealsense.com/depth-camera-d415/
- PVN3D: https://github.com/cvlab-columbia/pvn3d
- ClearGrasp: https://github.com/Shreeyak/cleargrasp
- TransCG: https://github.com/galaxies/depth_completion_dataset
- GGX microfacet model: https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
- SGM (Hirschmuller): https://ieeexplore.ieee.org/document/4359315
- Census Transform: https://link.springer.com/chapter/10.1007/3-540-57942-9_14
- Vulkan Ray Tracing: https://www.khronos.org/blog/ray-tracing-in-vulkan
- ManiSkill: https://github.com/haosulab/ManiSkill
- AlexNet perceptual loss: https://arxiv.org/abs/1603.08155
- SAC (Soft Actor-Critic): https://arxiv.org/abs/1801.01290

---

这篇paper的beauty在于：它不试图用learning-based方法去"猜"domain gap，而是从physics first principles出发，faithfully reproduce real sensor的mechanism。虽然工程复杂度高于rasterization或GAN-based方法，但得到的fidelity和generalizability是其他方法难以匹敌的。尤其是对transparent/specular等optically challenging materials的处理，physics-grounded方法是唯一能正确capture material-dependent depth error patterns的途径。
