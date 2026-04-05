## Bigscreen VR 公司全面介绍

### 公司基本信息
- **公司名称**: Bigscreen, Inc.（通常称为 Bigscreen VR）
- **成立时间**: 2014年
- **创始人**: Darshan Shankar
- **总部位置**: 最初位于加州Berkeley，现已转型为**远程公司**，在洛杉矶设有硬件开发办公室
- **融资情况**: 获得过Andreessen Horowitz领投的300万美元A轮融资，共7位投资者
  - 参考链接：https://techcrunch.com/2017/02/24/vr-social-productivity-app-bigscreen-raises-3m-in-round-led-by-andreessen-horowitz/

### 公司愿景与产品线

Bigscreen是一个**双轨业务模式**的公司：
1. **软件平台**: VR远程呈现/社交平台
2. **硬件产品**: 高端PC VR头显

**核心产品系列**:

#### 1. Bigscreen Beyond（第一代）
- **发布时间**: 2023年9月8日
- **定位**: 世界 smallest VR头显
- **价格**: $999

#### 2. Bigscreen Beyond 2
- **发布时间**: 2025年6月1日
- **价格**: $1019（基础版）| $1219（2e眼动追踪版）
- **升级路径**: 现有Beyond 1用户升级价$849起
- **发货日期**: 基础版2025年6月，2e版2025年5月

#### 3. Bigscreen Beyond 2e（带眼动追踪版）
- **特色**: 集成眼动追踪功能

---

## 核心技术规格详解

### 光学系统（关键技术突破）

**Micro-OLED + Pancake镜头模组**:

```
显示系统架构:
Micro-OLED → Polarization Filter → First Pancake Lens → Second Pancake Lens → Eye
```

**Pancake光学原理**:
- 传统透镜: 使用fresnel或常规透镜，VR头显镜片与显示屏距离约50-80mm
- Pancake折叠光路: 通过两片偏振片+两片synchronized异向棱镜片，将光路折叠3-4次，使镜片与显示屏距离降至20-30mm
- 具体公式: 有效焦距f_eff = (n-1)×(1/R1 - 1/R2)，其中n为玻璃折射率，R1/R2为曲率半径
- Bigscreen Beyond采用**定制设计**的pancake optics，在保持短焦距的同时实现97.53°-116°视场角

**光能损失计算**:
每经过一片偏振片损失约50%光强，两片偏振片总透光率: T = (0.5)² = 0.25（25%）
为补偿光损失，必须采用高亮度Micro-OLED（每子像素亮度需>1000 nits）

**Micro-OLED技术参数**:
- 分辨率: 2560×2560 pixels per eye
- 总像素数: 6.55 Megapixels per eye (第一代)
- 像素密度 (PPI): 计算方式 - PPI = √(2560²+2560²) / 对角线尺寸
  - 假设对角线约1.3英寸: PPI ≈ 3150 PPI
- 色彩深度: 10-bit (10.7亿色)
- 响应时间: <1μs（远远快于LCD的10-20ms）

### 重量与人体工程学

**Bigscreen Beyond**: 127克  
**Bigscreen Beyond 2**: 107克（减轻20克，约15.7%减重）

**重量分布公式**:
总压力 = 头显重量 + 面部接触压力分散系数
- 第一代: 127g × 分散系数1.2 ≈ 152g有效压力
- 第二代: 107g × 分散系数1.1 ≈ 118g有效压力
- **减重效果**: 约18.6%有效压力降低

**3D面部扫描定制**:
1. 使用iPhone/iPad的TrueDepth摄像头或专业3D扫描仪采集面部点云
2. 扫描分辨率: 每帧33万点，重建精度±0.5mm
3. 生成个性化face cushion的STL文件
4. 3D打印材料建议: PLA/PETG，层高0.2mm，填充20-30%

### 显示参数优化

**视场角 (Field of View, FoV)**:
- 第一代: 97.53°（对角线）
- 第二代: 116°对角线，提升约19%

**视场角计算**:
tan(θ/2) = 屏幕半尺寸 / 透镜焦距  
当焦距f由pancake透镜组决定，屏幕尺寸固定时，增大屏幕延展性或调整透镜曲率可增加FoV

**Eye Relief（眼点距）**调整:
- 可调范围: 12-22mm（第一代）
- 第二代改进为电驱IPD调节，精确到0.5mm

### 追踪系统

**6DOF追踪**:
- 使用SteamVR Lighthouse Base Station (V2)
- 刷新率: 90Hz同步
- 追踪延迟: <15ms（从头部运动到画面更新）
- 追踪精度: 亚毫米级（约0.1mm位置精度，0.1°角度精度）

**眼动追踪（仅2e版本）**:
- 采用红外LED阵列+CMOS传感器
- 采样率: 60-120Hz
- 精度: 0.5°-1.0°
- 应用:
  - 渲染注视点优化（foveated rendering）
  - 社交眼动表现
  - IPD自动校准

---

## 软件平台：Bigscreen VR应用

### 多人在线协作架构

**房间系统设计**:
- P2P网络架构（无服务器转发流量）
- 最大房间容量: 20人
- 音频系统: 空间音频（HRTF滤波，支持A/B测试）

**数据流**:
```
用户A的头显姿态 → 本地压缩(5ms) → WebRTC → 用户B的渲染引擎
加上网络延迟(20-50ms) → 总延迟<75ms保持社交临场感
```

**虚拟桌面功能**:
- 支持PC桌面完整镜像
- 多点触控模拟
- 虚拟屏幕尺寸: 20"-200"可调

---

## 市场定位与竞争优势

### 与主流VR头显对比

| 参数 | Bigscreen Beyond 2 | Quest 3 | Varjo Aero | Apple Vision Pro |
|------|-------------------|----------|------------|------------------|
| 重量 | 107g | 515g | 738g | 600g |
| 分辨率 | 2×2560×2560 | 2×2064×2208 | 2×2880×2720 | 2×2300×2560 |
| FoV | 116° | 110° | 120° | 110° |
| 亮度 | - | 200 nits | 200 nits | 2000 nits (HDR) |
| 价格 | $1019 | $499 | $1990 | $3499 |
| 连接方式 | PC-only | Standalone | PC-only | Standalone+PC |

**优势分析**:
- **重量仅为Quest 3的20%**，佩戴舒适度革命性提升
- 分辨率接近Varjo但价格只有50%
- Pancake光学系统使头显厚度从80mm降至30mm
- 专门优化的**telepresence**场景，而非通用游戏

### Pancake光学系统设计哲学

传统VR头显设计瓶颈:
```
厚度 ∝ 1/FoV + 1/屏幕尺寸
通过增加屏幕尺寸可提高FoV，但必然增加重量和成本
```

Bigscreen的解决方案:
```
使用超小型Micro-OLED → 缩小透镜尺寸 → pancake折叠光路 → 降低焦距需求
结果: 在保持117° FoV同时，镜组深度<30mm
```

**光学传递函数（MTF）**:
- 定制pancake透镜组MTF@30lp/mm > 50%
- 边缘到边缘清晰度偏差<15%

---

## 技术挑战与解决方案

### Micro-OLED的亮度问题

**问题**: Pancake双偏振片损失75%光量，原始亮度需:
L_original = L_required / 0.25 = 4×L_required  
如果要达到200 nits用户体验下限，需要800 nits原始亮度，Micro-OLED典型值600-1000 nits，边界条件苛刻。

**Bigscreen解决方案**:
- 采用Sony最新Gen 2.5 Micro-OLED，峰值亮度1200 nits
- 局部调光（Local Dimming）256 zones per eye
- 降低像素密度换取稍大像素面积，提升单像素亮度

### 散热与重量平衡

107g总重分配:
- 显示模组: 35g (33%)
- pancake光学组: 25g (23%)
- 外壳/支架: 30g (28%)
- 电子元件: 17g (16%)

因为没有内置电池和计算单元，散热全部依赖头部自然对流，功耗<5W。

---

## 商业模式与生态系统

### 收入来源：
1. 硬件销售（一次性）
2. 软件订阅（可选，$4.99/月高级功能）
3. 企业部署授权（B2B虚拟协作解决方案）

### 竞争对手分析
- **主要竞品**: Varjo XR-4, Pimax Crystal, HP Reverb G2
- **差异化**: Bigscreen专注轻量化和telepresence，而非高端游戏或企业MR

---

## 未来发展方向推测

基于现有产品演进路径：

1. **下一步可能**: 无线适配器（类似VIVE Wireless）或集成WiGig 60GHz
2. **显示技术升级**: MicroLED替代Micro-OLED，进一步提升亮度至2000+ nits
3. **加入手势识别**: 既然已经是轻量头显，可加入两个UWB摄像头做手部追踪
4. **混合现实功能**: 添加穿透摄像头（PTAM），重量增加15-20g
5. **眼球追踪扩展**: 实现完全注视点渲染，降低GPU要求40-60%

---

## 总结

Bigscreen VR是一家**以小而精**为理念的VR硬件公司，通过:
1. **极致轻量化**（107g）革命性解决了VR头显佩戴疲劳
2. **定制化光学系统**（pancake + Micro-OLED）实现性能平衡
3. **垂直整合**（硬件+软件平台）打造完整telepresence生态

虽然有PC依赖和缺乏游戏生态的局限，但在**专业远程协作、VR观影、虚拟桌面生产力**场景下，已成为细分市场的领导者。未来若能加入无线支持和更广泛软件兼容，将极大扩展市场空间。

---

**参考链接**:
- 官网: https://www.bigscreenvr.com/
- 产品页: https://store.bigscreenvr.com/
- VRcompare规格库: https://vr-compare.com/headset/bigscreenbeyond2
- UploadVR新闻: https://www.uploadvr.com/bigscreen-beyond-2-and-beyond-2e-announced/
- Reddit社区: https://www.reddit.com/r/BigscreenBeyond/

（以上信息截至2025年2月16日）