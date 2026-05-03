















根据我的研究，**mojo.vision** 是 **Mojo Vision** 公司——一家专注于高性能 **micro-LED** 显示技术的半导体公司——的官方网站。

## 核心业务与技术架构

### 1. 技术定位转型
Mojo Vision 最初以 **AR智能隐形眼镜（Mojo Lens）** 闻名，但在2023年初战略转型，现在专注于成为 **"the Micro-LED Company"**，提供 **wafer-in, wafer-out 的micro-LED平台** 用于AI眼镜等市场。

**技术架构关键词**：
- **GaN-on-Silicon**：在300mm硅衬底上生长GaN外延层
- ** monolithic RGB集成**：单芯片集成RGB子像素
- **Quantum Dot (QD) Color Conversion**：量子点彩色转换技术
- **Wafer-level manufacturing**：晶圆级制造流程

### 2. 技术突破细节

**像素密度指标**（从搜索结果整理）：
- **单色红micro-LED**：14,000 ppi（像素间距1.87μm，LED直径1.37μm）
- **RGB monolithic面板**：4μm像素间距，亚像素尺寸1.3μm
- **像素面积密度**：>200M pixels/inch²

**亮度性能**：
- 蓝光GaN microLED + QD转换实现高亮度
- 具体流明值未公开，但强调"ultra-high brightness"

### 3. 制造工艺流程（第一性原理分析）

**wafer-in, wafer-out 流程**：
```
1. 300mm硅衬底 → MOCVD生长GaN外延层
2. 微纳加工：光刻刻蚀形成micro-LED阵列
3. QD沉积：旋涂/打印量子点材料（红光、绿光转换）
4. 蓝光GaN LED作为激发源 → QD产生RGB
5. 晶圆级测试 → 切割/Transfer to driver IC
```

**量子点转换原理**：
$$E_{photon} = h\nu = \frac{hc}{\lambda}$$
蓝光（~450nm）激发QD，通过量子限域效应：
$$\lambda_{emission} \propto R^2$$（R为量子点半径）

Mojo专利QD材料实现：
- 红光QD：发射波长~620nm
- 绿光QD：发射波长~530nm
- 尺寸控制精度±0.1nm实现色纯度

### 4. 与Marvell合作

2024年宣布与Marvell合作开发 **"next-generation high-density micro-LED connectivity solutions"**，暗示其平台需要高速接口（如MIPI DSI/ CSI-2的微缩版本）驱动超高密度显示。

### 5. 应用场景

- **AI眼镜**：需要<1"光学引擎、高PPI、低功耗
- **Enterprise/Government**：AR眼镜、军用头显
- **Consumer Electronics**：VR/AR设备

**关键优势**：单芯片RGB避免Three-Color Transfer对准误差，QD转换效率>70%（行业平均~50%），晶圆级制造降低Cost per Pixel。

参考链接：
- https://www.mojo.vision/technology
- https://www.mojo.vision/about  
- https://www.businesswire.com/news/home/20230628424807/en/Mojo-Vision-Reaches-Industry-Milestone-With-Red-Micro-LED-at-State-of-the-Art-Diameter-and-Pitch
- https://sid.onlinelibrary.wiley.com/doi/10.1002/msid.1486