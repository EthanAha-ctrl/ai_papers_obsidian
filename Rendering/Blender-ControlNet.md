我来帮您解析这个 GitHub 项目的内容。让我获取该页面的详细信息。

# Blender-ControlNet 项目详解

这个项目是一个**Blender 插件/脚本**，实现了在 Blender 内部直接使用 ControlNet 生成 AI 图像的功能。下面我将从**技术架构、工作流程、核心机制**等多个维度详细解析。

## 一、项目核心功能

**Blender-ControlNet** 允许用户在 Blender 中通过其内置的**Compositor（合成器）**节点系统生成各种 control maps，然后通过 API 将这些条件图发送给 Stable Diffusion + ControlNet 进行图像生成。

**GitHub 链接**: https://github.com/coolzilj/Blender-ControlNet

相关依赖：
- **AUTOMATIC1111/stable-diffusion-webui** (SD WebUI)
- **Mikubill/sd-webui-controlnet** (ControlNet 扩展)

---

## 二、技术架构分析

### 2.1 系统拓扑

```
Blender (3D Scene)
   └── Compositor Nodes → 生成四种 Control Maps
          ├── Canny Edge Map
          ├── Depth Map
          ├── OpenPose / Bone Map
          └── Segmentation Map
   └── HTTP API (localhost:7860)
          └── Stable Diffusion WebUI + ControlNet
                 └── 生成最终图像 → 保存到 IMAGE_FOLDER
```

### 2.2 数据流程

1. **场景渲染**：用户在 Blender 中构建 3D 场景，按 **F12** 渲染
2. **Map 生成**：Blender Compositor 实时处理渲染结果，生成控制图
3. **API 调用**：脚本使用 `requests.post()` 将数据打包为 JSON 发送到 SD WebUI API
4. **ControlNet 处理**：SD WebUI 的 ControlNet 扩展接收多个 control units
5. **图像合成**：SD 根据 prompts + control maps 生成图像并返回

---

## 三、关键代码机制解析

### 3.1 API 请求参数结构

脚本定义了 `params` 字典，对应 SD WebUI 的 `/sdapi/v1/txt2img` API：

```python
params = {
    "prompt": "a room",
    "negative_prompt": "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality",
    "width": get_output_width(scene),  # 动态获取 Blender 输出分辨率
    "height": get_output_height(scene),
    "sampler_index": "DPM++ SDE Karras",  # ODE 采样器，SDE 版本
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,                # 去噪步数
    "cfg_scale": 7,             # Classifier-Free Guidance 强度
    "seed": -1,                 # -1 表示随机
    "denoising_strength": 0.5,  # 重绘幅度 (img2img 模式)
    "override_settings": {"CLIP_stop_at_last_layers": 2},  # CLIP 截断层
    "alwayson_scripts": {
        "controlnet": {
            "args": []  # 控制单元列表
        }
    }
}
```

**关键参数解释**：
- **`steps`**: DDPM/DDIM 采样的迭代次数，数值越高细节越好但速度越慢
- **`cfg_scale`**: $CFG = (1+\alpha)\epsilon_\theta(x_t,c) - \alpha\epsilon_\theta(x_t)$，控制条件强度
- **`CLIP_stop_at_last_layers`**: CLIP text encoder 的层截断，2 表示用倒数第二层特征

### 3.2 ControlNet Unit 结构

每种 ControlNet map 对应一个 `cn_units` 字典，以 `bone_cn_units`（OpenPose）为例：

```python
bone_cn_units = {
    "mask": "",                      # 遮罩图路径（可选）
    "module": "none",                # 预处理器（none 表示已预处理）
    "model": "diff_control_sd15_openpose_fp16 [1723948e]",  # ControlNet 模型哈希
    "weight": 1.1,                   # 控制权重系数
    "resize_mode": "Scale to Fit (Inner Fit)",
    "lowvram": False,
    "processor_res": 64,             # 预处理分辨率
    "threshold_a": 64,               # Canny 阈值低
    "threshold_b": 64,               # Canny 阈值高
    "guidance": 1,                   # 引导强度
    "guidance_start": 0,             # 控制开始步数 (t_start)
    "guidance_end": 1,               # 控制结束步数 (t_end)
    "guessmode": False               # 启用/禁用 ControlNet guess mode
}
```

**公式关联**：
ControlNet 的加噪过程修改为：
$$
\epsilon_\theta(x_t, c, c_{\text{control}}) = \epsilon_\theta(x_t, c) + w \cdot \text{ControlNet}(x_t, c_{\text{control}})
$$
其中 $w$ 即 `weight` 参数，`guidance_start` 和 `guidance_end` 定义了 $t$ 的区间。

### 3.3 Blender Compositor 节点图

项目依赖 Blender 的节点编辑器生成以下四种图：

#### （1）Canny Edge Map
- 使用 **Edge Detect** 节点检测边缘
- 通过 **ColorRamp** 调整对比度，输出二值化线条

#### （2）Depth Map
- 利用 **Render Layers** → **Z** 通道（深度缓冲）
- 转换为灰度图并归一化

#### （3）OpenPose / Bone Map
- 使用 Blender 骨骼系统，通过 **Freestyle** 或 **Shader** 在关键点绘制圆点
- 连接骨骼的线段绘制

#### （4）Segmentation Map
- 通过 **ID Mask** 节点，基于物体 ID 生成彩色分割图
- 预定义 150 种颜色（`seg.py` 脚本生成材质）

---

## 四、安装与配置步骤技术细节

### 4.1 A1111 API 模式启动

编辑 `webui-user.bat`：
```bat
set COMMANDLINE_ARGS=--api
```
`--api` 标志启用 REST API，监听 `http://localhost:7860`。

### 4.2 ControlNet 扩展设置

在 Mikubill/sd-webui-controlnet 的 Settings 中：
1. 启用 **"Allow other script to control this extension"**
2. 修改 **"Multi ControlNet: Max models amount"** 增加支持数量（默认重启生效）

### 4.3 Blender 脚本集成

将 `multicn.py` 代码复制到 Blender 的 **Scripting** 工作区，修改参数：
- `IMAGE_FOLDER`: 输出路径（`//sd_results` 表示相对 blend 文件路径）
- 选择要发送的 ControlNet 类型（`is_send_canny`, `is_send_depth` 等）

---

## 五、示例使用场景

### 场景：房间室内设计

1. 在 Blender 中建模一个房间，放置家具
2. 设置摄像机角度，渲染得到基础图像
3. Compositor 同时生成：
   - **Depth Map**: 保留空间布局
   - **Segmentation**: 保留物体类别（床、桌子、墙）
4. 脚本发送到 SD WebUI：
   ```json
   {
     "prompt": "a cozy bedroom with modern furniture, photorealistic, 8k",
     "negative_prompt": "blurry, deformed, low quality",
     "alwayson_scripts": {
       "controlnet": {
         "args": [
           { "model": "diff_control_sd15_depth_fp16", "weight": 1.2, "image": "<base64_depth>" },
           { "model": "diff_control_sd15_seg_fp16", "weight": 1.0, "image": "<base64_seg>" }
         ]
       }
     }
   }
   ```
5. SD 生成与 3D 布局一致但风格更逼真的图像

---

## 六、技术优势与局限

### 6.1 优势
- **精确空间控制**: 深度图确保透视一致性
- **物体级控制**: 分割图允许对特定区域使用不同 prompt
- **非侵入式**: 不修改 SD WebUI 源码，仅通过 API 调用
- **Blender 集成**: 艺术家可在熟悉的环境工作

### 6.2 局限（当前项目状态）
- **动画支持未实现**（Todo 中）
- 依赖 A1111 本地运行，资源消耗大
- ControlNet 模型哈希必须匹配本地已下载模型
- 分辨率受限于 Blender 渲染尺寸

---

## 七、扩展可能性

### 7.1 实现动画支持
- 使用 Blender **Animation** 系统逐帧渲染
- 批量发送 API 请求，保持 `seed` 连续性
- 时序平滑：对 ControlNet weight 进行关键帧插值

### 7.2 多 ControlNet 融合策略
可 implement 加权混合：
$$
I_{gen} = \sum_{i=1}^{N} w_i \cdot \text{ControlNet}_\theta(i)(x_t, c_i)
$$
其中 $\sum w_i = 1$，或在 API 层直接并行发送。

---

## 八、参考链接

- **项目主页**: https://github.com/coolzilj/Blender-ControlNet
- **SD WebUI**: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- **ControlNet 扩展**: https://github.com/Mikubill/sd-webui-controlnet
- **ControlNet 论文**: https://arxiv.org/abs/2302.05543

---

## 九、总结

**Blender-ControlNet** 是一个桥梁工具，将 3D DCC（数字内容创作）与生成式 AI 深度结合。它利用 Blender 的强大渲染和节点系统作为**条件生成前端**，通过 SD WebUI 的 ControlNet 实现**风格化图像合成**。核心技术在于：
1. **Compositor 作为预处理管线**
2. **REST API 作为通信协议**
3. **ControlNet multi-unit 作为条件注入机制**

这为 3D 艺术家提供了一种高效的工作流：先精确建模布局，再用 AI 填充纹理和风格，做到**结构可控、创意自由**。