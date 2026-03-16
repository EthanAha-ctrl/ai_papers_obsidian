



# OpenManus vs OpenClaw：它们是同类的东西吗？

基于我对两个项目的深入分析，**它们不是同类产品**。虽然都叫"OpenXX"并且都是AI Agent框架，但它们的设计哲学、技术栈和应用场景有**本质区别**。

## 📊 核心差异对比表

| 维度 | OpenManus | OpenClaw |
|------|-----------|----------|
| **定位** | 学术研究型 Agent框架 | 个人助手型 Agent平台 |
| **语言栈** | Python + LangChain | Node.js/TypeScript |
| **架构模式** | 双执行机制 (Agent/Flow) | Gateway 控制平面 + RPC |
| **目标用户** | 研究人员、开发者 | 个人用户、多设备用户 |
| **核心能力** | 复杂任务规划+RL调优 | 实时消息交互+多通道 |
| **运行模式** | 任务驱动 (batch/async) | 常驻服务 (always-on) |
| **数据流** | 轨迹数据集训练 | 实时对话流 |
| **部署方式** | 容器化/本地开发 | 本地服务+远程访问 |
| **许可证** | Apache 2.0 (开源) | MIT (开源) |

## 🔬 技术架构深度解析

### 1. OpenManus：分层Agent架构

```python
# 继承树结构
BaseAgent
    └─ ReActAgent (推理-行动循环)
        └─ ToolCallAgent (工具调用管理)
            └─ Manus (最终用户接口)
```

**双执行模式**：

**Agent模式**（`main.py`）：
```
User request → Manus.run() → ToolCallAgent.think() 
→ LLM决策工具 → act() → 执行 → 返回结果
```

**Flow模式**（`run_flow.py`）：
```
User request → PlanningFlow.create_initial_plan() 
→ 多步骤执行 (get_step → select_executor → execute) 
→ summary生成
```

**核心技术栈**：
- **LLM抽象层**：支持 OpenAI、Anthropic、本地模型
- **工具系统**：统一 `execute(name, input)` 接口
- **PlanningFlow**：状态机管理计划生命周期
- **数据集格式**：统一Agent轨迹格式（instruction, input, output, steps）

### 2. OpenClaw：Gateway控制平面架构

```
[多通道输入] 
    ↓
┌──────────────────────────────┐
│   Gateway (WS control plane) │
│   ws://127.0.0.1:18789       │
└─────────────┬────────────────┘
              │
    ┌─────────┼─────────┐
    ↓         ↓         ↓
Pi Agent   CLI    WebChat UI
(RPC)              (Control UI)
```

**核心子系统**：

1. **Gateway WebSocket网络**
   - 单一WS控制平面
   - 会话管理、存在感、配置、cron、webhook
   - Event-driven架构

2. **多通道适配层**
   - WhatsApp (Baileys)
   - Telegram (grammY)
   - Slack (Bolt)
   - Discord (discord.js)
   - Google Chat (Chat API)
   - Signal (signal-cli)
   - iMessage (imsg) + BlueBubbles
   - Teams, Matrix, Zalo等

3. **Pi Agent Runtime (RPC模式)**
   ```typescript
   // 简化版RPC调用
   interface PiAgent {
     invoke(tool: string, input: any): Promise<ToolResult>;
     streaming: boolean;  // 支持流式输出
   }
   ```

4. **设备节点系统**
   - iOS/Android节点：相机、屏幕录制、位置
   - macOS节点：`system.run`、通知、Canvas
   - 通过 `node.invoke` 调用设备本地能力

5. **Canvas + A2UI**
   - Agent驱动的可视化工作空间
   - 支持 `canvas.push/reset/eval/snapshot`
   - 实时交互界面

6. **Skills Registry (ClawHub)**
   - 技能自动发现和安装
   - 社区驱动的扩展系统

## 🎯 设计哲学对比

### OpenManus：任务优化导向
- **核心理念**：通过规划+执行完成复杂任务
- **数据驱动**：依赖轨迹数据集进行RL调优
- **研究性质**：关注Agent能力的边界和提升方法
- **批处理友好**：适合离线任务、benchmark测试
- **环境抽象**：支持标准Agent环境（Webshop、ALFWorld等）

### OpenClaw：人类交互导向
- **核心理念**：个人AI助手，随时响应
- **交互优先**：多通道、实时对话
- **本地控制**：强调隐私、低延迟、离线能力
- **设备集成**：深度整合移动设备和桌面能力
- **持续服务**：Daemon模式，24/7运行

## 🔧 关键技术特性对比

### 1. 推理能力

**OpenManus**：
```python
# ReAct模式
while not finished:
    thought = llm.generate(prompt_with_history)
    action = parse_action(thought)
    observation = tool.execute(action)
    history.add(thought, action, observation)
```

**OpenClaw**：
```typescript
// 实时流式响应
const stream = await agent.invoke({
  message: userInput,
  thinkingLevel: "high",  // 思考深度控制
  streaming: true
});
for await (const chunk of stream) {
  sendToChannel(chunk);
}
```

### 2. 工具系统

**OpenManus**：
- 工具有明确分类（browser、canvas、cron等）
- 通过 `available_tools` 动态选择
- ToolResult标准化格式

**OpenClaw**：
- 工具分为：Channel工具、Node工具、系统工具
- 权限模型：TCC (macOS)、Screen Recording等
- 设备本地工具vs远程工具

### 3. 状态管理

**OpenManus**：
- AgentState: IDLE, RUNNING, FINISHED
- PlanningTool管理计划状态（created、in_progress、completed）
- 面向任务生命周期

**OpenClaw**：
- Session模型：`main`会话、群组隔离
- Presence和typing indicators
- 会话持久化和上下文压缩

### 4. 安全模型

**OpenManus**：
- 主要关注任务执行安全性
- 工具权限管理
- 沙箱环境支持

**OpenClaw**：
- 多信道安全配对（`dmPolicy="pairing"`）
- 密码/Tailscale认证
- 设备权限系统（macOS TCC）
- Webhook安全、会话剪枝

## 📈 数据规模与生态

### OpenManus
- **数据集**：50,793条轨迹（4-20轮次）
- **环境**：Webshop、ALFWorld、GAIA、OSWorld
- **社区**：3,899 stars（OpenManus-RL）
- **组织**：OpenManus/Ulab-UIUC + MetaGPT
- **发布频率**：研究导向，定期更新

### OpenClaw
- **stars**：约3,000（快速上升）
- **频道支持**：12+即时通讯平台
- **设备支持**：macOS/iOS/Android/Web
- **发布频道**：stable/beta/dev（semantic versioning）
- **社区**：Discord活跃、Nix包支持

## 🎪 典型使用场景

### OpenManus适合：
1. **学术研究**：Agent能力benchmark、RL算法实验
2. **复杂任务自动化**：研究数据收集、多步骤工作流
3. **强化学习训练**：轨迹数据生成、策略优化
4. **Agent环境开发**：新环境集成、工具扩展
5. **大规模批处理**：批量任务执行、结果分析

### OpenClaw适合：
1. **个人助理**：日程管理、信息检索、快速问答
2. **多平台同步**：手机/电脑无缝切换
3. **实时监控**：系统通知、消息转发
4. **家庭自动化**：通过节点控制智能家居
5. **团队协作**：多通道群组管理、自动化回复

## 🔮 未来发展方向

### OpenManus
- **RL调优**：基于PPO/DPO的Agent策略优化
- **多Agent协作**：Agent间协调和数据共享
- **环境扩展**：更多模拟环境（机器人、游戏）
- **Benchmark标准化**：统一评估标准

### OpenClaw
- **AI硬件集成**：更多设备类型的节点支持
- **语音增强**：实时语音交互优化
- **隐私计算**：本地模型部署、联邦学习
- **生态系统**：技能商店、社区插件市场

## 💎 结论

**它们是"远亲"而非"同类"**：

- **共同点**：
  - 都使用LLM作为推理引擎
  - 都支持工具调用能力
  - 都是开源项目
  - 都关注Agent能力边界

- **本质差异**：
  - OpenManus是**研究框架**（research framework），面向Agent能力的**系统性提升**
  - OpenClaw是**应用平台**（application platform），面向**个人用户的便利性**

你可以把它们类比为：
- **OpenManus ≈ TensorFlow/PyTorch**（底层研究框架）
- **OpenClaw ≈ Siri/Alexa**（终端用户产品）

在实际应用中，你可以：
1. 使用 **OpenManus** 研究新的Agent算法，生成高质量轨迹数据
2. 将这些数据用于训练 **OpenClaw** 的Agent能力
3. 最终部署 **OpenClaw** 给终端用户使用

两者可以**互补**而非竞争关系。

---

**参考链接**：
- OpenManus GitHub: https://github.com/OpenManus/OpenManus-RL
- OpenClaw GitHub: https://github.com/openclaw/openclaw
- OpenManus文档: https://openmanus.org
- OpenClaw文档: https://docs.openclaw.org