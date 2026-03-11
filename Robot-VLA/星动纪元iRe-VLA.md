# 二、强化学习应用在VLA的三大难点？

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rs2TibHDn2AkuLfEPq5hDSibGAsFqImkjl2DicMfHuUaUTDqKMKzo0x0Ig/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

图注：VLA RL的难点

理论上，强化学习（RL）可以让机器人通过与环境互动、试错来持续进步，**但是这其实不是一件容易的事情**。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rfNlI2UEoticcjXDzKLc0Cr7mSzkKSVK800wzWHBjtia8WV8bc37aJ2mQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

图注：LLM和具身在RL上的区别

将类似 GPT 这样的大模型与强化学习结合**（如 RLHF）在聊天机器人领域非常成功**，但在控制物理机器人时却困难重重：

- **环境差异**：聊天机器人是在离线数据集上训练的，而机器人需要在物理世界中实时探索。物理任务通常周期长、奖励稀疏（做完一整套动作才算成功），这使得学习非常困难。
    

- **模型坍塌与不稳定性**：研究发现，如果直接对巨大的 VLA 模型（数十亿参数）进行在线强化学习，模型很容易出现“灾难性遗忘”或训练崩溃，导致性能甚至不如微调前。
    

- **算力负担**：在本地机器上对几十亿参数的模型进行全量梯度的强化学习更新，对硬件要求极高，通常超出了本地机器人控制器的算力极限。
    

# 三、星动纪元iRe-VLA最先突破VLA强化学习困境，也是π*0.6的引用来源

## 对于VLA的强化学习困境，行业内其实有三种类型的解决方案：

- **第一种：外挂式干预：**一些尝试不敢轻易触碰庞大的 VLA 参数。比如**V-GPS (Value-Guided Policy Steering)**训练一个通用的价值函数，在推理时，让 VLA 生成多个候选动作，用价值函数对它们进行打分和重排序，选择最好的动作执行；**DSRL训练一个小型的 RL 策略来优化扩散模型的输入噪声，通过改变输入噪声来“引导”冻结的 VLA 生成高价值动作**。这种方法虽然安全，但 VLA没有真正发生质变。
    

- **第二种：暴力美学：**以 **VLAC** 为代表的工作尝试直接用 PPO 等算法全量微调 VLA。虽然勇气可嘉，但大模型在 RL 训练中极易出现灾难性遗忘和模型坍塌（Model Collapse），且对算力的要求很高。
    

- **第三种是从探索到内化的循环。**让我们眼前一亮的是一篇以前没有跟踪过的，清华和UC Berkeley的《Improving Vision-Language-Action Model with Online Reinforcement Learning》（通过在线强化学习改进视觉-语言-动作模型），来自于清华大学助理教授、星动纪元创始人陈建宇老师组。星动纪元这项研究是全球最早将在线RL引入VLA的工作，在ICRA发表，π*0.6 也引用了该工作，是中美两方在RL上的顶尖对话。
    

**这两篇文章代表了第三种路径。**它们不再盲目地套用 RL 算法，而是**利用监督微调（SFT）将 RL 探索出的高价值行为（成功轨迹或高优势动作）稳定地内化为模型的原生能力。**

π*0.6不在此详细赘述。我们来看下iRe-VLA。

**iRe-VLA的作者设计了一个两阶段循环迭代的学习流程。这个流程的核心思想是：分而治之，动静结合。**

## 星动纪元：iRe-VLA 模型架构设计

VLA 模型由两部分组成：

**VLM 主干（大脑）**：使用预训练的大型视觉-语言模型（如 BLIP-2），负责理解图像和指令，拥有丰富的世界知识。

**Action Head（四肢）**：一个轻量级的动作输出层（由 Token Learner 和 MLP 构成），负责将 VLM 的深层特征转化为具体的机器人控制信号（如机械臂的移动、夹爪的开合）。

为了提高效率，作者还使用了 **LoRA**（低秩适应）技术，避免全量微调所有参数。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2r2m8zDbNhsblkVkRVTCQvibX3Tl7MdsOaJHVOfz78zhFamBtrsuoU7WQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

图注：模型架构

## 核心流程：两个阶段的交替

iRe-VLA 方法不是一次性训练，而是在以下两个阶段中反复迭代：

### 第一阶段：在线强化学习（探索与发现）

  

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rGmWyHIFSsaBkWscCoYCSJB2rmC9dUy3OCv0XANFSptriayjeRs6CrkQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

  

图注：稳定探索

在这个阶段，机器人的目标是去试错，探索如何完成新任务。

- **冻结大脑（Freeze VLM）**：为了防止模型崩溃和减少计算量，作者**冻结**了巨大的 VLM 主干参数。
    
- **只练四肢（Train Action Head）**：仅训练轻量级的 Action Head。同时引入一个Critic Head（评价网络）来辅助训练。
    
- **优势**：因为只更新很少的参数，训练非常**稳定**，而且计算量很小，可以在本地机器（如单张 4090 显卡）上高效运行。机器人通过不断尝试，找到了一些能够成功完成任务的轨迹（Success Trajectories）。
    

### 第二阶段：监督学习（巩固与内化）

在第一阶段，机器人可能只是碰巧学会了操作，为了让这种能力真正融入模型，需要进行第二阶段。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rQibkwgnbSDKydJ9r0ZZGm9MvLjPIibJm7de8ich2hgAOq2Q5l3KYWExdw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

图注：融合与升华

- **全模型微调**：解冻 VLM 主干，对整个模型（包括 LoRA 参数）进行训练。
    
- **混合数据**：训练数据不仅包含第一阶段探索到的新成功轨迹，还混合了原始的专家示范数据。
    
- **优势**：这不仅利用了大模型的强大表达能力来记住新技能，还因为混合了旧数据，有效防止了灾难性遗忘（即学会了新任务，忘了旧任务）。这一步计算量大，通常放在云端服务器（如 A100 集群）上进行。
    
    ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rlUko37S1khx7riadRaqOPicsPgvvnB0VmcX7lqY6Cd1RsRqaeftGxjJg/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)
    

图注：两阶段

**总结**：机器人先在“小参数模式”下大胆探索（阶段1），找到方法后，再在“全参数模式”下把经验固化到大脑中（阶段2），如此循环往复。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2reUpUDtDoNlrbICGACUMxI4DmcTqPiaoR1nIeCcOptjhJFqjO54tHEqg/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=10)

图注：循环往复

# 三、 实验结果与分析

  

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rXxBv6krQicmNahMeEuibWSVIQRArE5F2UJ1sq9D4zEjdicwrLFEn67pmg/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=11)

图注：三种情况的实验结果分析

作者在仿真环境（MetaWorld, Franka Kitchen）和真实世界（Panda 机械臂）中进行了大量实验，验证了该方法的有效性。

## 训练稳定性对比

实验显示，如果使用标准的 PPO 算法直接微调 VLA 模型，成功率曲线震荡剧烈，甚至在很多任务上性能下降（变差了）。而 iRe-VLA 的曲线则稳步上升，证明了“分阶段冻结参数”对于稳定训练至关重要。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rbSBadSOC4rSJ5vGPLXx7A8VnJV7ZMcog2Gzok2CJ4R5YUWbqY7SG6w/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=12)

图注：曲线对比

## 仿真环境表现

  

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rDqIu7eFMEsL2qIcVRoOiaGr7707cna96gwMHGEicSk0lPpqQh1INhhiaQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=13)

图注：仿真环境中具备压倒性优势

**MetaWorld & Franka Kitchen**：在这些基准测试中，iRe-VLA 不仅在原本学过的任务上表现更好（例如从 43% 提升到 83%），还能通过在线探索学会完全没见过的任务。

**对比 SFT**：相比仅进行监督微调的模型，经过 iRe-VLA 迭代后的模型在所有任务类别（专家任务、RL 训练任务、未见过的测试任务）上的成功率都有显著提升。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2roNicRAXdNiaXgGK6ZweicHgWk2LVRHicWDWemFtbtuwMdRbffg2cOaV7fA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=14)

图注：不同后训练策略的对比

## 真实世界挑战（Real-World Panda）

这是最令人印象深刻的部分。作者让机器人去抓取它从未见过的物体（如形状不规则的茄子、胡萝卜）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rpuFqjtHltYOGfTOvCDicoiadzZcxQWOJ9z3l1XzNbVkJ4CNjnuujVQzA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=15)

图注：真实世界的提升

- **初始状态**：仅靠专家数据（SFT），机器人抓取这些新物体的成功率只有 35% 左右。
    

- **训练后**：经过 iRe-VLA 的在线学习（利用 SACfD 算法提高样本效率），抓取成功率飙升到了 80%。
    

- **泛化能力**：更有趣的是，训练后的模型去抓取**完全未参与训练**的第三类物体，成功率也从 37% 提升到了 61%。这说明通过强化学习，模型不仅学会了抓茄子，还变得更聪明、更通用了。
    
    ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2r2VXk8Yf40gIqqTZgug9PGO5juElSLg3dAR9wme6ic2BQMOtB6V0J9CQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)
    

图注：实验和成功率

## 消融实验：为什么要解冻 VLM？

作者做了一个对比实验：如果在第二阶段依然冻结 VLM，只训练 Action Head（即 iRe-VLA-freeze），效果如何？

结果显示，如果不解冻 VLM，模型的性能提升会遇到瓶颈。这证明了在第二阶段解冻大模型参数是必要的，这样才能利用大模型深层的特征表示能力来彻底掌握复杂技能，并提升泛化性。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2ra5A0q7lV0zh4NFceoicEAO0NueeVKpdkWdgwJcSuVhDzG7AoYibicUTicw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=17)

图注：消融实验

# 四、 结论与意义

这篇文章提出了一种切实可行的方案，解决了大模型在机器人控制中落地难的问题。

- **稳定性**：解决了大模型直接上 RL 容易训崩的问题。
    
- **经济性**：巧妙地分配了算力，让本地机器人负责轻量级探索，云端服务器负责重量级消化，符合实际部署场景。
    
- **持续学习**：证明了机器人可以通过自我探索，在不遗忘旧技能的前提下，不断掌握新物体和新任务的操作技能。
    
    ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/gtMexOnrtBZO8y1rdFxZsGiaCMft45F2rvVc2fQLe4s4eLfyW86DHxFAkGLU0Era2mR9eoD0UFkgRlrKanDJASA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=18)
    

图注：该架构的优点

国内的星动纪元的iRe-VLA 的基础上，海外的PI π*0.6，都为我们揭示出了VLA在线强化学习技术的发展前景。这条路还有很多未尽的研究话题，比如如何高效探索与稀疏奖励下的新技能学习，如何面向大规模 VLA 构造稳定可扩展 RL 算法等。


