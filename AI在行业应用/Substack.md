




让我尝试直接解析该链接：



搜索工具暂时没有返回结果，但我对 Substack 非常了解，让我基于第一性原理来拆解这个平台：

---

## Substack 是什么？

**一句话概括：** Substack 是一个让独立创作者（writer, journalist, podcaster）通过 **paid newsletter（付费邮件订阅）** 模式直接从读者获得收入的平台。

---

### 🧠 第一性原理解读：为什么 Substack 存在？

传统媒体生态有一个核心矛盾：

$$
\text{Creator} \xrightarrow{\text{content}} \text{Platform (Ad-driven)} \xrightarrow{\text{attention}} \text{Advertiser} \xrightarrow{\$} \text{Creator}
$$

这个链条中，Creator 的收入被 Platform 和 Advertiser 中间层抽走，AND Creator 对自己 audience 没有 direct ownership（用户关系属于平台）。

Substack 的核心洞察是：**如果 Creator 直接拥有 subscriber 的 email address，并直接收费，中间层就可以被消除。**

$$
\text{Creator} \xrightarrow{\text{content via email}} \text{Subscriber} \xrightarrow{\$/month} \text{Creator}
$$

**Substack 只在这个直接关系中充当基础设施 provider，而非 content gatekeeper。**

---

### 🏗️ 核心功能架构

| 模块 | 功能 | 技术要点 |
|------|------|----------|
| **Publishing** | Web-based editor，支持 rich text, images, embeds, podcast audio | Markdown-like editor，输出为 web post + email |
| **Email Delivery** | 每篇 post 自动发送给所有 subscriber | 底层使用大规模 email sending infrastructure（自有 + 第三方如 Amazon SES） |
| **Payments** | Stripe 集成，支持 free / paid subscription tier | 订阅价格由 creator 自定，通常 $5–$50/month |
| **Podcast Hosting** | 内建 podcast RSS feed 生成 | audio file hosting + Apple Podcasts/Spotify 分发 |
| **Community** | Comments, Chat, Notes（类似 Twitter 的 social feed） | 构建 creator-centric 社区，增加 engagement |
| **Analytics** | Open rate, click rate, subscriber growth dashboard | 关键 metric：$\text{Open Rate} = \frac{\text{Emails Opened}}{\text{Emails Delivered}}$ |

---

### 💰 商业模式

Substack 的 revenue model 极其简单：

$$
R_{\text{Substack}} = \sum_{i=1}^{N_{\text{paid\_creators}}} 0.10 \times S_i
$$

其中：
- $N_{\text{paid\_creators}}$ = 使用 paid subscription 功能的 creator 数量
- $S_i$ = 第 $i$ 个 creator 的总 subscription revenue
- **10%** = Substack 的 take rate（从付费订阅收入中抽取）

对比传统媒体/平台：
- YouTube：约 45% take rate（ad revenue share）
- Apple App Store：30% take rate
- Substack：**仅 10%**，AND creator 拥有 email list，随时可以离开

这就是 Substack 的核心价值主张：**ownership + simplicity + low friction**。

---

### 🔑 关键设计哲学

1. **Creator Owns the Audience**：subscriber 的 email address 可以随时导出，creator 不是平台的 hostage
2. **No Algorithm**：不像 social media 靠 algorithm 决定谁看到什么，Substack 的 content delivery 是 **deterministic** 的——你订阅了，你就收到邮件
3. **Free Speech Stance**：Substack 采取 minimal content moderation 立场，这既是卖点也是争议点
4. **Network Effect via Notes**：2023 年推出的 Notes 功能尝试在独立 newsletter 之间建立 discovery network，形成 $\text{Creator}_A \to \text{Cross-promote} \to \text{Creator}_B$ 的网络效应

---

### 📊 一些关键数据（截至 2025 年大致范围）

| 指标 | 数值 |
|------|------|
| Active paid subscriptions | ~3M+ |
| Creators earning $100K+/year | ~100+ |
| Total creators on platform | ~35,000+（活跃发布者） |
| Company valuation | ~$650M（2021 Series B） |
| Founded | 2017 by Chris Best, Jairaj Sethi, Hamish McKenzie |

---

### 🤔 Substack vs 竞品对比

| 维度 | Substack | Ghost | Beehiiv | Patreon |
|------|----------|-------|---------|---------|
| Take rate | 10% | 0%（self-hosted） | 0-8% | 5-12% |
| Email ownership | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| Self-hosting | ❌ No | ✅ Yes | ❌ No | ❌ No |
| Podcast 支持 | ✅ Yes | ❌ No | ❌ No | ❌ Limited |
| Social/Discovery | ✅ Notes | ❌ No | ✅ Limited | ✅ Limited |
| 开源 | ❌ No | ✅ Yes | ❌ No | ❌ No |

---

### 总结

**Substack 本质上是一个 "email-first 的创作者经济基础设施"**——它把 newsletter 这个最古老的互联网内容形式，与现代支付系统（Stripe）和社区功能结合，让独立创作者可以在不依赖广告和算法的情况下，直接从读者那里获得可持续收入。

> 核心公式：**Direct Relationship + Paid Subscription + Low Take Rate = Creator Independence**

参考链接：
- Substack About 页面：https://substack.com/about
- Substack 官方博客：https://on.substack.com
- Wikipedia：https://en.wikipedia.org/wiki/Substack