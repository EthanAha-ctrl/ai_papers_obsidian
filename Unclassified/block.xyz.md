







搜索引擎暂时没有返回结果，但我对 **Block, Inc. (block.xyz)** 非常熟悉，下面给你做一个全面的深度解析。

---

# Block, Inc. — 全面解析

## 一、公司概况

**Block, Inc.**（NYSE: **SQ**），前身是 **Square, Inc.**，由 **Jack Dorsey** 和 **Jim McKelvey** 于 **2009** 年创立。2021年12月，公司正式更名为 **Block**，官网迁移至 **block.xyz**。更名的原因是公司业务已远超原来的 "Square" 品牌（POS card reader），需要一个大伞品牌来涵盖多个独立业务线。

---

## 二、核心业务线（五大板块）

### 1. 🟩 Square — Seller Ecosystem（商家/卖方生态系统）

这是 Block 的起家业务，也是公司最核心的收入引擎。

**核心产品矩阵：**

| 产品 | 功能 | 目标用户 |
|------|------|----------|
| Square POS (Point of Sale) | 移动/固定端收银系统 | 中小商家 |
| Square Reader | 磁条/芯片/NFC card reader，插入手机即可刷卡 | 个体商户 |
| Square Terminal | 独立无线POS终端 | 零售/餐饮 |
| Square Register | 一体化POS硬件 | 中型商家 |
| Square Online | 电商建站工具 | 线上线下融合的商家 |
| Square Invoices | 电子发票 | 服务型商家 |
| Square Banking | 商业贷款、储蓄账户 | 需流动资金的商家 |

**商业模式核心公式：**

$$\text{Revenue}_{\text{Square}} = \sum_{i=1}^{N} (T_i \times r_i) + H_i$$

其中：
- $T_i$ = 第 $i$ 个商家的 **Gross Payment Volume (GPV)**（总交易额）
- $r_i$ = **Take Rate**（费率），通常为 **2.6% + $0.10** 每笔交易
- $H_i$ = 来自硬件销售、订阅服务的收入
- $N$ = 活跃商家数量

> 关键直觉：Square 的收入与商家 GMV (Gross Merchandise Volume) 正相关，是一个 **transaction-based revenue model**。

---

### 2. 💚 Cash App — Consumer Ecosystem（消费者生态系统）

Cash App 是 Block 的第二个增长引擎，面向个人消费者。

**功能模块：**

| 功能 | 描述 |
|------|------|
| P2P Payment | 个人间转账（类Venmo） |
| Cash Card | Visa借记卡，直接从Cash App余额消费 |
| Bitcoin Trading | 买卖Bitcoin（Block从中收取spread） |
| Stock Investing | 零手续费股票投资（碎片化股份） |
| Cash App Savings | 高收益储蓄账户 |
| Borrow | 小额短期贷款 |
| Cash App Taxes | 免费报税工具（原Credit Karma Tax） |

**Cash App 商业模式核心公式：**

$$\text{Revenue}_{\text{Cash App}} = \underbrace{B_{\text{trading}}}_{\text{Bitcoin交易}} + \underbrace{F_{\text{Cash Card}}}_{\text{Debit card interchange fee}} + \underbrace{S_{\text{subscription}}}_{\text{订阅服务}} + \underbrace{L_{\text{Borrow}}}_{\text{贷款利息}}$$

> **关键洞察**：Cash App 的 Bitcoin Revenue 看似巨大（数十亿美元），但 **Bitcoin Gross Profit** 很低（margin ~1-2%），因为 Block 本质上是 Bitcoin 的 "pass-through" 中介。真正赚钱的是 **Bitcoin spread**：

$$\text{Bitcoin Spread} = P_{\text{sell}} - P_{\text{buy}} - P_{\text{market}}$$

其中 $P_{\text{sell}}$ 是卖给用户的价格，$P_{\text{buy}}$ 是从市场购入的价格。

---

### 3. 🎵 TIDAL — Music Streaming

2021年，Block 收购了 **TIDAL**（Jay-Z 创立的高保真音乐流媒体平台）的大部分股权。

- 高保真 / 无损音频流媒体（**HiFi / MQA / Dolby Atmos**）
- 艺人优先的版权分成模型
- 目前尚未成为主要收入贡献者
- 战略目的：探索 **创作者经济 (Creator Economy)**，与 Cash App / Square 形成协同

---

### 4. ⚡ Spiral — Open-Source Bitcoin Development

原名 **Square Crypto**，后更名为 **Spiral**。

- 使命：**contribute to Bitcoin open-source software**
- 不追求直接利润，而是投资 Bitcoin 生态的基础设施
- 代表项目：
  - **Lightning Development Kit (LDK)**：让开发者更容易集成 Lightning Network
  - **Bitcoin Development Kit (BDK)**：Bitcoin钱包开发工具包

**第一性原理分析：**

$$U_{\text{Block}} = f(\text{Bitcoin Adoption}) \times g(\text{Block's Position in Ecosystem})$$

Block 押注：如果 Bitcoin 被大规模采用，Block 作为最早拥抱 Bitcoin 的金融科技公司，将获得巨大的 **strategic optionality**（战略期权价值）。

---

### 5. 🔮 TBD — Decentralized Platform

TBD 是 Block 的子公司，由 **Mike Brock** 领导，专注于构建 **decentralized identity** 和 **decentralized data** 的基础设施。

核心项目：**tbDEX** — 一个 **decentralized exchange protocol**

**tbDEX 的设计哲学：**

传统的 fiat-to-crypto on-ramp 需要通过中心化交易所（CEX），存在以下问题：
- KYC/AML 合规成本高
- 用户隐私泄露
- 中心化点被攻击风险

tbDEX 的方案：

$$\text{P2P Fiat-Crypto Exchange} = \underbrace{\text{Decentralized Messaging}}_{\text{dfinity/secure comm}} + \underbrace{\text{Verifiable Credentials (VC)}}_{\text{DID-based identity}} + \underbrace{\text{Reputation System}}_{\text{Trust without central authority}}$$

- **DID (Decentralized Identifier)**：W3C 标准，让用户控制自己的身份信息
- **Verifiable Credentials**：可验证凭证，让用户选择性披露信息（例如：证明 >18岁，而不暴露具体年龄）

---

## 三、Block 的整体商业架构图

```
                    ┌─────────────────────┐
                    │     Block, Inc.      │
                    │   (block.xyz)        │
                    │  NYSE: SQ            │
                    └──────────┬───────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐    ┌───────▼───────┐    ┌──────▼──────┐
    │   Square    │    │   Cash App    │    │   TIDAL     │
    │ (Seller)    │    │ (Consumer)   │    │ (Music)     │
    │ B2B         │    │ B2C          │    │ B2C         │
    └─────────────┘    └───────────────┘    └─────────────┘
           │                   │
    ┌──────▼──────┐    ┌───────▼───────┐
    │   Spiral    │    │     TBD       │
    │ (Bitcoin    │    │ (Decentralized│
    │  Open Src)  │    │  Platform)   │
    └─────────────┘    └───────────────┘
```

**飞轮效应 (Flywheel)：**

$$\text{Square Sellers} \xrightarrow{\text{accept Cash App}} \text{Cash App Users} \xrightarrow{\text{buy Bitcoin}} \text{Bitcoin Ecosystem} \xrightarrow{\text{Spiral/TBD builds infra}} \text{More adoption}$$

---

## 四、财务关键指标（近年趋势）

| 指标 | FY2022 | FY2023 | 说明 |
|------|--------|--------|------|
| 总收入 | ~$17.6B | ~$21.9B | 含大量Bitcoin pass-through revenue |
| 总Gross Profit | ~$5.8B | ~$7.5B | 看profit而非revenue |
| Cash App MAU | ~51M | ~56M | 月活用户 |
| Square GPV | ~$210B | ~$220B | Gross Payment Volume |
| Bitcoin Revenue | ~$7.1B | ~$9.5B | 但profit margin仅~1-2% |
| 毛利率 | ~33% | ~34% | 逐步提升 |

> ⚠️ **关键陷阱**：很多投资者看到 Block 的 revenue 巨大就以为公司很赚钱，但实际上 **Bitcoin Revenue 是低毛利业务**。正确做法是看 **Gross Profit** 和 **Gross Margin**。

---

## 五、竞争优势（Moat Analysis）

### 1. **Network Effects (网络效应)**

Square 的商家越多 → Cash App 用户越多 → 更多商家接受 Cash App Pay → 更多商家使用 Square

$$V_{\text{network}} \propto n^2 \quad (\text{Metcalfe's Law})$$

### 2. **Switching Costs (转换成本)**

商家一旦使用 Square 的整套系统（POS + Online + Banking + Payroll），迁移成本极高。

### 3. **Data Advantage (数据优势)**

Block 掌握两端的交易数据：
- **Square 端**：商家的销售数据、现金流
- **Cash App 端**：消费者的消费行为

这些数据使 Block 在 **underwriting（放贷评估）** 方面有独特优势，例如 Square Loan 的审批可以在几分钟内完成。

$$P(\text{default}) = f(\underbrace{\text{transaction history}}_{\text{Square data}}, \underbrace{\text{spending patterns}}_{\text{Cash App data}})$$

---

## 六、与竞争对手对比

| 维度         | Block (Square/Cash App) | Stripe | PayPal/Venmo   | Adyen      |
| ---------- | ----------------------- | ------ | -------------- | ---------- |
| 目标客户       | SMB + Consumer          | 全栈开发者  | Consumer + SMB | Enterprise |
| 硬件生态       | ✅ 强                     | ❌      | ❌              | ❌          |
| Bitcoin 集成 | ✅ 深度                    | ❌      | 有限             | ❌          |
| P2P        | Cash App                | ❌      | Venmo          | ❌          |
| 开放协议       | TBD/tbDEX               | ❌      | ❌              | ❌          |

---

## 七、战略方向总结

Block 的长期战略可以用一个公式概括：

$$\text{Block's Thesis} = \underbrace{\text{Financial Inclusion}}_{\text{serve the underserved}} + \underbrace{\text{Bitcoin as Money}}_{\text{long-term bet}} + \underbrace{\text{Decentralization}}_{\text{TBD/Spiral}}$$

1. **短期**：Square 和 Cash App 继续驱动收入增长，扩大用户基础
2. **中期**：Cash App 成为"超级APP"（类似微信支付/支付宝），整合支付、投资、储蓄、报税
3. **长期**：Bitcoin 成为全球互联网的原生货币，Block 是最大的基础设施提供者和入口

---

## 八、总结一句话

> **Block (block.xyz) 是一家以技术驱动的金融科技公司，通过 Square 服务商家、Cash App 服务消费者、TIDAL 服务音乐创作者、Spiral 和 TBD 构建去中心化/Bitcoin 基础设施，最终押注于一个更加开放、去中心化、包容的全球金融体系。**

---

**参考链接：**
- 官网：https://block.xyz
- Investor Relations：https://investor.block.com
- Cash App：https://cash.app
- Square：https://squareup.com
- Spiral：https://spiral.xyz
- TBD：https://tbd.website
- TIDAL：https://tidal.com