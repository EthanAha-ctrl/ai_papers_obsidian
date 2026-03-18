# Carta 公司深度解析

## 一、公司基本概况

**Carta** (原名 eShares, Inc.) 是一家美国的 **financial technology** 公司，专注于 **equity management** 和 **cap table management** 领域。

### 基本信息：
- **Founded**: 2012年
- **Founder**: Henry Ward (CEO)
- **Headquarters**: San Francisco, California
- **Former Name**: eShares, Inc.
- **Business Model**: B2B SaaS (Software as a Service)
- **Valuation**: 约 $7.9 billion (截至 2021年 Series G)

---

## 二、Core Product & Business Lines

### 2.1 Cap Table Management (核心产品)

**Cap Table** (Capitalization Table) 是 Carta 的 foundational product。

**数学模型**：

$$\text{Ownership Percentage} = \frac{N_{shares}}{N_{total\_fully\_diluted}} \times 100\%$$

其中：
- $N_{shares}$ = 某股东持有的股份数
- $N_{total\_fully\_diluted}$ = 完全稀释后的总股份数

**Fully Diluted Share Count 计算**：

$$N_{fully\_diluted} = N_{common} + N_{preferred} + N_{options} + N_{warrants} + N_{convertible\_notes}$$

### 2.2 产品矩阵

| Product Line | Description | Target Customer |
|-------------|-------------|-----------------|
| Carta Launch | 免费 cap table 工具 | Early-stage startups |
| Carta Pro | 高级 equity 管理 | Growth-stage companies |
| Carta Enterprise | 复杂 equity 结构 | Large corporations |
| Carta Fund Admin | VC/PE fund administration | Venture Capital firms |
| Carta Total Compensation | Compensation benchmarking | HR & Finance teams |
| Carta Liquidity | Secondary market transactions | Employees & Investors |

---

## 三、Technical Architecture 深度解析

### 3.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Web App    │  │  Mobile App │  │  API Integrations   │  │
│  │  (React)    │  │  (React     │  │  (REST/GraphQL)     │  │
│  │             │  │   Native)   │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Kong API Gateway / AWS API Gateway                     ││
│  │  - Authentication (OAuth 2.0, JWT)                      ││
│  │  - Rate Limiting                                        ││
│  │  - Request Routing                                      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Microservices Layer                       │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────┐ │
│  │  User     │ │  Cap      │ │  Equity   │ │  Compliance │ │
│  │  Service  │ │  Table    │ │  Plan     │ │  Service    │ │
│  │           │ │  Service  │ │  Service  │ │             │ │
│  └───────────┘ └───────────┘ └───────────┘ └─────────────┘ │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────┐ │
│  │  Document │ │  Valuation│ │  Reporting│ │  Notification│ │
│  │  Service  │ │  Service  │ │  Service  │ │  Service    │ │
│  └───────────┘ └───────────┘ └───────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  PostgreSQL │  │  Redis      │  │  Elasticsearch      │  │
│  │  (Primary)  │  │  (Cache)    │  │  (Search)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  S3         │  │  Snowflake  │  │  Kafka              │  │
│  │  (Documents)│  │  (Analytics)│  │  (Event Stream)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Equity Valuation Methodology

Carta 使用多种 **valuation methods**：

#### 3.2.1 409A Valuation (Fair Market Value)

**Black-Scholes Model for Option Pricing**：

$$C = S_0 N(d_1) - X e^{-rT} N(d_2)$$

$$P = X e^{-rT} N(-d_2) - S_0 N(-d_1)$$

其中：
- $C$ = Call option price
- $P$ = Put option price
- $S_0$ = Current stock price
- $X$ = Strike price
- $r$ = Risk-free interest rate
- $T$ = Time to maturity (years)
- $N(d)$ = Cumulative standard normal distribution function
- $\sigma$ = Volatility

$$d_1 = \frac{\ln(S_0/X) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

#### 3.2.2 OPM (Option Pricing Method) for Allocating Enterprise Value

```
┌─────────────────────────────────────────────────────────┐
│                Enterprise Value                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │                  Senior Debt                      │ │
│  │        (First claim on assets)                    │ │
│  └───────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐ │
│  │              Preferred Stock                      │ │
│  │        (Liquidation preference)                   │ │
│  └───────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐ │
│  │              Common Stock                         │ │
│  │        (Residual claim)                           │ │
│  └───────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐ │
│  │              Options/Warrants                     │ │
│  │        (Conversion rights)                        │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 四、Business Model & Unit Economics

### 4.1 Revenue Model

**Pricing Structure** (示例):

| Tier | Monthly Price | Features |
|------|--------------|----------|
| Launch | $0 | Basic cap table, up to 25 stakeholders |
| Starter | $99/month | Up to 50 stakeholders, basic reporting |
| Growth | $249/month | Up to 100 stakeholders, advanced analytics |
| Scale | Custom | Unlimited stakeholders, enterprise features |

### 4.2 Unit Economics 分析

**LTV (Lifetime Value) 计算**：

$$LTV = \frac{ARPU \times Gross Margin}{Churn Rate}$$

其中：
- $ARPU$ = Average Revenue Per User
- $Gross Margin$ = 毛利率 (SaaS 公司通常 >70%)
- $Churn Rate$ = 客户流失率

**CAC (Customer Acquisition Cost)**：

$$CAC = \frac{\text{Total Sales \& Marketing Cost}}{\text{Number of New Customers}}$$

**LTV/CAC Ratio** (健康标准):

$$\frac{LTV}{CAC} \geq 3$$

### 4.3 Financial Data (估算)

| Metric | Value (Est.) | Notes |
|--------|-------------|-------|
| ARR (Annual Recurring Revenue) | ~$400M+ | 2023 estimate |
| Gross Margin | ~75-80% | Typical SaaS |
| NRR (Net Revenue Retention) | ~120%+ | Strong expansion revenue |
| Employees | ~2,500+ | Global workforce |

---

## 五、Competitive Landscape

### 5.1 竞争对手对比

| Company | Strengths | Weaknesses | Market Position |
|---------|-----------|------------|-----------------|
| **Carta** | Comprehensive product, strong brand, large network | Complex for small startups | Market Leader |
| **Pulley** | Simpler UX, lower price point | Fewer features | Challenger |
| **AngelList** | Integrated with fundraising | Less cap table focus | Niche |
| **Shareworks** (Morgan Stanley) | Enterprise features | Complex, expensive | Enterprise |
| **Equity Edge** | Mid-market focus | Limited brand awareness | Mid-market |
| **Capshare** (acquired by Solium) | Legacy installed base | Outdated tech | Declining |

### 5.2 Network Effects 分析

```
                    ┌─────────────────┐
                    │   Startups      │
                    │  (More Data)    │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────┐
│                    CARTA PLATFORM                     │
│                                                       │
│   Network Effect:                                    │
│   More Startups → More VCs → More Startups          │
│                                                       │
└─────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     VCs         │
                    │  (More Portfolio│
                    │   Management)   │
                    └─────────────────┘
```

---

## 六、Technology Stack (推断)

### 6.1 Backend Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python, Go | Primary backend languages |
| Framework | Django, Flask | Web framework |
| API | REST, GraphQL | API design |
| Database | PostgreSQL | Primary data store |
| Cache | Redis | Session & data cache |
| Queue | Celery, RabbitMQ | Task processing |
| Search | Elasticsearch | Full-text search |

### 6.2 Frontend Technologies

| Layer | Technology |
|-------|-----------|
| Framework | React.js |
| State Management | Redux, Context API |
| Styling | CSS Modules, Styled Components |
| Testing | Jest, Cypress |

### 6.3 Infrastructure

| Component | Technology |
|-----------|-----------|
| Cloud Provider | AWS (primary) |
| Container | Docker, Kubernetes |
| CI/CD | Jenkins, CircleCI |
| Monitoring | Datadog, PagerDuty |
| Logging | ELK Stack |

---

## 七、Key Algorithms & Technical Challenges

### 7.1 Waterfall Analysis Algorithm

**Liquidation Waterfall** 是 equity management 的核心技术难题：

```
Algorithm: Calculate Waterfall Distribution

Input: Exit Value E, Cap Table C, Liquidation Preferences L

1. Sort securities by seniority (debt → preferred → common)
2. For each security class i (from most senior):
   a. Calculate payout P_i = min(L_i, remaining_value)
   b. remaining_value -= P_i
   c. If conversion is beneficial:
      - Calculate conversion_value
      - Choose max(P_i, conversion_value)
3. Distribute remaining to common shareholders
4. Apply participation rights if applicable

Output: Distribution amounts for each shareholder
```

### 7.2 Pro Forma Cap Table Calculation

**Post-Money Valuation**:

$$V_{post} = \frac{I}{F}$$

其中：
- $V_{post}$ = Post-money valuation
- $I$ = Investment amount
- $F$ = Fraction of company purchased

**Price Per Share**:

$$P_{share} = \frac{V_{pre}}{N_{pre}}$$

其中：
- $V_{pre}$ = Pre-money valuation
- $N_{pre}$ = Pre-money fully diluted shares

### 7.3 Dilution Modeling

**Dilution Effect Formula**:

$$Ownership_{new} = \frac{N_{existing}}{N_{existing} + N_{new}} \times Ownership_{old}$$

**Anti-Dilution Protection** (Weighted Average):

$$P_{new} = \frac{N_{old} \times P_{old} + N_{new} \times P_{down}}{N_{old} + N_{new}}$$

其中：
- $P_{new}$ = Adjusted conversion price
- $P_{old}$ = Original conversion price
- $P_{down}$ = Price in down round
- $N_{old}$ = Original shares
- $N_{new}$ = New shares issued

---

## 八、Data Flow Architecture

### 8.1 Equity Transaction Processing

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Document   │───▶│   Parse &    │───▶│   Validate   │
│   Upload     │    │   Extract    │    │   Rules      │
│  (PDF/Doc)   │    │   (OCR/NLP)  │    │   Engine     │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                               ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Generate   │◀───│   Calculate  │◀───│   Cap Table  │
│   Reports    │    │   Ownership  │    │   Update     │
│  (409A, etc) │    │   Changes    │    │   (Database) │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 8.2 Real-time Cap Table Sync

**Conflict Resolution Strategy**:

```
┌─────────────────────────────────────────────────────────┐
│                  Event Sourcing Model                    │
│                                                          │
│   Event Log:                                            │
│   [E1, E2, E3, E4, E5, ...]                             │
│                                                          │
│   State = f(Event Log)                                  │
│                                                          │
│   Advantages:                                           │
│   - Audit trail                                         │
│   - Time-travel queries                                 │
│   - Conflict resolution via event replay                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 九、Compliance & Security

### 9.1 Regulatory Compliance

| Regulation | Description | Carta's Compliance Approach |
|-----------|-------------|----------------------------|
| **SEC** | Securities regulations | Broker-dealer license for certain services |
| **409A** | IRS valuation requirements | Built-in 409A valuation services |
| **SOX** | Sarbanes-Oxley Act | Audit trails, internal controls |
| **GDPR** | EU data protection | Data handling procedures |
| **CCPA** | California privacy law | Privacy controls for CA residents |
| **SOC 2** | Security framework | Type II certification |

### 9.2 Security Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Security Layers                       │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 1: Network Security                        │  │
│  │  - VPC isolation                                  │  │
│  │  - WAF (Web Application Firewall)                 │  │
│  │  - DDoS protection                                │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 2: Application Security                    │  │
│  │  - OAuth 2.0 / SAML SSO                           │  │
│  │  - MFA (Multi-Factor Authentication)              │  │
│  │  - RBAC (Role-Based Access Control)               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 3: Data Security                           │  │
│  │  - Encryption at rest (AES-256)                   │  │
│  │  - Encryption in transit (TLS 1.3)                │  │
│  │  - Key management (AWS KMS)                       │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 4: Audit & Monitoring                      │  │
│  │  - Comprehensive logging                          │  │
│  │  - Real-time alerting                             │  │
│  │  - Third-party penetration testing                │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 十、Funding History & Valuation

### 10.1 Funding Rounds

| Round | Date | Amount | Valuation | Lead Investor |
|-------|------|--------|-----------|---------------|
| Series A | 2014 | ~$4M | Undisclosed | Various |
| Series B | 2015 | $14.5M | ~$70M | Spark Capital |
| Series C | 2017 | $42M | ~$240M | Social Capital |
| Series D | 2018 | $80M | ~$800M | Thrive Capital |
| Series E | 2019 | $300M | $1.7B | Andreessen Horowitz |
| Series F | 2020 | $150M | $3.1B | Lightspeed |
| Series G | 2021 | $500M | $7.9B | Silver Lake |

### 10.2 Cap Table Evolution (模拟)

假设公司从 Seed 到 Series G 的 founder dilution 路径：

$$Ownership_{founder}^{final} = Ownership_{seed} \times \prod_{i=1}^{n} (1 - d_i)$$

其中 $d_i$ = 第 i 轮的稀释比例

---

## 十一、Key Challenges & Controversies

### 11.1 Data Privacy Concerns (2024)

Carta 曾面临关于 **secondary market data usage** 的争议，被指控利用客户数据来推广自己的流动性产品。

**Key Issues**:
- Data ownership boundaries
- Conflicts of interest
- Transparency concerns

### 11.2 Competitive Pressure

**Disruption Threats**:
- **Pulley**: 更简单的 UI，更低的价格
- **AngelList Stack**: 一体化 startup 服务
- **In-house solutions**: 大公司自建系统

---

## 十二、Future Strategy & Market Opportunity

### 12.1 TAM (Total Addressable Market) Analysis

**Market Sizing**:

$$TAM = N_{startups} \times ARPU_{avg} + N_{VCs} \times ARPU_{VC} + N_{public\_cos} \times ARPU_{enterprise}$$

估算：
- Startups globally: ~300,000+ venture-backed
- VC firms: ~10,000+ globally
- Public companies: Potentially 40,000+ globally

### 12.2 Product Roadmap (推测)

```
┌─────────────────────────────────────────────────────────┐
│                   Product Evolution                       │
│                                                          │
│   Phase 1 (2012-2016):                                  │
│   ▸ Core Cap Table Management                           │
│                                                          │
│   Phase 2 (2016-2019):                                  │
│   ▸ 409A Valuations                                     │
│   ▸ Scenario Modeling                                   │
│   ▸ Board Management                                    │
│                                                          │
│   Phase 3 (2019-2022):                                  │
│   ▸ Fund Administration                                 │
│   ▸ Total Compensation                                  │
│   ▸ Liquidity Solutions                                 │
│                                                          │
│   Phase 4 (2022-Future):                                │
│   ▸ International Expansion                             │
│   ▸ Public Company Readiness                            │
│   ▸ AI-Powered Insights                                 │
│   ▸ Integrated Financial Services                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 十三、First Principles Analysis

### 13.1 从第一性原理看 Carta 的价值

**Problem Statement**:
- Equity ownership 记录 historically 是 spreadsheet-based
- 复杂的 cap table 难以追踪和更新
- Compliance 需要专业知识
- Information asymmetry between founders, employees, and investors

**Solution Layer**:

$$Value = \frac{Accuracy \times Speed \times Compliance}{Cost \times Complexity}$$

Carta 通过数字化和自动化提升 numerator，降低 denominator。

### 13.2 Moat Analysis

| Moat Type | Strength | Description |
|-----------|----------|-------------|
| **Network Effects** | High | 更多公司使用 → 更多 VC 使用 → 更多公司使用 |
| **Switching Costs** | High | 数据迁移复杂，workflow 集成 |
| **Brand** | Medium | 行业 standard 认知 |
| **Regulatory** | Medium | Broker-dealer license 门槛 |

---

## 十四、References & Resources

### 官方资源
- Carta Official Website: https://carta.com/
- Carta Blog: https://carta.com/blog/
- Carta Documentation: https://support.carta.com/

### Funding & Valuation
- Crunchbase Profile: https://www.crunchbase.com/organization/carta
- PitchBook Data: https://pitchbook.com/profiles/company/89574-16

### Industry Reports
- Carta Annual Equity Report: https://carta.com/equity-report/
- State of Private Markets: https://carta.com/insights/

### Technical Resources
- 409A Valuations Guide: https://carta.com/blog/409a-valuations/
- Cap Table Management Best Practices: https://carta.com/blog/cap-table-management/

### Competitors
- Pulley: https://pulley.com/
- AngelList: https://angel.co/
- Morgan Stanley Shareworks: https://www.shareworks.com/

### News & Analysis
- TechCrunch Coverage: https://techcrunch.com/tag/carta/
- Fortune Article: https://fortune.com/company/carta/

---

## 十五、Summary: Building Intuition

### 核心洞察：

1. **Carta 本质上是 "Equity 的 Single Source of Truth"**
   - 从 spreadsheet → Database → Platform 的演进
   - 数据 accuracy 和 compliance 是核心 value prop

2. **Network Effect 是核心 moat**
   - 两端 marketplace: Startups ↔ VCs
   - 每个 new customer 增加 platform value

3. **Business Model 是经典的 B2B SaaS**
   - Recurring revenue
   - High gross margin
   - Land-and-expand strategy

4. **Technology 难点在 domain complexity，不在 technical complexity**
   - Waterfall calculations
   - Regulatory compliance
   - Edge cases handling

5. **Future 是成为 Private Market Financial Operating System**
   - From cap table → Full equity lifecycle
   - Integration with other financial services