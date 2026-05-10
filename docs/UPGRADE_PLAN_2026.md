# 2api.ai → AgentOps Control Plane: Kế Hoạch Tái Định Vị 2026

> **Phiên bản:** v1.0 (đề xuất, chờ duyệt)
> **Ngày:** 10/05/2026
> **Branch:** `claude/codebase-review-upgrade-plan-BkUl5`
> **Tác giả:** Engineering review + 3 luồng deep research thị trường

---

## TL;DR (đọc trong 60 giây)

**Vấn đề:** Dự án hiện tại đang định vị là "LLM Gateway giống OpenRouter/LiteLLM". Đây là **đại dương đỏ máu (saturation 10/10)**. Vercel và Cloudflare đã offer 0% markup, OpenRouter đã có $50M ARR + 400 models, LiteLLM dominate OSS, Portkey vừa bị Palo Alto mua $120-140M (30/04/2026), Helicone bị Mintlify mua, Langfuse bị ClickHouse mua. Nếu tiếp tục đi thẳng, chúng ta DOA.

**Cơ hội thật:** Trong khi mọi người đua nhau làm gateway, **lớp Operations cho AI Agents trong production vẫn là blue ocean**. Có 10 nỗi đau cụ thể được trả $300-10,000/tháng/khách mà chưa ai giải quyết tử tế. Quan trọng nhất:
1. **Agent loop $47,000 chạy 11 ngày không có kill switch** — dev đang khóc
2. **Anthropic giảm cache TTL từ 1h xuống 5min không thông báo** — không ai detect được provider drift
3. **Reasoning tokens (o3, Claude thinking) tính tiền hidden tokens không thể attribute được**
4. **Per-customer cost attribution** — SaaS đi từ $15K → $60K/tháng không biết khách nào đốt
5. **EU AI Act enforce 02/08/2026** — 78% executives KHÔNG pass nổi audit, đang panic mua

**Đề xuất pivot:** Đổi từ "Generic LLM Gateway" → **"AgentOps Control Plane"** — lớp control + observability + safety + compliance dành riêng cho **AI Agents trong production**, có thể bolt-on lên TRÊN OpenRouter/Bedrock/Vercel (không phải đối đầu, mà cộng sinh).

**Kết quả mong đợi 6 tháng:**
- 100 paying customers
- $30-80K MRR
- 5-10 case studies "saved $X cost / prevented Y outage"
- SOC 2 Type II + EU AI Act compliance pack
- Series A-ready metrics

---

## 1. Hiện Trạng Codebase (Engineering Audit)

### 1.1 Điểm mạnh (giữ và build trên đó)
- **Error taxonomy + semantic safety** (`src/core/errors.py`, `src/routing/fallback.py`): Phân biệt infra vs semantic error, không fallback giữa stream — production-grade, hơn nhiều competitor.
- **Routing engine** đa chiến lược (cost / latency / quality / balanced) với scoring composable — `src/routing/strategies.py:115-421`.
- **Circuit breaker** thread-safe đầy đủ states — `src/routing/circuit_breaker.py:109-227`.
- **Test coverage tốt**: 420 tests, 8,439 LOC tests / 25,247 LOC src (~33%).
- **Adapter abstraction** sạch sẽ, dễ mở rộng — `src/adapters/base.py:44-226`.
- **Observability stack** đầy đủ Prometheus + OTel + structured logs.

### 1.2 Điểm yếu nghiêm trọng (phải sửa trước khi sell)

| # | Gap | File | Impact |
|---|-----|------|--------|
| 1 | **Single-instance only**: rate limit, circuit breaker, usage đều in-memory | `src/usage/limits.py`, `src/routing/router.py:136-145` | Không thể scale, không thể bán enterprise |
| 2 | **Không có Admin UI** | (chưa có) | Không thể bán cho non-engineering team |
| 3 | **Fernet key không có rotation/versioning** | `src/security/encryption.py:35-88` | Lộ key = compromise toàn bộ history |
| 4 | **God file `router.py` 879 LOC** trộn 4 concerns | `src/routing/router.py` | Khó test, khó refactor |
| 5 | **Chỉ 3 providers** (OpenAI/Anthropic/Google) | `src/adapters/*` | OpenRouter có 400, LiteLLM có 140+ |
| 6 | **Không có semantic cache, batch API, files API, fine-tuning routing** | (chưa có) | Mất 40-50% enterprise use cases |
| 7 | **Không có MCP server gateway** | (chưa có) | Bỏ lỡ trào lưu 78% adoption |
| 8 | **Không có prompt injection detection / PII redaction enforcement** | `src/observability/logging.py` | Không pass HIPAA/GDPR |
| 9 | **Usage tracking dead-letter queue thiếu** | `src/usage/tracker.py:358-375` | Mất billing data nếu DB crash |
| 10 | **Health/circuit breaker state KHÔNG có alert hook** | (chưa có) | Ops mù, support hell |

### 1.3 Đánh giá tổng thể
- **Engineering quality:** 7/10 (foundation tốt)
- **Production-readiness for SaaS:** 3/10 (single-instance, không UI, không alert)
- **Differentiation vs market:** 1/10 (nothing unique)
- **Verdict:** Code base là vốn liếng tốt nhưng chiến lược product cần PIVOT.

---

## 2. Bản Đồ Thị Trường (May 2026) — Đỏ Máu vs Xanh Dương

### 2.1 Saturation Map (1=blue ocean, 10=red ocean)

| Sub-niche | Saturation | Ghi chú |
|-----------|:----------:|---------|
| "Yet another LLM proxy/gateway" | **10** | DOA — Vercel/CF 0% markup, OpenRouter scale, LiteLLM OSS |
| LLM observability/tracing chung | **9** | 4 vụ M&A trong 12 tháng (Langfuse, Helicone, Braintrust raise $80M) |
| Prompt management | **8** | Bundled trong mọi obs platform |
| Agent frameworks | **8** | LangGraph, CrewAI, OpenAI Agents SDK saturate |
| Eval/testing chung | **6** | Braintrust raise nhưng "closed-loop eval" còn mở |
| AI cost / FinOps chung | **7** | Mọi gateway claim, nhưng CFO-targeted SaaS còn lỗ hổng |
| Edge AI inference | **6** | Cloudflare dominate |
| Multi-modal routing | **5** | Phần video/voice còn mở |
| Agent observability/debugging | **5** | 6 platform consolidating, agent-specific failure modes còn mở |
| **Vertical AI gateway (legal/health/fin)** | **4** | EU AI Act forcing buys |
| **Compliance/SOC2/HIPAA layer cho LLM** | **4** | 78% không pass audit |
| **MCP infrastructure** | **3** | $40M+ funded, greenfield |
| **On-prem/sovereign AI gateway** | **3** | EU enforcement 02/08/2026 |
| **Embedded gateway-as-SDK trong vertical SaaS** | **2** | Biggest blue ocean |

### 2.2 Tin tức M&A nóng (chứng minh thị trường đang consolidate)
- **30/04/2026**: Palo Alto Networks mua Portkey (~$120-140M) → tích hợp vào Prisma AIRS
- **24/03/2026**: LiteLLM bị **supply-chain compromise** (PyPI credential harvester + K8s lateral movement) → reputation damage lớn
- **03/03/2026**: Mintlify mua Helicone → dev đang migrate ra (LLMeter, Respan publish migration guides)
- **02/2026**: Braintrust raise $80M Series B @ $800M (eval-driven routing thesis)
- **01/2026**: ClickHouse mua Langfuse
- **2025**: Martian (LLM router) ~$1.3B valuation
- **17 & 19/02/2026**: OpenRouter outage 2 lần trong 1 tuần — không có SLA
- **02/08/2026**: EU AI Act high-risk obligations enforce — phạt €35M hoặc 7% revenue

### 2.3 Bài học từ M&A
- **Pure LLM proxy** không còn defensible → bị các platform lớn nuốt
- **Compliance + governance** là moat thật sự ($25-500K ACV)
- **Agent-native security** là wedge mới (Operant AI raise $13.5M, Runlayer $11M)
- **Eval + closed-loop** là layer cao hơn được trả tiền

---

## 3. Top 10 Nỗi Đau Underserved (Có Khách Trả Tiền THẬT)

Ranked theo willingness-to-pay từ research dev complaints (Reddit, HN, dev.to, GitHub Issues, X):

| # | Pain | WTP/tháng | Tại sao chưa giải quyết tốt |
|---|------|----------:|------------------------------|
| 1 | **Hard kill switch cho agent loops** với per-agent / per-customer budget cap | $500-2,000 | Tools chỉ alert, không HALT atomically. Có dev mất $47,000/11 ngày |
| 2 | **Provider drift detection** (catch khi Anthropic/OpenAI silently change pinned model behavior, cache TTL...) | $300-1,500 | Mọi prompt-versioning tool track YOUR changes, không ai monitor PROVIDER |
| 3 | **Reasoning-token cost attribution** (o3/Claude thinking/Gemini deep think tính tiền hidden tokens) | $400-2,000 | Helicone/Langfuse show total, không slice "reasoning vs visible" by feature |
| 4 | **Per-customer/per-feature cost attribution** map ra invoice | $500-2,000 | Traceloop làm được basic, không ai handle passthrough billing |
| 5 | **MCP gateway** với auth + secret rotation + scope limit | $1,000-5,000 | 200K servers exposed, multiple CVEs (CVE-2025-49596, CVE-2026-22252). Portkey/LiteLLM admit chưa có full support |
| 6 | **Deterministic agent replay** + earliest-failure-step detection cross-framework | $400-2,000 | LangSmith locked LangChain, Polly closed, CodeTracer research-grade |
| 7 | **PII/secret detection trong agentic flows** (tool outputs, MCP calls, browser DOM) | $1,500-5,000 | Imperva/Lasso work pre-LLM, agentic data exfil bypass |
| 8 | **Streaming-aware multi-provider failover** với token-level resume | $300-1,000 | Portkey/LiteLLM retry whole call, không ai resume mid-stream |
| 9 | **Voice/Realtime cost telemetry** (audio-in/out/silence/text breakdown) | $200-800 | OpenAI dashboard opaque, obs tools không decode audio meter |
| 10 | **Compliance-audit logging** (GDPR/HIPAA/EU AI Act) tamper-proof, retention dài, không log-pricing punish | $2,000-10,000 | Portkey cap log + retention ngắn, Langfuse self-host on you |

**Tổng addressable budget per customer:** $7,200 - $34,300/tháng nếu giải được nhiều pain — tức là sales một deal = giá trị 5-20 deals của OpenRouter.

---

## 4. The Pivot: "AgentOps Control Plane"

### 4.1 Định vị mới
**Không** là "Gateway thay thế OpenRouter" → ai cũng làm rồi.
**Là** "Control plane cho AI Agents in production" — bolt-on TRÊN gateway hiện có.

> **One-liner:** *"The runtime layer that keeps your AI agents from losing $47K, leaking PII, or failing compliance audits — drop in 2 lines, works with any LLM gateway."*

### 4.2 Vì sao positioning này thắng
1. **Không đối đầu commoditized layer** (routing) — chúng ta dùng OpenRouter/Bedrock/Vercel làm upstream
2. **Hợp tác > thay thế** → ít friction, dễ install hơn rip-and-replace
3. **Riêng cho agents** — agent failures là khủng hoảng 2026, chat-completion gateway không cover được
4. **EU AI Act tạo urgency** — buyers panic mua trước 02/08/2026
5. **Higher margin** — bán subscription + usage tier (KHÔNG % take rate) = 70-90% gross margin
6. **Defensible** — closed-loop data (mỗi kill switch decision improve next), compliance certs là moat 12-18 tháng

### 4.3 Sản phẩm cốt lõi (5 modules)

```
┌────────────────────────────────────────────────────────────────────────┐
│                  AGENTOPS CONTROL PLANE                                 │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  GUARDIAN       │  │  ATTRIBUTION    │  │  DRIFT WATCHDOG         │ │
│  │  (kill switch + │  │  (per-customer/ │  │  (provider regression   │ │
│  │   budget cap +  │  │   feature cost +│  │   detection, cache TTL  │ │
│  │   PII redaction)│  │   reasoning     │  │   alerts, cron tests)   │ │
│  │                 │  │   token slice)  │  │                         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────────────────────────────────┐ │
│  │  REPLAY         │  │  COMPLIANCE PACK                             │ │
│  │  (deterministic │  │  (EU AI Act audit log, HIPAA BAA-ready,      │ │
│  │   agent replay, │  │   tamper-proof hash chain, BYOK + BYOC)      │ │
│  │   failure step  │  │                                              │ │
│  │   detection)    │  │                                              │ │
│  └─────────────────┘  └──────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  MCP GATEWAY (capability scope, signed servers, secret rotation) │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                              ↓ proxies / passes through
┌────────────────────────────────────────────────────────────────────────┐
│   YOUR EXISTING STACK (unchanged): OpenRouter / Bedrock / Vercel /      │
│                            LiteLLM / direct OpenAI / Anthropic / etc.   │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Tận dụng codebase hiện có
- **80% code base hiện có là ASSET** cho positioning mới:
  - Routing/fallback engine → reuse cho upstream gateway selection
  - Adapter pattern → reuse cho upstream connectors (OpenRouter, Bedrock...)
  - Error taxonomy + semantic safety → còn quan trọng hơn cho agents
  - Streaming normalizer → cần cho token-level resume
  - Observability stack → upgrade thành agent-grade traces
- **20% phải rewrite hoặc add mới**: Redis-backed state, MCP gateway, kill switch primitives, replay engine, audit log layer

---

## 5. Roadmap 28 Tuần (4 Phase)

### **PHASE 1 — Foundation Hardening + Wedge MVP (Tuần 1-6)**

> Mục tiêu: Sửa critical infra gaps + ship killer feature đầu tiên (Guardian).

**Engineering hardening:**
- [ ] Tách `router.py` (879 LOC) → `ProviderSelector` + `FallbackOrchestrator` + `StreamingManager` (3 files <300 LOC mỗi)
- [ ] Migrate state in-memory → Redis: rate limits, circuit breaker, usage aggregator (token bucket via ZADD)
- [ ] Fernet key versioning + scheduled re-encryption pipeline (`encryption.py`)
- [ ] Dead letter queue cho usage records (Redis stream với retry)
- [ ] EventBus cho alerts (Slack/PagerDuty/webhook handler)
- [ ] Add 3 providers ưu tiên: **AWS Bedrock** (enterprise), **Groq** (fastest inference), **OpenRouter passthrough** (anti-fragile)

**Wedge MVP — "Guardian" (Pain #1, #4, #7):**
- [ ] **Hard kill switch primitive**: per-agent_id budget cap (USD/tokens/duration), atomic enforce qua Redis lock + middleware
- [ ] **Per-customer attribution**: extend usage tracker với `customer_id`, `feature_id`, `agent_id` dimensions; SQL view aggregate
- [ ] **PII redactor inline**: regex + NER (presidio) pre-prompt + on tool outputs, configurable allow/deny list
- [ ] **Admin dashboard MVP** (Next.js): keys, usage by customer, kill switch config, real-time spend
- [ ] **Demo**: "Watch us stop a $47K loop in 4 seconds" — viral content

**Deliverables Phase 1:**
- Production-ready single-cluster (2-instance HA via Redis)
- 6 providers (3 cũ + Bedrock + Groq + OpenRouter)
- Guardian module ship-able
- Admin UI v1
- Tài liệu + demo video

---

### **PHASE 2 — Market Wedge: Agent Ops Differentiation (Tuần 7-12)**

> Mục tiêu: Ship 4 modules còn lại, build moat sâu vào agent-specific telemetry.

**Modules:**
- [ ] **MCP Gateway** (Pain #5):
  - Capability scoping (mỗi MCP tool có max permissions)
  - Secret rotation tự động (no long-lived static secrets)
  - Path traversal protection
  - SBOM + signed servers registry
  - Built-in audit log của mọi tool call
- [ ] **Drift Watchdog** (Pain #2):
  - Daily/hourly probe gửi prompt cố định lên mỗi pinned model
  - Embedding distance + judge-score delta alert
  - Track cache hit rate trend + alert on sudden drop
  - Public dashboard "provider reliability tracker" (PR weapon)
- [ ] **Reasoning Token Analytics** (Pain #3):
  - Track `thinking_tokens` separately từ visible output cho o3/o4/Claude/Gemini
  - Slice by feature/customer/prompt_template
  - Pre-request budget warning ("này request có thể tốn 50K thinking tokens")
- [ ] **Replay Engine** (Pain #6):
  - Capture toàn bộ agent trajectory (LLM calls + tool calls + state) thành deterministic trace
  - Replay với same/different model
  - "Earliest failure step" via binary search + judge
  - Cross-framework (LangGraph, CrewAI, raw OpenAI Agents SDK)
- [ ] **Streaming token-level resume** (Pain #8): nếu provider drop mid-stream, fallback resume từ token cuối cùng (an toàn semantic — chỉ với deterministic model + low temp)

**Deliverables Phase 2:**
- 5/10 pain points addressed
- Public Drift Watchdog (free PR + lead gen)
- 10 paying design partners
- $5-15K MRR

---

### **PHASE 3 — Compliance & Sovereignty Pack (Tuần 13-20)**

> Mục tiêu: Unlock enterprise ($2.5K-10K MRR per deal) thông qua compliance moat.

- [ ] **EU AI Act Article 26 audit log**:
  - Append-only, hash-chained, signed (Ed25519)
  - 7-year retention
  - Auto-generate audit reports (CSV/PDF)
  - "Risk classification" tagging mỗi request
- [ ] **HIPAA pack**: BAA template, PHI detector, allowed-region routing, encrypted at rest with customer-managed keys
- [ ] **SOC 2 Type II** preparation (Drata or Vanta), control mapping
- [ ] **BYOK enforcement**: customer-provided HSM keys, all provider keys encrypted với customer key
- [ ] **BYOC deployment**: Terraform modules cho AWS, Azure, GCP, **OVH, Scaleway, IONOS** (EU sovereign)
- [ ] **One-click sovereign deploy**: helm chart + signed images + air-gapped mode
- [ ] **Voice/Realtime telemetry** (Pain #9): audio-in/out/silence/text breakdown per session, alert on cost anomaly

**Deliverables Phase 3:**
- 7/10 pain points addressed
- SOC 2 Type II in progress (Q4 audit)
- HIPAA-ready (BAA-signable)
- 3 EU sovereign deploys live
- $30-50K MRR
- Series A pitchable metrics

---

### **PHASE 4 — Scale & Verticalization (Tuần 21-28)**

> Mục tiêu: $100K MRR + 100 paying customers + 2 vertical packages live.

- [ ] **Vertical packages**:
  - **Healthcare**: HIPAA BAA + medical PII patterns + ICD-10 vocab + 21 CFR Part 11 audit
  - **Finance**: PCI DSS scope reduction + SOX controls + FINRA archiving
  - **Legal**: privilege detection + Bates numbering + redaction policies
  - (Optional: Customer Support — refund decision audit)
- [ ] **Agent identity / delegation**: OAuth 2.1 + DPoP scoped tokens cho MCP servers ("Auth0 for agents" lite)
- [ ] **Self-serve onboarding**: 5-min Quickstart, sandbox API keys, sample apps
- [ ] **Public benchmark page**: "We saved $X for customer Y" — auto-generated case studies từ real data
- [ ] **Marketplace**: pre-built guardrails (e.g., "PII redactor — Vietnamese ID", "Cost cap — agent loop")
- [ ] **Public Drift Watchdog**: free public dashboard tracking OpenAI/Anthropic/Google reliability — viral SEO + lead gen
- [ ] **Documentation**: cookbook, video tutorials, migration guides từ Helicone/Portkey

**Deliverables Phase 4:**
- 10/10 pain points addressed
- 100 paying customers
- $80-150K MRR
- 3 vertical packages live
- Series A ready

---

## 6. Monetization & Pricing

### 6.1 Pricing Tiers (KHÔNG % take rate — fight 0% markup race)

| Tier | Price | Bao gồm | Persona |
|------|------:|---------|---------|
| **Free** | $0 | 100K agent steps/mo, 7-day retention, single tenant, community Discord | Solo dev, OSS evaluators |
| **Pro** | $99/mo | 1M steps, 30-day retention, kill switches, drift alerts, basic PII redaction | Indie hackers, early-stage startups |
| **Team** | $499/mo | 10M steps, 90-day retention, per-customer attribution, RBAC, SSO, dedicated Slack | Series A startups, AI products |
| **Business** | $2,499/mo | 100M steps, 1-year retention, MCP gateway, replay engine, SLA 99.9% | Scaling SaaS ($50-500K/mo AI spend) |
| **Enterprise** | $10K+/mo | Unlimited, BYOC sovereign, EU AI Act pack, HIPAA BAA, SOC2 Type II, dedicated CSM | Healthcare, finance, gov, EU regulated |

**Add-ons:**
- Vertical pack (Healthcare/Finance/Legal): +$2,000/mo
- Custom audit log retention: $0.0001 per log/month
- Premium support 24/7: +$1,500/mo
- BYOC management: $5,000 one-time + $500/mo

### 6.2 Tại sao tier này thắng
- **Pro $99/mo** = lower friction hơn LangSmith $39/seat (vì $99 covers cả team)
- **Team $499/mo** = sweet spot cho SaaS Series A (Portkey + Helicone + ad hoc tools cũ tốn $300-800)
- **Enterprise $10K+** = cùng range Portkey/LiteLLM Enterprise nhưng có sovereignty + agent specifics
- **Gross margin 80-90%** vs OpenRouter 5% take rate (thua 15-20x revenue per dollar of cost)

---

## 7. Go-To-Market Strategy

### 7.1 OSS-First (Apache 2.0)
- Theo playbook Portkey/Langfuse/LiteLLM: open-source core gateway + adapters + Guardian primitives
- Closed-source (cloud-only): Drift Watchdog dashboard, Replay UI, Compliance Pack, Vertical packs
- License: Apache 2.0 + BSL cho enterprise modules (như HashiCorp/Sentry pattern)

### 7.2 Launch Sequence
1. **Tuần 6**: HN "Show HN: We stopped a $47K agent loop in 4 seconds" + demo video
2. **Tuần 8**: Product Hunt với Drift Watchdog public dashboard
3. **Tuần 10**: Sponsor LiteLLM migration guide (post-supply-chain attack — capture displaced users)
4. **Tuần 12**: AI Engineer Summit booth + lightning talk
5. **Tuần 16**: EU AI Act compliance webinar series (lead gen pre-deadline)
6. **Tuần 20**: First vertical case study (healthcare), DEF CON / Black Hat MCP security talk
7. **Tuần 24**: Public benchmark "AgentOps cost savings report 2026" — viral

### 7.3 Distribution Wedges
- **Sponsor migration tools** từ Helicone (acquired) → us
- **Free tier qua Vercel/Cloudflare integrations** — embed our SDK in their AI Gateway templates
- **MCP server registry** — own the trusted registry
- **Discord community** + bi-weekly office hours
- **Vertical SaaS partnerships** (e.g., Cal.com health, Athelas, Akute Health for healthcare)

### 7.4 Content moat
- **Public reliability tracker** (always-on uptime/drift dashboard cho mọi LLM provider) → SEO + brand
- **Annual "State of AI Agent Operations" report** với data từ platform
- **Open-source benchmark suite** cho agent failures

---

## 8. Phân tích Risk & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Portkey (now Palo Alto) ship same features | High | High | Move fast — wedge 6 tháng, lock-in vertical packs sớm |
| OpenRouter add Guardian | Medium | High | Build deeper agent-specific (replay, drift) — họ tập trung consumer, ta tập trung B2B |
| LiteLLM thêm MCP gateway | High | Medium | Họ có security debt từ March 2026 attack — ta lead with security posture |
| Compliance certs trễ → mất enterprise deals | Medium | High | Bắt đầu Drata Day 1, target SOC 2 Type II Q4 2026 |
| Hyperscaler (AWS, Azure) bundle free | Medium | Medium | Multi-cloud sovereignty là moat — họ không bao giờ làm cross-cloud |
| Underestimate engineering effort | High | High | Phase 1 MUST land Tuần 6 hoặc cut scope ngay |
| Founder/team không có healthcare domain | Medium | Medium | Hire vertical advisor sớm (CMO healthcare expert) |
| MCP standard thay đổi | Low | Medium | Track Linux Foundation Agentic AI Foundation, contribute upstream |

---

## 9. Đầu Tư Cần Thiết

### 9.1 Engineering (28 tuần)
- 2 Senior Backend (Python/Go): full-time
- 1 Frontend (Next.js): full-time
- 1 DevOps/SRE: 50% (Phase 1-2), 100% (Phase 3-4)
- 1 Security engineer: contract Phase 3 (compliance)
- **Tổng:** ~$280-400K (tùy senior level và geo)

### 9.2 Compliance & Tools
- Drata/Vanta SOC 2: ~$15K/year
- HIPAA legal review: ~$10K
- Pen test: ~$25K (Phase 3)
- Tools (Datadog, Sentry, etc): ~$2K/mo

### 9.3 Marketing & Sales
- Content / DevRel hire: $80-120K
- Conference sponsorships: $30K (AI Engineer + DEF CON)
- Paid ads (LinkedIn for enterprise): $20K
- Design / brand: $15K

**Total 6 tháng burn:** ~$500-700K
**Break-even:** Tuần 24-28 nếu hit $80-100K MRR

---

## 10. Câu hỏi cần ANH duyệt (4 quyết định lớn)

Trước khi code line đầu tiên, em cần ANH chốt 4 thứ:

### Q1: Pivot có chấp nhận không?
- **A.** Đồng ý pivot toàn diện sang AgentOps Control Plane (recommend)
- **B.** Pivot một phần — giữ vibe gateway nhưng add Guardian như feature
- **C.** Không pivot — tiếp tục đua OpenRouter (em không recommend, sẽ DOA)

### Q2: Beachhead market đầu tiên?
- **A.** Indie / SaaS Series A ($99-499/mo, volume play) — nhanh feedback nhưng nhỏ ARR
- **B.** Mid-market ($499-2,499/mo) — sweet spot, balance
- **C.** Enterprise EU sovereign ($10K+/mo) — to nhưng cần SOC2/compliance trước, sales cycle 3-6 tháng
- **D.** Vertical (healthcare hoặc finance) ngay từ đầu — narrow + deep

### Q3: OSS strategy?
- **A.** Apache 2.0 mọi thứ trừ cloud dashboard (Portkey playbook)
- **B.** BSL (giới hạn không cloud-host) — bảo vệ tốt hơn, friction cao hơn
- **C.** Closed-source toàn bộ + free tier — không cần care OSS

### Q4: Tốc độ vs an toàn?
- **A.** Aggressive: ship Phase 1 trong 4 tuần (cut scope, accept tech debt)
- **B.** Balanced: 6 tuần như plan
- **C.** Conservative: 8-10 tuần, refactor đàng hoàng

---

## 11. Action Items Tuần 1 (nếu duyệt)

Ngay khi ANH duyệt, em sẽ làm:

1. **Day 1-2**: Setup Redis dev cluster, scaffold module mới (`src/agentops/`)
2. **Day 3-5**: Refactor `router.py` → 3 files; viết test
3. **Day 6-10**: Implement Guardian primitives (kill switch + budget cap atomic Redis)
4. **Day 11-14**: PII redactor + per-customer attribution
5. **Tuần 3**: Bedrock + Groq + OpenRouter passthrough adapters
6. **Tuần 4**: Admin dashboard MVP (Next.js + tailwind)
7. **Tuần 5**: Demo video "$47K loop stopped in 4 seconds"
8. **Tuần 6**: Internal beta với 3 design partners

Mỗi tuần em commit lên branch này + send PR review.

---

## Phụ lục A — Danh sách 60+ nguồn research

(file riêng — đã có trong research notes, có thể attach nếu cần)

## Phụ lục B — Threat model & competitive intelligence detailed

(file riêng — sẵn sàng deep dive khi anh cần)

---

**END OF PLAN — chờ ANH duyệt 4 câu hỏi ở Section 10.**
