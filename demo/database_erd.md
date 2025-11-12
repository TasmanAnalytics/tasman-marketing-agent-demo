# Database ERD - Marketing Analytics Demo

## Entity Relationship Diagram

```
┌─────────────────────────┐
│   dim_campaigns         │
│─────────────────────────│
│ 🔑 campaign_id (PK)     │
│   channel               │◄───────────┐
│   campaign_name         │            │
└─────────────────────────┘            │
            ▲                           │
            │                           │
            │                           │
            │                           │
┌───────────┴─────────────┐    ┌───────┴──────────────┐
│   fact_ad_spend         │    │   fact_sessions      │
│─────────────────────────│    │──────────────────────│
│   campaign_id (FK) ─────┘    │ 🔑 session_id (PK)   │
│   date                  │    │   campaign_id (FK) ──┘
│   spend                 │    │   device_type        │
│   impressions           │    │   converted_flag     │
│   clicks                │    │   session_start_time │
└─────────────────────────┘    └──────────┬───────────┘
                                          │
                                          │
                                          │
                              ┌───────────▼───────────┐
                              │   fact_orders         │
                              │───────────────────────│
                              │ 🔑 order_id (PK)      │
                              │   session_id (FK) ────┘
                              │   revenue             │
                              │   order_timestamp     │
                              └───────────────────────┘


┌─────────────────────────┐    ┌─────────────────────────┐
│   dim_customers         │    │   dim_adgroups          │
│─────────────────────────│    │─────────────────────────│
│ 🔑 customer_id (PK)     │    │ 🔑 adgroup_id (PK)      │
│   region                │    │   (not used in demo)    │
└─────────────────────────┘    └─────────────────────────┘


┌─────────────────────────┐
│   dim_creatives         │
│─────────────────────────│
│ 🔑 creative_id (PK)     │
│   (not used in demo)    │
└─────────────────────────┘
```

## Key Relationships

### 1. Ad Spend Attribution
```
fact_ad_spend → dim_campaigns
```
- **Purpose**: Track spend, impressions, clicks by campaign
- **Key field**: `campaign_id`

### 2. Session Attribution
```
fact_sessions → dim_campaigns
```
- **Purpose**: Track sessions and conversions by campaign
- **Key field**: `campaign_id`

### 3. Revenue Attribution (Last-Touch)
```
fact_orders → fact_sessions → dim_campaigns
```
- **Purpose**: Attribute revenue to campaigns via last-touch
- **Critical path**: `order.session_id = session.session_id = session.campaign_id = campaign.campaign_id`
- **⚠️ Enforced in semantic layer**: This join path is MANDATORY for revenue metrics

## Metrics Calculated

### From fact_ad_spend + dim_campaigns:
- **Spend by Channel**: `SUM(spend) GROUP BY channel`
- **Impressions/Clicks by Channel**: `SUM(impressions), SUM(clicks) GROUP BY channel`

### From fact_sessions + dim_campaigns:
- **Conversions by Channel**: `SUM(converted_flag) GROUP BY channel`
- **Sessions by Channel**: `COUNT(*) GROUP BY channel`

### From fact_orders + fact_sessions + dim_campaigns:
- **Revenue by Channel**: `SUM(revenue) GROUP BY channel` (via session attribution)
- **Orders by Channel**: `COUNT(order_id) GROUP BY channel`

### Derived Metrics:
- **ROAS** = Revenue / Spend (by channel)
- **CAC** = Spend / Conversions (by channel)
- **CTR** = Clicks / Impressions
- **CVR** = Conversions / Sessions

## Data Model Principles

✅ **Star Schema**: Dimension tables (dim_*) + Fact tables (fact_*)
✅ **Clear Attribution**: Revenue flows through session → campaign
✅ **Time-based**: All facts have timestamps for windowing
✅ **Denormalized**: Channel stored in dim_campaigns for fast aggregation
✅ **Semantic Layer Enforced**: Join rules prevent incorrect queries

## Why This Structure Works

1. **Separation of Concerns**: Ad spend vs sessions vs orders tracked independently
2. **Flexible Attribution**: Can change attribution model by changing join path
3. **Performance**: Aggregations are fast with proper indexes on foreign keys
4. **Data Integrity**: Enforced relationships prevent orphaned records
5. **Semantic Safety**: LLM cannot generate incorrect joins

---

**Note**: This is a simplified demo schema. Production systems typically have:
- More dimension tables (products, geo, time)
- More fact grain levels (hourly, daily aggregates)
- Slowly changing dimensions (SCD Type 2)
- Additional attribution models (multi-touch)
