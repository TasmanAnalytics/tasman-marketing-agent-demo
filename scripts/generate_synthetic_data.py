"""
Generate synthetic marketing data for demo notebooks.
"""

import duckdb
import random
from datetime import datetime, timedelta
import numpy as np

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
OUTPUT_PATH = 'data/synthetic_data.duckdb'
START_DATE = datetime.now() - timedelta(days=180)
END_DATE = datetime.now()

# Channels with different CAC characteristics
CHANNELS = [
    {'channel': 'search', 'avg_cac': 45, 'avg_roas': 3.2},
    {'channel': 'social', 'avg_cac': 35, 'avg_roas': 2.8},
    {'channel': 'display', 'avg_cac': 65, 'avg_roas': 1.9},
    {'channel': 'referral', 'avg_cac': 25, 'avg_roas': 4.1},  # Best performer
    {'channel': 'email', 'avg_cac': 15, 'avg_roas': 5.2},  # Best performer
]

DEVICES = ['desktop', 'mobile', 'tablet']
REGIONS = ['north_america', 'europe', 'asia_pacific', 'latam']

print("Generating synthetic marketing data...")

# Create database
conn = duckdb.connect(OUTPUT_PATH)

# 1. dim_campaigns
print("Creating dim_campaigns...")
campaigns = []
campaign_id = 1
for channel_info in CHANNELS:
    for i in range(4):  # 4 campaigns per channel
        campaigns.append({
            'campaign_id': campaign_id,
            'campaign_name': f"{channel_info['channel']}_campaign_{i+1}",
            'channel': channel_info['channel']
        })
        campaign_id += 1

conn.execute("""
    CREATE TABLE dim_campaigns (
        campaign_id INTEGER PRIMARY KEY,
        campaign_name VARCHAR,
        channel VARCHAR
    )
""")
conn.executemany(
    "INSERT INTO dim_campaigns VALUES (?, ?, ?)",
    [(c['campaign_id'], c['campaign_name'], c['channel']) for c in campaigns]
)

# 2. dim_adgroups
print("Creating dim_adgroups...")
adgroups = []
adgroup_id = 1
for campaign in campaigns:
    for i in range(3):  # 3 adgroups per campaign
        adgroups.append({
            'adgroup_id': adgroup_id,
            'campaign_id': campaign['campaign_id'],
            'adgroup_name': f"adgroup_{adgroup_id}"
        })
        adgroup_id += 1

conn.execute("""
    CREATE TABLE dim_adgroups (
        adgroup_id INTEGER PRIMARY KEY,
        campaign_id INTEGER,
        adgroup_name VARCHAR
    )
""")
conn.executemany(
    "INSERT INTO dim_adgroups VALUES (?, ?, ?)",
    [(a['adgroup_id'], a['campaign_id'], a['adgroup_name']) for a in adgroups]
)

# 3. dim_creatives
print("Creating dim_creatives...")
creatives = []
creative_id = 1
for adgroup in adgroups:
    for i in range(2):  # 2 creatives per adgroup
        creatives.append({
            'creative_id': creative_id,
            'adgroup_id': adgroup['adgroup_id'],
            'creative_name': f"creative_{creative_id}"
        })
        creative_id += 1

conn.execute("""
    CREATE TABLE dim_creatives (
        creative_id INTEGER PRIMARY KEY,
        adgroup_id INTEGER,
        creative_name VARCHAR
    )
""")
conn.executemany(
    "INSERT INTO dim_creatives VALUES (?, ?, ?)",
    [(c['creative_id'], c['adgroup_id'], c['creative_name']) for c in creatives]
)

# 4. dim_products (simple for now)
print("Creating dim_products...")
products = [{'id': i, 'name': f'Product_{i}', 'price': random.uniform(20, 200)}
            for i in range(1, 51)]

conn.execute("""
    CREATE TABLE dim_products (
        id INTEGER PRIMARY KEY,
        name VARCHAR,
        price DECIMAL(10, 2)
    )
""")
conn.executemany(
    "INSERT INTO dim_products VALUES (?, ?, ?)",
    [(p['id'], p['name'], p['price']) for p in products]
)

# 5. dim_customers
print("Creating dim_customers...")
customers = []
for i in range(1, 5001):
    customers.append({
        'customer_id': i,
        'region': random.choice(REGIONS),
        'signup_date': START_DATE + timedelta(days=random.randint(0, 180))
    })

conn.execute("""
    CREATE TABLE dim_customers (
        customer_id INTEGER PRIMARY KEY,
        region VARCHAR,
        signup_date TIMESTAMP
    )
""")
conn.executemany(
    "INSERT INTO dim_customers VALUES (?, ?, ?)",
    [(c['customer_id'], c['region'], c['signup_date']) for c in customers]
)

# 6. fact_ad_spend
print("Creating fact_ad_spend...")
ad_spend_records = []
current_date = START_DATE

while current_date <= END_DATE:
    for campaign in campaigns:
        # Get channel info
        channel_info = next(c for c in CHANNELS if c['channel'] == campaign['channel'])

        # Add some variance
        daily_spend = random.uniform(500, 2000)
        impressions = int(daily_spend * random.uniform(80, 120))
        clicks = int(impressions * random.uniform(0.01, 0.04))

        ad_spend_records.append({
            'date': current_date.date(),
            'campaign_id': campaign['campaign_id'],
            'spend': round(daily_spend, 2),
            'impressions': impressions,
            'clicks': clicks
        })

    current_date += timedelta(days=1)

conn.execute("""
    CREATE TABLE fact_ad_spend (
        date DATE,
        campaign_id INTEGER,
        spend DECIMAL(10, 2),
        impressions INTEGER,
        clicks INTEGER
    )
""")

# Batch insert for performance
batch_size = 1000
for i in range(0, len(ad_spend_records), batch_size):
    batch = ad_spend_records[i:i+batch_size]
    conn.executemany(
        "INSERT INTO fact_ad_spend VALUES (?, ?, ?, ?, ?)",
        [(r['date'], r['campaign_id'], r['spend'], r['impressions'], r['clicks']) for r in batch]
    )

print(f"  Inserted {len(ad_spend_records)} ad spend records")

# 7. fact_sessions
print("Creating fact_sessions...")
sessions = []
session_id = 1
current_date = START_DATE

while current_date <= END_DATE:
    # Generate sessions for each campaign
    for campaign in campaigns:
        # Get channel info
        channel_info = next(c for c in CHANNELS if c['channel'] == campaign['channel'])

        # Number of sessions varies by channel
        num_sessions = random.randint(10, 50)

        for _ in range(num_sessions):
            # Conversion rate varies by channel
            if channel_info['channel'] == 'referral':
                cvr = random.uniform(0.08, 0.12)  # Recent anomaly - higher CVR
            elif channel_info['channel'] == 'email':
                cvr = random.uniform(0.12, 0.18)
            elif channel_info['channel'] == 'search':
                cvr = random.uniform(0.05, 0.08)
            else:
                cvr = random.uniform(0.02, 0.06)

            converted = random.random() < cvr

            sessions.append({
                'session_id': session_id,
                'campaign_id': campaign['campaign_id'],
                'timestamp': current_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                ),
                'converted_flag': converted,
                'device_type': random.choice(DEVICES)
            })
            session_id += 1

    current_date += timedelta(days=1)

conn.execute("""
    CREATE TABLE fact_sessions (
        session_id INTEGER PRIMARY KEY,
        campaign_id INTEGER,
        timestamp TIMESTAMP,
        converted_flag BOOLEAN,
        device_type VARCHAR
    )
""")

# Batch insert
batch_size = 1000
for i in range(0, len(sessions), batch_size):
    batch = sessions[i:i+batch_size]
    conn.executemany(
        "INSERT INTO fact_sessions VALUES (?, ?, ?, ?, ?)",
        [(s['session_id'], s['campaign_id'], s['timestamp'], s['converted_flag'], s['device_type'])
         for s in batch]
    )

print(f"  Inserted {len(sessions)} session records")

# 8. fact_orders
print("Creating fact_orders...")
orders = []
order_id = 1

# Only converted sessions generate orders
converted_sessions = [s for s in sessions if s['converted_flag']]

for session in converted_sessions:
    # Get campaign channel
    campaign = next(c for c in campaigns if c['campaign_id'] == session['campaign_id'])
    channel_info = next(c for c in CHANNELS if c['channel'] == campaign['channel'])

    # Revenue varies by channel ROAS
    # Calculate expected revenue based on spend and ROAS
    avg_revenue = channel_info['avg_cac'] * channel_info['avg_roas']
    revenue = random.uniform(avg_revenue * 0.7, avg_revenue * 1.3)

    orders.append({
        'order_id': order_id,
        'session_id': session['session_id'],
        'order_timestamp': session['timestamp'] + timedelta(minutes=random.randint(1, 30)),
        'revenue': round(revenue, 2)
    })
    order_id += 1

conn.execute("""
    CREATE TABLE fact_orders (
        order_id INTEGER PRIMARY KEY,
        session_id INTEGER,
        order_timestamp TIMESTAMP,
        revenue DECIMAL(10, 2)
    )
""")

conn.executemany(
    "INSERT INTO fact_orders VALUES (?, ?, ?, ?)",
    [(o['order_id'], o['session_id'], o['order_timestamp'], o['revenue']) for o in orders]
)

print(f"  Inserted {len(orders)} order records")

# Summary statistics
print("\n" + "="*60)
print("Data Generation Summary")
print("="*60)
print(f"Campaigns: {len(campaigns)}")
print(f"Ad Groups: {len(adgroups)}")
print(f"Creatives: {len(creatives)}")
print(f"Products: {len(products)}")
print(f"Customers: {len(customers)}")
print(f"Ad Spend Records: {len(ad_spend_records)}")
print(f"Sessions: {len(sessions)}")
print(f"Orders: {len(orders)}")
print(f"\nDatabase saved to: {OUTPUT_PATH}")

# Verify with a quick query
print("\n" + "="*60)
print("Sample CAC by Channel (Last 90 Days)")
print("="*60)

result = conn.execute("""
    WITH spend AS (
        SELECT
            c.channel,
            SUM(s.spend) as spend
        FROM fact_ad_spend s
        INNER JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
        WHERE s.date >= CURRENT_DATE - INTERVAL '90' DAY
        GROUP BY c.channel
    ),
    conversions AS (
        SELECT
            c.channel,
            SUM(CASE WHEN s.converted_flag THEN 1 ELSE 0 END) as conversions
        FROM fact_sessions s
        INNER JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
        WHERE s.timestamp >= CURRENT_DATE - INTERVAL '90' DAY
        GROUP BY c.channel
    )
    SELECT
        COALESCE(sp.channel, cv.channel) as channel,
        COALESCE(sp.spend, 0) as spend,
        COALESCE(cv.conversions, 0) as conversions,
        CASE
            WHEN COALESCE(cv.conversions, 0) > 0
            THEN COALESCE(sp.spend, 0) / NULLIF(cv.conversions, 0)
            ELSE NULL
        END as cac
    FROM spend sp
    FULL OUTER JOIN conversions cv ON sp.channel = cv.channel
    ORDER BY cac ASC
""").fetchall()

for row in result:
    print(f"  {row[0]:15} CAC: ${row[3]:.2f} (spend: ${row[1]:.2f}, conversions: {row[2]})")

conn.close()
print("\n✓ Data generation complete!")
