#!/usr/bin/env python3
"""
Create sample DuckDB database with synthetic marketing data.

This script generates a sample marketing.duckdb file for testing the
Tasman Agentic Analytics system.
"""

import duckdb
from pathlib import Path
from datetime import datetime, timedelta
import random

# Configuration
DB_PATH = Path(__file__).parent.parent / "data" / "marketing.duckdb"
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
NUM_CAMPAIGNS = 20
NUM_ADGROUPS = 50
NUM_CREATIVES = 100
NUM_PRODUCTS = 200
NUM_CUSTOMERS = 5000


def create_database():
    """Create sample database with schema and data."""

    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing database: {DB_PATH}")

    # Connect to new database
    conn = duckdb.connect(str(DB_PATH))
    print(f"Creating database: {DB_PATH}")

    # Create dimension tables
    create_dimensions(conn)

    # Create fact tables
    create_facts(conn)

    # Summary
    print("\n✅ Database created successfully!")
    print("\nTable sizes:")
    for table in ['dim_campaigns', 'dim_adgroups', 'dim_creatives',
                  'dim_products', 'dim_customers',
                  'fact_ad_spend', 'fact_sessions', 'fact_orders']:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")

    conn.close()


def create_dimensions(conn):
    """Create dimension tables with sample data."""

    # dim_campaigns
    print("\nCreating dim_campaigns...")
    conn.execute("""
        CREATE TABLE dim_campaigns (
            campaign_id INTEGER PRIMARY KEY,
            channel TEXT,
            campaign_name TEXT,
            start_date DATE,
            end_date DATE,
            objective TEXT
        )
    """)

    channels = ['Google', 'Facebook', 'Instagram', 'TikTok', 'LinkedIn']
    objectives = ['Awareness', 'Consideration', 'Conversion']

    for i in range(1, NUM_CAMPAIGNS + 1):
        channel = random.choice(channels)
        objective = random.choice(objectives)
        start = START_DATE + timedelta(days=random.randint(0, 90))
        end = start + timedelta(days=random.randint(30, 180))

        conn.execute("""
            INSERT INTO dim_campaigns VALUES (?, ?, ?, ?, ?, ?)
        """, [i, channel, f"{channel}_{objective}_{i}", start, end, objective])

    # dim_adgroups
    print("Creating dim_adgroups...")
    conn.execute("""
        CREATE TABLE dim_adgroups (
            adgroup_id INTEGER PRIMARY KEY,
            campaign_id INTEGER,
            audience TEXT,
            placement TEXT
        )
    """)

    audiences = ['18-24', '25-34', '35-44', '45-54', '55+']
    placements = ['Feed', 'Stories', 'Search', 'Display', 'Video']

    for i in range(1, NUM_ADGROUPS + 1):
        campaign_id = random.randint(1, NUM_CAMPAIGNS)
        audience = random.choice(audiences)
        placement = random.choice(placements)

        conn.execute("""
            INSERT INTO dim_adgroups VALUES (?, ?, ?, ?)
        """, [i, campaign_id, audience, placement])

    # dim_creatives
    print("Creating dim_creatives...")
    conn.execute("""
        CREATE TABLE dim_creatives (
            creative_id INTEGER PRIMARY KEY,
            adgroup_id INTEGER,
            format TEXT,
            asset_url TEXT
        )
    """)

    formats = ['Image', 'Video', 'Carousel', 'Collection']

    for i in range(1, NUM_CREATIVES + 1):
        adgroup_id = random.randint(1, NUM_ADGROUPS)
        format = random.choice(formats)

        conn.execute("""
            INSERT INTO dim_creatives VALUES (?, ?, ?, ?)
        """, [i, adgroup_id, format, f"https://assets.example.com/{i}.jpg"])

    # dim_products
    print("Creating dim_products...")
    conn.execute("""
        CREATE TABLE dim_products (
            sku TEXT PRIMARY KEY,
            category TEXT,
            subcategory TEXT,
            price FLOAT,
            margin FLOAT,
            brand TEXT
        )
    """)

    categories = {
        'Electronics': ['Phones', 'Laptops', 'Tablets', 'Accessories'],
        'Clothing': ['Shirts', 'Pants', 'Shoes', 'Accessories'],
        'Home': ['Furniture', 'Decor', 'Kitchen', 'Bedding']
    }
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

    for i in range(1, NUM_PRODUCTS + 1):
        category = random.choice(list(categories.keys()))
        subcategory = random.choice(categories[category])
        brand = random.choice(brands)
        price = round(random.uniform(10, 500), 2)
        margin = round(random.uniform(0.2, 0.6) * price, 2)

        conn.execute("""
            INSERT INTO dim_products VALUES (?, ?, ?, ?, ?, ?)
        """, [f"SKU{i:05d}", category, subcategory, price, margin, brand])

    # dim_customers
    print("Creating dim_customers...")
    conn.execute("""
        CREATE TABLE dim_customers (
            customer_id INTEGER PRIMARY KEY,
            acquisition_channel TEXT,
            first_visit_date DATE,
            region TEXT
        )
    """)

    channels = ['Google', 'Facebook', 'Instagram', 'TikTok', 'LinkedIn', 'Organic']
    regions = ['North', 'South', 'East', 'West', 'Central']

    for i in range(1, NUM_CUSTOMERS + 1):
        channel = random.choice(channels)
        region = random.choice(regions)
        first_visit = START_DATE + timedelta(days=random.randint(0, 300))

        conn.execute("""
            INSERT INTO dim_customers VALUES (?, ?, ?, ?)
        """, [i, channel, first_visit, region])


def create_facts(conn):
    """Create fact tables with sample data."""

    # fact_ad_spend
    print("\nCreating fact_ad_spend...")
    conn.execute("""
        CREATE TABLE fact_ad_spend (
            date DATE,
            campaign_id INTEGER,
            adgroup_id INTEGER,
            creative_id INTEGER,
            spend FLOAT,
            impressions INTEGER,
            clicks INTEGER
        )
    """)

    current_date = START_DATE
    while current_date <= END_DATE:
        # Generate data for 5-10 random campaign/adgroup/creative combos per day
        for _ in range(random.randint(5, 10)):
            campaign_id = random.randint(1, NUM_CAMPAIGNS)
            adgroup_id = random.randint(1, NUM_ADGROUPS)
            creative_id = random.randint(1, NUM_CREATIVES)

            spend = round(random.uniform(100, 5000), 2)
            impressions = int(spend * random.uniform(10, 100))
            clicks = int(impressions * random.uniform(0.01, 0.05))

            conn.execute("""
                INSERT INTO fact_ad_spend VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [current_date, campaign_id, adgroup_id, creative_id,
                  spend, impressions, clicks])

        current_date += timedelta(days=1)

    # fact_sessions
    print("Creating fact_sessions...")
    conn.execute("""
        CREATE TABLE fact_sessions (
            session_id TEXT PRIMARY KEY,
            customer_id INTEGER,
            date DATE,
            campaign_id INTEGER,
            adgroup_id INTEGER,
            creative_id INTEGER,
            utm_source TEXT,
            utm_medium TEXT,
            utm_campaign TEXT,
            device TEXT,
            pages_viewed INTEGER,
            converted_flag BOOLEAN
        )
    """)

    devices = ['Desktop', 'Mobile', 'Tablet']
    utm_sources = ['Google', 'Facebook', 'Instagram', 'TikTok', 'LinkedIn']
    utm_mediums = ['cpc', 'display', 'social', 'video']

    session_count = 0
    current_date = START_DATE
    while current_date <= END_DATE:
        # Generate 10-50 sessions per day
        for _ in range(random.randint(10, 50)):
            session_count += 1
            customer_id = random.randint(1, NUM_CUSTOMERS)
            campaign_id = random.randint(1, NUM_CAMPAIGNS)
            adgroup_id = random.randint(1, NUM_ADGROUPS)
            creative_id = random.randint(1, NUM_CREATIVES)

            device = random.choice(devices)
            utm_source = random.choice(utm_sources)
            utm_medium = random.choice(utm_mediums)
            pages_viewed = random.randint(1, 20)
            converted = random.random() < 0.15  # 15% conversion rate

            conn.execute("""
                INSERT INTO fact_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [f"S{session_count:08d}", customer_id, current_date,
                  campaign_id, adgroup_id, creative_id,
                  utm_source, utm_medium, f"camp_{campaign_id}",
                  device, pages_viewed, converted])

        current_date += timedelta(days=1)

    # fact_orders
    print("Creating fact_orders...")
    conn.execute("""
        CREATE TABLE fact_orders (
            order_id TEXT PRIMARY KEY,
            session_id TEXT,
            customer_id INTEGER,
            order_timestamp TIMESTAMP,
            sku TEXT,
            quantity INTEGER,
            revenue FLOAT,
            margin FLOAT
        )
    """)

    # Get converted sessions
    converted_sessions = conn.execute("""
        SELECT session_id, customer_id, date
        FROM fact_sessions
        WHERE converted_flag = TRUE
    """).fetchall()

    order_count = 0
    for session_id, customer_id, date in converted_sessions:
        # Each converted session generates 1-3 orders
        num_orders = random.randint(1, 3)

        for _ in range(num_orders):
            order_count += 1

            # Get random product
            sku = f"SKU{random.randint(1, NUM_PRODUCTS):05d}"
            product = conn.execute("""
                SELECT price, margin FROM dim_products WHERE sku = ?
            """, [sku]).fetchone()

            if product:
                price, margin = product
                quantity = random.randint(1, 5)
                revenue = round(price * quantity, 2)
                total_margin = round(margin * quantity, 2)

                # Random time on that day
                timestamp = datetime.combine(date, datetime.min.time()) + \
                           timedelta(hours=random.randint(0, 23),
                                   minutes=random.randint(0, 59))

                conn.execute("""
                    INSERT INTO fact_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [f"O{order_count:08d}", session_id, customer_id,
                      timestamp, sku, quantity, revenue, total_margin])


if __name__ == "__main__":
    print("=" * 80)
    print("Tasman Agentic Analytics - Sample Data Generator")
    print("=" * 80)
    create_database()
    print(f"\n✅ Database saved to: {DB_PATH}")
    print("\nYou can now run the Jupyter notebook!")
    print("=" * 80)
