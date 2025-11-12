#!/usr/bin/env python3
"""
Generate ERD diagram for marketing analytics database schema.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
dim_color = '#E3F2FD'  # Light blue for dimensions
fact_color = '#FFF3E0'  # Light orange for facts
border_color = '#1976D2'
fact_border_color = '#F57C00'

def draw_table(ax, x, y, width, height, title, fields, is_fact=False):
    """Draw a table box with title and fields."""
    color = fact_color if is_fact else dim_color
    border = fact_border_color if is_fact else border_color

    # Draw main box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        edgecolor=border,
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)

    # Title
    ax.text(x + width/2, y + height - 0.15, title,
            ha='center', va='top', fontsize=11, fontweight='bold',
            color=border)

    # Separator line
    ax.plot([x + 0.1, x + width - 0.1], [y + height - 0.35, y + height - 0.35],
            color=border, linewidth=1)

    # Fields
    field_y = y + height - 0.55
    for field in fields:
        if field.startswith('🔑'):
            # Primary key
            ax.text(x + 0.15, field_y, field,
                   ha='left', va='top', fontsize=9, fontweight='bold',
                   color='#D84315')
        elif 'FK' in field:
            # Foreign key
            ax.text(x + 0.15, field_y, field,
                   ha='left', va='top', fontsize=9, fontweight='bold',
                   color='#1565C0')
        else:
            # Regular field
            ax.text(x + 0.15, field_y, field,
                   ha='left', va='top', fontsize=9)
        field_y -= 0.2

def draw_arrow(ax, x1, y1, x2, y2, label=None, style='->'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color='#424242',
        linewidth=2,
        connectionstyle="arc3,rad=0.2"
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.1, label,
               ha='center', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

# Draw dim_campaigns (top center)
draw_table(ax, 6, 9.5, 3.5, 1.8,
          'dim_campaigns',
          ['🔑 campaign_id (PK)', 'channel', 'campaign_name'],
          is_fact=False)

# Draw fact_ad_spend (left)
draw_table(ax, 1, 7, 3.5, 1.8,
          'fact_ad_spend',
          ['campaign_id (FK)', 'date', 'spend', 'impressions', 'clicks'],
          is_fact=True)

# Draw fact_sessions (right)
draw_table(ax, 11, 7, 3.5, 1.8,
          'fact_sessions',
          ['🔑 session_id (PK)', 'campaign_id (FK)', 'device_type', 'converted_flag', 'session_start_time'],
          is_fact=True)

# Draw fact_orders (bottom center)
draw_table(ax, 6, 4, 3.5, 1.5,
          'fact_orders',
          ['🔑 order_id (PK)', 'session_id (FK)', 'revenue', 'order_timestamp'],
          is_fact=True)

# Draw unused dimensions (bottom)
draw_table(ax, 1, 1.5, 3, 1.2,
          'dim_customers',
          ['🔑 customer_id (PK)', 'region'],
          is_fact=False)

draw_table(ax, 5.5, 1.5, 3, 1.2,
          'dim_adgroups',
          ['🔑 adgroup_id (PK)', '(not used in demo)'],
          is_fact=False)

draw_table(ax, 10, 1.5, 3, 1.2,
          'dim_creatives',
          ['🔑 creative_id (PK)', '(not used in demo)'],
          is_fact=False)

# Draw relationships
# fact_ad_spend -> dim_campaigns
draw_arrow(ax, 4.5, 8.2, 6, 10.2, 'campaign_id', '->')

# fact_sessions -> dim_campaigns
draw_arrow(ax, 11, 8.2, 9.5, 10.2, 'campaign_id', '->')

# fact_orders -> fact_sessions
draw_arrow(ax, 9, 5.2, 11.5, 7, 'session_id', '->')

# Add title
ax.text(8, 11.5, 'Marketing Analytics Database Schema',
       ha='center', va='top', fontsize=16, fontweight='bold')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=dim_color, edgecolor=border_color, linewidth=2, label='Dimension Table'),
    mpatches.Patch(facecolor=fact_color, edgecolor=fact_border_color, linewidth=2, label='Fact Table'),
    mlines.Line2D([], [], color='#424242', marker='>', markersize=8, linewidth=2, label='Foreign Key Relationship')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

# Add key notes
note_text = """Key Data Flow:
• Ad Spend: fact_ad_spend → dim_campaigns (by channel)
• Conversions: fact_sessions → dim_campaigns (by channel)
• Revenue (Last-Touch): fact_orders → fact_sessions → dim_campaigns

Critical Metrics:
• CAC = Spend / Conversions
• ROAS = Revenue / Spend"""

ax.text(0.5, 5.5, note_text,
       ha='left', va='top', fontsize=9, family='monospace',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDE7', edgecolor='#F9A825', linewidth=2))

# Save
plt.tight_layout()
plt.savefig('/Users/thomas/Documents/GitHub/tasman-marketing-agent/demo/database_erd.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ ERD saved to demo/database_erd.png")
plt.close()
