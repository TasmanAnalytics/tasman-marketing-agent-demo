"""Generate architecture diagram for the microagentic analytics system."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
COLOR_USER = '#E8F4F8'
COLOR_AGENT = '#B8E6B8'
COLOR_CORE = '#FFE6B8'
COLOR_DATA = '#E8D4F0'
COLOR_EXTERNAL = '#FFD4D4'

# Title
ax.text(5, 11.5, 'Tasman Agentic Analytics - Microagentic Architecture',
        ha='center', va='top', fontsize=18, fontweight='bold')
ax.text(5, 11.1, 'Local-first design with minimal LLM usage',
        ha='center', va='top', fontsize=12, style='italic', color='gray')

# ==================== USER LAYER ====================
user_box = FancyBboxPatch((0.5, 9.5), 9, 1, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=COLOR_USER, linewidth=2)
ax.add_patch(user_box)
ax.text(5, 10.2, 'USER QUESTION', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(5, 9.8, '"show ad spend per channel over time"', ha='center', va='center',
        fontsize=10, style='italic')

# ==================== SEARCH AGENT (Orchestrator) ====================
search_agent_box = FancyBboxPatch((1, 7.5), 8, 1.5, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor=COLOR_AGENT, linewidth=2)
ax.add_patch(search_agent_box)
ax.text(5, 8.7, 'SearchAgent (Orchestrator)', ha='center', va='center',
        fontsize=13, fontweight='bold')
ax.text(5, 8.35, '• End-to-end workflow coordination', ha='center', va='center', fontsize=9)
ax.text(5, 8.05, '• Manages: Triage → SQL Gen → Execute → Visualize → Summarize',
        ha='center', va='center', fontsize=9)
ax.text(5, 7.75, 'agents/agent_search.py', ha='center', va='center',
        fontsize=8, style='italic', color='gray')

# Arrow from user to SearchAgent
arrow1 = FancyArrowPatch((5, 9.5), (5, 9.0), arrowstyle='->', lw=2, color='black')
ax.add_patch(arrow1)

# ==================== AGENT LAYER ====================
# TriageAgent
triage_box = FancyBboxPatch((0.5, 5.5), 2.8, 1.5, boxstyle="round,pad=0.08",
                            edgecolor='black', facecolor=COLOR_AGENT, linewidth=1.5)
ax.add_patch(triage_box)
ax.text(1.9, 6.6, 'TriageAgent', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.9, 6.25, '• Rule-based classifier', ha='left', va='center', fontsize=8, transform=ax.transData)
ax.text(1.9, 6.0, '• Search vs Analysis', ha='left', va='center', fontsize=8)
ax.text(1.9, 5.75, '• Role inference', ha='left', va='center', fontsize=8)
ax.text(1.9, 5.55, 'agents/agent_triage.py', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# TextToSQLAgent
sql_box = FancyBboxPatch((3.6, 5.5), 2.8, 1.5, boxstyle="round,pad=0.08",
                         edgecolor='black', facecolor=COLOR_AGENT, linewidth=1.5)
ax.add_patch(sql_box)
ax.text(5.0, 6.6, 'TextToSQLAgent', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5.0, 6.25, '• Template matching', ha='left', va='center', fontsize=8)
ax.text(5.0, 6.0, '• SQL generation', ha='left', va='center', fontsize=8)
ax.text(5.0, 5.75, '• Validation', ha='left', va='center', fontsize=8)
ax.text(5.0, 5.55, 'agents/agent_text_to_sql.py', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# VizAgent (embedded in SearchAgent)
viz_box = FancyBboxPatch((6.7, 5.5), 2.8, 1.5, boxstyle="round,pad=0.08",
                         edgecolor='black', facecolor=COLOR_AGENT, linewidth=1.5)
ax.add_patch(viz_box)
ax.text(8.1, 6.6, 'AutoViz', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8.1, 6.25, '• Chart type detection', ha='left', va='center', fontsize=8)
ax.text(8.1, 6.0, '• Auto visualization', ha='left', va='center', fontsize=8)
ax.text(8.1, 5.75, '• Result summary', ha='left', va='center', fontsize=8)
ax.text(8.1, 5.55, 'core/viz.py', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# Arrows from SearchAgent to sub-agents
arrow2 = FancyArrowPatch((2.5, 7.5), (1.9, 7.0), arrowstyle='->', lw=1.5, color='black')
ax.add_patch(arrow2)
ax.text(2.1, 7.3, '1', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

arrow3 = FancyArrowPatch((5, 7.5), (5, 7.0), arrowstyle='->', lw=1.5, color='black')
ax.add_patch(arrow3)
ax.text(5.3, 7.3, '2', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

arrow4 = FancyArrowPatch((7.5, 7.5), (8.1, 7.0), arrowstyle='->', lw=1.5, color='black')
ax.add_patch(arrow4)
ax.text(7.9, 7.3, '4', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# ==================== CORE LOGIC LAYER ====================
# LocalTriage
local_triage_box = FancyBboxPatch((0.5, 3.5), 2.8, 1.5, boxstyle="round,pad=0.08",
                                  edgecolor='black', facecolor=COLOR_CORE, linewidth=1.5)
ax.add_patch(local_triage_box)
ax.text(1.9, 4.6, 'LocalTriage', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.9, 4.25, '• Keyword matching', ha='left', va='center', fontsize=8)
ax.text(1.9, 4.0, '• Confidence scoring', ha='left', va='center', fontsize=8)
ax.text(1.9, 3.75, '• No LLM needed', ha='left', va='center', fontsize=8, color='green', fontweight='bold')
ax.text(1.9, 3.55, 'core/triage_local.py', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# LocalTextToSQL
local_sql_box = FancyBboxPatch((3.6, 3.5), 2.8, 1.5, boxstyle="round,pad=0.08",
                               edgecolor='black', facecolor=COLOR_CORE, linewidth=1.5)
ax.add_patch(local_sql_box)
ax.text(5.0, 4.6, 'LocalTextToSQL', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5.0, 4.25, '• Fuzzy template match', ha='left', va='center', fontsize=8)
ax.text(5.0, 4.0, '• Parameter filling', ha='left', va='center', fontsize=8)
ax.text(5.0, 3.75, '• 6 SQL templates', ha='left', va='center', fontsize=8, color='green', fontweight='bold')
ax.text(5.0, 3.55, 'core/local_text_to_sql.py', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# DuckDBConnector
db_box = FancyBboxPatch((6.7, 3.5), 2.8, 1.5, boxstyle="round,pad=0.08",
                        edgecolor='black', facecolor=COLOR_CORE, linewidth=1.5)
ax.add_patch(db_box)
ax.text(8.1, 4.6, 'DuckDBConnector', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8.1, 4.25, '• Read-only execution', ha='left', va='center', fontsize=8)
ax.text(8.1, 4.0, '• LIMIT enforcement', ha='left', va='center', fontsize=8)
ax.text(8.1, 3.75, '• Schema validation', ha='left', va='center', fontsize=8)
ax.text(8.1, 3.55, 'core/duckdb_connector.py', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# Arrows from agents to core logic
arrow5 = FancyArrowPatch((1.9, 5.5), (1.9, 5.0), arrowstyle='->', lw=1.5, color='black', linestyle='--')
ax.add_patch(arrow5)

arrow6 = FancyArrowPatch((5.0, 5.5), (5.0, 5.0), arrowstyle='->', lw=1.5, color='black', linestyle='--')
ax.add_patch(arrow6)

# Arrow from SearchAgent to DB (step 3)
arrow7 = FancyArrowPatch((7.5, 7.8), (8.1, 5.0), arrowstyle='->', lw=1.5, color='black')
ax.add_patch(arrow7)
ax.text(7.9, 6.5, '3', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# ==================== LLM FALLBACK LAYER ====================
llm_box = FancyBboxPatch((0.5, 1.8), 4, 1.2, boxstyle="round,pad=0.08",
                         edgecolor='red', facecolor=COLOR_EXTERNAL, linewidth=2, linestyle='--')
ax.add_patch(llm_box)
ax.text(2.5, 2.7, 'LLM Fallback (Optional)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='red')
ax.text(2.5, 2.4, '• Only when confidence < 0.6', ha='left', va='center', fontsize=8)
ax.text(2.5, 2.15, '• OpenAI GPT-4o-mini / Anthropic Claude', ha='left', va='center', fontsize=8)
ax.text(2.5, 1.9, '• Filesystem cache for responses', ha='left', va='center', fontsize=8)
ax.text(2.5, 1.85, 'core/llm_clients.py', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# Dashed arrows to LLM (fallback paths)
arrow8 = FancyArrowPatch((1.4, 3.5), (1.8, 3.0), arrowstyle='<->', lw=1.5,
                        color='red', linestyle='--')
ax.add_patch(arrow8)
ax.text(1.1, 3.2, 'fallback\nif conf<0.6', ha='center', va='center',
        fontsize=7, color='red', style='italic')

arrow9 = FancyArrowPatch((4.5, 3.5), (3.2, 3.0), arrowstyle='<->', lw=1.5,
                        color='red', linestyle='--')
ax.add_patch(arrow9)
ax.text(4.2, 3.2, 'fallback\nif no match', ha='center', va='center',
        fontsize=7, color='red', style='italic')

# ==================== DATA LAYER ====================
# Config files
config_box = FancyBboxPatch((5.0, 1.8), 2.2, 1.2, boxstyle="round,pad=0.08",
                            edgecolor='black', facecolor=COLOR_DATA, linewidth=1.5)
ax.add_patch(config_box)
ax.text(6.1, 2.7, 'Configuration', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(6.1, 2.45, '• schema.json', ha='left', va='center', fontsize=8)
ax.text(6.1, 2.2, '• sql_templates.yaml', ha='left', va='center', fontsize=8)
ax.text(6.1, 1.95, '• business_context.yaml', ha='left', va='center', fontsize=8)
ax.text(6.1, 1.85, 'config/', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# Database
data_box = FancyBboxPatch((7.5, 1.8), 2, 1.2, boxstyle="round,pad=0.08",
                          edgecolor='black', facecolor=COLOR_DATA, linewidth=1.5)
ax.add_patch(data_box)
ax.text(8.5, 2.7, 'DuckDB', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8.5, 2.45, '8 tables', ha='center', va='center', fontsize=8)
ax.text(8.5, 2.2, '3 fact, 5 dim', ha='center', va='center', fontsize=8)
ax.text(8.5, 1.95, 'Read-only', ha='center', va='center', fontsize=8,
        color='green', fontweight='bold')
ax.text(8.5, 1.85, 'data/marketing.duckdb', ha='center', va='center',
        fontsize=7, style='italic', color='gray')

# Arrows to data layer
arrow10 = FancyArrowPatch((5.0, 4.0), (6.1, 3.0), arrowstyle='<-', lw=1, color='gray', linestyle=':')
ax.add_patch(arrow10)

arrow11 = FancyArrowPatch((8.1, 3.5), (8.5, 3.0), arrowstyle='<->', lw=1.5, color='black')
ax.add_patch(arrow11)

# ==================== OUTPUT LAYER ====================
output_box = FancyBboxPatch((1, 0.3), 8, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=COLOR_USER, linewidth=2)
ax.add_patch(output_box)
ax.text(5, 1.15, 'OUTPUT', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(5, 0.85, 'SQL Query | DataFrame | Chart (PNG) | Natural Language Summary',
        ha='center', va='center', fontsize=10)
ax.text(5, 0.55, 'Observability: confidence scores, methods used, LLM calls tracked',
        ha='center', va='center', fontsize=9, style='italic', color='gray')

# Arrow to output
arrow12 = FancyArrowPatch((8.1, 5.5), (7, 1.5), arrowstyle='->', lw=2, color='black')
ax.add_patch(arrow12)
ax.text(7.7, 3.5, '5', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# ==================== LEGEND ====================
legend_elements = [
    mpatches.Patch(facecolor=COLOR_AGENT, edgecolor='black', label='Agent Layer (Orchestration)'),
    mpatches.Patch(facecolor=COLOR_CORE, edgecolor='black', label='Core Logic Layer (Local-first)'),
    mpatches.Patch(facecolor=COLOR_EXTERNAL, edgecolor='red', label='LLM Fallback (Optional)', linestyle='--'),
    mpatches.Patch(facecolor=COLOR_DATA, edgecolor='black', label='Data & Configuration'),
    mlines.Line2D([], [], color='black', linewidth=2, label='Primary flow'),
    mlines.Line2D([], [], color='red', linewidth=1.5, linestyle='--', label='Fallback flow'),
]

ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, -0.15),
          ncol=3, frameon=True, fontsize=9)

# Add watermark
ax.text(9.5, 0.1, 'Phase 1 v0.1.0', ha='right', va='bottom',
        fontsize=8, color='gray', style='italic')

plt.tight_layout()
plt.savefig('/Users/thomas/Documents/GitHub/tasman-marketing-agent/docs/microagentic_architecture.png',
            dpi=300, bbox_inches='tight')
print("✅ Architecture diagram saved to: docs/microagentic_architecture.png")

# Also save as PDF for high quality
plt.savefig('/Users/thomas/Documents/GitHub/tasman-marketing-agent/docs/microagentic_architecture.pdf',
            bbox_inches='tight')
print("✅ PDF version saved to: docs/microagentic_architecture.pdf")
