import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os

# Load trade log data
try:
    df = pd.read_csv('paxg_trades_log.csv')
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Extract date parts
    df['year'] = df['exit_time'].dt.year
    df['month'] = df['exit_time'].dt.month
    df['month_name'] = df['exit_time'].dt.month_name()
    df['week'] = df['exit_time'].dt.isocalendar().week
    df['day_of_week'] = df['exit_time'].dt.day_name()
    df['hour_utc'] = df['exit_time'].dt.hour
    
    # Assign Trading Sessions (Approximated UTC)
    def assign_session(hour):
        if 0 <= hour < 8:
            return 'Asian Session (00:00-08:00 UTC)'
        elif 8 <= hour < 13:
            return 'London Morning (08:00-13:00 UTC)'
        elif 13 <= hour < 17:
            return 'NY / London Overlap (13:00-17:00 UTC)'
        elif 17 <= hour < 22:
            return 'New York Afternoon (17:00-22:00 UTC)'
        else:
            return 'Late NY / Sydney (22:00-24:00 UTC)'
            
    df['session'] = df['hour_utc'].apply(assign_session)
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

INITIAL_EQUITY = 10000.0

def calc_profit_pct(group):
    net_pnl = group['pnl'].sum()
    pct_return = (net_pnl / INITIAL_EQUITY) * 100
    return round(pct_return, 2)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)
fig = plt.figure(tight_layout=True)

# 1. Trading Session Profit %
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
session_order = [
    'Asian Session (00:00-08:00 UTC)',
    'London Morning (08:00-13:00 UTC)',
    'NY / London Overlap (13:00-17:00 UTC)',
    'New York Afternoon (17:00-22:00 UTC)',
    'Late NY / Sydney (22:00-24:00 UTC)'
]
session_pct = df.groupby('session').apply(calc_profit_pct).reset_index(name='Profit_Percentage')
session_pct['session'] = pd.Categorical(session_pct['session'], categories=session_order, ordered=True)
session_pct = session_pct.sort_values('session')
norm1 = plt.Normalize(session_pct['Profit_Percentage'].min(), session_pct['Profit_Percentage'].max())
sm1 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm1)
sns.barplot(x='session', y='Profit_Percentage', data=session_pct, ax=ax1, palette=sm1.to_rgba(session_pct['Profit_Percentage']))
ax1.axhline(0.0, color='black', linestyle='--', alpha=0.7)
ax1.set_title('Net Profit (%) by Trading Session (UTC)', fontsize=15, fontweight='bold')
ax1.set_xlabel('Market Session')
ax1.set_ylabel('Net Profit (%)')
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom' if p.get_height() >= 0 else 'top', fontsize=11, fontweight='bold')

# 2. Day of Week Profit %
ax2 = plt.subplot2grid((3, 2), (1, 0))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_pct = df.groupby('day_of_week').apply(calc_profit_pct).reset_index(name='Profit_Percentage')
dow_pct['day_of_week'] = pd.Categorical(dow_pct['day_of_week'], categories=day_order, ordered=True)
dow_pct = dow_pct.sort_values('day_of_week')
norm2 = plt.Normalize(dow_pct['Profit_Percentage'].min(), dow_pct['Profit_Percentage'].max())
sm2 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm2)
sns.barplot(x='day_of_week', y='Profit_Percentage', data=dow_pct, ax=ax2, palette=sm2.to_rgba(dow_pct['Profit_Percentage']))
ax2.axhline(0.0, color='black', linestyle='--', alpha=0.7)
ax2.set_title('Net Profit (%) by Day of the Week', fontsize=14, fontweight='bold')
ax2.set_xlabel('Day')
ax2.set_ylabel('Profit (%)')
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom' if p.get_height() >= 0 else 'top')

# 3. Monthly Profit %
ax3 = plt.subplot2grid((3, 2), (1, 1))
monthly_pct = df.groupby('month').apply(calc_profit_pct).reset_index(name='Profit_Percentage')
monthly_pct['month_name'] = monthly_pct['month'].apply(lambda x: calendar.month_abbr[x])
norm3 = plt.Normalize(monthly_pct['Profit_Percentage'].min(), monthly_pct['Profit_Percentage'].max())
sm3 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm3)
sns.barplot(x='month_name', y='Profit_Percentage', data=monthly_pct, ax=ax3, palette=sm3.to_rgba(monthly_pct['Profit_Percentage']))
ax3.axhline(0.0, color='black', linestyle='--', alpha=0.7) 
ax3.set_title('Net Profit (%) by Month (Seasonality)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Profit (%)')
for p in ax3.patches:
    ax3.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom' if p.get_height() >= 0 else 'top')

# 4. Yearly Profit %
ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
yearly_pct = df.groupby('year').apply(calc_profit_pct).reset_index(name='Profit_Percentage')
norm4 = plt.Normalize(yearly_pct['Profit_Percentage'].min(), yearly_pct['Profit_Percentage'].max())
sm4 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm4)
sns.barplot(x='year', y='Profit_Percentage', data=yearly_pct, ax=ax4, palette=sm4.to_rgba(yearly_pct['Profit_Percentage']))
ax4.axhline(0.0, color='black', linestyle='--', alpha=0.7)
ax4.set_title('Yearly Net Profit (%)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Profit (%)')
for p in ax4.patches:
    ax4.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom' if p.get_height() >= 0 else 'top', fontsize=11, fontweight='bold')

# Save the figure
output_path = '/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/profit_sessions_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Charts successfully generated and saved to: {output_path}")

# Export session metrics for markdown
session_df = df.groupby('session').agg(
    total_trades=('pnl', 'count'),
    wins=('pnl', lambda x: (x > 0).sum()),
    losses=('pnl', lambda x: (x <= 0).sum()),
    pct_return=('pnl', lambda x: round(x.sum() / INITIAL_EQUITY * 100, 2))
).reset_index()
session_df.to_csv('paxg_sessions.csv', index=False)
