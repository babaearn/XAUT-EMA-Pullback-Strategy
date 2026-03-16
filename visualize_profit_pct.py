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
    df['date'] = df['exit_time'].dt.date
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

INITIAL_EQUITY = 10000.0

def calc_profit_pct(group):
    # Total PnL for the period as a percentage of initial equity
    net_pnl = group['pnl'].sum()
    pct_return = (net_pnl / INITIAL_EQUITY) * 100
    return round(pct_return, 2)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
fig = plt.figure(tight_layout=True)

# 1. Yearly Profit %
ax1 = plt.subplot2grid((3, 2), (0, 0))
yearly_pct = df.groupby('year').apply(calc_profit_pct).reset_index(name='Profit_Percentage')
# use diverging palette centered at 0
norm1 = plt.Normalize(yearly_pct['Profit_Percentage'].min(), yearly_pct['Profit_Percentage'].max())
sm1 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm1)
sns.barplot(x='year', y='Profit_Percentage', data=yearly_pct, ax=ax1, palette=sm1.to_rgba(yearly_pct['Profit_Percentage']))
ax1.axhline(0.0, color='black', linestyle='--', alpha=0.7) # Breakeven line at 0%
ax1.set_title('Yearly Net Profit (%)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Profit (%)')
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom' if p.get_height() >= 0 else 'top')

# 2. Monthly Profit % (Aggregated across years)
ax2 = plt.subplot2grid((3, 2), (0, 1))
monthly_pct = df.groupby('month').apply(calc_profit_pct).reset_index(name='Profit_Percentage')
monthly_pct['month_name'] = monthly_pct['month'].apply(lambda x: calendar.month_abbr[x])
norm2 = plt.Normalize(monthly_pct['Profit_Percentage'].min(), monthly_pct['Profit_Percentage'].max())
sm2 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm2)
sns.barplot(x='month_name', y='Profit_Percentage', data=monthly_pct, ax=ax2, palette=sm2.to_rgba(monthly_pct['Profit_Percentage']))
ax2.axhline(0.0, color='black', linestyle='--', alpha=0.7) 
ax2.set_title('Overall Monthly Net Profit (%) (Seasonality)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Profit (%)')
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom' if p.get_height() >= 0 else 'top')

# 3. Day of Week Profit %
ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_pct = df.groupby('day_of_week').apply(calc_profit_pct).reset_index(name='Profit_Percentage')
dow_pct['day_of_week'] = pd.Categorical(dow_pct['day_of_week'], categories=day_order, ordered=True)
dow_pct = dow_pct.sort_values('day_of_week')
norm3 = plt.Normalize(dow_pct['Profit_Percentage'].min(), dow_pct['Profit_Percentage'].max())
sm3 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm3)
sns.barplot(x='day_of_week', y='Profit_Percentage', data=dow_pct, ax=ax3, palette=sm3.to_rgba(dow_pct['Profit_Percentage']))
ax3.axhline(0.0, color='black', linestyle='--', alpha=0.7)
ax3.set_title('Net Profit (%) by Day of the Week', fontsize=14, fontweight='bold')
ax3.set_xlabel('Day')
ax3.set_ylabel('Profit (%)')
for p in ax3.patches:
    ax3.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom' if p.get_height() >= 0 else 'top')

# 4. Weekly Profit % Heatmap (Year x Week)
ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
weekly_matrix = df.groupby(['year', 'week']).apply(calc_profit_pct).unstack(level=1)
vmax = max(abs(weekly_matrix.max().max()), abs(weekly_matrix.min().min()))
sns.heatmap(weekly_matrix, cmap="RdYlGn", center=0.0, annot=False, ax=ax4, linewidths=.5, vmin=-vmax, vmax=vmax, cbar_kws={'label': 'Net Profit (%)'})
ax4.set_title('Weekly Net Profit (%) Heatmap (Red < 0% < Green)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Week of Year')
ax4.set_ylabel('Year')

# Save the figure
output_path = '/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/profit_percentage_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Charts successfully generated and saved to: {output_path}")
