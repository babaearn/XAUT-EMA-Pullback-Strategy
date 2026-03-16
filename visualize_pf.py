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

def calc_pf(group):
    gross_profit = group[group['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(group[group['pnl'] < 0]['pnl'].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0
    return round(gross_profit / gross_loss, 2)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
fig = plt.figure(tight_layout=True)

# 1. Yearly PF
ax1 = plt.subplot2grid((3, 2), (0, 0))
yearly_pf = df.groupby('year').apply(calc_pf).reset_index(name='Profit_Factor')
sns.barplot(x='year', y='Profit_Factor', data=yearly_pf, ax=ax1, palette=sns.color_palette("coolwarm", len(yearly_pf)))
ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7) # Breakeven line
ax1.set_title('Yearly Profit Factor', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Profit Factor')
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

# 2. Monthly PF (Aggregated across years to find worst months generally)
ax2 = plt.subplot2grid((3, 2), (0, 1))
monthly_pf = df.groupby('month').apply(calc_pf).reset_index(name='Profit_Factor')
monthly_pf['month_name'] = monthly_pf['month'].apply(lambda x: calendar.month_abbr[x])
sns.barplot(x='month_name', y='Profit_Factor', data=monthly_pf, ax=ax2, palette=sns.color_palette("coolwarm", 12))
ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7) 
ax2.set_title('Overall Monthly Profit Factor (Seasonality)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Profit Factor')
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

# 3. Day of Week PF
ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_pf = df.groupby('day_of_week').apply(calc_pf).reset_index(name='Profit_Factor')
dow_pf['day_of_week'] = pd.Categorical(dow_pf['day_of_week'], categories=day_order, ordered=True)
dow_pf = dow_pf.sort_values('day_of_week')
sns.barplot(x='day_of_week', y='Profit_Factor', data=dow_pf, ax=ax3, palette=sns.color_palette("coolwarm", 7))
ax3.axhline(1.0, color='red', linestyle='--', alpha=0.7)
ax3.set_title('Profit Factor by Day of the Week', fontsize=14, fontweight='bold')
ax3.set_xlabel('Day')
ax3.set_ylabel('Profit Factor')
for p in ax3.patches:
    ax3.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

# 4. Weekly PF Heatmap (Year x Week)
ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
weekly_matrix = df.groupby(['year', 'week']).apply(calc_pf).unstack(level=1)
sns.heatmap(weekly_matrix, cmap="RdYlGn", center=1.0, annot=False, ax=ax4, linewidths=.5, cbar_kws={'label': 'Profit Factor'})
ax4.set_title('Weekly Profit Factor Heatmap (Red < 1.0 < Green)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Week of Year')
ax4.set_ylabel('Year')

# Save the figure
output_path = '/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/profit_factor_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Charts successfully generated and saved to: {output_path}")
