# Sports Performance Analysis - Football/Soccer
# Domain: Sports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("bright")

# Generate synthetic football player data
np.random.seed(456)

n_players = 200

positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F', 'Team G', 'Team H']

data = {
    'player_id': range(1, n_players + 1),
    'player_name': [f'Player_{i}' for i in range(1, n_players + 1)],
    'age': np.random.randint(18, 36, n_players),
    'position': np.random.choice(positions, n_players),
    'team': np.random.choice(teams, n_players),
    'matches_played': np.random.randint(15, 40, n_players),
    'minutes_played': np.random.randint(500, 3500, n_players),
    'goals': np.random.poisson(3, n_players),
    'assists': np.random.poisson(2, n_players),
    'shots': np.random.randint(10, 100, n_players),
    'passes_completed': np.random.randint(200, 2000, n_players),
    'pass_accuracy': np.random.uniform(65, 95, n_players).round(1),
    'tackles': np.random.randint(10, 80, n_players),
    'interceptions': np.random.randint(5, 60, n_players),
    'fouls': np.random.randint(5, 40, n_players),
    'yellow_cards': np.random.randint(0, 8, n_players),
    'red_cards': np.random.randint(0, 2, n_players),
    'rating': np.random.uniform(6.0, 9.0, n_players).round(1)
}

df = pd.DataFrame(data)

# Adjust stats based on position
for idx, row in df.iterrows():
    if row['position'] == 'Forward':
        df.at[idx, 'goals'] = np.random.poisson(8)
        df.at[idx, 'assists'] = np.random.poisson(4)
        df.at[idx, 'shots'] = np.random.randint(40, 120)
    elif row['position'] == 'Midfielder':
        df.at[idx, 'goals'] = np.random.poisson(4)
        df.at[idx, 'assists'] = np.random.poisson(6)
        df.at[idx, 'passes_completed'] = np.random.randint(800, 2500)
    elif row['position'] == 'Defender':
        df.at[idx, 'goals'] = np.random.poisson(1)
        df.at[idx, 'tackles'] = np.random.randint(40, 100)
        df.at[idx, 'interceptions'] = np.random.randint(30, 80)
    elif row['position'] == 'Goalkeeper':
        df.at[idx, 'goals'] = 0
        df.at[idx, 'assists'] = 0
        df.at[idx, 'saves'] = np.random.randint(50, 150)

# Add saves column for all players (primarily for goalkeepers)
df['saves'] = np.random.randint(0, 20, n_players)
for idx, row in df.iterrows():
    if row['position'] == 'Goalkeeper':
        df.at[idx, 'saves'] = np.random.randint(50, 150)

# Calculate per-90-minute stats
df['goals_per_90'] = (df['goals'] / df['minutes_played'] * 90).round(2)
df['assists_per_90'] = (df['assists'] / df['minutes_played'] * 90).round(2)
df['goal_contributions'] = df['goals'] + df['assists']

print("=" * 80)
print("FOOTBALL/SOCCER PERFORMANCE ANALYSIS")
print("=" * 80)

# 1. Data Overview
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total Players: {len(df)}")
print(f"\nFirst few records:")
print(df.head(10))

# 2. Player Demographics
print("\n\n2. PLAYER DEMOGRAPHICS")
print("-" * 80)
print(f"Position Distribution:")
print(df['position'].value_counts())
print(f"\nTeam Distribution:")
print(df['team'].value_counts())
print(f"\nAge Statistics:")
print(df['age'].describe())

# 3. Overall Performance Statistics
print("\n\n3. OVERALL PERFORMANCE STATISTICS")
print("-" * 80)
print(f"Total Goals Scored: {df['goals'].sum()}")
print(f"Total Assists: {df['assists'].sum()}")
print(f"Total Matches Played: {df['matches_played'].sum()}")
print(f"Average Player Rating: {df['rating'].mean():.2f}")
print(f"\nTop Performers:")
print(df.nlargest(10, 'rating')[['player_name', 'position', 'team', 'rating', 'goals', 'assists']])

# 4. Position-wise Analysis
print("\n\n4. POSITION-WISE ANALYSIS")
print("-" * 80)
position_stats = df.groupby('position').agg({
    'goals': 'sum',
    'assists': 'sum',
    'passes_completed': 'mean',
    'pass_accuracy': 'mean',
    'tackles': 'mean',
    'rating': 'mean'
}).round(2)
print(position_stats)

# 5. Top Scorers
print("\n\n5. TOP 10 SCORERS")
print("-" * 80)
top_scorers = df.nlargest(10, 'goals')[['player_name', 'position', 'team', 'goals', 'matches_played', 'goals_per_90']]
print(top_scorers)

# 6. Top Assist Providers
print("\n\n6. TOP 10 ASSIST PROVIDERS")
print("-" * 80)
top_assists = df.nlargest(10, 'assists')[['player_name', 'position', 'team', 'assists', 'matches_played']]
print(top_assists)

# 7. Team Performance
print("\n\n7. TEAM PERFORMANCE ANALYSIS")
print("-" * 80)
team_stats = df.groupby('team').agg({
    'goals': 'sum',
    'assists': 'sum',
    'rating': 'mean',
    'player_id': 'count'
}).round(2)
team_stats.columns = ['Total Goals', 'Total Assists', 'Avg Rating', 'Squad Size']
team_stats = team_stats.sort_values('Total Goals', ascending=False)
print(team_stats)

# 8. Disciplinary Records
print("\n\n8. DISCIPLINARY ANALYSIS")
print("-" * 80)
print(f"Total Yellow Cards: {df['yellow_cards'].sum()}")
print(f"Total Red Cards: {df['red_cards'].sum()}")
print(f"Total Fouls: {df['fouls'].sum()}")
print(f"\nMost Disciplined Teams:")
discipline = df.groupby('team')[['yellow_cards', 'red_cards', 'fouls']].sum().sort_values('yellow_cards')
print(discipline)

# 9. Passing Analysis
print("\n\n9. PASSING ANALYSIS")
print("-" * 80)
print(f"Top 10 Passers:")
top_passers = df.nlargest(10, 'passes_completed')[['player_name', 'position', 'team', 'passes_completed', 'pass_accuracy']]
print(top_passers)
print(f"\nAverage Pass Accuracy by Position:")
print(df.groupby('position')['pass_accuracy'].mean().round(2))

# 10. Defensive Analysis
print("\n\n10. DEFENSIVE ANALYSIS")
print("-" * 80)
print(f"Top 10 Defenders (by tackles):")
top_defenders = df.nlargest(10, 'tackles')[['player_name', 'position', 'team', 'tackles', 'interceptions']]
print(top_defenders)

# 11. Age Analysis
print("\n\n11. AGE-BASED PERFORMANCE")
print("-" * 80)
df['age_group'] = pd.cut(df['age'], bins=[17, 23, 28, 35, 40], labels=['18-23', '24-28', '29-35', '36+'])
age_performance = df.groupby('age_group')[['rating', 'goals', 'assists']].mean().round(2)
print(age_performance)

# 12. Visualizations
print("\n\n12. GENERATING VISUALIZATIONS")
print("-" * 80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Goals by Position
ax1 = plt.subplot(3, 3, 1)
position_goals = df.groupby('position')['goals'].sum().sort_values(ascending=False)
position_goals.plot(kind='bar', color='dodgerblue')
plt.title('Total Goals by Position', fontsize=12, fontweight='bold')
plt.xlabel('Position')
plt.ylabel('Goals')
plt.xticks(rotation=45, ha='right')

# Plot 2: Top 10 Scorers
ax2 = plt.subplot(3, 3, 2)
top_10_scorers = df.nlargest(10, 'goals')
plt.barh(range(len(top_10_scorers)), top_10_scorers['goals'], color='orange')
plt.yticks(range(len(top_10_scorers)), top_10_scorers['player_name'])
plt.title('Top 10 Scorers', fontsize=12, fontweight='bold')
plt.xlabel('Goals')
plt.gca().invert_yaxis()

# Plot 3: Team Performance
ax3 = plt.subplot(3, 3, 3)
team_goals = df.groupby('team')['goals'].sum().sort_values(ascending=False)
team_goals.plot(kind='bar', color='green')
plt.title('Goals by Team', fontsize=12, fontweight='bold')
plt.xlabel('Team')
plt.ylabel('Total Goals')
plt.xticks(rotation=45, ha='right')

# Plot 4: Goals vs Assists Scatter
ax4 = plt.subplot(3, 3, 4)
colors = {'Forward': 'red', 'Midfielder': 'blue', 'Defender': 'green', 'Goalkeeper': 'orange'}
for position in df['position'].unique():
    pos_data = df[df['position'] == position]
    plt.scatter(pos_data['goals'], pos_data['assists'], 
               label=position, alpha=0.6, s=50, c=colors[position])
plt.title('Goals vs Assists by Position', fontsize=12, fontweight='bold')
plt.xlabel('Goals')
plt.ylabel('Assists')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Pass Accuracy Distribution
ax5 = plt.subplot(3, 3, 5)
plt.hist(df['pass_accuracy'], bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.title('Pass Accuracy Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Pass Accuracy (%)')
plt.ylabel('Number of Players')
plt.axvline(df['pass_accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {df["pass_accuracy"].mean():.1f}%')
plt.legend()

# Plot 6: Player Ratings by Position
ax6 = plt.subplot(3, 3, 6)
df.boxplot(column='rating', by='position', ax=ax6)
plt.title('Player Ratings by Position', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Position')
plt.ylabel('Rating')

# Plot 7: Age Distribution
ax7 = plt.subplot(3, 3, 7)
plt.hist(df['age'], bins=15, color='teal', alpha=0.7, edgecolor='black')
plt.title('Age Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Number of Players')

# Plot 8: Disciplinary Records
ax8 = plt.subplot(3, 3, 8)
discipline_data = df.groupby('position')[['yellow_cards', 'red_cards']].sum()
discipline_data.plot(kind='bar', ax=ax8, color=['yellow', 'red'])
plt.title('Disciplinary Records by Position', fontsize=12, fontweight='bold')
plt.xlabel('Position')
plt.ylabel('Number of Cards')
plt.xticks(rotation=45, ha='right')
plt.legend(['Yellow Cards', 'Red Cards'])

# Plot 9: Minutes Played vs Rating
ax9 = plt.subplot(3, 3, 9)
plt.scatter(df['minutes_played'], df['rating'], alpha=0.5, c=df['goals'], cmap='Reds')
plt.colorbar(label='Goals')
plt.title('Minutes Played vs Rating', fontsize=12, fontweight='bold')
plt.xlabel('Minutes Played')
plt.ylabel('Rating')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sports_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'sports_analysis_dashboard.png'")

# 13. Advanced Metrics
print("\n\n13. ADVANCED PERFORMANCE METRICS")
print("-" * 80)
df['shot_accuracy'] = (df['goals'] / df['shots'] * 100).round(1)
df['involvement_score'] = df['goal_contributions'] / df['matches_played']

print("Most Efficient Forwards (Goals per Shot):")
forwards = df[df['position'] == 'Forward'].nlargest(5, 'shot_accuracy')[['player_name', 'goals', 'shots', 'shot_accuracy']]
print(forwards)

print("\nMost Involved Players (Goal Contributions per Match):")
involved = df.nlargest(10, 'involvement_score')[['player_name', 'position', 'goal_contributions', 'matches_played', 'involvement_score']]
print(involved)

# 14. Key Insights
print("\n\n14. KEY INSIGHTS & RECOMMENDATIONS")
print("-" * 80)
print("\nKey Findings:")
print(f"1. Top Scorer: {df.nlargest(1, 'goals')['player_name'].values[0]} with {df['goals'].max()} goals")
print(f"2. Highest Rated Player: {df.nlargest(1, 'rating')['player_name'].values[0]} (Rating: {df['rating'].max()})")
print(f"3. Best Team: {team_stats.index[0]} with {team_stats.iloc[0]['Total Goals']:.0f} goals")
print(f"4. Average Pass Accuracy: {df['pass_accuracy'].mean():.1f}%")
print(f"5. Most Disciplined Position: {discipline.groupby(df['position'])['yellow_cards'].sum().idxmin()}")

print("\nCoaching Recommendations:")
print("• Focus on improving shot accuracy for forwards")
print("• Develop passing skills for players below 75% accuracy")
print("• Implement disciplinary training for high-foul positions")
print("• Balance youth development with experienced player retention")
print("• Optimize playing time based on performance ratings")
print("• Strengthen defensive tactics for teams with low tackle/interception rates")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Save dataset
df.to_csv('sports_player_data.csv', index=False)
print("\n✓ Dataset saved as 'sports_player_data.csv'")