# Social Media Sentiment Analysis
# Domain: Social Media / Marketing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Generate synthetic social media data
np.random.seed(101)

n_posts = 2500

# Sample post templates
positive_posts = [
    "Love this product! #amazing",
    "Best purchase ever! Highly recommend",
    "Excellent service! Very satisfied",
    "Great quality! Will buy again",
    "Fantastic experience! Thank you",
    "Amazing product! Exceeded expectations",
    "Superb quality! Very happy",
    "Outstanding service! Five stars"
]

negative_posts = [
    "Terrible experience. Very disappointed",
    "Poor quality. Not worth the money",
    "Bad service. Would not recommend",
    "Worst purchase ever. Total waste",
    "Disappointed with the product",
    "Horrible experience. Never again",
    "Low quality. Not satisfied",
    "Terrible customer service"
]

neutral_posts = [
    "Product arrived today",
    "Checking out this new item",
    "Just received my order",
    "Testing this product",
    "Product as described",
    "Standard quality",
    "Average experience",
    "It's okay, nothing special"
]

# Generate timestamps
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 10, 20)
date_range = (end_date - start_date).days

timestamps = [start_date + timedelta(days=np.random.randint(0, date_range), 
                                      hours=np.random.randint(0, 24),
                                      minutes=np.random.randint(0, 60)) 
              for _ in range(n_posts)]

# Generate data
sentiments = np.random.choice(['Positive', 'Negative', 'Neutral'], n_posts, p=[0.55, 0.25, 0.20])
platforms = np.random.choice(['Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'TikTok'], n_posts)
post_types = np.random.choice(['Text', 'Image', 'Video', 'Link'], n_posts, p=[0.4, 0.3, 0.2, 0.1])

posts = []
for sentiment in sentiments:
    if sentiment == 'Positive':
        posts.append(np.random.choice(positive_posts))
    elif sentiment == 'Negative':
        posts.append(np.random.choice(negative_posts))
    else:
        posts.append(np.random.choice(neutral_posts))

data = {
    'post_id': range(1, n_posts + 1),
    'timestamp': timestamps,
    'platform': platforms,
    'post_type': post_types,
    'content': posts,
    'sentiment': sentiments,
    'likes': np.random.randint(0, 1000, n_posts),
    'shares': np.random.randint(0, 300, n_posts),
    'comments': np.random.randint(0, 150, n_posts),
    'reach': np.random.randint(100, 50000, n_posts),
    'impressions': np.random.randint(500, 100000, n_posts)
}

# Adjust engagement based on sentiment
for i in range(n_posts):
    if sentiments[i] == 'Positive':
        data['likes'][i] = int(data['likes'][i] * 1.5)
        data['shares'][i] = int(data['shares'][i] * 1.3)
    elif sentiments[i] == 'Negative':
        data['comments'][i] = int(data['comments'][i] * 1.4)  # Negative posts get more comments

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['engagement_rate'] = ((df['likes'] + df['shares'] + df['comments']) / df['reach'] * 100).round(2)

print("=" * 80)
print("SOCIAL MEDIA SENTIMENT ANALYSIS")
print("=" * 80)

# 1. Data Overview
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total Posts Analyzed: {len(df)}")
print(f"Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"\nFirst few records:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())

# 2. Sentiment Distribution
print("\n\n2. SENTIMENT DISTRIBUTION")
print("-" * 80)
sentiment_counts = df['sentiment'].value_counts()
sentiment_pct = (sentiment_counts / len(df) * 100).round(2)
sentiment_summary = pd.DataFrame({
    'Count': sentiment_counts,
    'Percentage': sentiment_pct
})
print(sentiment_summary)

# 3. Platform Analysis
print("\n\n3. PLATFORM ANALYSIS")
print("-" * 80)
platform_counts = df['platform'].value_counts()
print("Posts by Platform:")
print(platform_counts)

print("\nSentiment by Platform:")
platform_sentiment = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
print(platform_sentiment.round(2))

# 4. Content Type Analysis
print("\n\n4. CONTENT TYPE ANALYSIS")
print("-" * 80)
print("Posts by Type:")
print(df['post_type'].value_counts())

print("\nSentiment by Content Type:")
type_sentiment = pd.crosstab(df['post_type'], df['sentiment'], normalize='index') * 100
print(type_sentiment.round(2))

# 5. Engagement Metrics
print("\n\n5. ENGAGEMENT METRICS")
print("-" * 80)
print("Overall Engagement Statistics:")
engagement_stats = df[['likes', 'shares', 'comments', 'reach', 'impressions', 'engagement_rate']].describe()
print(engagement_stats.round(2))

print("\nEngagement by Sentiment:")
sentiment_engagement = df.groupby('sentiment')[['likes', 'shares', 'comments', 'engagement_rate']].mean().round(2)
print(sentiment_engagement)

# 6. Top Performing Posts
print("\n\n6. TOP PERFORMING POSTS")
print("-" * 80)
print("Top 10 Posts by Engagement Rate:")
top_posts = df.nlargest(10, 'engagement_rate')[['post_id', 'platform', 'sentiment', 'likes', 'shares', 'comments', 'engagement_rate']]
print(top_posts)

# 7. Time Series Analysis
print("\n\n7. TIME SERIES ANALYSIS")
print("-" * 80)
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

# Daily sentiment trends
daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
print("\nRecent Daily Sentiment Distribution:")
print(daily_sentiment.tail(10))

# Hourly patterns
print("\nPosts by Hour of Day:")
hourly_posts = df['hour'].value_counts().sort_index()
print(hourly_posts)

# Day of week patterns
print("\nPosts by Day of Week:")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_posts = df['day_of_week'].value_counts().reindex(day_order)
print(day_posts)

# 8. Sentiment Score
print("\n\n8. SENTIMENT SCORING")
print("-" * 80)
# Assign sentiment scores
sentiment_score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['sentiment_score'] = df['sentiment'].map(sentiment_score_map)

overall_sentiment_score = df['sentiment_score'].mean()
print(f"Overall Sentiment Score: {overall_sentiment_score:.3f}")
print(f"(Range: -1 to +1, where -1 is very negative and +1 is very positive)")

# Sentiment score by platform
platform_score = df.groupby('platform')['sentiment_score'].mean().sort_values(ascending=False)
print("\nSentiment Score by Platform:")
print(platform_score.round(3))

# 9. Hashtag Analysis
print("\n\n9. HASHTAG ANALYSIS")
print("-" * 80)
all_hashtags = []
for content in df['content']:
    hashtags = re.findall(r'#(\w+)', content)
    all_hashtags.extend(hashtags)

if all_hashtags:
    hashtag_counts = Counter(all_hashtags)
    print("Top Hashtags:")
    for tag, count in hashtag_counts.most_common(10):
        print(f"  #{tag}: {count} times")
else:
    print("No hashtags found in posts")

# 10. Reach and Impressions Analysis
print("\n\n10. REACH AND IMPRESSIONS ANALYSIS")
print("-" * 80)
print("Total Reach:", f"{df['reach'].sum():,}")
print("Total Impressions:", f"{df['impressions'].sum():,}")
print("Average Reach per Post:", f"{df['reach'].mean():,.0f}")
print("Average Impressions per Post:", f"{df['impressions'].mean():,.0f}")

print("\nReach by Sentiment:")
reach_by_sentiment = df.groupby('sentiment')['reach'].sum().sort_values(ascending=False)
print(reach_by_sentiment)

print("\nReach by Platform:")
reach_by_platform = df.groupby('platform')['reach'].sum().sort_values(ascending=False)
print(reach_by_platform)

# 11. Visualizations
print("\n\n11. GENERATING VISUALIZATIONS")
print("-" * 80)

fig = plt.figure(figsize=(18, 14))

# Plot 1: Sentiment Distribution
ax1 = plt.subplot(3, 4, 1)
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45, ha='right')

# Plot 2: Sentiment Pie Chart
ax2 = plt.subplot(3, 4, 2)
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightgray'], startangle=90)
plt.title('Sentiment Distribution (%)', fontsize=12, fontweight='bold')
plt.ylabel('')

# Plot 3: Platform Distribution
ax3 = plt.subplot(3, 4, 3)
platform_counts.plot(kind='bar', color='skyblue')
plt.title('Posts by Platform', fontsize=12, fontweight='bold')
plt.xlabel('Platform')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45, ha='right')

# Plot 4: Sentiment by Platform
ax4 = plt.subplot(3, 4, 4)
platform_sentiment.plot(kind='bar', stacked=True, color=['green', 'red', 'gray'], ax=ax4)
plt.title('Sentiment Distribution by Platform', fontsize=12, fontweight='bold')
plt.xlabel('Platform')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')

# Plot 5: Engagement Rate by Sentiment
ax5 = plt.subplot(3, 4, 5)
sentiment_engagement['engagement_rate'].plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Avg Engagement Rate by Sentiment', fontsize=12, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Engagement Rate (%)')
plt.xticks(rotation=45, ha='right')

# Plot 6: Likes by Sentiment
ax6 = plt.subplot(3, 4, 6)
df.boxplot(column='likes', by='sentiment', ax=ax6)
plt.title('Likes Distribution by Sentiment', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Sentiment')
plt.ylabel('Likes')

# Plot 7: Hourly Post Distribution
ax7 = plt.subplot(3, 4, 7)
hourly_posts.plot(kind='bar', color='orange')
plt.title('Posts by Hour of Day', fontsize=12, fontweight='bold')
plt.xlabel('Hour')
plt.ylabel('Number of Posts')
plt.xticks(rotation=0)

# Plot 8: Day of Week Distribution
ax8 = plt.subplot(3, 4, 8)
day_posts.plot(kind='bar', color='purple')
plt.title('Posts by Day of Week', fontsize=12, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45, ha='right')

# Plot 9: Content Type Distribution
ax9 = plt.subplot(3, 4, 9)
df['post_type'].value_counts().plot(kind='bar', color='teal')
plt.title('Posts by Content Type', fontsize=12, fontweight='bold')
plt.xlabel('Content Type')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45, ha='right')

# Plot 10: Reach by Platform
ax10 = plt.subplot(3, 4, 10)
reach_by_platform.plot(kind='barh', color='steelblue')
plt.title('Total Reach by Platform', fontsize=12, fontweight='bold')
plt.xlabel('Total Reach')
plt.ylabel('Platform')

# Plot 11: Sentiment Score by Platform
ax11 = plt.subplot(3, 4, 11)
platform_score.plot(kind='barh', color='coral')
plt.title('Sentiment Score by Platform', fontsize=12, fontweight='bold')
plt.xlabel('Sentiment Score')
plt.ylabel('Platform')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

# Plot 12: Daily Sentiment Trend
ax12 = plt.subplot(3, 4, 12)
# Get last 30 days for clearer visualization
recent_daily = daily_sentiment.tail(30)
recent_daily.plot(kind='area', stacked=True, alpha=0.7, color=['green', 'red', 'gray'], ax=ax12)
plt.title('Sentiment Trend (Last 30 Days)', fontsize=12, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment', loc='upper left')

plt.tight_layout()
plt.savefig('sentiment_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'sentiment_analysis_dashboard.png'")

# 12. Word Cloud Data Preparation
print("\n\n12. WORD FREQUENCY ANALYSIS")
print("-" * 80)
from collections import Counter
import string

# Extract words from posts
all_words = []
for content in df['content']:
    # Remove punctuation and convert to lowercase
    words = content.lower().translate(str.maketrans('', '', string.punctuation)).split()
    # Filter out common words
    stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'this', 'that']
    words = [w for w in words if w not in stopwords and len(w) > 2]
    all_words.extend(words)

word_counts = Counter(all_words)
print("Top 20 Most Common Words:")
for word, count in word_counts.most_common(20):
    print(f"  {word}: {count} times")

# Word analysis by sentiment
print("\nMost Common Words by Sentiment:")
for sentiment in ['Positive', 'Negative', 'Neutral']:
    sentiment_posts = df[df['sentiment'] == sentiment]['content']
    sentiment_words = []
    for content in sentiment_posts:
        words = content.lower().translate(str.maketrans('', '', string.punctuation)).split()
        words = [w for w in words if w not in stopwords and len(w) > 2]
        sentiment_words.extend(words)
    
    sentiment_word_counts = Counter(sentiment_words)
    print(f"\n{sentiment}:")
    for word, count in sentiment_word_counts.most_common(5):
        print(f"  {word}: {count} times")

# 13. Engagement Correlation Analysis
print("\n\n13. ENGAGEMENT CORRELATION ANALYSIS")
print("-" * 80)
engagement_metrics = df[['likes', 'shares', 'comments', 'reach', 'impressions', 'engagement_rate']]
correlation_matrix = engagement_metrics.corr()
print(correlation_matrix.round(3))

# 14. Best Posting Times
print("\n\n14. OPTIMAL POSTING TIMES")
print("-" * 80)
hourly_engagement = df.groupby('hour')['engagement_rate'].mean().sort_values(ascending=False)
print("Best Hours to Post (by Engagement Rate):")
print(hourly_engagement.head(5).round(2))

day_engagement = df.groupby('day_of_week')['engagement_rate'].mean().reindex(day_order).sort_values(ascending=False)
print("\nBest Days to Post (by Engagement Rate):")
print(day_engagement.round(2))

# 15. Key Insights
print("\n\n15. KEY INSIGHTS & RECOMMENDATIONS")
print("-" * 80)
print("\nKey Findings:")
print(f"1. Overall Sentiment: {overall_sentiment_score:.3f} ({sentiment_counts.index[0]} dominant)")
print(f"2. Total Posts Analyzed: {len(df):,}")
print(f"3. Total Reach: {df['reach'].sum():,}")
print(f"4. Average Engagement Rate: {df['engagement_rate'].mean():.2f}%")
print(f"5. Best Performing Platform: {platform_score.index[0]} (Score: {platform_score.iloc[0]:.3f})")
print(f"6. Most Engaging Content Type: {df.groupby('post_type')['engagement_rate'].mean().idxmax()}")

print("\nSentiment Insights:")
positive_pct = sentiment_pct['Positive']
negative_pct = sentiment_pct['Negative']
if positive_pct > 60:
    print("• Strong positive sentiment indicates good brand health")
elif positive_pct > 40:
    print("• Moderate positive sentiment with room for improvement")
else:
    print("• Low positive sentiment requires immediate attention")

if negative_pct > 30:
    print("• High negative sentiment needs urgent addressing")
elif negative_pct > 15:
    print("• Moderate negative sentiment should be monitored")
else:
    print("• Negative sentiment is within acceptable range")

print("\nMarketing Recommendations:")
print("• Focus content creation on high-engagement platforms")
print(f"• Post during peak hours: {hourly_engagement.head(3).index.tolist()}")
print(f"• Prioritize {df.groupby('post_type')['engagement_rate'].mean().idxmax()} content")
print("• Address negative sentiment through improved customer service")
print("• Engage more with positive posts to amplify reach")
print("• Monitor sentiment trends daily for quick response")
print("• Create content around top-performing keywords")
print("• Leverage positive sentiment for testimonials and marketing")

print("\nAction Items:")
print("• Respond to negative comments within 24 hours")
print("• Create crisis management protocol for sentiment drops")
print("• Implement sentiment-based alert system")
print("• A/B test content types for optimization")
print("• Schedule posts during high-engagement periods")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Save dataset
df.to_csv('social_media_sentiment_data.csv', index=False)
print("\n✓ Dataset saved as 'social_media_sentiment_data.csv'")
print("\nAll 5 projects completed successfully!")
print("Files generated:")
print("  1. ecommerce_sales_data.csv & ecommerce_analysis_dashboard.png")
print("  2. healthcare_patient_data.csv & healthcare_analysis_dashboard.png")
print("  3. sports_player_data.csv & sports_analysis_dashboard.png")
print("  4. telecom_churn_data.csv & churn_analysis_dashboard.png")
print("  5. social_media_sentiment_data.csv & sentiment_analysis_dashboard.png")