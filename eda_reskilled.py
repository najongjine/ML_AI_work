import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'Malgun Gothic' # For Korean characters if needed, or default
plt.rcParams['axes.unicode_minus'] = False

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def run_eda():
    print("Loading dataset...")
    df = pd.read_csv('ai_company_adoption.csv')
    
    # 1. Basic statistics for reskilled_employees
    reskilled_stats = df['reskilled_employees'].describe()
    print("\n--- Basic Statistics for Reskilled Employees ---")
    print(reskilled_stats)
    
    # 2. Reskilled Employees by Industry
    plt.figure(figsize=(12, 6))
    industry_reskilled = df.groupby('industry')['reskilled_employees'].mean().sort_values(ascending=False)
    sns.barplot(x=industry_reskilled.values, y=industry_reskilled.index)
    plt.title('Average Reskilled Employees by Industry')
    plt.xlabel('Average Reskilled Employees')
    plt.tight_layout()
    plt.savefig('plots/reskilled_by_industry.png')
    
    # 3. Reskilled Employees by Company Size
    plt.figure(figsize=(10, 5))
    size_reskilled = df.groupby('company_size')['reskilled_employees'].mean().sort_values(ascending=False)
    sns.barplot(x=size_reskilled.index, y=size_reskilled.values)
    plt.title('Average Reskilled Employees by Company Size')
    plt.ylabel('Average Reskilled Employees')
    plt.tight_layout()
    plt.savefig('plots/reskilled_by_size.png')
    
    # 4. Correlation Analysis
    # We want to see how reskilling relates to AI Maturity, Job Displacement, and Revenue Growth
    cols_to_corr = ['reskilled_employees', 'jobs_displaced', 'ai_maturity_score', 'revenue_growth_percent', 'ai_adoption_rate']
    corr_matrix = df[cols_to_corr].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap: Reskilling and Key Metrics')
    plt.tight_layout()
    plt.savefig('plots/reskilled_correlation.png')
    
    # 5. Reskilled vs Jobs Displaced (Scatter plot with sample)
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(1000) # Sample for better visualization in scatter
    sns.regplot(data=sample_df, x='jobs_displaced', y='reskilled_employees', scatter_kws={'alpha':0.3})
    plt.title('Jobs Displaced vs. Reskilled Employees (Sample of 1000)')
    plt.tight_layout()
    plt.savefig('plots/displaced_vs_reskilled.png')

    # 6. Industry Summary Table
    industry_summary = df.groupby('industry').agg({
        'reskilled_employees': 'mean',
        'jobs_displaced': 'mean',
        'jobs_created': 'mean',
        'revenue_growth_percent': 'mean',
        'ai_investment_per_employee': 'mean',
        'ai_budget_percentage': 'mean',
        'ai_training_hours': 'mean'
    }).sort_values(by='reskilled_employees', ascending=False)
    
    industry_summary.to_csv('reskilled_industry_summary.csv')
    print("\n--- Industry Summary (Top 5 by Reskilling) ---")
    print(industry_summary.head())

    # 7. AI Investment vs Reskilling
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(1000)
    sns.scatterplot(data=sample_df, x='ai_investment_per_employee', y='reskilled_employees', alpha=0.5)
    plt.title('AI Investment per Employee vs. Reskilled Employees (Sample 1000)')
    plt.tight_layout()
    plt.savefig('plots/investment_vs_reskilled.png')

    # 8. Budget Percentage Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ai_budget_percentage'], bins=30, kde=True)
    plt.title('Distribution of AI Budget Percentage')
    plt.xlabel('AI Budget Percentage (%)')
    plt.tight_layout()
    plt.savefig('plots/budget_distribution.png')

    # 9. AI Training Hours Analysis
    print("\n--- Basic Statistics for AI Training Hours ---")
    print(df['ai_training_hours'].describe())

    plt.figure(figsize=(12, 6))
    industry_training = df.groupby('industry')['ai_training_hours'].mean().sort_values(ascending=False)
    sns.barplot(x=industry_training.values, y=industry_training.index, palette='viridis')
    plt.title('Average Annual AI Training Hours per Employee by Industry')
    plt.xlabel('Average Training Hours')
    plt.tight_layout()
    plt.savefig('plots/training_hours_by_industry.png')

    # 10. Training Hours Correlation
    plt.figure(figsize=(10, 6))
    training_corr_cols = ['ai_training_hours', 'reskilled_employees', 'innovation_score', 'ai_maturity_score']
    sns.heatmap(df[training_corr_cols].corr(), annot=True, cmap='YlGnBu')
    plt.title('Correlation: Training Hours, Reskilling, and Innovation')
    plt.tight_layout()
    plt.savefig('plots/training_correlation.png')

    # 11. Total Investment Estimate (Simplified)
    df['estimated_total_ai_investment'] = df['ai_investment_per_employee'] * df['num_employees']
    total_investment_stats = df['estimated_total_ai_investment'].describe()
    print("\n--- Estimated Total AI Investment Stats (USD) ---")
    print(total_investment_stats)

    print("\nEDA completed. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    run_eda()
