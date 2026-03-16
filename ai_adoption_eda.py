import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic' # For Korean support on Windows
plt.rcParams['axes.unicode_minus'] = False

def run_eda(file_path):
    print(f"--- Loading data from {file_path} ---")
    df = pd.read_csv(file_path)
    
    # 1. Basic Cleaning and Feature Engineering
    # Calculate Net Job Change
    df['net_job_change'] = df['jobs_created'] - df['jobs_displaced']
    
    # Check for missing values
    missing = df[['jobs_displaced', 'jobs_created']].isnull().sum()
    print("\nMissing values in job columns:")
    print(missing)
    
    # 2. Descriptive Statistics
    stats = df[['jobs_displaced', 'jobs_created', 'net_job_change']].describe()
    print("\nDescriptive Statistics for Employment Change:")
    print(stats)
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # 3. Visualizations
    
    # A. Distribution of Jobs Displaced vs Created
    plt.figure(figsize=(12, 6))
    sns.histplot(df['jobs_displaced'], color='red', label='Jobs Displaced', kde=True, alpha=0.5)
    sns.histplot(df['jobs_created'], color='blue', label='Jobs Created', kde=True, alpha=0.5)
    plt.title('Distribution of Jobs Displaced and Created')
    plt.xlabel('Number of Jobs')
    plt.legend()
    plt.savefig('plots/job_distribution.png')
    plt.close()
    
    # B. Net Job Change by Industry (Top 10)
    plt.figure(figsize=(14, 8))
    industry_impact = df.groupby('industry')['net_job_change'].mean().sort_values()
    sns.barplot(x=industry_impact.values, y=industry_impact.index, palette='viridis')
    plt.title('Average Net Job Change by Industry')
    plt.xlabel('Average Net Change (Created - Displaced)')
    plt.tight_layout()
    plt.savefig('plots/industry_job_impact.png')
    plt.close()
    
    # C. Jobs Displaced vs Created by Industry
    industry_counts = df.groupby('industry')[['jobs_displaced', 'jobs_created']].mean().sort_values(by='jobs_displaced', ascending=False)
    industry_counts.plot(kind='bar', figsize=(15, 7), color=['lightcoral', 'cornflowerblue'])
    plt.title('Average Jobs Displaced vs Created by Industry')
    plt.ylabel('Average Number of Jobs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/industry_comparison.png')
    plt.close()

    # D. Impact by Company Size
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='company_size', y='net_job_change', data=df)
    plt.title('Net Job Change Distribution by Company Size')
    plt.savefig('plots/company_size_impact.png')
    plt.close()
    
    # E. Correlation Analysis
    cols_to_corr = [
        'jobs_displaced', 'jobs_created', 'net_job_change', 
        'ai_adoption_rate', 'task_automation_rate', 
        'productivity_change_percent', 'revenue_growth_percent'
    ]
    # Filter only numeric columns that exist
    numeric_cols = [col for col in cols_to_corr if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap: AI Adoption and Workforce Impact')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    print("\nVisualizations saved in the 'plots/' directory.")
    
    # 4. Summary Findings (Example)
    top_displacing = df.groupby('industry')['jobs_displaced'].mean().idxmax()
    top_creating = df.groupby('industry')['jobs_created'].mean().idxmax()
    net_positive = (df['net_job_change'] > 0).sum() / len(df) * 100
    
    print("\n--- Key Findings ---")
    print(f"Industry with highest average job displacement: {top_displacing}")
    print(f"Industry with highest average job creation: {top_creating}")
    print(f"Percentage of companies with overall positive net job change: {net_positive:.1f}%")

if __name__ == "__main__":
    csv_file = "ai_company_adoption.csv"
    if os.path.exists(csv_file):
        run_eda(csv_file)
    else:
        print(f"Error: {csv_file} not found.")
