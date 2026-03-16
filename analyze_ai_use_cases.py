import pandas as pd
import os

def analyze_use_cases():
    file_path = 'ai_company_adoption.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    # 1. Global AI Use Case Distribution
    use_case_counts = df['ai_use_case'].value_counts()
    use_case_percentages = df['ai_use_case'].value_counts(normalize=True) * 100
    
    summary_global = pd.DataFrame({
        'Count': use_case_counts,
        'Percentage (%)': use_case_percentages
    })
    
    print("\nGlobal AI Use Case Distribution:")
    print(summary_global)
    
    # 2. Top Use Cases by Industry
    industry_use_pivot = df.groupby(['industry', 'ai_use_case']).size().unstack(fill_value=0)
    
    print("\nTop AI Use Cases by Industry:")
    industry_summary = []
    for industry in industry_use_pivot.index:
        top_cases = industry_use_pivot.loc[industry].sort_values(ascending=False).head(3)
        cases_str = ", ".join([f"{uc} ({count})" for uc, count in top_cases.items()])
        industry_summary.append({'Industry': industry, 'Top 3 Use Cases': cases_str})
    
    industry_df = pd.DataFrame(industry_summary)
    print(industry_df.to_string(index=False))

    # 3. Correlation with Productivity Change (Average by Use Case)
    avg_impact = df.groupby('ai_use_case')['productivity_change_percent'].mean().sort_values(ascending=False)
    print("\nAverage Productivity Increase by Use Case (%):")
    print(avg_impact)

    # Save to file
    with open('ai_use_case_report.txt', 'w', encoding='utf-8') as f:
        f.write("Global AI Use Case Distribution\n")
        f.write("==============================\n")
        f.write(summary_global.to_string())
        f.write("\n\nTop AI Use Cases by Industry\n")
        f.write("============================\n")
        f.write(industry_df.to_string(index=False))
        f.write("\n\nAverage Productivity Increase by Use Case\n")
        f.write("=========================================\n")
        f.write(avg_impact.to_string())

if __name__ == "__main__":
    analyze_use_cases()
