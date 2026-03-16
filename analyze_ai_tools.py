import pandas as pd
import os

def analyze():
    file_path = 'ai_company_adoption.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    # 1. Global AI Primary Tool Distribution
    tool_counts = df['ai_primary_tool'].value_counts()
    tool_percentages = df['ai_primary_tool'].value_counts(normalize=True) * 100
    
    summary_global = pd.DataFrame({
        'Count': tool_counts,
        'Percentage (%)': tool_percentages
    })
    
    print("\nGlobal AI Primary Tool Distribution:")
    print(summary_global)
    
    # 2. AI Tool by Industry
    # Top 3 tools per industry
    industry_tool_pivot = df.groupby(['industry', 'ai_primary_tool']).size().unstack(fill_value=0)
    
    print("\nTop AI Tools by Industry:")
    industry_summary = []
    for industry in industry_tool_pivot.index:
        top_tools = industry_tool_pivot.loc[industry].sort_values(ascending=False).head(3)
        tools_str = ", ".join([f"{tool} ({count})" for tool, count in top_tools.items()])
        industry_summary.append({'Industry': industry, 'Top 3 Tools': tools_str})
    
    industry_df = pd.DataFrame(industry_summary)
    print(industry_df.to_string(index=False))

    # Save to file for easy reading
    with open('ai_tool_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("Global AI Primary Tool Distribution\n")
        f.write("==================================\n")
        f.write(summary_global.to_string())
        f.write("\n\nTop AI Tools by Industry\n")
        f.write("========================\n")
        f.write(industry_df.to_string(index=False))

if __name__ == "__main__":
    analyze()
