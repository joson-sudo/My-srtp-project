import pandas as pd

df = pd.read_csv("data/sample_data.csv")

print("--- 表格前 5 行数据 ---")
print(df.head())

print("\n--- 表格的统计信息 ---")
print(df.describe())
