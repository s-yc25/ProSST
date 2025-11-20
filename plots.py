import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取饱和突变预测结果
df = pd.read_csv("saturation_pred.csv")  # 包含 mutant 和 pred_score 列

# 画直方图查看得分分布
plt.figure(figsize=(8,5))
sns.histplot(df['pred_score'], bins=50, kde=True)
plt.xlabel("Predicted score (log probability difference)")
plt.ylabel("Frequency")
plt.title("Distribution of saturation mutagenesis scores")
plt.savefig("1.pdf")

# 可选：按序列位置画平均得分曲线
# 假设 mutant 格式为 "A10G" 或多突变 "A10G:C15T"
# 先拆分成单个突变
records = []
for idx, row in df.iterrows():
    for sub_mut in row['mutant'].split(":"):
        wt = sub_mut[0]
        pos = int(sub_mut[1:-1])
        mt = sub_mut[-1]
        records.append({'pos': pos, 'pred_score': row['pred_score']})

df_expanded = pd.DataFrame(records)
pos_avg = df_expanded.groupby('pos')['pred_score'].mean()

plt.figure(figsize=(12,4))
plt.plot(pos_avg.index, pos_avg.values, marker='o')
plt.xlabel("Residue position")
plt.ylabel("Average pred_score")
plt.title("Position-wise average mutation effect")
plt.grid(True)
plt.savefig("2.pdf")

