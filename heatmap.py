import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取预测结果
df = pd.read_csv("saturation_pred.csv")  # mutant, pred_score

# 20 种氨基酸
aa_list = list("ACDEFGHIKLMNPQRSTVWY")

# 展开每个突变到单个位置
records = []
for idx, row in df.iterrows():
    for sub_mut in row['mutant'].split(":"):
        wt = sub_mut[0]
        pos = int(sub_mut[1:-1])
        mt = sub_mut[-1]
        records.append({'pos': pos, 'mut': mt, 'pred_score': row['pred_score']})

df_expanded = pd.DataFrame(records)

# 构建热图矩阵
positions = sorted(df_expanded['pos'].unique())
heatmap_matrix = pd.DataFrame(index=positions, columns=aa_list, dtype=float)

for _, row in df_expanded.iterrows():
    heatmap_matrix.at[row['pos'], row['mut']] = row['pred_score']

# 如果有缺失的突变分数，可以填充 0 或 NaN
heatmap_matrix = heatmap_matrix.fillna(0)

# 画热图
plt.figure(figsize=(20,8))
sns.heatmap(
    heatmap_matrix,
    cmap='RdBu_r',  # 红负蓝正，颜色越深表示突变越有害
    center=0,
    cbar_kws={'label': 'Predicted score'}
)
plt.xlabel("Mutated Amino Acid")
plt.ylabel("Residue Position")
plt.title("Saturation Mutagenesis Prediction Heatmap")
plt.savefig("3.pdf")

