import sys
from Bio import SeqIO
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForMaskedLM, AutoTokenizer
from prosst.structure.get_sst_seq import SSTPredictor

# -------------------------------
# 参数
# -------------------------------
fasta_file = "data/La_N.fasta"
pdb_file = "data/pred_La_N.pdb"
top_k = 5  # 每轮保留 top-k 突变组合
num_rounds = 3  # 三轮突变

# -------------------------------
# 载入 ProSST
# -------------------------------
prosst_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
prosst_tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
predictor = SSTPredictor(structure_vocab_size=2048)

# -------------------------------
# 读取序列和结构
# -------------------------------
residue_sequence = str(SeqIO.read(fasta_file, "fasta").seq)
seq_len = len(residue_sequence)
structure_sequence = predictor.predict_from_pdb(pdb_file)[0]['2048_sst_seq']
structure_sequence_offset = [i + 3 for i in structure_sequence]

tokenized_res = prosst_tokenizer([residue_sequence], return_tensors='pt')
input_ids = tokenized_res['input_ids']
attention_mask = tokenized_res['attention_mask']
structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    outputs = prosst_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=structure_input_ids
    )
logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()
vocab = prosst_tokenizer.get_vocab()
aa_letters = sorted(vocab.keys(), key=lambda x: vocab[x])

# -------------------------------
# 初始化候选突变：每个位点所有氨基酸（排除 WT）
# -------------------------------
candidates = []
for pos, wt in enumerate(residue_sequence):
    for aa in aa_letters:
        if aa != wt:
            candidates.append([(pos, aa)])

# -------------------------------
# 三轮 top-k 搜索
# -------------------------------
for round_idx in range(num_rounds):
    print(f"=== Round {round_idx + 1} ===")
    scores = []

    # 计算每个候选组合的总分
    for muts in candidates:
        score = sum(logits[pos, vocab[aa]] - logits[pos, vocab[residue_sequence[pos]]] for pos, aa in muts)
        scores.append(score)

    # 按总分排序，取 top-k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_candidates = [candidates[i] for i in top_indices]
    top_scores = [scores[i].item() for i in top_indices]

    # -------------------------------
    # 构建热图数据：行 Top-k 组合，列 蛋白质序列位置
    # -------------------------------
    heat_data = []
    annotations = []
    for muts in top_candidates:
        row = [0] * seq_len
        row_annot = [""] * seq_len
        for pos, aa in muts:
            row[pos] = logits[pos, vocab[aa]].item() - logits[pos, vocab[residue_sequence[pos]]].item()
            row_annot[pos] = aa
        heat_data.append(row)
        annotations.append(row_annot)

    # 按总分从高到低排序
    sorted_indices = sorted(range(len(top_scores)), key=lambda i: top_scores[i], reverse=True)
    heat_data = [heat_data[i] for i in sorted_indices]
    annotations = [annotations[i] for i in sorted_indices]
    top_scores = [top_scores[i] for i in sorted_indices]

    # -------------------------------
    # 绘制热图
    # -------------------------------
    plt.figure(figsize=(seq_len/3, top_k))
    sns.heatmap(heat_data, annot=annotations, fmt="", cmap="RdBu_r", center=0,
                cbar_kws={'label':'Score'}, xticklabels=range(1, seq_len+1))
    plt.title(f"Top-{top_k} mutation combinations Round {round_idx + 1}")
    plt.xlabel("Protein sequence position")
    plt.ylabel("Top-k combinations (sorted by score)")
    plt.tight_layout()
    plt.savefig("topk_v3.pdf")

    # -------------------------------
    # 构建下一轮候选：在 top-k 基础上，每个组合再增加一个新的突变（不同于已有突变位点）
    # -------------------------------
    new_candidates = []
    for muts in top_candidates:
        used_pos = {pos for pos, _ in muts}
        for pos, wt in enumerate(residue_sequence):
            if pos in used_pos:
                continue
            for aa in aa_letters:
                if aa != wt:
                    new_candidates.append(muts + [(pos, aa)])
    candidates = new_candidates

