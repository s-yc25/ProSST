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
top_k = 5          # 每轮保留 top-k 突变组合（可自行调整）
num_rounds = 3      # 总共 3 轮突变

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
# 初始化：所有单点突变
# -------------------------------
candidates = []
for pos, wt in enumerate(residue_sequence):
    for aa in aa_letters:
        if aa != wt:
            candidates.append([(pos, aa)])

print(f"初始单点突变数量: {len(candidates)}")


# -------------------------------
# 多轮 top-k 搜索
# -------------------------------
for round_idx in range(num_rounds):
    print(f"\n==============================")
    print(f"         Round {round_idx + 1}")
    print(f"==============================")

    scores = []

    # 计算每个候选组合的总分
    for muts in candidates:
        score = sum(
            logits[pos, vocab[aa]] - logits[pos, vocab[residue_sequence[pos]]]
            for pos, aa in muts
        )
        scores.append(score)

    # 取 top-k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_candidates = [candidates[i] for i in top_indices]
    top_scores = [scores[i].item() for i in top_indices]

    # -------------------------------
    # 打印本轮 top-k 结果
    # -------------------------------
    print(f"本轮候选数量: {len(candidates)}")
    print(f"选出的 Top-{top_k} 组合:")

    for rank, (muts, sc) in enumerate(zip(top_candidates, top_scores), start=1):
        mut_desc = ", ".join([f"{pos+1}{residue_sequence[pos]}→{aa}" for pos, aa in muts])
        print(f"  #{rank}: score = {sc:.4f} | {mut_desc}")

    # -------------------------------
    # 构建下一轮候选：top-k 每个组合扩展新的突变
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


