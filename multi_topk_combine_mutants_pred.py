import torch
import pandas as pd
from Bio import SeqIO
from transformers import AutoModelForMaskedLM, AutoTokenizer
from prosst.structure.get_sst_seq import SSTPredictor
import itertools

# --------------------------
# 配置参数
# --------------------------
fasta_file = "data/La_N.fasta"
pdb_file = "data/pred_La_N.pdb"

num_rounds = 3          # 最大突变轮数
top_k = 20              # 每轮选取 top-k 突变组合
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# 加载模型
# --------------------------
prosst_model = AutoModelForMaskedLM.from_pretrained(
    "AI4Protein/ProSST-2048", trust_remote_code=True
).to(device)
prosst_tokenizer = AutoTokenizer.from_pretrained(
    "AI4Protein/ProSST-2048", trust_remote_code=True
)

predictor = SSTPredictor(structure_vocab_size=2048, device=device)

# --------------------------
# 读取序列 & 结构
# --------------------------
residue_sequence = str(SeqIO.read(fasta_file, 'fasta').seq)
structure_sequence = predictor.predict_from_pdb(pdb_file)[0]['2048_sst_seq']
structure_sequence_offset = [i + 3 for i in structure_sequence]

# token 化
tokenized_res = prosst_tokenizer([residue_sequence], return_tensors='pt').to(device)
input_ids = tokenized_res['input_ids']
attention_mask = tokenized_res['attention_mask']
structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0).to(device)

# 预测 logits
with torch.no_grad():
    outputs = prosst_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=structure_input_ids
    )
logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze(0)  # [L, 20]

vocab = prosst_tokenizer.get_vocab()
aa_letters = list("ACDEFGHIKLMNPQRSTVWY")
aa_to_idx = {aa: vocab[aa] for aa in aa_letters}

# --------------------------
# 多轮迭代贪心选择
# --------------------------
sequence_length = len(residue_sequence)
current_combinations = [("", residue_sequence, 0.0)]  # (突变描述, 序列, 得分)

for round_idx in range(num_rounds):
    print(f"--- Round {round_idx+1} ---")
    next_combinations = []

    for desc, seq, score in current_combinations:
        # 所有单点突变
        for pos in range(sequence_length):
            wt = seq[pos]
            for mt in aa_letters:
                if mt == wt:
                    continue
                mutated_seq = seq[:pos] + mt + seq[pos+1:]
                mutated_desc = f"{desc}:{wt}{pos+1}{mt}" if desc else f"{wt}{pos+1}{mt}"
                # 计算得分
                pred = logits[pos, aa_to_idx[mt]] - logits[pos, aa_to_idx[wt]]
                new_score = score + pred.item()
                next_combinations.append((mutated_desc, mutated_seq, new_score))

    # 选 top-k 进入下一轮
    next_combinations.sort(key=lambda x: x[2], reverse=True)  # 得分越高越好
    current_combinations = next_combinations[:top_k]

# --------------------------
# 保存结果
# --------------------------
df = pd.DataFrame(current_combinations, columns=["mutant", "mutated_sequence", "pred_score"])
df.to_csv("multi_mutants_topk.csv", index=False)
print("Saved top-k multi-mutants to multi_mutants_topk.csv")

