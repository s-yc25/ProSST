import torch
import pandas as pd
from Bio import SeqIO
from transformers import AutoModelForMaskedLM, AutoTokenizer
from prosst.structure.get_sst_seq import SSTPredictor
import itertools

# -------------------------------
# 初始化模型
# -------------------------------
prosst_model = AutoModelForMaskedLM.from_pretrained(
    "AI4Protein/ProSST-2048", trust_remote_code=True
)
prosst_tokenizer = AutoTokenizer.from_pretrained(
    "AI4Protein/ProSST-2048", trust_remote_code=True
)
predictor = SSTPredictor(structure_vocab_size=2048)

# -------------------------------
# 输入蛋白序列
# -------------------------------
residue_sequence = str(SeqIO.read('data/La_N.fasta', 'fasta').seq)
seq_len = len(residue_sequence)

# 预测结构序列
structure_sequence = predictor.predict_from_pdb("data/pred_La_N.pdb")[0]['2048_sst_seq']
structure_sequence_offset = [i + 3 for i in structure_sequence]

# Tokenize
tokenized_res = prosst_tokenizer([residue_sequence], return_tensors='pt')
input_ids = tokenized_res['input_ids']
attention_mask = tokenized_res['attention_mask']
structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

# 获取 logits
with torch.no_grad():
    outputs = prosst_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=structure_input_ids
    )
logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()
vocab = prosst_tokenizer.get_vocab()

# -------------------------------
# 多突变生成
# -------------------------------
aa_list = list("ACDEFGHIKLMNPQRSTVWY")
mutant_results = []

max_mutations = 3  # 可以改成 2 或 3，控制组合数量

# 生成所有位置组合
positions = list(range(seq_len))
for num_mut in range(1, max_mutations+1):
    for pos_combo in itertools.combinations(positions, num_mut):
        # 对每个组合生成所有氨基酸替换
        aa_options = [aa_list for _ in pos_combo]
        for muts in itertools.product(*aa_options):
            # 跳过未突变的情况
            if all(residue_sequence[pos] == muts[i] for i, pos in enumerate(pos_combo)):
                continue
            # 构建突变字符串，例如 A10G:C15T
            mutant_str = ":".join(f"{residue_sequence[pos]}{pos+1}{muts[i]}" for i, pos in enumerate(pos_combo))
            # 构建突变序列
            mutated_seq = list(residue_sequence)
            for i, pos in enumerate(pos_combo):
                mutated_seq[pos] = muts[i]
            mutated_seq = "".join(mutated_seq)
            # 计算突变得分
            score = sum(logits[pos, vocab[muts[i]]] - logits[pos, vocab[residue_sequence[pos]]]
                        for i, pos in enumerate(pos_combo))
            mutant_results.append((mutant_str, mutated_seq, score.item()))

# -------------------------------
# 保存结果
# -------------------------------
df = pd.DataFrame(mutant_results, columns=["mutant", "mutated_sequence", "pred_score"])
df.to_csv("data/multi_saturation_mutants_pred.csv", index=False)
print("Prediction complete. Saved to data/multi_saturation_mutants_pred.csv")

