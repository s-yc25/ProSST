from transformers import AutoModelForMaskedLM, AutoTokenizer
from prosst.structure.get_sst_seq import SSTPredictor
from Bio import SeqIO
import torch
import pandas as pd

# 加载模型
prosst_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
prosst_tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
predictor = SSTPredictor(structure_vocab_size=2048)

# 输入蛋白序列和结构
residue_sequence = str(SeqIO.read('data/La_N.fasta', 'fasta').seq)
structure_sequence = predictor.predict_from_pdb("data/pred_La_N.pdb")[0]['2048_sst_seq']
structure_sequence_offset = [i + 3 for i in structure_sequence]

# Tokenize
tokenized_res = prosst_tokenizer([residue_sequence], return_tensors='pt')
input_ids = tokenized_res['input_ids']
attention_mask = tokenized_res['attention_mask']
structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

# 获取模型输出
with torch.no_grad():
    outputs = prosst_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=structure_input_ids
    )
logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()  # 去掉CLS/SEP

# 生成饱和突变列表
aa_list = list("ACDEFGHIKLMNPQRSTVWY")
saturation_mutants = []
for i, wt in enumerate(residue_sequence):
    for mt in aa_list:
        if mt != wt:
            saturation_mutants.append(f"{wt}{i+1}{mt}")

# 计算每个突变的预测分数
vocab = prosst_tokenizer.get_vocab()
pred_scores = []
for mutant in saturation_mutants:
    wt = mutant[0]
    idx = int(mutant[1:-1]) - 1
    mt = mutant[-1]
    score = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
    pred_scores.append((mutant, score.item()))

# 保存到 CSV
df_saturation = pd.DataFrame(pred_scores, columns=['mutant', 'pred_score'])
df_saturation.to_csv("saturation_pred.csv", index=False)
print("Saturation mutagenesis predictions saved to saturation_pred.csv")

