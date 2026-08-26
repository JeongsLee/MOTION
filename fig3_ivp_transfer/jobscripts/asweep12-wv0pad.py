import io
io.open("/eu/code_mirror/code/train/configs/poseidon_finetune_combo_158m_wv0pad_sweep.py","w",encoding="utf-8").write('"""Anchor sweep of the DEFINITIVE wave arm (wv0pad: 2IC + pad + i>=0): checks the anchor-0 hole is\nclosed and gives the swept closed-benchmark row. EVAL-ONLY."""\nfrom .poseidon_finetune_combo_158m_wv0_pad import Cfg as _Base\n\n\nclass Cfg(_Base):\n    eval_anchor_sweep = (0, 1, 2, 4, 8)\n    eval_lead_cap = 12\n')
print("CFG_OK")
