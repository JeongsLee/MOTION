"""Phase-1 Task-1 BEST recipe (unified-lup MASKING) scaled to ~152M (geo_layers 9->13, same scale as the
dsup flagship but NO dummy-supervision). Tests: does scaling the best recipe help (vs 114M masking
10.49/6.35), and does masking beat dsup at the SAME 152M scale (dsup flagship 10.88/6.64)?"""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    geo_layers = 13               # 9 -> 13 encoder depth: ~152M (matches flagship scale)
    save_dir = "results/prose_150M_fluid5_6x_unified_155m_mask"
