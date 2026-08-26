"""PROSE-FD SWE multi-GPU run (data-parallel, jit, no-checkpoint, 1 sample/replica)."""
class Cfg:
    Nx=128; T_in=10; Nt=11; T_final=1.0; domain_L=1.0
    N_p=64; n_modes=32; max_order=32
    d_geom=224; geo_layers=5; expert_hidden=192; n_channels=6; desc_dim=6
    experts=("convective","diffusive","reaction","elliptic","shock")
    lambda_gate=1e-3
    dataset="shallow_water"; n_total=900; t_num=20; t_step=5
    steps=6000; batch=1; lr=3e-4         # batch = per-replica (1 sample → fits 1 H100, no ckpt)
    jit_compile=True; grad_checkpoint=False
    save_dir="results/prose_swe_mgpu"; ckpt_every=1000; seed=0
