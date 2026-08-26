"""Tiny local 1-GPU validation of the MirroredStrategy training path (real SWE 128²)."""
class Cfg:
    Nx=128; T_in=10; Nt=11; T_final=1.0; domain_L=1.0
    N_p=8; n_modes=4; max_order=4
    d_geom=16; geo_layers=2; expert_hidden=16; n_channels=6; desc_dim=6
    experts=("convective","diffusive","reaction","elliptic","shock")
    lambda_gate=1e-3
    dataset="shallow_water"; n_total=40; t_num=20; t_step=5
    steps=20; batch=1; lr=1e-3; jit_compile=True; grad_checkpoint=False; seed=0
    save_dir=None
