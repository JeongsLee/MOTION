# MOTION pretrained weights

Weights are hosted on Zenodo (CC-BY-4.0): **DOI: [10.5281/zenodo.22104636](https://doi.org/10.5281/zenodo.22104636)**.
Each archive is a plain `tar` of one folder below; extract into `weights/` at the repository root.
Per-file SHA-256 sums: `weights/SHA256SUMS.release` in the Zenodo record (`SHA256SUMS.npz`)
and per-archive sums in `SHA256SUMS.tar`.

All headline numbers are class-averaged relative L2 in physical (denormalized) space.

| archive | model | arch | params | headline | used in |
|---|---|---|---|---|---|
| `MOTION_weights_motion_156.9m.tar` | MOTION (stage-2 fluctuation) | d896 / depth12 / d_w48 | 156.9M | 7.03% (19-family val) | Fig. 1, Fig. 4 |
| `MOTION_weights_motion_s_20.7m.tar` | MOTION-S (stage-2 fluctuation) | d384 / depth8 / d_w16 | 20.7M | 8.22% (19-family val) | Fig. 1 |
| `MOTION_weights_bylfa_114m.tar` | 6-family joint-benchmark model | — | 114M | 2.774% (6-family) | Fig. 2 |
| `MOTION_weights_ivp_161.9m_pretrain.tar` | IVP-mode pretrain on the Poseidon corpus (800k steps) | combo_158m | 161.9M | median rel-L1 12.30% | Fig. 3 |
| `MOTION_weights_ivp_161.9m_finetuned.tar` | six per-task N=64 few-shot finetuned arms | 〃 | 〃 | NS-PwC AR 3.30 / Poisson 6.25 / Wave 11.89–31.27 | Fig. 3 |

`motion_s_20.7m/m1s2_figrender/` is a stage-4 branch used only for early figure renders; it is
not the protocol model.

Checkpoints are TF 2.15 named-array `.npz` files and are loaded positionally by the training
scripts — use the same TF major version. Score each checkpoint in the numeric mode it was trained
in (bf16); see the reproduction notes in `README.md`.
