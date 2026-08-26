"""Family conditioner — the gate α_k(family) over operator experts (DESIGN.md §4.3, §6.1).

v1 uses option (a): a learned map from a *structured PDE descriptor* (one-hot family +
normalised coefficients) to per-expert gate weights. This is deliberately the simplest
gate; §6.1 escalates to symbolic-operator-token (b) and in-context (c) later.

The gate is the headline interpretability object: after training, α_k per family should
recover the known operator composition (Heat ≈ diffusive; Burgers ≈ advective+small
diffusive; ADR ≈ advective+diffusive+reaction). See DESIGN.md §7.
"""
from __future__ import annotations
import tensorflow as tf


class FamilyGate(tf.Module):
    """Maps a PDE descriptor → softplus gate weights, one per expert (global scalar gate)."""

    def __init__(self, n_experts, hidden=32, name="family_gate"):
        super().__init__(name=name)
        self.n_experts = int(n_experts)
        self.h = tf.keras.layers.Dense(hidden, activation="gelu", name="g_h")
        # bias-init slightly positive so every expert is mildly active at start.
        self.out = tf.keras.layers.Dense(n_experts, name="g_out",
                                         bias_initializer=tf.keras.initializers.Constant(0.5))

    def __call__(self, descriptor):
        """descriptor: (B, d_desc). Returns α: (B, n_experts), non-negative."""
        a = self.out(self.h(descriptor))
        return tf.nn.softplus(a)                              # non-negative, unbounded

    @property
    def trainable_variables(self):
        return list(self.h.trainable_variables) + list(self.out.trainable_variables)


class FieldGate(tf.Module):
    """Field-conditioned operator gate (DESIGN.md §6.1 option c). α_k from REGIME-DISCRIMINATIVE
    statistics of the observed field — NOT the equation label. Rationale:
      • regime-aware: a high-Re vs low-Re flow look different (sharper gradients, finer scales), so α
        varies per-sample with the observed regime (the descriptor gate gave one constant α/family).
      • transferable: hand-fed dimensionless numbers (Re, Ma, Da…) are different/undefined for an
        unseen equation at fine-tune; a field always exists, so a field-conditioned gate routes any
        new PDE. (Poseidon takes no equation params for exactly this reason.)
    v1 fed GLOBAL-POOLED h_geom and the gate COLLAPSED to a constant α (std≈0) — spatial averaging
    washes out the regime signal (regime lives in spatial STRUCTURE, not feature means). v2 feeds
    explicit per-channel field statistics [mean, std, mean|∇u|, mean|Δu|] computed in model.operator
    — these directly encode amplitude + length-scale (|∇u|/std ∝ inverse scale ∝ regime), are cheap,
    equation-agnostic, and don't collapse. descriptor kept as a weak dropout-prior → field-primary,
    label-usable, and works at desc=0 transfer. Interpretability: correlate learned α with Re/Ma/Da."""

    def __init__(self, n_experts, hidden=64, desc_dropout=0.2, name="field_gate"):
        super().__init__(name=name)
        self.n_experts = int(n_experts)
        self.desc_dropout = float(desc_dropout)
        # LayerNorm the regime stats: their scale drifts during training; normalising the gate input
        # keeps α dependent on the relative regime signature, not the drifting magnitude (also avoids
        # the runaway feedback bigger-stats→bigger-α→bigger-W→NaN seen with raw features).
        self.ln = tf.keras.layers.LayerNormalization(name="fg_ln")
        self.h1 = tf.keras.layers.Dense(hidden, activation="gelu", name="fg_h1")
        self.h2 = tf.keras.layers.Dense(hidden, activation="gelu", name="fg_h2")
        self.out = tf.keras.layers.Dense(n_experts, name="fg_out",
                                         bias_initializer=tf.keras.initializers.Constant(0.5))

    def __call__(self, feat, descriptor, training=False):
        """feat: (B, F) precomputed regime statistics (model._regime_features)."""
        x = self.ln(tf.cast(feat, tf.float32))
        d = tf.cast(descriptor, tf.float32)
        if training and self.desc_dropout > 0.0:             # randomly hide the label → field-primary
            keep = tf.cast(tf.random.uniform([tf.shape(d)[0], 1]) >= self.desc_dropout, tf.float32)
            d = d * keep
        x = tf.concat([x, d], axis=-1)
        a = self.out(self.h2(self.h1(x)))
        return tf.nn.softplus(a)                              # non-negative, unbounded

    @property
    def trainable_variables(self):
        return (list(self.ln.trainable_variables) + list(self.h1.trainable_variables)
                + list(self.h2.trainable_variables) + list(self.out.trainable_variables))
