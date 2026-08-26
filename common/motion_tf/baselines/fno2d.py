"""Compact 2-D FNO baseline for ACE (Phase-1 reference operator).

Task (identical to our model's): map u0 (B,128,128) → full Nt-frame trajectory
(B,Nt,128,128). Output frames are produced as Nt output channels of a standard 2-D FNO
(FFT2D, low-freq corner modes, pointwise skip). This is the from-scratch FNO reference
against which Accuracy/Efficiency Gain is measured (Poseidon Table-1 convention).
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf


class SpectralConv2d(tf.keras.layers.Layer):
    """FFT2D spectral conv: keep 2 low-freq corner blocks (kh × kw), complex-mul, iFFT."""
    def __init__(self, in_ch, out_ch, modes_h, modes_w, name=None):
        super().__init__(name=name)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kh, self.kw = modes_h, modes_w
        scale = 1.0 / (in_ch * out_ch)
        init = tf.keras.initializers.RandomNormal(stddev=scale ** 0.5)
        for i in (1, 2):
            setattr(self, f"W{i}_re", self.add_weight(
                name=f"W{i}_re", shape=(in_ch, out_ch, modes_h, modes_w), initializer=init))
            setattr(self, f"W{i}_im", self.add_weight(
                name=f"W{i}_im", shape=(in_ch, out_ch, modes_h, modes_w), initializer=init))

    def _cmul(self, x, W_re, W_im):
        return tf.einsum("bihw,iohw->bohw", x, tf.complex(W_re, W_im))

    def call(self, x):                              # x: (B,H,W,C)
        B = tf.shape(x)[0]; H = tf.shape(x)[1]; Wd = tf.shape(x)[2]
        kh, kw = self.kh, self.kw
        xp = tf.transpose(x, [0, 3, 1, 2])          # (B,C,H,W)
        xh = tf.signal.fft2d(tf.cast(xp, tf.complex64))
        c1 = self._cmul(xh[:, :, :kh, :kw], self.W1_re, self.W1_im)   # top-left
        c2 = self._cmul(xh[:, :, -kh:, :kw], self.W2_re, self.W2_im)  # bottom-left
        zW = tf.zeros([B, self.out_ch, kh, Wd - kw], tf.complex64)
        row1 = tf.concat([c1, zW], axis=-1)
        row2 = tf.concat([c2, zW], axis=-1)
        zmid = tf.zeros([B, self.out_ch, H - 2 * kh, Wd], tf.complex64)
        out_h = tf.concat([row1, zmid, row2], axis=2)
        out = tf.math.real(tf.signal.ifft2d(out_h))                  # (B,out,H,W)
        return tf.transpose(out, [0, 2, 3, 1])


class FNO2D(tf.Module):
    def __init__(self, cfg, name="fno2d"):
        super().__init__(name=name)
        self.cfg = cfg
        w = cfg.fno_width
        mh = mw = cfg.fno_modes
        self.lift = tf.keras.layers.Conv2D(w, 1, name="lift")        # [u0, x, y] → w
        self.spec = [SpectralConv2d(w, w, mh, mw, name=f"spec{i}")
                     for i in range(cfg.fno_layers)]
        self.lin = [tf.keras.layers.Conv2D(w, 1, name=f"lin{i}")
                    for i in range(cfg.fno_layers)]
        self.proj1 = tf.keras.layers.Conv2D(128, 1, activation="gelu", name="proj1")
        self.proj2 = tf.keras.layers.Conv2D(cfg.Nt, 1, name="proj2")  # Nt output frames
        xs = np.linspace(0.0, 1.0, cfg.Nx, endpoint=False, dtype=np.float32)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        self._coords = tf.constant(np.stack([gx, gy], -1)[None], tf.float32)
        self._build()

    def _build(self):
        d = tf.zeros((1, self.cfg.Nx, self.cfg.Nx, 1))
        _ = self.call(d[..., 0], None, None)

    def call(self, u0, descriptor=None, coeffs=None):  # signature-compatible w/ ours
        B = tf.shape(u0)[0]
        coords = tf.broadcast_to(self._coords, [B, self.cfg.Nx, self.cfg.Nx, 2])
        x = self.lift(tf.concat([u0[..., None], coords], axis=-1))
        for sp, ln in zip(self.spec, self.lin):
            x = tf.nn.gelu(sp(x) + ln(x))
        out = self.proj2(self.proj1(x))                # (B,H,W,Nt)
        out = tf.transpose(out, [0, 3, 1, 2])          # (B,Nt,H,W)
        # hard IC: anchor frame 0 to u0 (fair: our model also satisfies IC exactly)
        ic = u0[:, None]
        return tf.concat([ic, out[:, 1:]], axis=1)

    @property
    def trainable_variables(self):
        out = list(self.lift.trainable_variables) + list(self.proj1.trainable_variables) \
            + list(self.proj2.trainable_variables)
        for l in self.spec + self.lin:
            out += list(l.trainable_variables)
        return out
