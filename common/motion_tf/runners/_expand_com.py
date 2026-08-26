"""Expand the com_ns cache with Turb (512→128) data WITHOUT removing the existing Rand shards (so a
running Task-1 job that already indexed the Rand shards is unaffected — it just ignores the appended
Turb shards). _write_comns_shards resumes: re-walks Rand (skips existing shards), then appends Turb."""
import os, sys
sys.path.insert(0, "/tmp/work")
from motion_tf.data import prose

PRE = os.environ.get("PREBUILT_DIR", "/code-vol/data/prose/prebuilt")
d = os.path.join(PRE, "com_ns_n100000_t20_shards")
print(f"expanding com_ns (Rand kept + Turb appended) -> {d}", flush=True)
w = prose._write_comns_shards(100000, 20, d)
print(f"com_ns total trajectories now: {w}", flush=True)
print("COM_EXPAND_DONE", flush=True)
