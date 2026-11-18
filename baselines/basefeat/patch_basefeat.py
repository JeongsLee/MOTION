"""Patch the ConvEmbedder of felix-lyx/bcat (fd0176d) or felix-lyx/prose prose_fd (eaf598d) to use basefeat.

  python patch_basefeat.py bcat  /tmp/bc/src
  python patch_basefeat.py prose /tmp/pr/prose_fd

Copies basefeat.py into <src>/models/ and edits models/embedder.py (ConvEmbedder only, non-early-conv path):
  - first patch-embedding conv: in_channels = data_dim + n_extra(data_dim)
  - encode(): x = features(x) right after the (b t) c h w rearrange
With BASEFEAT unset/off, n_extra = 0 and features(x) = x, so the model is identical to the unpatched one.
"""
import os
import shutil
import sys

kind, src = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "basefeat.py"), os.path.join(src, "models", "basefeat.py"))
p = os.path.join(src, "models", "embedder.py")
s = open(p).read()
if "basefeat" in s:
    print("already patched"); sys.exit(0)

c0 = s.index("class ConvEmbedder")
c1 = s.index("\nclass ", c0 + 10) if "\nclass " in s[c0 + 10:] else len(s)
body = s[c0:c1]


def sub(text, old, new):
    assert text.count(old) == 1, (kind, old)
    return text.replace(old, new)


if kind == "bcat":
    body = sub(body, "            self.in_proj = nn.Conv2d(\n                in_channels=data_dim,",
               "            self.in_proj = nn.Conv2d(\n                in_channels=data_dim + _bf.n_extra(data_dim),")
    body = sub(body, '        data = rearrange(data[:, skip_len:], "b t h w c -> (b t) c h w")\n',
               '        data = rearrange(data[:, skip_len:], "b t h w c -> (b t) c h w")\n        data = _bf.features(data)\n')
elif kind == "prose":
    body = sub(body, "            self.conv_proj = nn.Sequential(\n                nn.Conv2d(\n                    in_channels=data_dim,",
               "            self.conv_proj = nn.Sequential(\n                nn.Conv2d(\n                    in_channels=data_dim + _bf.n_extra(data_dim),")
    body = sub(body, '        data = rearrange(data, "b t h w c -> (b t) c h w")\n        data = self.conv_proj(data)',
               '        data = rearrange(data, "b t h w c -> (b t) c h w")\n        data = _bf.features(data)\n        data = self.conv_proj(data)')
else:
    raise SystemExit("kind must be bcat or prose")

s = s[:c0] + body + s[c1:]
imp = "try:\n    from . import basefeat as _bf\nexcept ImportError:\n    import basefeat as _bf\n"
i = s.index("\n", s.index("import torch.nn as nn")) + 1
s = s[:i] + imp + s[i:]
open(p, "w").write(s)
print(f"patched {p} (BASEFEAT={os.environ.get('BASEFEAT', 'off')} at runtime)")
