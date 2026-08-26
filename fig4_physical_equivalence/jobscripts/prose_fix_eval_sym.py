import io
p = 'evaluate.py'
s = io.open(p).read()
if 'FULLTEST_RELNORM' not in s:
    anchor = '\n                # computing eval metrics'
    snippet = '''
                if True:  # FULLTEST_RELNORM: our-formula running mean over the FULL test set
                    import numpy as _np
                    if not hasattr(self, "_ft_acc"): self._ft_acc = {}
                    _dim = data_output.shape[-1]
                    _o = (data_output.float() * d["std"] + d["mean"]) if self.params.normalize else data_output.float()
                    _o = _o.cpu().numpy(); _t = samples["data"][..., :_dim].cpu().numpy()[:, -_o.shape[1]:]
                    _num = _np.sqrt(((_o - _t) ** 2).sum((2, 3, 4))); _den = _np.sqrt((_t ** 2).sum((2, 3, 4))) + 1e-12
                    _a = self._ft_acc.setdefault(type, [])
                    _a.append((_num / _den).mean(1))
                    _c = _np.concatenate(_a)
                    print(f"FULLTEST_RELNORM {type} n={len(_c)} mean={float(_c.mean())*100:.3f}%", flush=True)
                    if not hasattr(self, "_ft_pred"): self._ft_pred = {}; self._ft_gt = {}; self._ft_saved = set()
                    self._ft_pred.setdefault(type, []).append(_o[:, :10]); self._ft_gt.setdefault(type, []).append(_t[:, :10])
                    if len(_c) >= 100 and type not in self._ft_saved:
                        _P = _np.concatenate(self._ft_pred[type])[:100]; _G = _np.concatenate(self._ft_gt[type])[:100]
                        import os as _os
                        _np.savez(f"/eu/_prosesym_{_os.environ.get('FLIPTAG','id')}_{type}.npz", pred=_P.astype("float32"), gt=_G.astype("float32"))
                        self._ft_saved.add(type); print(f"DUMP_SAVED {type} {_P.shape}", flush=True)
'''
    # reset the accumulator at each evaluate() call (per-epoch curve, not run-cumulative)
    ra = 'all_results = {}'
    k = s.index(ra)
    ls = s.rindex('\n', 0, k) + 1
    ind = s[ls:k]
    s = s[:k + len(ra)] + '\n' + ind + 'self._ft_acc = {}' + s[k + len(ra):]
    i = s.index(anchor)
    s = s[:i] + snippet + s[i:]
    io.open(p, 'w').write(s)
print('EVALPATCH3 OK')
