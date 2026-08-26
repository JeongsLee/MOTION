#!/bin/bash
# Re-render Figure 1 from the SAVED DUMP, locally, with no GPU and no cluster access.
#
# The GPU job (fig1-m2, 08-14) wrote three data files plus a metadata file; everything the
# figure shows is in them.  Layout, colour, label and panel-composition work therefore never
# needs the model again -- only this script.  Re-running the dump is required ONLY if the
# underlying quantity changes (different checkpoint, different lead time, different families
# or heads), never for presentation.
#
#   fig1_data.npz   panels b (GT vs prediction, 2D families + 3D isosurfaces) and c (19-family
#                   knockout damage / family-specific contribution heat map)
#   fig1d_ko.npz    panel d (spectral-space head knockouts)
#   fig1e_ko.npz    panel e (per-family knockout response)
#   fig1_meta.json  checkpoint path, family list, mechanism list -- provenance of the above
#
# The displayed mechanism names live in the MLAB maps inside the renderers, not in the dump:
# the dump stores internal head keys (shock, spectral, buoyancy, ...) and the renderers map
# them to the manuscript vocabulary (upwind transport, spectral nonlocal, gravity gradient,
# ...).  Renaming a mechanism in the paper is therefore a one-line edit here, not a re-dump.
#
# usage:  ./render_fig1_local.sh [DATA_DIR]        (default: the m2 dump directory below)

set -e
D="${1:-/mnt/e/motion_fig1_m2}"
cd "$(dirname "$0")"

for f in fig1_data.npz fig1_meta.json; do
  [ -f "$D/$f" ] || { echo "missing $D/$f -- download the dump first"; exit 1; }
done
python3 -c "import json;m=json.load(open('$D/fig1_meta.json'));print('ckpt   :',m['ckpt']);print('families:',len(m['families']),' mechanisms:',len(m['mechanisms']))"

echo "== panels b, c =="
python3 render_fig1.py "$D"

if [ -f "$D/fig1d_ko.npz" ]; then
  echo "== panel d (velocity channel, then last channel) =="
  python3 render_fig1d_ko.py "$D"
  KO_CHAN=last python3 render_fig1d_ko.py "$D"
fi

if [ -f "$D/fig1e_ko.npz" ]; then
  echo "== panel e =="
  python3 render_fig1e_ko.py "$D"
fi

echo "== written =="
ls -la "$D"/*.png | awk '{print $5, $9}'
echo
echo "To place them in the manuscript:"
echo "  cp $D/fig1b.png $D/fig1c.png ../manuscript_ncs/figs/"
echo "  cp <chosen panel-d png> ../manuscript_ncs/figs/fig1d_v2.png"
echo "  cp $D/fig1e_v3.png ../manuscript_ncs/figs/"
