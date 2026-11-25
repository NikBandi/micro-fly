"""Create the two flyvis-side inputs that build_bridge.py needs (vision environment).

    data/flyvis_nodes.csv        one row per flyvis cell: flyvis_index, type, u, v
    data/fib25-fib19_v2.2.json   flyvis's connectome description (copied from the flyvis package)

Usage (inside ~/flyduck/.venv-vision, from the repository folder):
    python tools/export_flyvis_inputs.py
"""
import glob
import os
import shutil

import pandas as pd
import flyvis
from flyvis import Network

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(out, exist_ok=True)

nodes = Network().connectome.nodes
df = pd.DataFrame({"type": [t.decode() if isinstance(t, bytes) else str(t) for t in nodes.type[:]],
                   "u": nodes.u[:], "v": nodes.v[:]})
df.insert(0, "flyvis_index", range(len(df)))
df.to_csv(os.path.join(out, "flyvis_nodes.csv"), index=False)
print(f"wrote data/flyvis_nodes.csv ({len(df):,} cells, {df.type.nunique()} types)")

pkg = os.path.dirname(flyvis.__file__)
hits = glob.glob(os.path.join(pkg, "**", "fib25-fib19_v2.2.json"), recursive=True)
if not hits:
    raise SystemExit("fib25-fib19_v2.2.json not found inside the flyvis package; "
                     "search your flyvis install / data folder for it and copy it to data/")
shutil.copy(hits[0], os.path.join(out, "fib25-fib19_v2.2.json"))
print(f"copied {hits[0]} -> data/fib25-fib19_v2.2.json")
