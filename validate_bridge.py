"""Sanity checks for bridge_map.csv (run after build_bridge.py)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("build_bridge.py").read().split("# --- 1. columnar neurons")[0])  # reuse loaders/helpers
b = pd.read_csv("bridge_map.csv")

# 1. Leave-one-out test of the connectivity method on neurons whose column IS known.
pos = col.set_index("root_id")[["p", "q"]]
XY = pd.DataFrame(fw_cart(pos.p, pos.q), index=pos.index, columns=["X", "Y"])
print("Connectivity-placement error on neurons with known columns (hex steps):")
for t in ["T4a", "T5c", "Tm3", "Mi4"]:
    ids = set(col[col.type == t].root_id)
    a = con[con.Postsynaptic_ID.isin(ids) & con.Presynaptic_ID.isin(XY.index)].rename(
        columns={"Postsynaptic_ID": "cell", "Presynaptic_ID": "partner"})
    o = con[con.Presynaptic_ID.isin(ids) & con.Postsynaptic_ID.isin(XY.index)].rename(
        columns={"Presynaptic_ID": "cell", "Postsynaptic_ID": "partner"})
    e = pd.concat([a, o])
    e = e[e.partner != e.cell].join(XY, on="partner")
    g = e.groupby("cell").apply(lambda s: pd.Series({
        "X": np.average(s.X, weights=s.Connectivity), "Y": np.average(s.Y, weights=s.Connectivity)}),
        include_groups=False)
    p, q = fw_axial(g.X.values, g.Y.values)
    tp, tq = pos.loc[g.index].p.values, pos.loc[g.index].q.values
    dp, dq = p - tp, q - tq
    dist = (np.abs(dp) + np.abs(dq) + np.abs(dp + dq)) / 2
    print(f"  {t:4s} n={len(dist):4d}  exact {np.mean(dist==0):.0%}  within 1 {np.mean(dist<=1):.0%}"
          f"  within 2 {np.mean(dist<=2):.0%}")

# 2. One FlyWire neuron per flyvis cell for strictly columnar types?
cm = b[(b.method == "column_file") & (b.flyvis_index >= 0)]
dup = cm.groupby(["side", "flyvis_index"]).size()
print(f"\nflyvis cells (per eye) driving more than one columnar FlyWire neuron: {(dup > 1).sum()} of {len(dup):,}")
print(f"flyvis columns (of 721) covered per eye by Mi1: "
      f"{cm[(cm.flyvis_type=='Mi1')&(cm.side=='right')].flyvis_index.nunique()} right, "
      f"{cm[(cm.flyvis_type=='Mi1')&(cm.side=='left')].flyvis_index.nunique()} left")

# 3. Hex maps of coverage
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
uu, vv = [], []
for u in range(-15, 16):
    for v in range(max(-15, -15 - u), min(15, 15 - u) + 1):
        uu.append(u); vv.append(v)
uu, vv = np.array(uu), np.array(vv)
for ax, side in zip(axes, ["left", "right"]):
    s = b[(b.side == side) & (b.method != "pooled")]
    inside = s[s.in_flyvis_eye]
    n_types = inside.groupby(["u", "v"]).flyvis_type.nunique()
    val = np.array([n_types.get((u, v), 0) for u, v in zip(uu, vv)])
    r, c = uu + vv / 2, vv                         # BoxEye layout (row, col)
    sc = ax.scatter(c, -r, c=val, cmap="viridis", s=22, marker="h", vmin=0, vmax=val.max())
    out = s[~s.in_flyvis_eye].drop_duplicates(["u", "v"])
    ax.scatter(out.v, -(out.u + out.v / 2), s=6, c="lightgray", marker="h",
               label="FlyWire columns outside flyvis's eye")
    ax.set_title(f"{side} eye: cell types mapped per flyvis column")
    ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower left", fontsize=7)
fig.colorbar(sc, ax=axes, shrink=0.7, label="# cell types mapped")
fig.savefig("bridge_hexmap.png", dpi=130, bbox_inches="tight")
print("\nsaved bridge_hexmap.png")
