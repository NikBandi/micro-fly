"""Build the flyvis <-> FlyWire bridge table.

For every FlyWire optic-lobe neuron whose cell type flyvis also models, find the
flyvis cell (type + hex column) that should drive it.

Inputs (in ./data):
  column_assignment.csv.gz                  FlyWire v783 column assignment (Matsliah et al. 2024)
  Supplemental_file1_neuron_annotations.tsv FlyWire v783 cell types (flywire_annotations repo)
  Connectivity_783.parquet                  FlyWire v783 connectivity (Shiu et al. model repo)
  flyvis_nodes.csv                          flyvis cell list: type, u, v, flyvis_index

Outputs:
  bridge_map.csv        one row per mapped FlyWire neuron
  bridge_meta.json      lattice alignment, centres, camera orientation, coverage numbers
  bridge_report.txt     human-readable coverage report

Method
  1. Columnar types (30 shared types): use the official FlyWire column (p, q).
  2. Other shared types: estimate each neuron's column from the synapse-weighted
     mean position of its columnar partners (inputs and outputs).
  3. Align the two hex lattices. Both use axial coordinates whose third neighbour is
     (+1, -1). We test all 12 lattice symmetries (6 rotations x mirror) and keep the one
     under which flyvis's T4/T5 preferred directions best match FlyWire's. A preferred
     direction is measured anatomically in both datasets as the vector from the
     preferred-side input (Mi9 for T4, Tm9 for T5) to the null-side inputs (Mi4 for
     T4, Tm1+Tm2 for T5).
  4. Translate so each eye's central column sits at flyvis (0, 0); keep neurons that
     land inside flyvis's 721-column eye (extent 15).
"""
import itertools
import json

import numpy as np
import pandas as pd

D = "data/"
EXTENT = 15

# --- name translation: flyvis type -> FlyWire cell_type(s) -------------------
NAME_MAP = {
    "TmY9": ["TmY9q", "TmY9q__perp"],
    "CT1(Lo1)": ["CT1"],
    "CT1(M10)": ["CT1"],
    "Am": ["Am", "Am1"],
}
# Single wide-field cells: one FlyWire neuron per eye, driven by the mean of all its
# flyvis compartments instead of one column.
POOLED = {"CT1(Lo1)", "CT1(M10)", "Am"}
# Photoreceptors are upstream of everything flyvis replaces, so they are not driven.
SKIP = {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"}

# --- load ----------------------------------------------------------------------
col = pd.read_csv(D + "column_assignment.csv.gz")
ann = pd.read_csv(D + "Supplemental_file1_neuron_annotations.tsv", sep="\t",
                  low_memory=False, usecols=["root_id", "cell_type", "side"])
con = pd.read_parquet(D + "Connectivity_783.parquet",
                      columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
fv = pd.read_csv(D + "flyvis_nodes.csv")
fv_types = list(dict.fromkeys(fv.type))


def fw_cart(p, q):
    """FlyWire axial (p, q) -> cartesian. Neighbours at unit distance."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return np.stack([(q - p) / 2.0, (p + q) * np.sqrt(3) / 2], -1)


def fw_axial(X, Y):
    """Inverse of fw_cart, rounded to the nearest lattice point."""
    s = Y / (np.sqrt(3) / 2)          # p + q
    d = 2 * X                          # q - p
    p, q = (s - d) / 2, (s + d) / 2
    return hex_round(p, q)


def hex_round(a, b):
    """Round fractional axial coords (neighbours ±(1,0), ±(0,1), ±(1,-1)) to the nearest hex.

    Standard cube rounding with x = a, z = b, y = -a - b.
    """
    x, z = np.asarray(a, float), np.asarray(b, float)
    y = -x - z
    rx, ry, rz = np.round(x), np.round(y), np.round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    return rx.astype(int), rz.astype(int)


# --- 1. columnar neurons -----------------------------------------------------------
rows = []
for t in fv_types:
    if t in SKIP:
        continue
    sub = col[col.type == t]
    for r in sub.itertuples():
        rows.append((r.root_id, r.hemisphere, t, r.p, r.q, "column_file"))
mapped = pd.DataFrame(rows, columns=["flywire_root_id", "side", "flyvis_type", "p", "q", "method"])
columnar_types = set(mapped.flyvis_type)

# --- 2. non-columnar neurons via connectivity --------------------------------------
pos = col.set_index("root_id")[["p", "q", "hemisphere"]]
XY = pd.DataFrame(fw_cart(pos.p, pos.q), index=pos.index, columns=["X", "Y"])
con_in = con[con.Presynaptic_ID.isin(XY.index)]      # columnar partner is the sender
con_out = con[con.Postsynaptic_ID.isin(XY.index)]    # (both directions are used)

extra = []
unmatched = []
pooled_rows = []
for t in fv_types:
    if t in SKIP or t in columnar_types:
        continue
    names = NAME_MAP.get(t, [t])
    cells = ann[ann.cell_type.isin(names)]
    if cells.empty:
        unmatched.append(t)
        continue
    if t in POOLED:
        for r in cells.itertuples():
            pooled_rows.append((r.root_id, r.side, t, np.nan, np.nan, "pooled"))
        continue
    ids = set(cells.root_id)
    a = con_in[con_in.Postsynaptic_ID.isin(ids)].rename(
        columns={"Postsynaptic_ID": "cell", "Presynaptic_ID": "partner"})
    b = con_out[con_out.Presynaptic_ID.isin(ids)].rename(
        columns={"Presynaptic_ID": "cell", "Postsynaptic_ID": "partner"})
    e = pd.concat([a, b]).join(XY, on="partner")
    g = e.groupby("cell").apply(
        lambda s: pd.Series({"X": np.average(s.X, weights=s.Connectivity),
                             "Y": np.average(s.Y, weights=s.Connectivity),
                             "syn": s.Connectivity.sum()}), include_groups=False)
    g = g[g.syn >= 5]                                   # need a few synapses to trust it
    p, q = fw_axial(g.X.values, g.Y.values)
    side = cells.set_index("root_id").side.reindex(g.index).values
    for rid, s, pp, qq in zip(g.index, side, p, q):
        extra.append((rid, s, t, pp, qq, "connectivity"))

mapped = pd.concat([mapped,
                    pd.DataFrame(extra, columns=mapped.columns)], ignore_index=True)

# --- 3. align the lattices with T4/T5 preferred directions -------------------------
PAIRS = {"T4": (["Mi4"], ["Mi9"]), "T5": (["Tm1", "Tm2"], ["Tm9"])}
SUBTYPES = [f"T{k}{d}" for k in "45" for d in "abcd"]

fvj = json.load(open(D + "fib25-fib19_v2.2.json"))


def flyvis_vec(tar):
    A, B = PAIRS[tar[:2]]

    def mean_off(srcs):
        offs, ws = [], []
        for e in fvj["edges"]:
            if e["tar"] == tar and e["src"] in srcs:
                for (uv, w) in e["offsets"]:
                    offs.append(uv)
                    ws.append(w)
        return np.average(np.array(offs, float), axis=0, weights=ws)

    # flyvis stores each offset as (target - source), so the position of the input
    # (source) relative to the T4/T5 cell (target) is the NEGATIVE of the offset.
    return -(mean_off(A) - mean_off(B))   # in (u, v) axial units


def flywire_vec(tar, hemi):
    A, B = PAIRS[tar[:2]]
    tids = col[(col.type == tar) & (col.hemisphere == hemi)].root_id
    sub = con[con.Postsynaptic_ID.isin(tids) & con.Presynaptic_ID.isin(pos.index)]
    pre = col.set_index("root_id").loc[sub.Presynaptic_ID]
    post = col.set_index("root_id").loc[sub.Postsynaptic_ID]
    d = fw_cart(pre.p, pre.q) - fw_cart(post.p, post.q)
    w = sub.Connectivity.values
    t = pre.type.values

    def m(ts):
        k = np.isin(t, ts)
        return np.average(d[k], axis=0, weights=w[k])

    return m(A) - m(B)                 # cartesian


# The 12 symmetries of a hex lattice in axial coords with neighbours
# {±(1,0), ±(0,1), ±(1,-1)}: integer 2x2 matrices that permute that set.
NEIGH = {(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)}
SYMS = []
for m in itertools.product([-1, 0, 1], repeat=4):
    M = np.array(m).reshape(2, 2)
    if abs(round(np.linalg.det(M))) != 1:
        continue
    if {tuple(M @ np.array(n)) for n in NEIGH} == NEIGH:
        SYMS.append(M)
assert len(SYMS) == 12, len(SYMS)

fv_vecs = {t: flyvis_vec(t) for t in SUBTYPES}
meta = {"alignment": {}}
best_M = {}
for hemi in ["right", "left"]:
    fw_vecs = {t: flywire_vec(t, hemi) for t in SUBTYPES}
    scores = []
    for M in SYMS:
        cos = []
        for t in SUBTYPES:
            a = fw_cart(*(M @ fv_vecs[t]))           # flyvis vec -> (p,q) -> cartesian
            b = fw_vecs[t]
            cos.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        scores.append((np.mean(cos), M, cos))
    scores.sort(key=lambda s: -s[0])
    best, M, cos = scores[0]
    best_M[hemi] = M
    meta["alignment"][hemi] = {
        "matrix_uv_to_pq": M.tolist(),
        "mean_cosine": round(float(best), 3),
        "runner_up_cosine": round(float(scores[1][0]), 3),
        "per_subtype_cosine": {t: round(float(c), 3) for t, c in zip(SUBTYPES, cos)},
        "flywire_pd_angles_deg": {t: round(float(np.degrees(np.arctan2(*fw_vecs[t][::-1]))), 1)
                                  for t in SUBTYPES},
    }

# --- 4. centre, convert to flyvis (u, v), keep inside the flyvis eye --------------
out = []
for hemi in ["right", "left"]:
    s = col[(col.hemisphere == hemi)].drop_duplicates("column_id")
    c = fw_cart(s.p, s.q).mean(0)
    cp, cq = fw_axial(np.array([c[0]]), np.array([c[1]]))
    centre = np.array([cp[0], cq[0]])
    meta.setdefault("centre_pq", {})[hemi] = centre.tolist()
    Minv = np.round(np.linalg.inv(best_M[hemi])).astype(int)
    m = mapped[mapped.side == hemi].copy()
    pq = np.stack([m.p.values, m.q.values], 1) - centre
    uv = pq @ Minv.T
    m["u"], m["v"] = uv[:, 0], uv[:, 1]
    m["in_flyvis_eye"] = (np.abs(m.u) <= EXTENT) & (np.abs(m.v) <= EXTENT) & \
                         (np.abs(m.u + m.v) <= EXTENT)
    out.append(m)
bridge = pd.concat(out, ignore_index=True)

# attach the flyvis node index (the row in flyvis's 45,669-cell output) -------------
fv_key = fv.set_index(["type", "u", "v"]).flyvis_index
idx = []
for r in bridge.itertuples():
    k = (r.flyvis_type, r.u, r.v)
    idx.append(int(fv_key[k]) if r.in_flyvis_eye and k in fv_key.index else -1)
bridge["flyvis_index"] = idx
# Lawf1/2 live on a sparser grid in flyvis: snap to the nearest flyvis Lawf cell.
for t in ["Lawf1", "Lawf2"]:
    grid = fv[fv.type == t][["u", "v", "flyvis_index"]].values
    k = (bridge.flyvis_type == t) & bridge.in_flyvis_eye
    for i in bridge.index[k]:
        du = grid[:, 0] - bridge.at[i, "u"]
        dv = grid[:, 1] - bridge.at[i, "v"]
        dist = (du * du + dv * dv + du * dv)           # hex metric
        bridge.at[i, "flyvis_index"] = int(grid[np.argmin(dist), 2])

pooled = pd.DataFrame(pooled_rows, columns=["flywire_root_id", "side", "flyvis_type", "p", "q", "method"])
pooled["flyvis_index"] = -2          # -2 = drive with the mean over all compartments
pooled["in_flyvis_eye"] = True
bridge = pd.concat([bridge, pooled], ignore_index=True)
bridge = bridge[["flywire_root_id", "side", "flyvis_type", "flyvis_index", "u", "v",
                 "p", "q", "method", "in_flyvis_eye"]]
bridge.to_csv("bridge_map.csv", index=False)

# --- camera orientation -------------------------------------------------------------
# World axes are defined from FlyWire anatomy: "up" = the T4c/T5c preferred direction
# (upward motion), "front-to-back" = the T4a/T5a preferred direction. These are carried
# through the chosen lattice symmetry into flyvis (u, v) and then into BoxEye image
# coordinates, where BoxEye samples column (u, v) at image (row, col) = (u + v/2, v).
# Result: how to rotate (and mirror) each half of the duck's camera image before BoxEye
# so that the brain's retinotopy matches the world.
def fw_cart_to_pq(vec):
    X, Y = vec
    s, d = Y / (np.sqrt(3) / 2), 2 * X
    return np.array([(s - d) / 2, (s + d) / 2])


meta["camera"] = {}
for hemi in ["right", "left"]:
    al = meta["alignment"][hemi]
    fwv = {t: flywire_vec(t, hemi) for t in SUBTYPES}
    unit = lambda x: x / np.linalg.norm(x)
    up = unit(sum(unit(fwv[f"T{k}c"]) - unit(fwv[f"T{k}d"]) for k in "45"))
    ftb = unit(sum(unit(fwv[f"T{k}a"]) - unit(fwv[f"T{k}b"]) for k in "45"))
    Minv = np.linalg.inv(best_M[hemi])

    def to_img(vec):           # -> image vector (x = col to the right, y = row upward)
        u, v = Minv @ fw_cart_to_pq(vec)
        row, colm = u + v / 2, v
        return unit(np.array([colm, -row]))

    U, F = to_img(up), to_img(ftb)
    # camera half-image: right eye -> front-to-back is +x; left eye -> front-to-back is -x
    cam_ftb = np.array([1.0, 0.0]) if hemi == "right" else np.array([-1.0, 0.0])
    # find rotation R (and optional mirror) taking camera (ftb, up) to BoxEye (F, U)
    best = None
    for mirror in (False, True):
        Mm = np.diag([-1.0, 1.0]) if mirror else np.eye(2)
        a_up = np.arctan2(U[1], U[0]) - np.arctan2(*(Mm @ [0, 1])[::-1])
        a_ft = np.arctan2(F[1], F[0]) - np.arctan2(*(Mm @ cam_ftb)[::-1])
        ang = np.angle(np.exp(1j * a_up) + np.exp(1j * a_ft))          # average angle
        err = np.degrees(abs(np.angle(np.exp(1j * (a_up - a_ft)))))
        if best is None or err < best[2]:
            best = (mirror, float(np.degrees(ang)), float(err))
    meta["camera"][f"{hemi}_eye"] = {
        "mirror_horizontally_first": best[0],
        "then_rotate_ccw_deg": round(best[1], 1),
        "axis_mismatch_deg": round(best[2], 1),
        "boxeye_up_xy": np.round(U, 3).tolist(),
        "boxeye_front_to_back_xy": np.round(F, 3).tolist(),
    }
meta["camera"]["note"] = ("Split the duck camera image into left and right halves. For each "
                          "half: optionally mirror left-right, then rotate counter-clockwise by "
                          "the given angle, then pass to BoxEye. Image y axis = up.")

# --- coverage report ----------------------------------------------------------------
lines = []
lines.append("flyvis <-> FlyWire bridge report\n")
for hemi in ["right", "left"]:
    al = meta["alignment"][hemi]
    lines.append(f"{hemi} eye: best lattice symmetry mean cos = {al['mean_cosine']} "
                 f"(runner-up {al['runner_up_cosine']}); centre (p,q) = {meta['centre_pq'][hemi]}")
for hemi in ["right", "left"]:
    c = meta["camera"][f"{hemi}_eye"]
    lines.append(f"camera {hemi} half: mirror={c['mirror_horizontally_first']}, rotate "
                 f"{c['then_rotate_ccw_deg']} deg CCW (up vs front-to-back axes differ from "
                 f"90 deg by {c['axis_mismatch_deg']} deg)")
lines.append("")
good = bridge[bridge.flyvis_index != -1]
cov = (bridge.groupby(["flyvis_type", "side"])
       .apply(lambda s: pd.Series({"flywire_cells": len(s),
                                   "mapped": int((s.flyvis_index != -1).sum()),
                                   "method": s.method.iloc[0]}), include_groups=False)
       .unstack("side"))
lines.append(cov.to_string())
lines.append(f"\nflyvis types with no FlyWire counterpart: {unmatched}")
lines.append(f"skipped (upstream photoreceptors): {sorted(SKIP)}")
lines.append(f"\nTOTAL FlyWire neurons mapped to a flyvis cell: {len(good):,} "
             f"(column file {int((good.method=='column_file').sum()):,}, "
             f"connectivity {int((good.method=='connectivity').sum()):,}, "
             f"pooled {int((good.method=='pooled').sum())})")
meta["unmatched_flyvis_types"] = unmatched
meta["n_mapped"] = int(len(good))
open("bridge_report.txt", "w").write("\n".join(lines))
json.dump(meta, open("bridge_meta.json", "w"), indent=2)
print("\n".join(lines[:4]))
print(lines[-1])
