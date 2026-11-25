"""Who drives a cell type in the Shiu brain model: summed signed synapses by presynaptic type.

Usage (any env with pandas + pyarrow):
    python inputs_of.py Connectivity_783.parquet data/Supplemental_file1_neuron_annotations.tsv LPLC2:right DNp01:right
"""
import sys, pandas as pd
c = pd.read_parquet(sys.argv[1]); a = pd.read_csv(sys.argv[2], sep="\t", low_memory=False,
     usecols=["root_id", "cell_type", "side", "top_nt", "super_class"])
FLYVIS = set("""L1 L2 L3 L4 L5 Lawf1 Lawf2 Am C2 C3 CT1 Mi1 Mi2 Mi3 Mi4 Mi9 Mi10 Mi11 Mi12 Mi13 Mi14 Mi15
T1 T2 T2a T3 T4a T4b T4c T4d T5a T5b T5c T5d Tm1 Tm2 Tm3 Tm4 Tm5a Tm5b Tm5c Tm5Y Tm9 Tm16 Tm20 Tm28 Tm30
TmY3 TmY4 TmY5a TmY9 TmY10 TmY13 TmY14 TmY15 TmY18""".split())
typ = dict(zip(a.root_id, a.cell_type.fillna(a.super_class.fillna("?"))))
side = dict(zip(a.root_id, a.side))
for target in sys.argv[3:]:
    t, _, s = target.partition(":")
    ids = set(a.root_id[(a.cell_type == t) & (a.side == (s or "right"))])
    sub = c[c.Postsynaptic_ID.isin(ids)].copy()
    sub["pre_type"] = sub.Presynaptic_ID.map(typ).fillna("?")
    sub["eye"] = sub.pre_type.isin(FLYVIS)
    g = sub.groupby(["pre_type", "eye"])["Excitatory x Connectivity"].sum().reset_index()
    g["per_cell"] = g["Excitatory x Connectivity"] / len(ids)
    exc, inh = g[g.per_cell > 0].per_cell.sum(), -g[g.per_cell < 0].per_cell.sum()
    eye_exc = g[(g.per_cell > 0) & g.eye].per_cell.sum(); eye_inh = -g[(g.per_cell < 0) & g.eye].per_cell.sum()
    print(f"\n=== {target} ({len(ids)} cells), synapses per cell: excit {exc:.0f} (from eye puppets {eye_exc:.0f}), "
          f"inhib {inh:.0f} (from eye puppets {eye_inh:.0f})")
    g = g.reindex(g.per_cell.abs().sort_values(ascending=False).index).head(18)
    for r in g.itertuples():
        print(f"  {r.pre_type:<14} {'EYE ' if r.eye else 'brain'} {r.per_cell:+8.1f}")
