from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REQUIRED_COLS = ["source", "target", "value"]
OPTIONAL_COLS = ["link_label", "link_color"]
NODE_COLS = ["node", "level", "order"]  # layout control


# ---------- Data helpers ----------
def default_df() -> pd.DataFrame:
    # Generic sample (domain-neutral). Includes a duplicate edge to demonstrate aggregation.
    return pd.DataFrame(
        [
            ["A", "B", 10, "A → B: 10", ""],
            ["A", "C", 5, "A → C: 5", ""],
            ["B", "D", 7, "B → D: 7", ""],
            ["C", "D", 4, "C → D: 4", ""],
            ["D", "E", 8, "D → E: 8", ""],
            ["A", "B", 2, "A → B: 2", ""],
        ],
        columns=["source", "target", "value", "link_label", "link_color"],
    )


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_COLS:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")
    for c in OPTIONAL_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[REQUIRED_COLS + OPTIONAL_COLS]
    return out


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    df2 = normalize_df(df)
    return df2.to_csv(index=False).encode("utf-8")


def df_from_uploaded_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return normalize_df(df)


def get_nodes_from_flows(df: pd.DataFrame) -> list[str]:
    df = normalize_df(df)
    s = df["source"].astype(str).str.strip()
    t = df["target"].astype(str).str.strip()
    nodes = pd.Index(pd.concat([s, t], ignore_index=True).unique()).tolist()
    return [n for n in nodes if n != ""]


def normalize_nodes_df(nodes_df: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """
    Ensures nodes_df covers all nodes and has columns: node, level, order.
    Missing nodes are added with defaults. Extra nodes are removed.
    """
    if nodes_df is None or nodes_df.empty:
        out = pd.DataFrame({"node": nodes, "level": 0, "order": list(range(len(nodes)))})
        return out[NODE_COLS]

    out = nodes_df.copy()
    for c in NODE_COLS:
        if c not in out.columns:
            out[c] = 0

    out["node"] = out["node"].astype(str).str.strip()
    out = out[out["node"].isin(nodes)]

    missing = [n for n in nodes if n not in set(out["node"].tolist())]
    if missing:
        add = pd.DataFrame(
            {"node": missing, "level": 0, "order": list(range(len(out), len(out) + len(missing)))}
        )
        out = pd.concat([out, add], ignore_index=True)

    out["level"] = pd.to_numeric(out["level"], errors="coerce").fillna(0).astype(int)
    out["order"] = pd.to_numeric(out["order"], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["level", "order", "node"], kind="stable").reset_index(drop=True)
    return out[NODE_COLS]


def aggregate_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate duplicate edges by (source,target).
    - value is summed.
    - link_color kept only if all non-empty colors are identical; else blank.
    - link_label cleared (auto hover labels are generated after aggregation).
    """
    df = normalize_df(df).copy()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[(df["source"] != "") & (df["target"] != "") & df["value"].notna()]
    if df.empty:
        return df

    def pick_color(series: pd.Series) -> str:
        vals = [v.strip() for v in series.fillna("").astype(str).tolist() if str(v).strip()]
        if not vals:
            return ""
        uniq = sorted(set(vals))
        return uniq[0] if len(uniq) == 1 else ""

    grouped = df.groupby(["source", "target"], as_index=False)
    out = grouped.agg(
        value=("value", "sum"),
        link_color=("link_color", pick_color),
    )
    out["link_label"] = ""
    return out[REQUIRED_COLS + OPTIONAL_COLS]


def nodes_df_to_csv_bytes(nodes_df: pd.DataFrame) -> bytes:
    out = nodes_df.copy()
    out = out[NODE_COLS]
    return out.to_csv(index=False).encode("utf-8")


def nodes_df_from_uploaded_csv(file) -> pd.DataFrame:
    """
    Reads nodes layout CSV with columns: node, level, order
    """
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    if "node" not in df.columns:
        raise ValueError("nodes.csv must include a 'node' column.")
    if "level" not in df.columns:
        df["level"] = 0
    if "order" not in df.columns:
        df["order"] = 0

    df["node"] = df["node"].astype(str).str.strip()
    df["level"] = pd.to_numeric(df["level"], errors="coerce").fillna(0).astype(int)
    df["order"] = pd.to_numeric(df["order"], errors="coerce").fillna(0).astype(int)

    return df[["node", "level", "order"]]


# ---------- Sankey build (optional fixed layout) ----------
def build_sankey(
    df: pd.DataFrame,
    title: str,
    value_suffix: str = "",
    use_fixed_layout: bool = False,
    nodes_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    df = normalize_df(df)

    tmp = df.copy()
    tmp["source"] = tmp["source"].astype(str).str.strip()
    tmp["target"] = tmp["target"].astype(str).str.strip()
    tmp["value"] = pd.to_numeric(tmp["value"], errors="raise").astype(float)

    tmp = tmp[(tmp["source"] != "") & (tmp["target"] != "") & tmp["value"].notna()]
    if tmp.empty:
        raise ValueError("No valid rows found. Add at least one flow with source, target, and value.")
    if (tmp["value"] < 0).any():
        raise ValueError("All values must be non-negative. Use direction via source → target, not negatives.")

    nodes = get_nodes_from_flows(tmp)

    arrangement = "snap"
    node_kwargs: dict[str, Any] = dict(
        pad=18,
        thickness=18,
        line=dict(width=0.5),
    )

    if use_fixed_layout:
        nodes_df = normalize_nodes_df(nodes_df if nodes_df is not None else pd.DataFrame(), nodes)
        min_level = int(nodes_df["level"].min())
        max_level = int(nodes_df["level"].max())
        span = max(1, max_level - min_level)

        x_map: dict[str, float] = {}
        y_map: dict[str, float] = {}

        for lvl in sorted(nodes_df["level"].unique()):
            level_nodes = nodes_df[nodes_df["level"] == lvl].sort_values(["order", "node"], kind="stable")
            k = len(level_nodes)
            ys = [0.5] if k == 1 else [0.05 + i * (0.90 / (k - 1)) for i in range(k)]
            for node, _order, y in zip(level_nodes["node"].tolist(), level_nodes["order"].tolist(), ys):
                x_map[node] = (lvl - min_level) / span
                y_map[node] = float(y)

        node_labels = nodes_df["node"].tolist()
        idx = {label: i for i, label in enumerate(node_labels)}
        node_kwargs["label"] = node_labels
        node_kwargs["x"] = [x_map[n] for n in node_labels]
        node_kwargs["y"] = [y_map[n] for n in node_labels]
        arrangement = "fixed"
    else:
        node_labels = pd.Index(pd.concat([tmp["source"], tmp["target"]], ignore_index=True).unique()).tolist()
        idx = {label: i for i, label in enumerate(node_labels)}
        node_kwargs["label"] = node_labels

    sources = tmp["source"].map(idx).tolist()
    targets = tmp["target"].map(idx).tolist()
    values = tmp["value"].tolist()

    has_custom = tmp["link_label"].notna().any() and (tmp["link_label"].astype(str).str.strip() != "").any()
    if has_custom:
        raw = tmp["link_label"].fillna("").astype(str).tolist()
        link_labels = [
            lbl if lbl.strip() else f"{s} → {t}: {v:g}{value_suffix}"
            for lbl, s, t, v in zip(raw, tmp["source"], tmp["target"], values)
        ]
    else:
        link_labels = [f"{s} → {t}: {v:g}{value_suffix}" for s, t, v in zip(tmp["source"], tmp["target"], values)]

    link_colors = None
    has_colors = tmp["link_color"].notna().any() and (tmp["link_color"].astype(str).str.strip() != "").any()
    if has_colors:
        link_colors = [c.strip() if str(c).strip() else None for c in tmp["link_color"].fillna("").astype(str).tolist()]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement=arrangement,
                node=node_kwargs,
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    label=link_labels,
                    color=link_colors,
                ),
            )
        ]
    )
    fig.update_layout(
        title_text=title,
        font=dict(size=12),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def fig_to_png_bytes(fig: go.Figure, width_px: int = 1600, height_px: int = 900, scale: int = 2) -> bytes:
    return fig.to_image(format="png", width=width_px, height=height_px, scale=scale)


def fig_to_html_bytes(fig: go.Figure) -> bytes:
    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


# ---------- Validation ----------
@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]


def validate_flows(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    try:
        df = normalize_df(df)
    except Exception as e:
        return ValidationResult(errors=[str(e)], warnings=[], stats={})

    tmp = df.copy()
    tmp["source"] = tmp["source"].astype(str).str.strip()
    tmp["target"] = tmp["target"].astype(str).str.strip()
    tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")

    missing_source = (tmp["source"] == "") | tmp["source"].isna()
    missing_target = (tmp["target"] == "") | tmp["target"].isna()
    missing_value = tmp["value"].isna()

    if missing_source.any() or missing_target.any() or missing_value.any():
        bad_rows = tmp[missing_source | missing_target | missing_value].index.tolist()
        warnings.append(
            "Some rows have missing source/target/value "
            f"(rows: {', '.join(map(str, bad_rows[:20]))}{'…' if len(bad_rows) > 20 else ''})."
        )

    if (tmp["value"].fillna(0) < 0).any():
        errors.append("Negative values found. Sankey link values must be non-negative.")

    if (tmp["value"].fillna(0) == 0).any():
        warnings.append("Some rows have value = 0. These links may render oddly or be invisible.")

    dup_mask = tmp[["source", "target"]].duplicated(keep=False)
    if dup_mask.any():
        warnings.append("Duplicate (source,target) pairs found. You can aggregate duplicates in Settings.")

    valid_rows = tmp.dropna(subset=["value"])
    valid_rows = valid_rows[(valid_rows["source"] != "") & (valid_rows["target"] != "")]
    sources = set(valid_rows["source"].tolist())
    targets = set(valid_rows["target"].tolist())
    nodes = sorted(sources | targets)

    only_out = sorted(sources - targets)
    only_in = sorted(targets - sources)
    if only_out:
        warnings.append(
            "Nodes with only outgoing links (no incoming): "
            f"{', '.join(only_out[:10])}{'…' if len(only_out) > 10 else ''}"
        )
    if only_in:
        warnings.append(
            "Nodes with only incoming links (no outgoing): "
            f"{', '.join(only_in[:10])}{'…' if len(only_in) > 10 else ''}"
        )

    stats["rows"] = int(len(df))
    stats["valid_rows"] = int(len(valid_rows))
    stats["nodes"] = int(len(nodes))
    stats["total_flow"] = float(valid_rows["value"].sum()) if not valid_rows.empty else 0.0

    return ValidationResult(errors=errors, warnings=warnings, stats=stats)


# ---------- Project save/load (JSON) ----------
def project_to_json_bytes(
    name: str,
    title: str,
    value_suffix: str,
    df: pd.DataFrame,
    aggregate_dupes: bool,
    use_fixed_layout: bool,
    nodes_df: pd.DataFrame,
) -> bytes:
    df2 = normalize_df(df)
    nodes_df2 = normalize_nodes_df(nodes_df, get_nodes_from_flows(df2))

    payload = {
        "schema_version": 2,
        "name": name,
        "title": title,
        "value_suffix": value_suffix,
        "settings": {
            "aggregate_duplicates": aggregate_dupes,
            "fixed_layout": use_fixed_layout,
        },
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "flows": df2.to_dict(orient="records"),
        "nodes": nodes_df2.to_dict(orient="records"),
        "columns": df2.columns.tolist(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def project_from_json_bytes(b: bytes) -> dict[str, Any]:
    payload = json.loads(b.decode("utf-8"))
    ver = int(payload.get("schema_version", 1))

    if ver == 1:
        name = str(payload.get("name", "Untitled"))
        title = str(payload.get("title", "Sankey Diagram"))
        value_suffix = str(payload.get("value_suffix", ""))
        flows = payload.get("flows", [])
        df = normalize_df(pd.DataFrame(flows))
        nodes = get_nodes_from_flows(df)
        nodes_df = normalize_nodes_df(pd.DataFrame(), nodes)
        return {
            "name": name,
            "title": title,
            "value_suffix": value_suffix,
            "aggregate_duplicates": False,
            "fixed_layout": False,
            "df": df,
            "nodes_df": nodes_df,
        }

    if ver != 2:
        raise ValueError("Unsupported project schema_version.")

    name = str(payload.get("name", "Untitled"))
    title = str(payload.get("title", "Sankey Diagram"))
    value_suffix = str(payload.get("value_suffix", ""))

    settings = payload.get("settings", {}) or {}
    aggregate_dupes = bool(settings.get("aggregate_duplicates", False))
    fixed_layout = bool(settings.get("fixed_layout", False))

    flows = payload.get("flows", [])
    df = normalize_df(pd.DataFrame(flows))

    nodes = get_nodes_from_flows(df)
    nodes_payload = payload.get("nodes", [])
    nodes_df = normalize_nodes_df(pd.DataFrame(nodes_payload), nodes)

    return {
        "name": name,
        "title": title,
        "value_suffix": value_suffix,
        "aggregate_duplicates": aggregate_dupes,
        "fixed_layout": fixed_layout,
        "df": df,
        "nodes_df": nodes_df,
    }


# ---------- UI state ----------
def ensure_state():
    if "datasets" not in st.session_state:
        base = default_df()
        st.session_state.datasets = [
            {
                "name": "Chart 1",
                "title": "Sankey Diagram",
                "value_suffix": "",
                "aggregate_duplicates": True,
                "fixed_layout": False,
                "df": normalize_df(base),
                "nodes_df": normalize_nodes_df(pd.DataFrame(), get_nodes_from_flows(base)),
            }
        ]
    if "active_idx" not in st.session_state:
        st.session_state.active_idx = 0


def set_active_idx(i: int):
    st.session_state.active_idx = i


# ---------- App ----------
st.set_page_config(page_title="Sankey Builder", layout="wide")
st.title("Sankey Builder")

ensure_state()

with st.sidebar:
    st.header("Datasets")

    names = [d["name"] for d in st.session_state.datasets]
    active = st.selectbox(
        "Active dataset",
        options=list(range(len(names))),
        format_func=lambda i: names[i],
        index=st.session_state.active_idx,
    )
    set_active_idx(active)

    st.divider()
    c_add, c_dup = st.columns(2)
    with c_add:
        if st.button("Add new", width="stretch"):
            new_df = normalize_df(default_df())
            st.session_state.datasets.append(
                {
                    "name": f"Chart {len(st.session_state.datasets) + 1}",
                    "title": "Sankey Diagram",
                    "value_suffix": "",
                    "aggregate_duplicates": True,
                    "fixed_layout": False,
                    "df": new_df,
                    "nodes_df": normalize_nodes_df(pd.DataFrame(), get_nodes_from_flows(new_df)),
                }
            )
            st.session_state.active_idx = len(st.session_state.datasets) - 1
            st.rerun()

    with c_dup:
        if st.button("Duplicate", width="stretch"):
            src = st.session_state.datasets[st.session_state.active_idx]
            st.session_state.datasets.append(
                {
                    "name": f"{src['name']} copy",
                    "title": src["title"],
                    "value_suffix": src["value_suffix"],
                    "aggregate_duplicates": src["aggregate_duplicates"],
                    "fixed_layout": src["fixed_layout"],
                    "df": normalize_df(src["df"].copy()),
                    "nodes_df": src["nodes_df"].copy(),
                }
            )
            st.session_state.active_idx = len(st.session_state.datasets) - 1
            st.rerun()

    c_ren, c_del = st.columns(2)
    with c_ren:
        if st.button("Rename", width="stretch"):
            st.session_state["_rename_mode"] = True
    with c_del:
        if st.button("Delete", width="stretch", disabled=len(st.session_state.datasets) <= 1):
            del st.session_state.datasets[st.session_state.active_idx]
            st.session_state.active_idx = max(0, st.session_state.active_idx - 1)
            st.rerun()

    if st.session_state.get("_rename_mode"):
        new_name = st.text_input(
            "New name",
            value=st.session_state.datasets[st.session_state.active_idx]["name"],
        )
        if st.button("Save name", width="stretch"):
            st.session_state.datasets[st.session_state.active_idx]["name"] = new_name.strip() or "Untitled"
            st.session_state["_rename_mode"] = False
            st.rerun()

    st.divider()
    st.header("Import")

    up_proj = st.file_uploader("Import Project (.json)", type=["json"], key="import_project")
    if up_proj is not None:
        try:
            proj = project_from_json_bytes(up_proj.read())
            st.session_state.datasets.append(
                {
                    "name": proj["name"],
                    "title": proj["title"],
                    "value_suffix": proj["value_suffix"],
                    "aggregate_duplicates": proj["aggregate_duplicates"],
                    "fixed_layout": proj["fixed_layout"],
                    "df": normalize_df(proj["df"]),
                    "nodes_df": proj["nodes_df"],
                }
            )
            st.session_state.active_idx = len(st.session_state.datasets) - 1
            st.success("Project imported.")
            st.rerun()
        except Exception as e:
            st.error(f"Project import failed: {e}")

    up_csv = st.file_uploader("Import Flows (.csv)", type=["csv"], key="import_csv")
    if up_csv is not None:
        try:
            df = df_from_uploaded_csv(up_csv)
            ds = st.session_state.datasets[st.session_state.active_idx]
            ds["df"] = df
            ds["nodes_df"] = normalize_nodes_df(ds.get("nodes_df", pd.DataFrame()), get_nodes_from_flows(df))
            st.success("CSV imported into active dataset.")
            st.rerun()
        except Exception as e:
            st.error(f"CSV import failed: {e}")

    up_nodes = st.file_uploader("Import Node Layout (nodes.csv)", type=["csv"], key="import_nodes_csv")
    if up_nodes is not None:
        try:
            incoming_nodes_df = nodes_df_from_uploaded_csv(up_nodes)

            ds = st.session_state.datasets[st.session_state.active_idx]
            flow_nodes = get_nodes_from_flows(ds["df"])

            incoming_set = set(incoming_nodes_df["node"].tolist())
            flow_set = set(flow_nodes)

            unknown = sorted(incoming_set - flow_set)
            missing = sorted(flow_set - incoming_set)

            filtered = incoming_nodes_df[incoming_nodes_df["node"].isin(flow_nodes)].copy()

            base_nodes = normalize_nodes_df(ds.get("nodes_df", pd.DataFrame()), flow_nodes)

            merged = base_nodes.merge(filtered, on="node", how="left", suffixes=("_base", ""))
            merged["level"] = merged["level"].where(merged["level"].notna(), merged["level_base"]).astype(int)
            merged["order"] = merged["order"].where(merged["order"].notna(), merged["order_base"]).astype(int)

            merged = merged[["node", "level", "order"]].sort_values(["level", "order", "node"], kind="stable").reset_index(drop=True)
            ds["nodes_df"] = merged

            if unknown:
                st.warning(
                    "Some nodes in nodes.csv are not present in the flow data and were ignored: "
                    + ", ".join(unknown[:15])
                    + ("…" if len(unknown) > 15 else "")
                )
            if missing:
                st.info(
                    "Some flow nodes were not present in nodes.csv and kept default layout values: "
                    + ", ".join(missing[:15])
                    + ("…" if len(missing) > 15 else "")
                )

            st.success("Node layout imported into active dataset.")
            st.rerun()
        except Exception as e:
            st.error(f"Node layout import failed: {e}")

# Active dataset
ds = st.session_state.datasets[st.session_state.active_idx]
ds_name = ds["name"]

# Top controls
colA, colB, colC, colD = st.columns([1.2, 0.9, 0.9, 0.8])
with colA:
    ds["title"] = st.text_input("Chart title", value=ds["title"])
with colB:
    ds["value_suffix"] = st.text_input("Value suffix (optional)", value=ds["value_suffix"])
with colC:
    ds["aggregate_duplicates"] = st.checkbox(
        "Aggregate duplicate edges",
        value=bool(ds.get("aggregate_duplicates", True)),
    )
with colD:
    if st.button("Reset sample data", width="stretch"):
        ds["df"] = normalize_df(default_df())
        ds["nodes_df"] = normalize_nodes_df(pd.DataFrame(), get_nodes_from_flows(ds["df"]))
        st.rerun()

# Keep nodes table aligned with flows
nodes = get_nodes_from_flows(ds["df"])
ds["nodes_df"] = normalize_nodes_df(ds.get("nodes_df", pd.DataFrame()), nodes)

left, right = st.columns([1.1, 1.0], gap="large")

with left:
    st.subheader("Flow table")

    # IMPORTANT: pass ds["df"] directly to avoid "type twice" behavior.
    prev_df = ds["df"].copy(deep=True)

    edited_df = st.data_editor(
        ds["df"],
        num_rows="dynamic",
        width="stretch",
        column_config={
            "source": st.column_config.TextColumn("source", required=True),
            "target": st.column_config.TextColumn("target", required=True),
            "value": st.column_config.NumberColumn("value", required=True, min_value=0),
            "link_label": st.column_config.TextColumn("link_label (optional)"),
            "link_color": st.column_config.TextColumn("link_color (optional, e.g., rgba(0,0,0,0.25))"),
        },
        key=f"editor_flows_{st.session_state.active_idx}",
    )

    # Only update state if user changed something (prevents unnecessary reruns / editor resets)
    if not edited_df.equals(prev_df):
        ds["df"] = normalize_df(edited_df)
        ds["nodes_df"] = normalize_nodes_df(ds.get("nodes_df", pd.DataFrame()), get_nodes_from_flows(ds["df"]))

    st.divider()
    st.subheader("Layout control")

    ds["fixed_layout"] = st.checkbox(
        "Enable fixed layout (levels)",
        value=bool(ds.get("fixed_layout", False)),
    )

    if ds["fixed_layout"]:
        st.caption(
            "Assign each node a level (0..N). Levels map to left→right x-position. "
            "Order controls top→bottom placement within a level."
        )

        cL1, cL2, cL3 = st.columns(3)
        with cL1:
            if st.button("Auto levels (from flows)", width="stretch"):
                edges = normalize_df(ds["df"])
                edges["source"] = edges["source"].astype(str).str.strip()
                edges["target"] = edges["target"].astype(str).str.strip()
                edges["value"] = pd.to_numeric(edges["value"], errors="coerce")
                edges = edges[(edges["source"] != "") & (edges["target"] != "") & edges["value"].notna()]

                node_levels: dict[str, int] = {n: 0 for n in get_nodes_from_flows(ds["df"])}
                # Simple relaxation: push targets right of sources
                for _ in range(25):
                    changed = False
                    for s, t in zip(edges["source"], edges["target"]):
                        if node_levels.get(t, 0) <= node_levels.get(s, 0):
                            node_levels[t] = node_levels.get(s, 0) + 1
                            changed = True
                    if not changed:
                        break

                nd = ds["nodes_df"].copy()
                nd["level"] = nd["node"].map(lambda n: node_levels.get(n, 0)).astype(int)
                nd = nd.sort_values(["level", "order", "node"], kind="stable").reset_index(drop=True)
                ds["nodes_df"] = nd
                st.rerun()

        with cL2:
            if st.button("Reset levels", width="stretch"):
                nd = ds["nodes_df"].copy()
                nd["level"] = 0
                nd["order"] = list(range(len(nd)))
                ds["nodes_df"] = nd
                st.rerun()

        with cL3:
            if st.button("Refresh nodes", width="stretch"):
                ds["nodes_df"] = normalize_nodes_df(ds.get("nodes_df", pd.DataFrame()), get_nodes_from_flows(ds["df"]))
                st.rerun()

        ds["nodes_df"] = st.data_editor(
            ds["nodes_df"],
            num_rows="fixed",
            width="stretch",
            column_config={
                "node": st.column_config.TextColumn("node", disabled=True),
                "level": st.column_config.NumberColumn("level", min_value=0, step=1),
                "order": st.column_config.NumberColumn("order", min_value=0, step=1),
            },
            key=f"editor_nodes_{st.session_state.active_idx}",
        )

    st.divider()
    st.subheader("Export / Save")

    fig = None
    build_error = None
    try:
        plot_df = ds["df"]
        if ds["aggregate_duplicates"]:
            plot_df = aggregate_edges(plot_df)

        fig = build_sankey(
            plot_df,
            title=ds["title"],
            value_suffix=ds["value_suffix"],
            use_fixed_layout=bool(ds.get("fixed_layout", False)),
            nodes_df=ds.get("nodes_df", pd.DataFrame()),
        )
    except Exception as e:
        build_error = str(e)

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.download_button(
            "Export CSV",
            data=df_to_csv_bytes(ds["df"]),
            file_name=f"{ds_name.replace(' ', '_').lower()}_flows.csv",
            mime="text/csv",
            width="stretch",
        )

    with e2:
        if fig is not None:
            st.download_button(
                "Save chart (HTML)",
                data=fig_to_html_bytes(fig),
                file_name=f"{ds_name.replace(' ', '_').lower()}.html",
                mime="text/html",
                width="stretch",
            )
        else:
            st.button("Save chart (HTML)", disabled=True, width="stretch")

    with e3:
        if fig is not None:
            try:
                png = fig_to_png_bytes(fig)
                st.download_button(
                    "Export PNG",
                    data=png,
                    file_name=f"{ds_name.replace(' ', '_').lower()}.png",
                    mime="image/png",
                    width="stretch",
                )
            except Exception as e:
                st.button("Export PNG", disabled=True, width="stretch")
                st.caption(f"PNG export unavailable (kaleido?): {e}")
        else:
            st.button("Export PNG", disabled=True, width="stretch")

    with e4:
        proj_bytes = project_to_json_bytes(
            name=ds_name,
            title=ds["title"],
            value_suffix=ds["value_suffix"],
            df=ds["df"],
            aggregate_dupes=bool(ds.get("aggregate_duplicates", False)),
            use_fixed_layout=bool(ds.get("fixed_layout", False)),
            nodes_df=ds.get("nodes_df", pd.DataFrame()),
        )
        st.download_button(
            "Export Project (.json)",
            data=proj_bytes,
            file_name=f"{ds_name.replace(' ', '_').lower()}.json",
            mime="application/json",
            width="stretch",
        )

    # Optional: export node layout CSV for fixed layout usage
    if ds.get("nodes_df") is not None and not ds["nodes_df"].empty:
        st.download_button(
            "Export Node Layout (nodes.csv)",
            data=nodes_df_to_csv_bytes(ds["nodes_df"]),
            file_name=f"{ds_name.replace(' ', '_').lower()}_nodes.csv",
            mime="text/csv",
            width="stretch",
        )

with right:
    st.subheader("Preview")

    v = validate_flows(ds["df"])
    with st.expander("Validation", expanded=True):
        st.write(
            "Rows: **{rows}**  |  Valid rows: **{valid}**  |  Nodes: **{nodes}**  |  Total flow: **{total:g}**".format(
                rows=v.stats.get("rows", 0),
                valid=v.stats.get("valid_rows", 0),
                nodes=v.stats.get("nodes", 0),
                total=v.stats.get("total_flow", 0.0),
            )
        )
        if v.errors:
            st.error("\n".join(v.errors))
        if v.warnings:
            st.warning("\n".join(v.warnings))
        if not v.errors and not v.warnings:
            st.success("No issues detected.")

        if ds.get("aggregate_duplicates", False):
            st.caption("Aggregation is enabled: repeated (source,target) edges are summed for rendering.")

    if build_error:
        st.error(build_error)
        st.info("Fix the flow table values and the chart will render.")
    else:
        st.plotly_chart(fig, width="stretch")

st.caption("CSV tip: If any field contains commas in text fields, wrap those fields in double quotes in the CSV.")
