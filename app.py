from __future__ import annotations
import json
import os
import shlex
import subprocess
import time
from collections import defaultdict
from typing import Dict, Any, Iterable, Tuple, List, Set

import streamlit as st
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship

from pyvis.network import Network
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node as ANode, Edge as AEdge, Config as AConfig
from pathlib import Path
import csv as csv_mod
from io import StringIO
import shutil
from pathlib import Path


def _json_safe_value(v):
    if isinstance(v, Node):
        return {
            "__type__": "Node",
            "id": getattr(v, "element_id", None) or str(getattr(v, "id", None)),
            "labels": list(v.labels),
            "properties": dict(v.items()),
        }
    if isinstance(v, Relationship):
        return {
            "__type__": "Relationship",
            "type": v.type,
            "start": getattr(v.start_node, "element_id", None) or str(getattr(v.start_node, "id", None)),
            "end":   getattr(v.end_node,   "element_id", None) or str(getattr(v.end_node,   "id", None)),
            "properties": dict(v.items()),
        }
    if isinstance(v, Path):
        # represent as alternating list of nodes/rels
        parts = []
        for n in v.nodes:
            parts.append(_json_safe_value(n))
        # relationships are in order but not interleaved by default; optional to include:
        parts += [_json_safe_value(r) for r in v.relationships]
        return {"__type__": "Path", "parts": parts}

    if isinstance(v, list):
        return [_json_safe_value(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_json_safe_value(x) for x in v)
    if isinstance(v, dict):
        return {k: _json_safe_value(val) for k, val in v.items()}
    return v  # primitives are fine

def json_safe_rows(rows: list[dict]) -> list[dict]:
    return [{k: _json_safe_value(v) for k, v in row.items()} for row in rows]


def _stringify(v):
    # make nested lists/dicts CSV-safe
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return v

def nodes_to_table(nodes: List[Node]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return rows + header for nodes as a flat table."""
    rows = []
    all_keys: Set[str] = set()
    for n in nodes:
        nid = getattr(n, "element_id", None) or str(getattr(n, "id", None))
        labels = ":".join(list(n.labels))
        props = dict(n.items())
        row = {"id": nid, "labels": labels, **{k: _stringify(v) for k, v in props.items()}}
        rows.append(row)
        all_keys.update(row.keys())
    # stable ordering
    header = ["id", "labels"] + sorted([k for k in all_keys if k not in {"id", "labels"}])
    # ensure all rows have all columns
    for r in rows:
        for k in header:
            r.setdefault(k, "")
    return rows, header

def rels_to_table(rels: List[Relationship]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return rows + header for relationships as a flat table."""
    rows = []
    all_keys: Set[str] = set()
    for r in rels:
        sid = getattr(r.start_node, "element_id", None) or str(getattr(r.start_node, "id", None))
        tid = getattr(r.end_node,   "element_id", None) or str(getattr(r.end_node,   "id", None))
        props = dict(r.items())
        row = {
            "start_id": sid,
            "end_id": tid,
            "type": r.type,
            **{k: _stringify(v) for k, v in props.items()},
        }
        rows.append(row)
        all_keys.update(row.keys())
    header = ["start_id", "end_id", "type"] + sorted([k for k in all_keys if k not in {"start_id", "end_id", "type"}])
    for r in rows:
        for k in header:
            r.setdefault(k, "")
    return rows, header

def to_csv(rows: List[Dict[str, Any]], header: List[str]) -> str:
    buf = StringIO()
    w = csv_mod.DictWriter(buf, fieldnames=header, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in header})
    return buf.getvalue()


# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="Neo4j Graph Studio + CYP Loader", page_icon="🧬", layout="wide")
st.title("🧬 Neo4j Graph Studio (CYP4Z1 Example Loader)")

# --------------------------- Connection (sidebar) ---------------------------
with st.sidebar:
    st.subheader("Connection")
    uri = st.text_input("BOLT URI", value=st.secrets.get("NEO4J_URI", "neo4j+s://<your-aura-host>:7687"))
    user = st.text_input("Username", value=st.secrets.get("NEO4J_USER", "neo4j"))
    password = st.text_input("Password", type="password", value=st.secrets.get("NEO4J_PASSWORD", ""))
    database = st.text_input("Database", value=st.secrets.get("NEO4J_DATABASE", "neo4j"))
    connect_btn = st.button("Connect / Reconnect", type="primary")

@st.cache_resource(show_spinner=False)
def get_driver(uri_: str, user_: str, pw_: str):
    return GraphDatabase.driver(uri_, auth=(user_, pw_))

if "driver" not in st.session_state and uri and password:
    try:
        st.session_state.driver = get_driver(uri, user, password)
        st.session_state.db = database or "neo4j"
    except Exception as e:
        st.warning(f"Auto-connect failed: {e}")

if connect_btn:
    try:
        st.session_state.driver = get_driver(uri, user, password)
        st.session_state.db = database or "neo4j"
        st.success("Connected.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

driver = st.session_state.get("driver")
db = st.session_state.get("db", "neo4j")

# --------------------------- Neo4j helpers ---------------------------
def run_cypher(q: str, params: Dict[str, Any] | None = None):
    if not driver:
        return [], []
    with driver.session(database=db) as session:
        rs = session.run(q, params or {})
        return [dict(r) for r in rs], list(rs.keys())

# --------------------------- Graph extraction ---------------------------
def extract_graph(records: Iterable[dict]) -> Tuple[List[Node], List[Relationship]]:
    nodes: Dict[str, Node] = {}
    rels: List[Relationship] = []

    def add_node(n: Node | None):
        if not n: return
        nid = getattr(n, "element_id", None) or str(getattr(n, "id", None))
        if nid and nid not in nodes: nodes[nid] = n

    for rec in records:
        for v in rec.values():
            if isinstance(v, Node):
                add_node(v)
            elif isinstance(v, Relationship):
                rels.append(v); add_node(v.start_node); add_node(v.end_node)
            elif isinstance(v, (list, tuple)):
                for it in v:
                    if isinstance(it, Node): add_node(it)
                    elif isinstance(it, Relationship):
                        rels.append(it); add_node(it.start_node); add_node(it.end_node)
            elif isinstance(v, dict):
                for it in v.values():
                    if isinstance(it, Node): add_node(it)
                    elif isinstance(it, Relationship):
                        rels.append(it); add_node(it.start_node); add_node(it.end_node)
    return list(nodes.values()), rels

# --------------------------- Visualizers ---------------------------
def render_pyvis(nodes: List[Node], rels: List[Relationship], height=650, physics=True, hierarchical=False):
    net = Network(height=f"{height}px", width="100%", directed=True)

    if hierarchical:
        options = {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "levelSeparation": 150,
                    "nodeSpacing": 120,
                    "direction": "UD",
                }
            },
            "physics": {"enabled": False},
            "edges": {"smooth": False},
        }
    else:
        options = {
            "physics": {
                "enabled": bool(physics),
                "stabilization": {"iterations": 200},
            },
            "edges": {"smooth": False},
        }

    net.set_options(json.dumps(options))  # <-- valid JSON, not JS

    # degree for sizing
    deg = defaultdict(int)
    for r in rels:
        sid = getattr(r.start_node, "element_id", None) or str(getattr(r.start_node, "id", None))
        tid = getattr(r.end_node, "element_id", None) or str(getattr(r.end_node, "id", None))
        deg[sid] += 1
        deg[tid] += 1

    added: Set[str] = set()
    for n in nodes:
        nid = getattr(n, "element_id", None) or str(getattr(n, "id", None))
        if not nid or nid in added:
            continue
        added.add(nid)
        labels = ":".join(list(n.labels))
        props = dict(n.items())
        label_text = props.get("name") or props.get("title") or props.get("id") or labels or nid
        size = 12 + min(28, deg.get(nid, 0) * 2)
        title = "<br>".join([f"<b>{labels}</b>"] + [f"{k}: {v}" for k, v in props.items()])
        net.add_node(nid, label=str(label_text), title=title, shape="dot", value=size)

    for r in rels:
        sid = getattr(r.start_node, "element_id", None) or str(getattr(r.start_node, "id", None))
        tid = getattr(r.end_node, "element_id", None) or str(getattr(r.end_node, "id", None))
        net.add_edge(sid, tid, label=r.type)

    components.html(net.generate_html(), height=height, scrolling=True)
    
def render_agraph(nodes: List[Node], rels: List[Relationship], height=650, physics=True):
    a_nodes: List[ANode] = []; a_edges: List[AEdge] = []
    deg = defaultdict(int)
    for r in rels:
        sid = getattr(r.start_node, "element_id", None) or str(getattr(r.start_node, "id", None))
        tid = getattr(r.end_node, "element_id", None) or str(getattr(r.end_node, "id", None))
        deg[sid]+=1; deg[tid]+=1
    seen: Set[str] = set()
    for n in nodes:
        nid = getattr(n, "element_id", None) or str(getattr(n, "id", None))
        if not nid or nid in seen: continue
        seen.add(nid)
        labels = ":".join(list(n.labels)); props = dict(n.items())
        label_text = props.get("name") or props.get("title") or props.get("id") or labels or nid
        size = 10 + min(26, deg.get(nid, 0) * 2)
        a_nodes.append(ANode(id=nid, label=str(label_text), title=f"{labels}\n{json.dumps(props, ensure_ascii=False)}", size=size))
    for r in rels:
        sid = getattr(r.start_node, "element_id", None) or str(getattr(r.start_node, "id", None))
        tid = getattr(r.end_node, "element_id", None) or str(getattr(r.end_node, "id", None))
        a_edges.append(AEdge(source=sid, target=tid, label=r.type))
    cfg = AConfig(height=height, width="100%", directed=True, physics=physics, hierarchical=False)
    agraph(nodes=a_nodes, edges=a_edges, config=cfg)

# --------------------------- Pipeline commands ---------------------------
def commands_for_example(neo4j_uri: str, neo4j_password: str) -> List[List[str]]:
    u = neo4j_uri; pw = neo4j_password
    cmds = [
        "setup-data-folder",
        "collect-process-nodes --node_type Compound --enzyme_list CYP4Z1 --start_chunk 0",
        "collect-process-nodes --node_type BioAssay --enzyme_list CYP4Z1 --start_chunk 0",
        "collect-process-nodes --node_type Gene --enzyme_list CYP4Z1 --start_chunk 0",
        "collect-process-nodes --node_type Protein --enzyme_list CYP4Z1 --start_chunk 0",
        "collect-process-relationships --relationship_type Assay_Compound --start_chunk 0",
        "collect-process-relationships --relationship_type Assay_Gene --start_chunk 0",
        "collect-process-relationships --relationship_type Gene_Protein --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Gene --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Similarity --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Compound_Cooccurrence --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Gene_Cooccurrence --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Gene_Interaction --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Transformation --start_chunk 0",
        f"load-graph-nodes --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --label Compound",
        f"load-graph-nodes --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --label Compound",
        f"load-graph-nodes --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --label BioAssay",
        f"load-graph-nodes --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --label Protein",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Assay_Compound",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Assay_Gene",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Gene_Protein",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Compound_Gene",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Compound_Similarity",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Compound_Gene_Interaction",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Compound_Gene_CoOccurrence",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Compound_Compound_CoOccurrence",
        f"load-graph-relationships --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --relationship_type Compound_Transformation",
    ]
    return [shlex.split(c) for c in cmds]

def safe_rename(src: str, dst: str, cwd: str, retries: int = 10, delay: float = 0.2):
    """Cross-platform rename with light retry (helps on Windows if a writer closes late)."""
    base = Path(cwd)
    src_p = (base / src).resolve()
    dst_p = (base / dst).resolve()

    # Optional: wait until src appears (in case previous step is still flushing)
    for _ in range(retries):
        if src_p.exists():
            break
        time.sleep(delay)
    if not src_p.exists():
        raise FileNotFoundError(f"Rename source not found: {src_p}")

    # If destination exists, remove it so rename succeeds
    if dst_p.exists():
        dst_p.unlink()

    src_p.rename(dst_p)


def run_pipeline(commands: list[list[str]], extra_env: dict[str, str] | None = None, *, cwd: str):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    progress = st.progress(0)
    total = len(commands)
    logs: list[dict] = []

    st.write("**Starting CYP4Z1 example loader…**")
    for i, argv in enumerate(commands, start=1):
        cmd_str = " ".join(shlex.quote(a) for a in argv)
        st.write(f"**Step {i}/{total}** → `{cmd_str}`")
        t0 = time.perf_counter()

        # ---- intercept rename steps (portable) ----
        if argv and argv[0].lower() in {"mv", "move", "ren", "rename"}:
            if len(argv) != 3:
                raise RuntimeError(f"Rename expects 2 args (src dst). Got: {' '.join(argv)}")
            try:
                safe_rename(argv[1], argv[2], cwd=cwd)
                dt = time.perf_counter() - t0
                logs.append({"cmd": cmd_str, "returncode": 0, "seconds": round(dt, 3)})
                st.success(f"✅ Renamed {argv[1]} → {argv[2]} in {dt:.2f}s")
                progress.progress(int(i * 100 / total))
                continue  # IMPORTANT: don’t fall through to Popen on mv/ren
            except Exception as e:
                st.error(f"❌ Rename failed: {e}")
                raise

        # ---- normal external command ----
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                cwd=cwd,            # ensure Data/... resolves correctly
                shell=False,        # keep explicit argv execution
            )
        except FileNotFoundError:
            st.error(f"❌ Command not found: `{argv[0]}`. Ensure it’s installed and on PATH.")
            raise

        out_lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            out_lines.append(line)
            st.write(line)

        ret = proc.wait()
        dt = time.perf_counter() - t0
        logs.append({"cmd": cmd_str, "returncode": ret, "seconds": round(dt, 3)})

        if ret != 0:
            st.error(f"❌ Failed (exit {ret}) in {dt:.2f}s")
            raise RuntimeError(f"Command failed: {cmd_str}")

        st.success(f"✅ Finished in {dt:.2f}s")
        progress.progress(int(i * 100 / total))

    st.success("🎉 CYP4Z1 pipeline completed.")
    return logs

# --------------------------- Tabs (no expanders) ---------------------------
tab_loader, tab_workbench, tab_explorer, tab_visual = st.tabs(
    ["Example Loader", "Query Workbench", "Subgraph Explorer", "Visualize"]
)

# ----- Example Loader (no expanders) -----
with tab_loader:
    st.subheader("Run Example (CYP4Z1) — Multi-command Loader")
    colA, colB, colC = st.columns([2, 2, 2])
    with colA:
        wipe_first = st.toggle("Wipe existing graph first", value=False, help="Runs MATCH (n) DETACH DELETE n")
    with colB:
        show_cmds = st.toggle("Show commands before running", value=True)
    with colC:
        confirm = st.text_input("Type 'yes' to confirm", value="")
    run_btn = st.button("Run example pipeline", type="primary", use_container_width=True)

    if run_btn:
        if not uri or not password:
            st.error("Provide Aura URI/password in the sidebar.")
        elif confirm.lower() != "yes":
            st.warning("Please type 'yes' to confirm.")
        else:
            try:
                if wipe_first and driver:
                    run_cypher("MATCH (n) DETACH DELETE n")
                    st.info("Graph wiped.")
                cmds = commands_for_example(uri, password)
                target = Path("Data")
                if target.exists():
                    shutil.rmtree(target)
                    print(f"Removed folder: {target}")
                if show_cmds:
                    st.code("\n".join(" ".join(map(shlex.quote, c)) for c in cmds), language="bash")
                env = {"NEO4J_URI": uri, "NEO4J_PASSWORD": password, "NEO4J_USER": user or "neo4j", "NEO4J_DATABASE": db}
                logs = run_pipeline(cmds, extra_env=env, cwd=".")
            except Exception as e:
                st.error(f"Pipeline error: {e}")

# ----- Query Workbench -----
with tab_workbench:
    st.subheader("Cypher Workbench")
    c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
    with c1:
        preset = st.selectbox("Preset", [
            "MATCH (n) RETURN n LIMIT $limit",
            "MATCH (n)-[r]->(m) RETURN n,r,m LIMIT $limit",
            "MATCH p=(n)-[r]->(m) RETURN p LIMIT $limit",
            "CALL db.labels() YIELD label RETURN label LIMIT $limit",
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType LIMIT $limit",
        ], index=1)
    with c2:
        limit = st.slider("LIMIT", 25, 2000, 200, 25)
    with c3:
        profile = st.toggle("PROFILE", False)
    with c4:
        params_text = st.text_input("Params (JSON)", value=json.dumps({"limit": limit}))
        try:
            p = json.loads(params_text or "{}"); p["limit"] = limit; params_text = json.dumps(p)
        except Exception:
            pass

    cypher = st.text_area("Cypher", value=preset, height=140)
    try:
        params = json.loads(params_text or "{}")
    except Exception as e:
        params = {}
        st.warning(f"Params JSON invalid: {e}")

    # ---------- helpers ----------
    @st.cache_data(show_spinner=False)
    def _build_blobs(safe_rows: List[dict]) -> tuple[str, str, List[str]]:
        keys = list(safe_rows[0].keys()) if safe_rows else []
        json_blob = json.dumps(safe_rows, ensure_ascii=False, indent=2)
        def cell(v): return json.dumps(v, ensure_ascii=False)
        csv_text = ",".join(keys) + "\n" + "\n".join([",".join(cell(r.get(k)) for k in keys) for r in safe_rows])
        return json_blob, csv_text, keys

    go = st.button("Run query", type="primary")

    if go and driver:
        q = ("PROFILE " + cypher) if (profile and not cypher.strip().upper().startswith(("PROFILE","EXPLAIN"))) else cypher
        try:
            # Do the heavy work ONLY on button click
            rows, keys_from_db = run_cypher(q, params)
            st.session_state["_last_rows"] = rows

            safe = json_safe_rows(rows)
            json_blob, csv_text, keys = _build_blobs(safe)
            st.session_state["_qw_safe"] = safe
            st.session_state["_qw_json_blob"] = json_blob
            st.session_state["_qw_csv_text"] = csv_text
            st.session_state["_qw_keys"] = keys

            st.success(f"Returned {len(rows)} row(s).")

        except Exception as e:
            st.error(f"Query error: {e}")

    # ---------- reuse results on rerun (e.g., when clicking download) ----------
    safe = st.session_state.get("_qw_safe", [])
    json_blob = st.session_state.get("_qw_json_blob", "")
    csv_text = st.session_state.get("_qw_csv_text", "")
    keys = st.session_state.get("_qw_keys", [])

    if safe:
        st.dataframe(safe, use_container_width=True)
        d1, d2 = st.columns([1,1])
        with d1:
            st.download_button("⬇️ JSON", data=json_blob, file_name="cypher_result.json", mime="application/json", key="qw_json_dl", use_container_width=True)
        with d2: 
            st.download_button("⬇️ CSV", data=csv_text, file_name="cypher_result.csv", mime="text/csv", key="qw_csv_dl", use_container_width=True)

# ----- Subgraph Explorer -----
with tab_explorer:
    st.subheader("Subgraph Explorer")
    colx, coly, colz = st.columns([3, 2, 2])
    with colx:
        start_label = st.text_input("Start label (optional)", "")
        start_key   = st.text_input("Match key", "name")
        start_val   = st.text_input("Match value (optional)", "")
    with coly:
        hops = st.slider("Hops", 1, 3, 2)
    with colz:
        sub_limit = st.slider("Max rows", 50, 2000, 400, 50)

    fetch = st.button("Fetch subgraph", type="secondary")
    if fetch:
        try:
            if start_label and start_val:
                q = f"""
                MATCH (s:{start_label} {{{start_key}:$v}})
                CALL apoc.path.expandConfig(s, {{maxLevel:{hops}}}) YIELD path
                WITH path LIMIT $limit
                UNWIND nodes(path) AS n
                WITH DISTINCT n LIMIT $limit
                MATCH (n)-[r]-(m)
                RETURN n,r,m LIMIT $limit
                """
                rows, _ = run_cypher(q, {"v": start_val, "limit": sub_limit})
            else:
                rows, _ = run_cypher("MATCH (n)-[r]-(m) RETURN n,r,m LIMIT $limit", {"limit": sub_limit})
            st.success(f"Fetched {len(rows)} row(s).")
            st.session_state["_last_rows"] = rows
        except Exception as e:
            st.error(f"Subgraph error: {e}")

# ----- Visualize -----
with tab_visual:
    st.subheader("Visualize")
    rows = st.session_state.get("_last_rows", [])
    if not rows:
        st.info("Run a query (or fetch a subgraph) first.")
    else:
        v1, v2, v3, v4 = st.columns([1,1,1,1])
        with v1: renderer = st.selectbox("Renderer", ["Agraph (fast)", "PyVis (rich)"], index=0)
        with v2: physics = st.toggle("Enable Graph Physics", True)
        with v3: hierarchical = st.toggle("Hierarchical (PyVis only)", False)
        with v4: edge_cap = st.slider("Edge cap", 100, 5000, 2000, 100)

        nodes, rels = extract_graph(rows)
        # Build CSVs for downloads
        node_rows, node_header = nodes_to_table(nodes)
        rel_rows, rel_header   = rels_to_table(rels)
        nodes_csv = to_csv(node_rows, node_header)
        rels_csv  = to_csv(rel_rows, rel_header)

        dlc1, dlc2 = st.columns(2)
        with dlc1:
            st.download_button(
                "⬇️ Download Nodes (CSV)",
                data=nodes_csv,
                file_name="nodes.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dlc2:
            st.download_button(
                "⬇️ Download Relationships (CSV)",
                data=rels_csv,
                file_name="relationships.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if len(rels) > edge_cap: rels = rels[:edge_cap]
        st.caption(f"Rendering {len(nodes)} nodes / {len(rels)} relationships")

        if renderer.startswith("Agraph"):
            render_agraph(nodes, rels, height=650, physics=physics)
        else:
            render_pyvis(nodes, rels, height=650, physics=physics, hierarchical=hierarchical)
