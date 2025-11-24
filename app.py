from __future__ import annotations
import base64
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
from pathlib import Path
import csv as csv_mod
from io import StringIO
import shutil
import re


# Simple color mapping by label
LABEL_COLOR = {
    "Compound": "#4C78A8",
    "BioAssay": "#F58518",
    "Gene": "#54A24B",
    "Protein": "#E45756",
}

def _node_color(n: Node) -> str:
    """Pick a color based on the main label of the node."""
    for lab in n.labels:
        if lab in LABEL_COLOR:
            return LABEL_COLOR[lab]
    return "#9E9E9E"  # default gray

# compile once
_SECRET_FLAGS = {"--password", "--pass", "--pw", "--token", "--apikey", "--api-key", "--key", "--secret"}
_ENV_SECRET_KEYS = re.compile(r"(PASS(WORD)?|TOKEN|API[_-]?KEY|SECRET)", re.I)

# For URIs like neo4j+s://user:pass@host:7687 or schemes with credentials
_URI_CRED_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<user>[^:/@]+):(?P<pw>[^@]+)@")

# --------------------------- Config helpers ---------------------------
def _parse_enzyme_list(s: str) -> list[str]:
    # normalize: split comma/whitespace, uppercase, keep CYP-like tokens
    toks = [t.strip().upper() for t in re.split(r"[,\s]+", s or "") if t.strip()]
    # Very light validation; keep alnum/_/-
    return [t for t in toks if re.fullmatch(r"[A-Z0-9_\-]+", t)]


def _mask_uri(s: str) -> str:
    s = "" if s is None else (s if isinstance(s, str) else str(s))
    return _URI_CRED_RE.sub(lambda m: f"{m.group('scheme')}***:***@", s)

def _mask_equals_form(s: str) -> str:
    s = "" if s is None else (s if isinstance(s, str) else str(s))
    return re.sub(r"(--(?:password|pass|pw|token|apikey|api-key|key|secret))\s*=\s*([^\s]+)", r"\1=***", s, flags=re.I)

def redact_text(s: str) -> str:
    s = "" if s is None else (s if isinstance(s, str) else str(s))
    s = _mask_uri(s)
    s = _mask_equals_form(s)
    s = re.sub(r"(?i)(password|pass|pw|token|apikey|api-key|key|secret)\s*=\s*([^\s&]+)", r"\1=***", s)
    return s


def redact_argv(argv: list[str]) -> str:
    """Return a printable (masked) version of argv."""
    masked: list[str] = []
    it = iter(range(len(argv)))
    i = 0
    while i < len(argv):
        a = argv[i]
        a_masked = _mask_equals_form(_mask_uri(a))
        if a.lower() in _SECRET_FLAGS:
            masked.append(a)            # keep the flag
            if i + 1 < len(argv):
                masked.append("***")    # hide the next token
                i += 2
                continue
        masked.append(a_masked)
        i += 1
    return " ".join(shlex.quote(x) for x in masked)

def redact_env(env: dict[str, str]) -> dict[str, str]:
    """Return a shallow redacted copy of env for display."""
    shown = {}
    for k, v in env.items():
        if _ENV_SECRET_KEYS.search(k):
            shown[k] = "***"
        else:
            shown[k] = _mask_uri(v)
    return shown


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
st.set_page_config(page_title="Build Neo4J Graph using ChemGraphBuilder", layout="wide")
st.title("Build Neo4J Graph using ChemGraphBuilder")

# --------------------------- Connection (sidebar) ---------------------------
file_ = open("images/kg_icon.webp", "rb").read()
base64_image = base64.b64encode(file_).decode("utf-8")
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; padding-bottom: 10px;">
        <!-- Logo -->
        <div style="display: flex; align-items: center; margin-right: 10px;">
            <img src="data:image/png;base64,{base64_image}" alt="Logo" width="100" style="border-radius: 5px;">
        </div>
        <!-- Separator -->
        <div style="width: 4px; height: 50px; background-color: #ccc; margin-right: 10px;"></div>
        <!-- Text -->
        <div style="font-size: 20px; font-weight: bold; color: #112f5f;">
            ChemGraphBuilder
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
# --------------------------- Connection (sidebar) ---------------------------
with st.sidebar:
    st.subheader("Provide Neo4j Aura Credentials")

    source = st.radio(
        "Credential source",
        ["Enter manually", "Use st.secrets"],
        horizontal=True,
        index=0,
        help="Select where to read the connection details from.",
    )

    # Defaults (prefer secrets if available)
    default_uri  = "neo4j+s://<your-aura-host>:7687"
    default_user = "neo4j"
    default_db   = "neo4j"
    default_pwd  = ""
    default_enz  = st.secrets.get("ENZYMES")

    if source == "Use st.secrets":
        uri_input  = st.secrets.get("NEO4J_URI")
        user_input = st.secrets.get("NEO4J_USER")
        db_input   = st.secrets.get("NEO4J_DATABASE")
        pwd_input  = st.secrets.get("NEO4J_PASSWORD")
        enz_input  = default_enz
        st.caption("Using credentials from st.secrets (values are hidden).")
    else:
        uri_input  = st.text_input("Aura URI", value=default_uri)
        user_input = st.text_input("Username", value=default_user)
        pwd_input  = st.text_input("Password / Token", type="password", value=default_pwd)
        db_input   = st.text_input("Database", value=default_db)
        enz_input  = st.text_input("Enzyme list (comma-separated)", help="Example: CYP2J2,CYP2C9,CYP3A4")

    # Store in session for reuse everywhere
    st.session_state["conn_uri"]  = uri_input
    st.session_state["conn_user"] = user_input
    st.session_state["conn_pwd"]  = pwd_input
    st.session_state["conn_db"]   = db_input
    st.session_state["enz_list"]  = enz_input

    connect_btn = st.button("Connect / Reconnect", type="primary", use_container_width=True)

@st.cache_resource(show_spinner=False)
def get_driver(uri_: str, user_: str, pw_: str):
    return GraphDatabase.driver(uri_, auth=(user_, pw_))

# Attempt auto-connect if uri + pwd are present
if "driver" not in st.session_state and st.session_state.get("conn_uri") and st.session_state.get("conn_pwd"):
    try:
        st.session_state.driver = get_driver(
            st.session_state["conn_uri"],
            st.session_state["conn_user"],
            st.session_state["conn_pwd"]
        )
        st.session_state.db = st.session_state.get("conn_db") or "neo4j"
    except Exception as e:
        st.warning(f"Auto-connect failed: {e}")

if connect_btn:
    try:
        st.session_state.driver = get_driver(
            st.session_state["conn_uri"],
            st.session_state["conn_user"],
            st.session_state["conn_pwd"]
        )
        st.session_state.db = st.session_state.get("conn_db") or "neo4j"
        st.success("Connected.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

driver = st.session_state.get("driver")
db = st.session_state.get("conn_db", "neo4j")

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
        if not n:
            return
        nid = getattr(n, "element_id", None) or str(getattr(n, "id", None))
        if nid and nid not in nodes: 
            nodes[nid] = n

    for rec in records:
        for v in rec.values():
            if isinstance(v, Node):
                add_node(v)
            elif isinstance(v, Relationship):
                rels.append(v)
                add_node(v.start_node)
                add_node(v.end_node)
            elif isinstance(v, (list, tuple)):
                for it in v:
                    if isinstance(it, Node): 
                        add_node(it)
                    elif isinstance(it, Relationship):
                        rels.append(it)
                        add_node(it.start_node)
                        add_node(it.end_node)
            elif isinstance(v, dict):
                for it in v.values():
                    if isinstance(it, Node): 
                        add_node(it)
                    elif isinstance(it, Relationship):
                        rels.append(it)
                        add_node(it.start_node)
                        add_node(it.end_node)
    return list(nodes.values()), rels

# --------------------------- Visualizers ---------------------------
def render_pyvis(
    nodes: List[Node],
    rels: List[Relationship],
    height: int = 650,
    physics: bool = True,
    hierarchical: bool = False,
    show_labels: bool = True,
    min_label_degree: int = 1,
):
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
            "nodes": {
                "shape": "dot",
                "scaling": {"min": 5, "max": 35},
                "font": {"size": 12, "multi": "html"},
            },
            "edges": {
                "smooth": False,
                "arrows": {"to": {"enabled": True}},
                "font": {"size": 9, "align": "top"},
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True,
                "keyboard": True,
            },
        }
    else:
        options = {
            "physics": {
                "enabled": bool(physics),
                "stabilization": {"iterations": 300},
                "barnesHut": {"springLength": 120},
            },
            "nodes": {
                "shape": "dot",
                "scaling": {"min": 5, "max": 35},
                "font": {"size": 12, "multi": "html"},
            },
            "edges": {
                "smooth": {"type": "dynamic"},
                "arrows": {"to": {"enabled": True}},
                "font": {"size": 9, "align": "top"},
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True,
                "keyboard": True,
            },
        }

    net.set_options(json.dumps(options))

    # degree for sizing & label threshold
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

        # Tooltip (rich info)
        title = "<br>".join(
            [f"<b>{labels}</b>"]
            + [f"<b>{k}</b>: {v}" for k, v in props.items()]
        )

        # Label text – optional + only for higher-degree nodes
        base_label = props.get("name") or props.get("title") or props.get("id") or labels or nid
        if show_labels and deg.get(nid, 0) >= min_label_degree:
            label_text = str(base_label)
        else:
            label_text = ""  # no text on the node, just tooltip

        # Size based on degree (but modest)
        size = 5 + min(30, deg.get(nid, 0) * 1.5)

        net.add_node(
            nid,
            label=label_text,
            title=title,
            value=size,
            color=_node_color(n),
        )

    for r in rels:
        sid = getattr(r.start_node, "element_id", None) or str(getattr(r.start_node, "id", None))
        tid = getattr(r.end_node, "element_id", None) or str(getattr(r.end_node, "id", None))
        # Edge labels can be noisy; keep them but small (font options above)
        net.add_edge(sid, tid, label=r.type)

    components.html(net.generate_html(), height=height, scrolling=True)

    
# --------------------------- Pipeline commands ---------------------------
def commands_for_run(neo4j_uri: str, neo4j_password: str, enzymes: list[str] | str = None) -> List[List[str]]:
    """
    Build a runnable command list for the provided enzyme symbols.
    `enzymes` can be a list like ["CYP4X1","CYP4Z1"] or a comma-separated string.
    """
    # enzymes = enzymes or []
    # if isinstance(enzymes, str):
    #     enzymes = [t.strip() for t in enzymes.split(",") if t.strip()]
    # enzymes = ",".join(enzymes)

    u = neo4j_uri
    pw = neo4j_password

    core = [
        "setup-data-folder",
        f"collect-process-nodes --node_type Compound --enzyme_list {enzymes} --start_chunk 0",
        f"collect-process-nodes --node_type BioAssay --enzyme_list {enzymes} --start_chunk 0",
        f"collect-process-nodes --node_type Gene --enzyme_list {enzymes} --start_chunk 0",
        f"collect-process-nodes --node_type Protein --enzyme_list {enzymes} --start_chunk 0",
        "collect-process-relationships --relationship_type Assay_Compound --start_chunk 0",
        "collect-process-relationships --relationship_type Assay_Gene --start_chunk 0",
        "collect-process-relationships --relationship_type Gene_Protein --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Gene --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Similarity --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Compound_Cooccurrence --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Gene_Cooccurrence --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Gene_Interaction --start_chunk 0",
        "collect-process-relationships --relationship_type Compound_Transformation --start_chunk 0",
    ]

    loads = [
        f"load-graph-nodes --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --label Compound",
        f"load-graph-nodes --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --label BioAssay",
        f"load-graph-nodes --uri {shlex.quote(u)} --username neo4j --password {shlex.quote(pw)} --label Gene",
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

    return [shlex.split(c) for c in (core + loads)]


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

    st.write("**Starting {} example loader…**".format(st.secrets.get('ENZYMES')))
    for i, argv in enumerate(commands, start=1):
        cmd_str_safe = redact_argv(argv)
        st.write(f"**Step {i}/{total}** → `{cmd_str_safe}`")

        t0 = time.perf_counter()

        # ---- intercept rename steps (portable) ----
        if argv and argv[0].lower() in {"mv", "move", "ren", "rename"}:
            if len(argv) != 3:
                raise RuntimeError(f"Rename expects 2 args (src dst). Got: {' '.join(argv)}")
            try:
                safe_rename(argv[1], argv[2], cwd=cwd)
                dt = time.perf_counter() - t0
                logs.append({"cmd": cmd_str_safe, "returncode": 0, "seconds": round(dt, 3)})
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
            st.write(redact_text(line))


        ret = proc.wait()
        dt = time.perf_counter() - t0
        logs.append({"cmd": cmd_str_safe, "returncode": ret, "seconds": round(dt, 3)})

        if ret != 0:
            st.error(f"❌ Failed (exit {ret}) in {dt:.2f}s")
            raise RuntimeError(f"Command failed: {cmd_str_safe}")

        st.success(f"✅ Finished in {dt:.2f}s")
        progress.progress(int(i * 100 / total))

    st.success("🎉 {} pipeline completed.".format(enz_input))
    return logs

# --------------------------- Tabs (no expanders) ---------------------------
tab_loader, tab_workbench, tab_visual = st.tabs(
    ["Example Loader", "Query/Explore Neo4j Graph","Visualize"]
)

# ----- Example Loader (no expanders) -----
with tab_loader:
    st.subheader("Create Database — Use Your Enzymes or the Example")

    colA, colB = st.columns([2, 2])
    with colA:
        wipe_first = st.toggle("Wipe existing graph first", value=False, help="Runs MATCH (n) DETACH DELETE n")
    with colB:
        show_cmds = st.toggle("Show commands before running", value=True)

    mode = st.radio(
        "Run mode",
        ["Build from my enzyme list", "Run the example ({})".format(enz_input)],
        horizontal=True,
        index=0
    )

    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

    if run_btn:
        uri = st.session_state.get("conn_uri") or ""
        user = st.session_state.get("conn_user") or "neo4j"
        password = st.session_state.get("conn_pwd") or ""
        database = st.session_state.get("conn_db") or "neo4j"

        if not uri or not password:
            st.error("Provide Aura URI/password in the sidebar (or via st.secrets).")
        else:
            try:
                # # choose enzymes (user list or example)
                # if mode.startswith("Run the example"):
                #     enzymes = enz_input
                # else:
                #     enzymes = _parse_enzyme_list(st.session_state.get("enz_list") or "")

                # if not enzymes:
                #     st.error("Your enzyme list is empty/invalid.")
                #     st.stop()

                cmds = commands_for_run(neo4j_uri=uri, neo4j_password=password, enzymes=enz_input)

                if wipe_first and driver:
                    run_cypher("MATCH (n) DETACH DELETE n")
                    st.info("Graph wiped.")

                # Clean Data/ and show redacted commands if requested
                target = Path("Data")
                if target.exists():
                    import shutil
                    shutil.rmtree(target)
                    print(f"Removed folder: {target}")

                if show_cmds:
                    st.code("\n".join(redact_argv(c) for c in cmds), language="bash")

                # Pass safe ENV as well (tools can prefer these)
                env = {
                    "NEO4J_URI": uri,
                    "NEO4J_PASSWORD": password,
                    "NEO4J_USER": user or "neo4j",
                    "NEO4J_DATABASE": database,
                    "ENZYMES": enz_input,
                }

                logs = run_pipeline(cmds, extra_env=env, cwd=".")
            except Exception as e:
                st.error(f"Pipeline error: {e}")

# ----- Query Workbench -----
with tab_workbench:
    st.subheader("Cypher Workbench")
    c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
    with c1:
        preset = st.selectbox(
            "Preset",
            [
                # 1 – sample graph
                """MATCH (n)-[r]->(m)
                WITH n, r, m, rand() AS rrand
                RETURN n, r, m
                ORDER BY rrand
                LIMIT 100;""",

                # 2 – count nodes per label
                """MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(*) AS count
                ORDER BY count DESC""",

                # 3 – count relationships per type
                """MATCH ()-[r]->()
                RETURN type(r) AS relationshipType, count(*) AS count
                ORDER BY count DESC""",
            ],
            index=1,
        )

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

# ----- Visualize -----
with tab_visual:
    st.subheader("Visualize")
    rows = st.session_state.get("_last_rows", [])
    if not rows:
        st.info("Run a query (or fetch a subgraph) first.")
    else:
        v1, v2, v3 = st.columns([1, 1, 1])
        with v1:
            physics = st.toggle("Enable Graph Physics", True)
        with v2:
            hierarchical = st.toggle("Hierarchical layout", False)
        with v3:
            edge_cap = st.slider("Edge cap", 100, 5000, 2000, 100)

        show_labels = st.checkbox("Show node labels", value=True)
        min_label_degree = st.slider(
            "Min degree for showing label", 0, 10, 1, 1,
            help="Increase to show labels only for better-connected nodes."
        )

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

        # Limit edges for performance
        if len(rels) > edge_cap:
            rels = rels[:edge_cap]
        st.caption(f"Rendering {len(nodes)} nodes / {len(rels)} relationships (PyVis)")

        # Single renderer: PyVis
        render_pyvis(
            nodes,
            rels,
            height=650,
            physics=physics,
            hierarchical=hierarchical,
            show_labels=show_labels,
            min_label_degree=min_label_degree,
        )

# --------------------------- Footer (sidebar) ---------------------------

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <br>
        <div style="text-align: center;">
            <p style="font-size: 12px; color: gray;">
                © 2025 Asmaa A. Abdelwahab
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center;">
            <a href="https://github.com/asmaa-a-abdelwahab" target="_blank" style="text-decoration: none;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" style="width:40px; height:40px; margin-right: 10px;">
            </a>
            <a href="https://github.com/asmaa-a-abdelwahab" target="_blank" style="text-decoration: none;">
                <p style="font-size: 16px; font-weight: bold; color: black; margin: 0;">@asmaa-a-abdelwahab</p>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )