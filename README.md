# 📘 ChemGraphBuilder – Neo4j Graph Builder App  
*A Streamlit application for building, querying, and visualizing PubChem-derived chemical–biological knowledge graphs.*

---

## 🚀 Overview

This application provides a complete interface around the **ChemGraphBuilder** pipeline, enabling you to:

- **Build** a fully populated **Neo4j knowledge graph** from PubChem using enzyme-focused queries.
- **Explore** the graph using an interactive Cypher workbench.
- **Visualize** subgraphs with a rich, interactive **PyVis** network view.

Graph content includes:

- Nodes: `Compound`, `BioAssay`, `Gene`, `Protein`
- Relationships: `Assay_Compound`, `Assay_Gene`, `Gene_Protein`, `Compound_Gene`, similarity, co-occurrence, transformation, etc.

Node color scheme:

- Compound → blue  
- BioAssay → orange  
- Gene → green  
- Protein → red  
- Other labels → grey  

---

## 🧩 Features

| Feature | Description |
|--------|-------------|
| 🔌 Neo4j Connection Manager | Connect via manual input or `st.secrets` |
| 🧬 Enzyme-specific graph building | Use your own enzyme list or an example list |
| 🔁 End-to-end ETL pipeline | Runs ChemGraphBuilder CLI tools with progress & logging |
| 📝 Cypher Query Workbench | Run and profile queries, export JSON/CSV |
| 🌐 PyVis Graph Visualization | Interactive, color-coded network layouts |
| 📦 Data Export | Download nodes and relationships as flat CSV tables |

---

## 🛠 Installation

```bash
git clone https://github.com/asmaa-a-abdelwahab/ChemGraphBuilder-Demo
cd ChemGraphBuilder-Demo
pip install -r requirements.txt
streamlit run app.py
````

Make sure the **ChemGraphBuilder CLI tools** are on your PATH:

* `setup-data-folder`
* `collect-process-nodes`
* `collect-process-relationships`
* `load-graph-nodes`
* `load-graph-relationships`

---

## 🔐 Connecting to Neo4j

Use the **left sidebar**:

* **Aura URI** (e.g. `neo4j+s://<your-aura-host>:7687`)
* **Username**
* **Password / Token**
* **Database** (default: `neo4j`)
* **Enzyme list**, comma-separated (e.g. `CYP2J2,CYP2C9,CYP3A4`)

You can either:

* Select **Enter manually**, or
* Select **Use st.secrets** to load values from `st.secrets`.

Click **Connect / Reconnect**.

---

## 🧪 Tab 1 – Example Loader

This tab builds (or rebuilds) the Neo4j graph using ChemGraphBuilder.

### Controls

* **Wipe existing graph first**
  Runs:

  ```cypher
  MATCH (n) DETACH DELETE n
  ```
* **Show commands before running**
  Displays all CLI steps with secrets redacted.
* **Run mode**

  * Build from your enzyme list
  * Run the example list (`ENZYMES` from `st.secrets`)

### Pipeline steps

The app orchestrates:

1. `setup-data-folder`
2. `collect-process-nodes` for all node types
3. `collect-process-relationships` for all relationship types
4. `load-graph-nodes` into Neo4j
5. `load-graph-relationships` into Neo4j

All commands stream their logs into the UI, with a progress bar and error handling.

---

## 💻 Tab 2 – Query / Explore Neo4j Graph

A small **Cypher Workbench** to interact with the database.

### Presets

* **Random sample of graph**

  ```cypher
  MATCH (n)-[r]->(m)
  WITH n, r, m, rand() AS rrand
  RETURN n, r, m
  ORDER BY rrand
  LIMIT 100;
  ```
* **Count nodes per label**

  ```cypher
  MATCH (n)
  UNWIND labels(n) AS label
  RETURN label, count(*) AS count
  ORDER BY count DESC;
  ```
* **Count relationships per type**

  ```cypher
  MATCH ()-[r]->()
  RETURN type(r) AS relationshipType, count(*) AS count
  ORDER BY count DESC;
  ```

### Controls

* **LIMIT** slider (used in params as `"limit"`)
* **PROFILE** toggle – prepends `PROFILE` to your query
* **Params (JSON)** – custom query parameters
* **Run query** button

### Output

* Results displayed as a dataframe
* Download buttons:

  * `cypher_result.json`
  * `cypher_result.csv`

The last query result is cached and reused by the **Visualize** tab.

---

## 🌐 Tab 3 – Visualize (PyVis)

This tab turns the last Cypher result into an interactive **PyVis** graph.

### Requirements

Your last query should return nodes and relationships, e.g.:

```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100;
```

### Controls

* **Enable Graph Physics** – toggle force-directed layout
* **Hierarchical layout** – enables a top-down layout (PyVis hierarchical mode)
* **Edge cap** – limit the maximum number of relationships rendered
* **Show node labels** – show/hide labels on nodes
* **Min degree for showing label** – only show labels on well-connected nodes

### Exports

* **Download Nodes (CSV)** – `nodes.csv` with `id`, `labels`, and properties
* **Download Relationships (CSV)** – `relationships.csv` with `start_id`, `end_id`, `type`, and properties

Nodes are:

* **Colored** by label (Compound/BioAssay/Gene/Protein/Other)
* **Sized** by degree (more connected = larger)

Hover on nodes/edges to inspect details.

---

## 🧭 Typical Workflow

1. **Connect** to Neo4j via the sidebar.
2. **Build** the graph from Tab 1 (Example Loader).
3. **Query** the graph in Tab 2 (Cypher Workbench).
4. **Visualize** the subgraph from Tab 3 (Visualize).
5. **Export** results and node/edge tables as needed.

---

## 🩺 Troubleshooting

* **Connection failed**

  * Check URI scheme (`neo4j+s://` for Aura)
  * Verify username, password, and database name

* **CLI command not found**

  * Ensure ChemGraphBuilder tools are installed and on PATH.

* **Graph is too dense / slow**

  * Lower **Edge cap**
  * Increase **Min degree for showing label**
  * Hide labels or physics

* **No graph in Visualize tab**

  * Ensure your last query returns both nodes and relationships.

---

## 📄 License & Credits

Created by **Asmaa Abdelwahab** (2025).
Uses **Streamlit**, **Neo4j Python driver**, and **PyVis** for visualization.
