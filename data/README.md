# Data & provenance

The analytics engine consumes **graph files only** — it never reads a binary or
a packet capture. Graphs enter the engine in one of two ways:

1. **Synthetic (RMAT)** — generated on the fly by `core/graph/generator`. No files
   needed; fully reproducible from a seed. Used for the scalability study.
2. **Edge-list files** (`*.edges`) — plain text, one `u v` edge per line,
   `#` comments allowed. Placed in `data/graphs/`.

## Edge-list format
```
# ls.callgraph  — caller callee
0 5
0 12
5 12
```
Vertex ids are 0-based integers. Ids need not be dense; the loader infers
`num_vertices = max_id + 1`.

## Where graphs come from (later phases)

| Class | Extractor (Phase ≥2) | Source |
|---|---|---|
| Binary call-graph / CFG | `tools/extract_callgraph.py` (Ghidra/angr) | open binaries only |
| Network-flow graph | `tools/pcap_to_graph.py` | CICIDS2017, UNSW-NB15, CSE-CIC-IDS2018 |

## ⚠️ Licensing — read before committing any graph

- Only commit graphs derived from binaries whose license permits redistribution
  (coreutils, busybox, open-source firmware). **Never commit a graph extracted
  from proprietary or NDA firmware** — even the graph can leak structure.
- Public IDS datasets have their own terms; link to them, do not re-host.
- `sample.edges` in `data/graphs/` is the only graph tracked by git (see
  `.gitignore`); everything else is generated or downloaded locally.
