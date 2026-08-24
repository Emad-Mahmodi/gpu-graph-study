# tools/ — graph extractors

These turn real-world artifacts into the plain edge-list format the engine
consumes (`u v` per line). The analytics engine never runs them; they are a
separate, Python-only preprocessing step.

## extract_callgraph.py — binary → call-graph

Functions become vertices; `call` instructions become edges.

```bash
python tools/extract_callgraph.py --binary /bin/ls --out data/graphs/ls --stats
# -> data/graphs/ls.edges   (feed to run_bench --graph)
# -> data/graphs/ls.map     (vertex id -> function name)
```

Only needs `objdump` (GNU binutils) on PATH — no angr/Ghidra. Works on any ELF
or PE binary objdump can disassemble.

**On graph shape:** call-graphs of *large, statically-linked* binaries are
strongly power-law — a few hot helpers (allocators, logging, string ops) are
called from everywhere while most functions are called once. Small
dynamically-linked binaries like `/bin/ls` are shallow, because the real hubs
live in shared libraries (libc), not in the binary itself. For the skew study,
use big static binaries; the synthetic RMAT generator provides controlled,
large-scale skew independent of any single binary.

**License note:** only commit graphs derived from binaries you may redistribute
(see `data/README.md`). Never commit a graph from proprietary or NDA firmware —
the call structure alone can leak design.

## pcap_to_graph.py — network flow → host graph  *(planned, Phase 6)*

Hosts become vertices; flows become edges. Targets public IDS captures
(CICIDS2017, UNSW-NB15) for the cross-dataset generalization study.
