#!/usr/bin/env python3
"""
extract_callgraph.py — turn a real binary into a call-graph edge list.

This is the bridge that makes the study's graphs *domain graphs*, not just
synthetic RMAT: functions become vertices, `call` instructions become edges.
Call-graphs of real binaries are power-law and irregular (a handful of hot
helpers called from everywhere, most functions called from one place) — exactly
the skew the GPU tiers are built to handle.

The engine never parses a binary: this tool emits the same plain edge-list format
(`u v` per line) that core/graph/csr consumes, plus a sidecar mapping file
(vertex id -> function name).

Dependencies: just `objdump` (GNU binutils) on PATH. No angr / Ghidra needed,
so it runs anywhere. Works on ELF and PE binaries that objdump can disassemble.

Usage:
    python tools/extract_callgraph.py --binary /bin/ls --out data/graphs/ls
        -> writes data/graphs/ls.edges  and  data/graphs/ls.map

    python tools/extract_callgraph.py --binary a.out --out g --stats
"""
from __future__ import annotations  # allow "str | None" hints on Python < 3.10

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Disassemblers we know how to drive, in preference order. GNU objdump and
# LLVM's llvm-objdump produce compatible-enough output for the regexes below.
DISASSEMBLERS = ['objdump', 'llvm-objdump']


def find_disassembler(preferred: str | None) -> str:
    """Locate objdump / llvm-objdump on PATH (or use --objdump override)."""
    candidates = [preferred] if preferred else DISASSEMBLERS
    for c in candidates:
        if c and shutil.which(c):
            return c
    sys.exit(
        "error: no disassembler found. Install one of:\n"
        "  - LLVM (gives llvm-objdump):  winget install LLVM.LLVM\n"
        "  - GNU binutils (objdump)   :  via MSYS2 / your package manager\n"
        "Then reopen the shell, or pass --objdump <path>.")

# A function header line in objdump output, e.g.
#   0000000000001234 <malloc>:
FUNC_RE = re.compile(r'^[0-9a-fA-F]+\s+<([^>]+)>:')

# A call instruction, e.g.
#   1247:  call   1050 <printf@plt>
#   1247:  callq  0x1050 <printf>
# We capture the target symbol inside <...>. PLT stubs (<sym@plt>) are unified
# with their symbol so direct and PLT calls to the same function coincide.
CALL_RE = re.compile(r':\s*(?:call|callq|bl|blx|jal)\w*\s+.*<([^>+]+)(?:\+0x[0-9a-fA-F]+)?>')


def disassemble(binary: str, tool: str) -> str:
    """Return disassembly text from objdump / llvm-objdump, or exit clearly."""
    # llvm-objdump uses -d and understands --no-show-raw-insn too (recent
    # versions); older ones ignore it harmlessly, so we keep a fallback.
    cmds = [
        [tool, '-d', '--no-show-raw-insn', binary],
        [tool, '-d', binary],
    ]
    last_err = ''
    for cmd in cmds:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return out.stdout
        except FileNotFoundError:
            sys.exit(f"error: '{tool}' not found on PATH.")
        except subprocess.CalledProcessError as e:
            last_err = e.stderr.strip()
    sys.exit(f"error: {tool} failed on {binary}:\n{last_err}")


def normalize(name: str) -> str:
    """Collapse a PLT stub name onto its target symbol."""
    return name.split('@')[0]


def build_callgraph(disasm: str):
    """Parse disassembly into (edges, id_of_name). Vertices are functions."""
    id_of = {}          # function name -> vertex id
    edges = set()       # (caller_id, callee_id), de-duplicated
    current = None      # current caller id

    def vid(name: str) -> int:
        name = normalize(name)
        if name not in id_of:
            id_of[name] = len(id_of)
        return id_of[name]

    for line in disasm.splitlines():
        m = FUNC_RE.match(line)
        if m:
            current = vid(m.group(1))
            continue
        if current is None:
            continue
        c = CALL_RE.search(line)
        if c:
            callee = vid(c.group(1))
            if callee != current:          # drop self-recursion self-loops
                edges.add((current, callee))
    return edges, id_of


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--binary', required=True, help='path to the binary to analyze')
    ap.add_argument('--out', required=True,
                    help='output basename; writes <out>.edges and <out>.map')
    ap.add_argument('--stats', action='store_true', help='print graph statistics')
    ap.add_argument('--objdump', default=None,
                    help='disassembler to use (objdump / llvm-objdump / full path); '
                         'auto-detected if omitted')
    args = ap.parse_args()

    if not Path(args.binary).exists():
        sys.exit(f"error: binary not found: {args.binary}")

    tool = find_disassembler(args.objdump)
    disasm = disassemble(args.binary, tool)
    edges, id_of = build_callgraph(disasm)

    n = len(id_of)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(f'{args.out}.edges', 'w') as f:
        f.write(f'# call-graph of {args.binary}\n')
        f.write(f'# {n} functions (vertices), {len(edges)} call edges\n')
        f.write('# format: caller_id callee_id\n')
        for u, v in sorted(edges):
            f.write(f'{u} {v}\n')

    with open(f'{args.out}.map', 'w') as f:
        f.write('# vertex_id\tfunction_name\n')
        for name, i in sorted(id_of.items(), key=lambda kv: kv[1]):
            f.write(f'{i}\t{name}\n')

    print(f'wrote {args.out}.edges  ({n} vertices, {len(edges)} edges)')
    print(f'wrote {args.out}.map')

    # Stripped binaries expose no function symbols — only section labels like
    # ".text" — so no call-graph can be recovered this way. Say so plainly.
    only_sections = all(name.startswith('.') for name in id_of)
    if n <= 1 or len(edges) == 0 or only_sections:
        print('\nwarning: almost no functions/edges were found. This binary is '
              'likely STRIPPED (no symbols) — e.g. a production Windows system\n'
              '         binary. A lightweight objdump pass cannot recover a '
              'function-level call graph from it.\n'
              '         Use a binary WITH symbols (a locally compiled / '
              'non-stripped binary), or a proper RE tool (Ghidra/angr) for\n'
              '         stripped targets. The bundled data/graphs/ls.edges is a '
              'ready symbol-based example.', file=sys.stderr)

    if args.stats and n:
        outdeg = [0] * n
        indeg = [0] * n
        for u, v in edges:
            outdeg[u] += 1
            indeg[v] += 1
        # Most-called functions = the power-law hubs the study cares about.
        inv = {i: name for name, i in id_of.items()}
        hot = sorted(range(n), key=lambda i: indeg[i], reverse=True)[:8]
        print(f'\nstats: {n} vertices, {len(edges)} edges, '
              f'avg out-degree {len(edges)/n:.2f}, '
              f'max in-degree {max(indeg)} (the hottest hub)')
        print('most-called functions (the hubs that cause GPU load imbalance):')
        for i in hot:
            print(f'  in={indeg[i]:5d}  out={outdeg[i]:3d}  {inv[i]}')


if __name__ == '__main__':
    main()
