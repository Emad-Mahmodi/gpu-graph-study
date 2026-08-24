// tests/test_correctness.cpp
// Known-answer tests on the tiny hand-checkable graph. This is the safety net
// that later GPU tiers are diffed against. Exit code != 0 on any failure; CI
// runs this.
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../core/cpu/algorithms.hpp"
#include "../core/graph/csr.hpp"
#include "../core/graph/generator.hpp"

using namespace ggs;

static int failures = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) { std::cerr << "FAIL: " << (msg) << "\n"; ++failures; }\
        else         { std::cout << "ok:   " << (msg) << "\n"; }           \
    } while (0)

int main() {
    // make_test_graph: component A {0,1,2,3} (0-1,1-2,2-3,0-3, undirected),
    //                  component B {4,5,6} (triangle). Symmetrized.
    CsrGraph g = make_test_graph();

    CHECK(g.num_vertices == 7, "test graph has 7 vertices");
    CHECK(g.num_edges == 14, "test graph has 14 directed edges (7 undirected)");

    // --- BFS from 0 ------------------------------------------------------
    auto dist = seq::bfs(g, 0);
    CHECK(dist[0] == 0, "bfs dist[0]==0");
    CHECK(dist[1] == 1, "bfs dist[1]==1");
    CHECK(dist[3] == 1, "bfs dist[3]==1 (via chord 0-3)");
    CHECK(dist[2] == 2, "bfs dist[2]==2");
    CHECK(dist[4] == seq::UNREACHED, "bfs cannot reach other component (v4)");
    CHECK(dist[5] == seq::UNREACHED, "bfs cannot reach other component (v5)");

    // --- WCC -------------------------------------------------------------
    auto lbl = seq::wcc(g);
    CHECK(seq::count_components(lbl) == 2, "wcc finds exactly 2 components");
    CHECK(lbl[0] == lbl[1] && lbl[1] == lbl[2] && lbl[2] == lbl[3],
          "wcc: {0,1,2,3} share a label");
    CHECK(lbl[4] == lbl[5] && lbl[5] == lbl[6], "wcc: {4,5,6} share a label");
    CHECK(lbl[0] != lbl[4], "wcc: the two components differ");

    // --- PageRank --------------------------------------------------------
    auto pr = seq::pagerank(g);
    double sum = 0.0;
    for (double x : pr) sum += x;
    CHECK(std::abs(sum - 1.0) < 1e-6, "pagerank sums to 1.0");
    // In component A, vertices 0 and 2 have degree 2; 1 and 3 have degree 2 too
    // (it's a 4-cycle), so all four ranks are equal. Symmetric triangle B too.
    CHECK(std::abs(pr[0] - pr[2]) < 1e-9, "pagerank symmetric in cycle (0==2)");
    CHECK(std::abs(pr[4] - pr[5]) < 1e-9, "pagerank symmetric in triangle (4==5)");

    // --- RMAT sanity -----------------------------------------------------
    CsrGraph r = make_rmat(/*scale=*/10, /*ef=*/8, /*seed=*/42);
    auto rs = r.stats();
    CHECK(rs.num_vertices == 1024, "rmat scale 10 -> 1024 vertices");
    CHECK(rs.num_edges > 0, "rmat produced edges");
    CHECK(rs.max_degree > rs.avg_degree * 2, "rmat is skewed (max >> avg degree)");
    // Determinism: same seed => identical graph.
    CsrGraph r2 = make_rmat(10, 8, 42);
    CHECK(r.col_indices == r2.col_indices, "rmat is deterministic given a seed");

    std::cout << "\n" << (failures ? "TESTS FAILED: " : "ALL TESTS PASSED (")
              << failures << (failures ? " failure(s)\n" : " failures)\n");
    return failures ? 1 : 0;
}
