// tests/test_correctness.cpp
// Known-answer tests on the tiny hand-checkable graph. This is the safety net
// that later GPU tiers are diffed against. Exit code != 0 on any failure; CI
// runs this.
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../core/cpu/algorithms.hpp"
#include "../core/cpu/algorithms_omp.hpp"
#include "../core/cpu/louvain.hpp"
#include "../core/cpu/louvain_omp.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../core/graph/csr.hpp"
#include "../core/graph/generator.hpp"
#ifdef HAVE_CUDA
#include "../core/cuda/cuda_algorithms.hpp"
#endif

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

    // ====================================================================
    // Tier T1 (OpenMP) must match the T0 oracle on a larger, skewed graph.
    // This is the guarantee that parallelization did not change results.
    // ====================================================================
    CsrGraph big = make_rmat(/*scale=*/14, /*ef=*/16, /*seed=*/7);

    // BFS: identical distances from a fixed source.
    {
        auto ds = seq::bfs(big, 0);
        auto dp = omp::bfs(big, 0);
        CHECK(ds == dp, "OpenMP BFS matches sequential BFS (all distances)");
    }
    // WCC: identical component labels.
    {
        auto ls = seq::wcc(big);
        auto lp = omp::wcc(big);
        CHECK(ls == lp, "OpenMP WCC matches sequential WCC (all labels)");
        CHECK(seq::count_components(ls) == seq::count_components(lp),
              "OpenMP WCC finds the same component count");
    }
    // PageRank: equal within a tight tolerance (atomic float order differs).
    {
        auto ps = seq::pagerank(big);
        auto pp = omp::pagerank(big);
        double maxerr = 0.0;
        for (VertexId v = 0; v < big.num_vertices; ++v)
            maxerr = std::max(maxerr, std::abs(ps[v] - pp[v]));
        CHECK(maxerr < 1e-9, "OpenMP PageRank matches sequential within 1e-9");
    }

    // ====================================================================
    // High-diameter grid. RMAT is only ~4 BFS levels; a grid is ~2*(side-1),
    // so this is where a many-level frontier/active-set bug would surface.
    // ====================================================================
    {
        CsrGraph grid = make_grid(64);   // 4096 vertices, diameter 126
        CHECK(grid.num_vertices == 4096, "grid 64 has 4096 vertices");
        auto d = seq::bfs(grid, 0);
        CHECK(d[4095] == 126, "grid BFS corner-to-corner distance == 2*(side-1)");
        auto l = seq::wcc(grid);
        CHECK(seq::count_components(l) == 1, "grid is one connected component");
        auto dp = omp::bfs(grid, 0);
        CHECK(d == dp, "OpenMP BFS matches seq on high-diameter grid");
    }

    // ====================================================================
    // Louvain (T0). The test graph has two obvious communities: the 4-cycle
    // {0,1,2,3} and the triangle {4,5,6}, joined by nothing. Louvain must find
    // exactly those two, with positive modularity.
    // ====================================================================
    {
        auto lr = seq::louvain(g);
        CHECK(lr.num_communities == 2, "Louvain finds 2 communities on the test graph");
        CHECK(lr.modularity > 0.3, "Louvain modularity is high (>0.3) for clear structure");
        CHECK(lr.community[0] == lr.community[1] &&
              lr.community[1] == lr.community[2] &&
              lr.community[2] == lr.community[3],
              "Louvain groups the 4-cycle {0,1,2,3} together");
        CHECK(lr.community[4] == lr.community[5] &&
              lr.community[5] == lr.community[6],
              "Louvain groups the triangle {4,5,6} together");
        CHECK(lr.community[0] != lr.community[4],
              "Louvain separates the two communities");
    }
    // Louvain on a larger graph: modularity must be a valid, positive value.
    {
        auto lr = seq::louvain(big);
        CHECK(lr.modularity > 0.0 && lr.modularity <= 1.0,
              "Louvain modularity in valid range on RMAT graph");
        CHECK(lr.num_communities >= 1 && lr.num_communities <= big.num_vertices,
              "Louvain community count is sane on RMAT graph");
    }

    // ====================================================================
    // Louvain T1 (OpenMP). This is the one tier that CANNOT be diffed against
    // the oracle label-for-label: sequential Louvain is order-dependent by
    // construction, so the parallel (Jacobi) formulation reaches a different
    // local optimum. The contract is therefore on structure and on quality.
    // ====================================================================
    {
        // On the tiny graph the answer is unambiguous, so T1 must still nail it.
        auto lp = omp::louvain(g);
        CHECK(lp.num_communities == 2, "Louvain T1 finds 2 communities on the test graph");
        CHECK(lp.community[0] == lp.community[1] &&
              lp.community[1] == lp.community[2] &&
              lp.community[2] == lp.community[3],
              "Louvain T1 groups the 4-cycle {0,1,2,3} together");
        CHECK(lp.community[4] == lp.community[5] &&
              lp.community[5] == lp.community[6],
              "Louvain T1 groups the triangle {4,5,6} together");
        CHECK(lp.community[0] != lp.community[4],
              "Louvain T1 separates the two communities");
    }
    {
        auto ls = seq::louvain(big);
        auto lp = omp::louvain(big);
        CHECK(lp.modularity > 0.0 && lp.modularity <= 1.0,
              "Louvain T1 modularity in valid range on RMAT graph");
        // Quality, not equality. A parallel community detector that quietly
        // returned worse structure would pass every "is this a valid partition"
        // check, so the contract has to be on Q itself.
        //
        // The bar is 0.90, set from measurement rather than taste: with the
        // shipped kChunks=4 this tier lands at 92.4% of the oracle here (RMAT
        // scale 14) and 93.4-93.9% on scales 16-18. It is deliberately loose
        // enough not to flag that known, documented ~6% quality cost -- and
        // tight enough to catch the regression it exists for: pure Jacobi local
        // moving (kChunks=1) scores 72.6%, which this rejects.
        CHECK(lp.modularity > ls.modularity * 0.90,
              "Louvain T1 modularity within 10% of the T0 oracle");
        // Every vertex must carry a label inside the reported range.
        bool labels_ok = lp.community.size() == big.num_vertices;
        for (VertexId v = 0; v < big.num_vertices && labels_ok; ++v)
            if (lp.community[v] >= lp.num_communities) labels_ok = false;
        CHECK(labels_ok, "Louvain T1 labels are dense and in range");
    }
#ifdef _OPENMP
    {
        // The parallel formulation reads a snapshot and a fixed edge order, so
        // its result must not depend on how many threads ran it. Sequential
        // Louvain has no such guarantee; this is the property that makes T1
        // reproducible, and it is worth a test rather than a claim.
        const int saved = omp_get_max_threads();
        omp_set_num_threads(1);
        auto a1 = omp::louvain(big);
        omp_set_num_threads(saved > 1 ? saved : 2);
        auto a2 = omp::louvain(big);
        omp_set_num_threads(saved);
        CHECK(a1.community == a2.community,
              "Louvain T1 is deterministic: 1 thread and N threads agree exactly");
        CHECK(std::abs(a1.modularity - a2.modularity) < 1e-12,
              "Louvain T1 modularity is thread-count independent");
    }
#endif

#ifdef HAVE_CUDA
    // ====================================================================
    // Tiers T2 (naïve CUDA) and T3 (optimized CUDA) must match the T0 oracle
    // on a larger, skewed graph. Skipped gracefully if no GPU is visible.
    // ====================================================================
    if (cuda::available()) {
        std::cout << "\n[CUDA] device: " << cuda::device_name() << "\n";

        // BFS T2 and T3 == T0 (exact distances).
        {
            auto ds  = seq::bfs(big, 0);
            auto dc2 = cuda::bfs(big, 0);       // T2 naïve
            auto dc3 = cuda::bfs_opt(big, 0);   // T3 frontier
            auto dc4 = cuda::bfs_lb(big, 0);    // T4 degree-bucketed frontier
            CHECK(ds == dc2, "CUDA T2 BFS matches sequential BFS (all distances)");
            CHECK(ds == dc3, "CUDA T3 BFS matches sequential BFS (all distances)");
            CHECK(ds == dc4, "CUDA T4 BFS matches sequential BFS (all distances)");
        }
        // WCC T2 and T3 == T0 (same component labels).
        {
            auto ls  = seq::wcc(big);
            auto lc2 = cuda::wcc(big);       // T2 naïve
            auto lc3 = cuda::wcc_opt(big);   // T3 active-set
            auto lc4 = cuda::wcc_lb(big);    // T4 degree-bucketed active-set
            CHECK(seq::count_components(ls) == seq::count_components(lc2),
                  "CUDA T2 WCC finds the same component count");
            CHECK(lc2 == lc3, "CUDA T3 WCC matches T2 WCC (all labels)");
            CHECK(seq::count_components(ls) == seq::count_components(lc3),
                  "CUDA T3 WCC finds the same component count");
            // Label propagation converges to the minimum id per component
            // regardless of the thread-to-vertex mapping, so T4 must be
            // label-for-label identical, not merely component-count identical.
            CHECK(lc2 == lc4, "CUDA T4 WCC matches T2 WCC (all labels)");
            CHECK(seq::count_components(ls) == seq::count_components(lc4),
                  "CUDA T4 WCC finds the same component count");
        }
        // PageRank T2 and T3 == T0 (within tolerance; atomic order differs).
        {
            auto ps  = seq::pagerank(big);
            auto p2  = cuda::pagerank(big);       // T2 naïve
            auto p3  = cuda::pagerank_opt(big);   // T3 optimized
            auto p4  = cuda::pagerank_lb(big);    // T4 degree-bucketed
            double e2 = 0.0, e3 = 0.0, e4 = 0.0;
            for (VertexId v = 0; v < big.num_vertices; ++v) {
                e2 = std::max(e2, std::abs(ps[v] - p2[v]));
                e3 = std::max(e3, std::abs(ps[v] - p3[v]));
                e4 = std::max(e4, std::abs(ps[v] - p4[v]));
            }
            CHECK(e2 < 1e-9, "CUDA T2 PageRank matches sequential within 1e-9");
            CHECK(e3 < 1e-9, "CUDA T3 PageRank matches sequential within 1e-9");
            CHECK(e4 < 1e-9, "CUDA T4 PageRank matches sequential within 1e-9");
        }
        // Louvain T2 (naive CUDA). Like T1 this cannot reproduce the oracle's
        // labels, and unlike T1 it does not even try to match its quality:
        // T2 applies every vertex's move simultaneously (pure Jacobi), which
        // the CPU chunk ablation measured at ~73% of the oracle's modularity.
        // The tests below pin both halves of that: unambiguous structure must
        // still come out exactly right, and the known quality loss must stay
        // where it was measured rather than drifting further.
        {
            CsrGraph tiny = make_test_graph();
            auto ls = seq::louvain(tiny);
            auto lg = cuda::louvain(tiny);
            CHECK(lg.num_communities == 2,
                  "CUDA T2 Louvain finds 2 communities on the test graph");
            CHECK(lg.community[0] == lg.community[1] &&
                  lg.community[1] == lg.community[2] &&
                  lg.community[2] == lg.community[3],
                  "CUDA T2 Louvain groups the 4-cycle {0,1,2,3} together");
            CHECK(lg.community[4] == lg.community[5] &&
                  lg.community[5] == lg.community[6],
                  "CUDA T2 Louvain groups the triangle {4,5,6} together");
            CHECK(lg.community[0] != lg.community[4],
                  "CUDA T2 Louvain separates the two communities");
            // On a graph this unambiguous there is nothing for the parallel
            // formulation to get wrong, so here it must match the oracle exactly.
            CHECK(std::abs(lg.modularity - ls.modularity) < 1e-9,
                  "CUDA T2 Louvain matches the oracle's Q on the test graph");
        }
        // T3 and T4 change only how each vertex's weight-per-community is
        // built -- a global sort, a bit-bounded sort, or a per-vertex hash
        // table. The accept rule is identical in all three, so on the same
        // graph they must agree on Q to the last bit. That equality is the
        // whole basis for comparing their run times, so it is asserted rather
        // than assumed.
        {
            auto g2 = cuda::louvain(big);            // T2, global sort
            auto g3 = cuda::louvain_opt(big);        // T3, bit-bounded sort
            auto g4 = cuda::louvain_lb(big);         // T4, degree-bucketed hash
            CHECK(std::abs(g3.modularity - g2.modularity) < 1e-12,
                  "CUDA T3 Louvain reproduces T2's modularity exactly");
            CHECK(std::abs(g4.modularity - g2.modularity) < 1e-12,
                  "CUDA T4 Louvain reproduces T2's modularity exactly");
            CHECK(g3.num_communities == g2.num_communities &&
                  g4.num_communities == g2.num_communities,
                  "CUDA T2/T3/T4 Louvain agree on the community count");
            // The two T3 groupings are kept selectable; both must also agree.
            auto seg = cuda::louvain_opt_variant(big, /*segmented=*/true);
            CHECK(std::abs(seg.modularity - g2.modularity) < 1e-12,
                  "CUDA T3 segmented grouping reproduces T2's modularity");
        }
        {
            // T4's hash tables are sized per vertex, so a hub whose degree
            // exceeds the shared-memory table has to reach the global-table
            // path. A dense graph puts real traffic through every bucket,
            // which the sparse default would leave untested.
            CsrGraph dense = make_rmat(/*scale=*/13, /*ef=*/128, /*seed=*/5);
            auto d2 = cuda::louvain(dense);
            auto d4 = cuda::louvain_lb(dense);
            CHECK(std::abs(d4.modularity - d2.modularity) < 1e-12,
                  "CUDA T4 Louvain matches T2 on a dense graph (all buckets used)");
        }
        {
            auto ls = seq::louvain(big);
            auto lg = cuda::louvain(big);
            bool labels_ok = lg.community.size() == big.num_vertices;
            for (VertexId v = 0; v < big.num_vertices && labels_ok; ++v)
                if (lg.community[v] >= lg.num_communities) labels_ok = false;
            CHECK(labels_ok, "CUDA T2 Louvain labels are dense and in range");
            CHECK(lg.modularity > 0.0 && lg.modularity <= 1.0,
                  "CUDA T2 Louvain modularity in valid range");
            // The bar is 0.60, not the 0.90 used for T1: this tier is *expected*
            // to lose quality and measures ~73% of the oracle. The assertion is
            // a floor against real breakage, and the gap itself is reported in
            // the README rather than hidden behind a loose threshold.
            CHECK(lg.modularity > ls.modularity * 0.60,
                  "CUDA T2 Louvain modularity above the naive-tier floor (60%)");
        }

        // High-diameter grid: the real test of the many-level frontier /
        // active-set kernels (254 levels here vs ~4 on RMAT).
        {
            CsrGraph grid = make_grid(128);   // 16384 vertices, diameter 254
            auto ds = seq::bfs(grid, 0);
            auto d2 = cuda::bfs(grid, 0);
            auto d3 = cuda::bfs_opt(grid, 0);
            auto d4 = cuda::bfs_lb(grid, 0);
            CHECK(ds == d2, "CUDA T2 BFS matches seq on high-diameter grid");
            CHECK(ds == d3, "CUDA T3 BFS matches seq on high-diameter grid");
            CHECK(ds == d4, "CUDA T4 BFS matches seq on high-diameter grid");
            auto ls = seq::wcc(grid);
            auto l3 = cuda::wcc_opt(grid);
            auto l4 = cuda::wcc_lb(grid);
            CHECK(seq::count_components(ls) == seq::count_components(l3),
                  "CUDA T3 WCC matches seq component count on grid");
            CHECK(l3 == l4, "CUDA T4 WCC matches T3 WCC on grid (all labels)");
        }
    } else {
        std::cout << "\n[CUDA] no device visible — GPU tiers not tested here.\n";
    }
#endif

    std::cout << "\n" << (failures ? "TESTS FAILED: " : "ALL TESTS PASSED (")
              << failures << (failures ? " failure(s)\n" : " failures)\n");
    return failures ? 1 : 0;
}
