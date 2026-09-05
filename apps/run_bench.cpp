// apps/run_bench.cpp
// Unified CLI driver for the study. Loads or generates a graph, runs one
// algorithm at one tier, prints result summary + median timing as a CSV row.
//
// Usage:
//   run_bench --algo bfs --tier seq --graph data/graphs/foo.edges [--source 0]
//   run_bench --algo pagerank --tier seq --rmat 16 --ef 16 [--seed 1]
//   run_bench --algo wcc --tier seq --rmat 18 --ef 8
//
// Tiers available today: seq (T0). openmp/cuda tiers are added in later phases
// and will be dispatched here behind the same flags.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "../bench/timer.hpp"
#include "../core/cpu/algorithms.hpp"
#include "../core/cpu/algorithms_omp.hpp"
#include "../core/cpu/louvain.hpp"
#include "../core/graph/csr.hpp"
#include "../core/graph/generator.hpp"
#ifdef HAVE_CUDA
#include "../core/cuda/cuda_algorithms.hpp"
#endif

using namespace ggs;

namespace {

struct Args {
    std::string algo  = "bfs";
    std::string tier  = "seq";
    std::string graph;           // edge-list path (if empty, use rmat)
    int         grid_side = 0;   // if >0, use a side×side 2D grid instead
    int         rmat_scale = 16;
    int         edge_factor = 16;
    std::uint64_t seed = 1;
    VertexId    source = 0;
    int         repeats = 10;
    bool        symmetrize = true;   // undirected by default for wcc/pagerank
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(2); }
            return argv[++i];
        };
        if      (k == "--algo")   a.algo = next();
        else if (k == "--tier")   a.tier = next();
        else if (k == "--graph")  a.graph = next();
        else if (k == "--grid")   a.grid_side = std::stoi(next());
        else if (k == "--rmat")   a.rmat_scale = std::stoi(next());
        else if (k == "--ef")     a.edge_factor = std::stoi(next());
        else if (k == "--seed")   a.seed = std::stoull(next());
        else if (k == "--source") a.source = static_cast<VertexId>(std::stoul(next()));
        else if (k == "--repeats")a.repeats = std::stoi(next());
        else if (k == "--directed") a.symmetrize = false;
        else if (k == "--help") {
            std::cout << "see header of apps/run_bench.cpp for usage\n";
            std::exit(0);
        } else {
            std::cerr << "unknown flag: " << k << "\n"; std::exit(2);
        }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    // --- Load or generate the graph -------------------------------------
    CsrGraph g;
    std::string graph_name;
    if (!a.graph.empty()) {
        g = CsrGraph::load_edgelist(a.graph, a.symmetrize);
        graph_name = a.graph;
    } else if (a.grid_side > 0) {
        g = make_grid(static_cast<std::uint32_t>(a.grid_side));
        graph_name = "grid-" + std::to_string(a.grid_side) +
                     "x" + std::to_string(a.grid_side);
    } else {
        g = make_rmat(a.rmat_scale, a.edge_factor, a.seed, a.symmetrize);
        graph_name = "rmat-s" + std::to_string(a.rmat_scale) +
                     "-ef" + std::to_string(a.edge_factor);
    }
    auto s = g.stats();

    // --- Dispatch algorithm x tier --------------------------------------
    double ms = 0.0;
    std::string result_summary;

    const bool seq_tier = (a.tier == "seq");
    const bool omp_tier = (a.tier == "openmp");
    const bool cuda_tier = (a.tier == "cuda");           // T2 naïve
    const bool cuda_opt_tier = (a.tier == "cuda_opt");   // T3 optimized
    const bool cuda_lb_tier = (a.tier == "cuda_lb");     // T4 load-balanced
    const bool any_cuda = cuda_tier || cuda_opt_tier || cuda_lb_tier;
    if (!seq_tier && !omp_tier && !any_cuda) {
        std::cerr << "unknown tier '" << a.tier
                  << "'. Use seq, openmp, cuda, cuda_opt, or cuda_lb.\n";
        return 3;
    }
#ifndef HAVE_CUDA
    if (any_cuda) {
        std::cerr << "this binary was built without CUDA; rebuild on a machine "
                     "with the CUDA toolkit to use --tier cuda / cuda_opt / cuda_lb.\n";
        return 3;
    }
#else
    if (any_cuda && !cuda::available()) {
        std::cerr << "no CUDA device visible at runtime.\n";
        return 3;
    }
#endif

    // The GPU tiers T2/T3/T4 all implement bfs, pagerank and wcc.
    // Louvain is sequential-only (see the roadmap).
    if (any_cuda && a.algo == "louvain") {
        std::cerr << "louvain has no GPU tier; use --tier seq.\n";
        return 3;
    }

    if (a.algo == "bfs") {
        std::vector<std::int32_t> dist;
        ms = time_median_ms([&]{
            if (seq_tier)      dist = seq::bfs(g, a.source);
            else if (omp_tier) dist = omp::bfs(g, a.source);
#ifdef HAVE_CUDA
            else if (cuda_tier)     dist = cuda::bfs(g, a.source);      // T2
            else if (cuda_opt_tier) dist = cuda::bfs_opt(g, a.source);  // T3
            else                    dist = cuda::bfs_lb(g, a.source);   // T4
#endif
        }, a.repeats);
        std::int64_t reached = 0, maxd = 0;
        for (auto d : dist) if (d != seq::UNREACHED) { reached++; maxd = std::max<std::int64_t>(maxd, d); }
        result_summary = "reached=" + std::to_string(reached) +
                         " max_depth=" + std::to_string(maxd);
    } else if (a.algo == "pagerank") {
        std::vector<double> pr;
        ms = time_median_ms([&]{
            if (seq_tier)      pr = seq::pagerank(g);
            else if (omp_tier) pr = omp::pagerank(g);
#ifdef HAVE_CUDA
            else if (cuda_tier)     pr = cuda::pagerank(g);      // T2 naïve
            else if (cuda_opt_tier) pr = cuda::pagerank_opt(g);  // T3 optimized
            else                    pr = cuda::pagerank_lb(g);   // T4 load-balanced
#endif
        }, a.repeats);
        double sum = 0.0, mx = 0.0;
        for (double x : pr) { sum += x; mx = std::max(mx, x); }
        result_summary = "sum=" + std::to_string(sum) + " max_pr=" + std::to_string(mx);
    } else if (a.algo == "wcc") {
        std::vector<VertexId> lbl;
        ms = time_median_ms([&]{
            if (seq_tier)      lbl = seq::wcc(g);
            else if (omp_tier) lbl = omp::wcc(g);
#ifdef HAVE_CUDA
            else if (cuda_tier)     lbl = cuda::wcc(g);      // T2
            else if (cuda_opt_tier) lbl = cuda::wcc_opt(g);  // T3
            else                    lbl = cuda::wcc_lb(g);   // T4
#endif
        }, a.repeats);
        result_summary = "components=" + std::to_string(seq::count_components(lbl));
    } else if (a.algo == "louvain") {
        if (!seq_tier) {
            std::cerr << "louvain currently has a sequential (T0) tier only; use --tier seq.\n";
            return 3;
        }
        seq::LouvainResult lr;
        ms = time_median_ms([&]{ lr = seq::louvain(g); }, a.repeats);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "communities=%u modularity=%.4f levels=%d",
                      lr.num_communities, lr.modularity, lr.levels);
        result_summary = buf;
    } else {
        std::cerr << "unknown algo: " << a.algo << "\n";
        return 3;
    }

    // --- Emit a CSV row (append to results/*.csv from scripts) -----------
    // header: algo,tier,graph,vertices,edges,max_degree,avg_degree,median_ms,summary
    std::cout << a.algo << "," << a.tier << "," << graph_name << ","
              << s.num_vertices << "," << s.num_edges << ","
              << s.max_degree << "," << s.avg_degree << ","
              << ms << "," << result_summary << "\n";
    return 0;
}
