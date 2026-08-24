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
#include "../core/graph/csr.hpp"
#include "../core/graph/generator.hpp"

using namespace ggs;

namespace {

struct Args {
    std::string algo  = "bfs";
    std::string tier  = "seq";
    std::string graph;           // edge-list path (if empty, use rmat)
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
    if (!seq_tier && !omp_tier) {
        std::cerr << "tier '" << a.tier << "' not implemented yet "
                     "(GPU tiers land in a later phase). Use seq or openmp.\n";
        return 3;
    }

    if (a.algo == "bfs") {
        std::vector<std::int32_t> dist;
        ms = time_median_ms([&]{
            dist = seq_tier ? seq::bfs(g, a.source) : omp::bfs(g, a.source);
        }, a.repeats);
        std::int64_t reached = 0, maxd = 0;
        for (auto d : dist) if (d != seq::UNREACHED) { reached++; maxd = std::max<std::int64_t>(maxd, d); }
        result_summary = "reached=" + std::to_string(reached) +
                         " max_depth=" + std::to_string(maxd);
    } else if (a.algo == "pagerank") {
        std::vector<double> pr;
        ms = time_median_ms([&]{
            pr = seq_tier ? seq::pagerank(g) : omp::pagerank(g);
        }, a.repeats);
        double sum = 0.0, mx = 0.0;
        for (double x : pr) { sum += x; mx = std::max(mx, x); }
        result_summary = "sum=" + std::to_string(sum) + " max_pr=" + std::to_string(mx);
    } else if (a.algo == "wcc") {
        std::vector<VertexId> lbl;
        ms = time_median_ms([&]{
            lbl = seq_tier ? seq::wcc(g) : omp::wcc(g);
        }, a.repeats);
        result_summary = "components=" + std::to_string(seq::count_components(lbl));
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
