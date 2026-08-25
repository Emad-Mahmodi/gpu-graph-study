// core/graph/generator.hpp
// Synthetic graph generators. RMAT produces power-law, skewed-degree graphs
// that mimic the shape of real call-graphs and network-flow graphs — the
// exact regime this study is about. Deterministic given a seed (reproducible).
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "csr.hpp"

namespace ggs {

// Generate an RMAT graph with 2^scale vertices and edge_factor * 2^scale edges.
// Default a/b/c/d = Graph500 parameters (0.57, 0.19, 0.19, 0.05) -> heavy skew.
CsrGraph make_rmat(int scale, int edge_factor, std::uint64_t seed = 1,
                   bool symmetrize = true,
                   double a = 0.57, double b = 0.19, double c = 0.19);

// A 2D grid (mesh) graph: side*side vertices, 4-connectivity. Unlike RMAT this
// is a *high-diameter, low-degree* graph (diameter ≈ 2*(side-1)), the opposite
// regime — used to show where frontier/active-set GPU strategies pay off. One
// connected component. Undirected (symmetrized).
CsrGraph make_grid(std::uint32_t side);

// A tiny hand-checkable graph used in unit tests (known BFS/WCC answers).
// 7 vertices, two connected components:
//   component A: 0-1-2-3 (a path with an extra 0-3 chord)
//   component B: 4-5-6   (a triangle)
CsrGraph make_test_graph();

}  // namespace ggs
