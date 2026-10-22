#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <utility>

namespace strata::ornith {
// Text format: magic, layer/expert/embedding/FF dimensions, then for each layer
// its expert stride and all routing counts. Counts are a prior, never predictions.
struct CachePrior { std::vector<int> order; std::vector<double> score; };
inline bool read_cache_prior(const char* path, const std::vector<size_t>& strides,
                             int experts, int embedding, int ff, int top_k, double tokens,
                             std::vector<CachePrior>& result, std::string& error) {
    std::ifstream in(path);
    std::string magic; size_t layers = 0; int ne = 0, n = 0, f = 0;
    auto fail = [&]() { error = "invalid or incompatible Ornith cache profile"; result.clear(); return false; };
    if (!(in >> magic >> layers >> ne >> n >> f) || magic != "ORNITH_CACHE_V1" ||
        layers != strides.size() || ne != experts || n != embedding || f != ff ||
        !std::isfinite(tokens) || tokens < 0 || tokens > 65536) return fail();
    result.clear();
    for (size_t li = 0; li < layers; ++li) {
        size_t stride = 0; CachePrior p; p.score.resize(ne); p.order.resize(ne);
        if (!(in >> stride) || stride != strides[li]) return fail();
        double sum = 0;
        for (double& value : p.score) {
            if (!(in >> value) || !std::isfinite(value) || value < 0 || value > 1e15 || std::floor(value) != value) return fail();
            sum += value;
        }
        if (sum <= 0) return fail();
        std::iota(p.order.begin(), p.order.end(), 0);
        std::stable_sort(p.order.begin(), p.order.end(), [&](int a, int b) { return p.score[a] > p.score[b]; });
        for (double& value : p.score) value *= tokens * top_k / sum;
        result.push_back(std::move(p));
    }
    std::string trailing;
    if (in >> trailing) return fail();
    return true;
}
} // namespace strata::ornith
