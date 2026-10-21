// The opt-in singleton exception must not weaken other tensor shape checks.
#include "gguf_fixture.hpp"
#include "strata/core/family_layout.hpp"
#include <chrono>
#include <cstdio>

int main() {
    using namespace strata::core;
    ModelGeometry g;
    g.family = strata::ModelFamily::Qwen35Moe;
    g.hc = g.hc_lr = 0;
    g.n_layers = 1;
    g.n_mtp_layers = 0;
    g.n_embd = 256;
    g.n_ff = g.n_ff_shexp = 8;
    g.n_expert = 1;
    g.n_vocab = 8;
    g.n_head = g.n_head_kv = 1;
    g.head_dim = 8;
    g.qsa_interval = 1;
    const auto path = std::filesystem::temp_directory_path() /
        ("strata-ornith-layout-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    int failed = 0;
    for (int variant = 0; variant < 5; ++variant) {
        std::vector<fixture::Tensor> tensors;
        auto add = [&](const TensorWant& want, const std::string& prefix) {
            fixture::Tensor t;
            t.name = prefix + want.name;
            t.shape.assign(want.shape.begin(), want.shape.end());
            if (want.name == "ffn_gate_exps.weight") t.type = 19; // reader must know IQ1_S byte sizes
            if (want.name == "ffn_gate_inp_shexp.weight") {
                if (variant == 1) t.shape = {256, 1};
                if (variant == 2) t.shape = {256, 2};
                if (variant == 3) t.shape = {1, 256};
            }
            if (variant == 4 && want.name == "attn_norm.weight") t.shape = {256, 1};
            tensors.push_back(t);
        };
        for (const auto& t : global_tensors(g)) add(t, "");
        for (const auto& t : layer_tensors(g, 0)) add(t, "blk.0.");
        fixture::write(path, {fixture::str("general.architecture", "qwen35moe")}, tensors);
        {
            auto model = strata::GgufModel::open(path.string());
            for (bool opt_in : {false, true}) {
                std::string error;
                const bool accepted = check_gguf_layout(model, g, error, nullptr, opt_in);
                const bool expected = variant == 0 || (variant == 1 && opt_in);
                if (accepted != expected) {
                    std::fprintf(stderr, "variant %d opt-in %d: %s\n", variant, opt_in, error.c_str());
                    ++failed;
                }
            }
        }
    }
    std::filesystem::remove(path);
    std::printf("Ornith layout: %d failures in 10 checks\n", failed);
    return failed ? 1 : 0;
}
