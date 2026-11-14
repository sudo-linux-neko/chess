
#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <memory>
#include <array>
namespace binpack {
struct Position {
    std::vector<uint16_t> features;
    std::vector<float> weights;
    std::array<float, 3> wdl;
    float eval;
    int8_t game_result;
};
class BinpackDataset {
public:
    explicit BinpackDataset(const std::string& filepath);
    ~BinpackDataset();
    size_t size() const;
    Position operator[](size_t idx) const;
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
