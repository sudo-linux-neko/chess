
#include "binpack_loader.h"
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cstdint>
namespace binpack {
struct BinpackDataset::Impl {
    std::ifstream file;
    size_t num_positions = 0;
    std::vector<size_t> position_offsets;
    uint32_t version = 0;
    explicit Impl(const std::string& filepath) : file(filepath, std::ios::binary) {
        if (!file.is_open()) throw std::runtime_error("Failed to open: " + filepath);
        read_header(); build_index();
    }
    void read_header() {
        char magic[4]; file.read(magic, 4);
        if (std::strncmp(magic, "BINP", 4) != 0) throw std::runtime_error("Invalid magic");
        file.read(reinterpret_cast<char*>(&version), 4);
        file.read(reinterpret_cast<char*>(&num_positions), 8);
    }
    void build_index() {
        position_offsets.reserve(num_positions);
        size_t pos = file.tellg();
        for (size_t i = 0; i < num_positions; ++i) {
            position_offsets.push_back(pos);
            uint16_t num_features; file.read(reinterpret_cast<char*>(&num_features), 2);
            pos = file.tellg(); pos += num_features * 6 + 17; file.seekg(pos);
        }
    }
    Position read_position(size_t idx) {
        if (idx >= num_positions) throw std::out_of_range("Index out of range");
        file.seekg(position_offsets[idx]);
        Position pos; uint16_t num_features;
        file.read(reinterpret_cast<char*>(&num_features), 2);
        pos.features.resize(num_features); pos.weights.resize(num_features);
        file.read(reinterpret_cast<char*>(pos.features.data()), num_features * 2);
        file.read(reinterpret_cast<char*>(pos.weights.data()), num_features * 4);
        file.read(reinterpret_cast<char*>(pos.wdl.data()), 12);
        file.read(reinterpret_cast<char*>(&pos.eval), 4);
        file.read(reinterpret_cast<char*>(&pos.game_result), 1);
        return pos;
    }
};
BinpackDataset::BinpackDataset(const std::string& filepath) : pimpl_(std::make_unique<Impl>(filepath)) {}
BinpackDataset::~BinpackDataset() = default;
size_t BinpackDataset::size() const { return pimpl_->num_positions; }
Position BinpackDataset::operator[](size_t idx) const { return pimpl_->read_position(idx); }
}
