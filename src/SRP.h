#pragma once

#include <cereal/access.hpp>
#include <cstdint>
#include <vector>
#include <ctime>

namespace thirdai::hashing {

class SparseRandomProjection {

 public:
  SparseRandomProjection(uint32_t input_dim, uint32_t srps_per_table,
                         uint32_t num_tables, uint32_t seed = time(nullptr));

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const;

  uint32_t numTables() const {
    return _num_tables;
  }

  uint32_t range() const {
    return 1 << _srps_per_table;
  }

 private:
  uint32_t _num_tables, _srps_per_table, _total_num_srps, _dim, _sample_size;
  std::vector<int16_t> _random_bits;
  std::vector<uint32_t> _hash_indices;

  SparseRandomProjection() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_num_tables, _srps_per_table, _total_num_srps, _dim, _sample_size, _random_bits, _hash_indices);
  }
};

}  // namespace thirdai::hashing
