#pragma once

#include <cereal/access.hpp>
#include "TinyTable.h"
#include <memory>
#include <utility>

namespace thirdai::search {

template <typename LABEL_T>
class MaxFlash {
 public:
  MaxFlash(uint32_t num_tables, uint32_t range, LABEL_T num_elements,
           const std::vector<uint32_t>& hashes);

  float getScore(const std::vector<uint32_t>& query_hashes,
                 uint32_t num_elements, std::vector<uint32_t>& count_buffer,
                 const std::vector<float>& collision_count_to_sim) const;

  // Delete copy constructor and assignment
  MaxFlash(const MaxFlash&) = delete;
  MaxFlash& operator=(const MaxFlash&) = delete;

 private:

  MaxFlash<LABEL_T>() : _hashtable(0, 0, 0, std::vector<uint32_t>()){};

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hashtable);
  }
  
  hashtable::TinyTable<LABEL_T> _hashtable;
};

}  // namespace thirdai::search