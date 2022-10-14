#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "MaxFlash.h"
#include "SRP.h"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <utility>
#include "Utils.h"

namespace py = pybind11;
using EmbeddingArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace thirdai::search {

// TODO(josh): This class is NOT currently safe to call concurrently.
// Fix this.
/**
 * Represents a collection of documents. We can incrementally update documents
 * and estimate the ColBERT score (sum of max similarities) between a query
 * and a document.
 * LABEL_T is an unsigned numerical type, currently uint8_t, uin16_t, uin32_t.
 */
template <typename LABEL_T>
class MaxFlashArray {
 public:
  // This will own the hash function and delete it during the destructor
  // TODO(josh): Remove naked pointers from hash function library so moves will
  // work, then change this to a unique pointer.
  // Any documents passed in larger than max_doc_size or larger than
  // the max value of LABEL_T will be truncated.
  // TODO(josh): Change truncation to error?
  // TODO(josh): Make LABEL_T allowed to be different for each document, so
  // it is as space efficient as possible
  MaxFlashArray(hashing::SparseRandomProjection* function, uint32_t hashes_per_table,
                uint64_t max_doc_size = std::numeric_limits<LABEL_T>::max());

  // This needs to be public since it's a top level serialization target, but
  // DO NOT call it unless you are creating a temporary object to serialize
  // into.
  MaxFlashArray(){};

  uint64_t addDocument(const EmbeddingArray& batch);

  std::vector<float> getDocumentScores(
      const EmbeddingArray& query,
      const std::vector<uint32_t>& documents_to_query) const;

  // Delete copy constructor and assignment
  MaxFlashArray(const MaxFlashArray&) = delete;
  MaxFlashArray& operator=(const MaxFlashArray&) = delete;

 private:
  std::vector<uint32_t> hash(const EmbeddingArray& batch) const;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_max_allowable_doc_size, _hash_function, _maxflash_array,
            _collision_count_to_sim);
  }

  LABEL_T _max_allowable_doc_size;
  std::unique_ptr<hashing::SparseRandomProjection> _hash_function;
  std::vector<std::unique_ptr<MaxFlash<LABEL_T>>> _maxflash_array;
  std::vector<float> _collision_count_to_sim;
};

}  // namespace thirdai::search