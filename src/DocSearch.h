#pragma once

#include "EigenDenseWrapper.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include "MaxFlashArray.h"
#include "Utils.h"
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <memory>

// TODO(josh): Figure out why sometimes the query is retruning duplicates (but very rarely)

using Centroids = py::array_t<float, py::array::c_style | py::array::forcecast>;
using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using namespace std::chrono;

namespace thirdai::search {

// TODO(josh): This class is NOT currently safe to call concurrently.
// TODO(josh): Right now this only has support for documents
// with a max of 256 embeddings. If there are more than 256 embeddings, it
// silently truncates. This should be fixed with a dynamic tiny table size,
// but for now I think we should keep this a silent error. If we threw an
// error existing scripts would fail, and printing out a warning is inelegant
// (we may print out thousands of warnings).
/**
 * Represents a service that allows document addition, removal, and queries.
 * For now, can represent at most 2^32 - 1 documents.
 */
class DocSearch {
 public:
  DocSearch(uint32_t hashes_per_table, uint32_t num_tables, uint32_t dense_dim,
            const Centroids& centroids_input)
      : _dense_dim(dense_dim),
        _nprobe_query(2),
        _largest_internal_id(0),
        _num_centroids(centroids_input.request().shape.at(0)),
        _centroid_id_to_internal_id(_num_centroids) {

    auto buffer = centroids_input.request();
    uint32_t num_centroids = buffer.shape[0];
    uint32_t centroids_dense_dim = buffer.shape[1];

    if (dense_dim == 0 || num_tables == 0 || hashes_per_table == 0) {
      throw std::invalid_argument(
          "The passed in dense dimension, number of tables, and hashes per "
          "table must all be greater than 0.");
    }
    if (num_centroids == 0) {
      throw std::invalid_argument(
          "Must pass in at least one centroid, found 0.");
    }
    if (centroids_dense_dim != _dense_dim) {
      throw std::invalid_argument("The passed in centroids array must have dimension equal to dense_dim");
    }
    _nprobe_query = std::min<uint64_t>(centroids_input.size(), _nprobe_query);

    // We delay constructing this so that we can do sanitize input in
    // this constructor rather than in FastSRP and MaxFlashArray
    _document_array = std::make_unique<MaxFlashArray<uint8_t>>(
        new thirdai::hashing::SparseRandomProjection(dense_dim, hashes_per_table,
                                      num_tables),
        hashes_per_table);

    // Do a copy because the centroids are a reference
    float* centroid_data = static_cast<float*>(buffer.ptr);
    _centroids = MatrixXfRowMajor::Map(centroid_data, num_centroids, centroids_dense_dim);
    _centroids.transposeInPlace();
  }

  // Returns true if this is a new document, false if this was an old document
  // and we updated it.
  bool addDocument(const EmbeddingArray& doc_embeddings, const std::string& doc_id) {
    std::vector<uint32_t> centroid_ids = getNearestCentroids(doc_embeddings, 1);
    return addDocumentWithCentroids(doc_embeddings, doc_id, centroid_ids);
  }

  // Returns true if this is a new document, false if this was an old document
  // and we updated it.
  bool addDocumentWithCentroids(const EmbeddingArray& doc_embeddings,
                                const std::string& doc_id,
                                const std::vector<uint32_t>& doc_centroid_ids) {
    // The document array assigned the new document to an "internal_id" when
    // we add it, which now becomes the integer representing this document in
    // our system. They differ from the passed in doc_id, which is an arbitrary
    // string that uniquely identifies the document; the internal_id is the
    // next available smallest positive integer that from now on uniquely
    // identifies the document.
    uint32_t internal_id = _document_array->addDocument(doc_embeddings);
    _largest_internal_id = std::max(_largest_internal_id, internal_id);

    for (uint32_t centroid_id : doc_centroid_ids) {
      _centroid_id_to_internal_id.at(centroid_id).push_back(internal_id);
    }

    _doc_id_to_internal_id[doc_id] = internal_id;

    // We need to call resize here instead of simply push_back because the
    // internal_id we get assigned might not necessarily be equal to the
    // push_back index, since if we concurrently add two documents at the same
    // time race conditions might reorder who gets assigned an id first and
    // who gets to this line first.
    if (internal_id >= _internal_id_to_doc_id.size()) {
      _internal_id_to_doc_id.resize(internal_id + 1);
    }
    _internal_id_to_doc_id.at(internal_id) = doc_id;

    return true;
  }

  std::vector<std::string> query(
      const EmbeddingArray& embeddings, uint32_t top_k,
      uint32_t num_to_rerank) {
    std::vector<uint32_t> centroid_ids =
        getNearestCentroids(embeddings, _nprobe_query);
    return queryWithCentroids(embeddings, centroid_ids, top_k, num_to_rerank);
  }

  std::vector<std::string> queryWithCentroids(
      const EmbeddingArray& embeddings, const std::vector<uint32_t>& centroid_ids,
      uint32_t top_k, uint32_t num_to_rerank) {
    auto buffer = embeddings.request();
    uint32_t num_vectors_in_query = buffer.shape[0];
    uint32_t dense_dim = buffer.shape[1];
    if (dense_dim != _dense_dim) {
      throw std::invalid_argument("Invalid row dimension");      
    }
    if (num_vectors_in_query == 0) {
      throw std::invalid_argument("Need at least one query vector but found 0");
    }
    if (top_k == 0) {
      throw std::invalid_argument(
          "The passed in top_k must be at least 1, was 0");
    }
    if (top_k > num_to_rerank) {
      throw std::invalid_argument(
          "The passed in top_k must be <= the passed in num_to_rerank");
    }

    std::vector<uint32_t> top_k_internal_ids =
        frequencyCountCentroidBuckets(centroid_ids, num_to_rerank);
    
    std::vector<uint32_t> reranked =
        rankDocuments(embeddings, top_k_internal_ids);

    uint32_t result_size = std::min<uint32_t>(reranked.size(), top_k);
    std::vector<std::string> result(result_size);
    for (uint32_t i = 0; i < result_size; i++) {
      uint32_t internal_id = reranked.at(i);
      std::string doc_id = _internal_id_to_doc_id.at(internal_id);
      result.at(i) = doc_id;
    }

    return result;
  }

  // Delete copy constructor and assignment
  DocSearch(const DocSearch&) = delete;
  DocSearch& operator=(const DocSearch&) = delete;


  void serialize_to_file(const std::string& path) {
    std::ofstream filestream(path, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<DocSearch> deserialize_from_file(
      const std::string& path) {
    std::ifstream filestream(path, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<DocSearch> serialize_into(new DocSearch());
    iarchive(*serialize_into);
    return serialize_into;
  }


  // This method returns a permutation of the input internal_ids_to_rerank
  // sorted in descending order by the approximated score of that document.
  std::vector<uint32_t> rankDocuments(
      const EmbeddingArray& query_embeddings,
      const std::vector<uint32_t>& internal_ids_to_rerank) const {
    // This returns a vector of scores, where the ith score is the score of
    // the document with the internal_id at internal_ids_to_rerank[i]
    std::vector<float> document_scores = _document_array->getDocumentScores(
        query_embeddings, internal_ids_to_rerank);

    // This is a little confusing, these sorted_indices are indices into
    // the document_scores array and represent a ranking of the
    // internal_ids_to_rerank. We still need to use this ranking to permute and
    // return internal_ids_to_rerank, which we do below.
    std::vector<uint32_t> sorted_indices = argsort_descending(document_scores);

    std::vector<uint32_t> permuted_ids(internal_ids_to_rerank.size());
    for (uint32_t i = 0; i < permuted_ids.size(); i++) {
      permuted_ids[i] = internal_ids_to_rerank.at(sorted_indices[i]);
    }

    return permuted_ids;
  }

  // This method returns a permutation of the input internal_ids_to_rerank
  // sorted in descending order by the approximated score of that document.
  std::vector<float> getScores(
      const EmbeddingArray& query_embeddings,
      const std::vector<uint32_t>& internal_ids_to_rerank) const {
        return _document_array->getDocumentScores(query_embeddings, internal_ids_to_rerank);
    }

  
 private:

  DocSearch() { };

  uint32_t _dense_dim, _nprobe_query, _largest_internal_id, _num_centroids;
  // This is a uinque_ptr rather than the object itself so that we can delay
  // constructing it until after input sanitization; see the constructor for m
  // more information.
  std::unique_ptr<MaxFlashArray<uint8_t>> _document_array;
  // These are stored tranposed for ease of multiplication
  MatrixXfRowMajor _centroids;
  std::vector<std::vector<uint32_t>> _centroid_id_to_internal_id;
  std::unordered_map<std::string, uint32_t> _doc_id_to_internal_id;
  std::vector<std::string> _internal_id_to_doc_id;
  std::vector<int32_t> _count_buffer;

  // Finds the nearest nprobe centroids for each vector in the batch and
  // then concatenates all of the centroid ids across the batch.
  std::vector<uint32_t> getNearestCentroids(const EmbeddingArray& batch,
                                            uint32_t nprobe) const {
    auto buffer = batch.request();
    uint32_t num_vectors = buffer.shape[0];
    Eigen::Map<MatrixXfRowMajor> eigen_batch(static_cast<float *>(buffer.ptr), num_vectors, _dense_dim);
    MatrixXfRowMajor eigen_result = eigen_batch * _centroids;
    std::vector<uint32_t> nearest_centroids(num_vectors * nprobe);

#pragma omp parallel for default(none) \
    shared(batch, eigen_result, nprobe, nearest_centroids, num_vectors)
    for (uint32_t i = 0; i < num_vectors; i++) {
      std::vector<uint32_t> probe_results = argmax(eigen_result.row(i), nprobe);
      for (uint32_t p = 0; p < nprobe; p++) {
        nearest_centroids.at(i * nprobe + p) = probe_results.at(p);
      }
    }

    removeDuplicates(nearest_centroids);
    return nearest_centroids;
  }

  std::vector<uint32_t> frequencyCountCentroidBuckets(
      const std::vector<uint32_t>& centroid_ids, uint32_t top_k) {

    // This array gets zerod in between 
    if (_count_buffer.size() != _largest_internal_id + 1) {
      _count_buffer = std::vector<int32_t>(_largest_internal_id + 1, 0);
    }

    // This is a mild race condition but it shouldn't significantly affect results
    #pragma omp parallel for
    for (uint32_t centroid_id : centroid_ids) {
      for (uint32_t internal_id : _centroid_id_to_internal_id.at(centroid_id)) {
        _count_buffer[internal_id] += 1;
      }
    }

    // Find the top k by looping through the input again to know what counts
    // values to examine (we can avoid traversing all possible counts, many
    // of which are zero). Since we do it this way for performance we can't
    // use the argmax util function. We negate a counts element when we have
    // seen it so that if we come across it again (if there is more than one
    // occurence of it) we can ignore it.
    // Note also that the heap is a max heap, so we negate everything to make
    // it a min heap in effect.

    std::priority_queue<std::pair<int32_t, uint32_t>> heap;
    for (uint32_t centroid_id : centroid_ids) {
      for (uint32_t internal_id : _centroid_id_to_internal_id.at(centroid_id)) {
        if (_count_buffer.at(internal_id) < 0) {
          continue;
        }
        int32_t negative_count = _count_buffer.at(internal_id) * -1;
        _count_buffer.at(internal_id) = 0;
        if (heap.size() < top_k || negative_count < heap.top().first) {
          heap.emplace(negative_count, internal_id);
        }
        if (heap.size() > top_k) {
          heap.pop();
        }
      }
    }

    std::vector<uint32_t> result;
    while (!heap.empty()) {
      // Top is the pair with the smallest score still in the heap as the first
      // element and the internal_id as the second element, so we push back
      // the second element, the internal_id, into the result vector.
      result.push_back(heap.top().second);
      heap.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
  }

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dense_dim, _nprobe_query, _largest_internal_id, _num_centroids,
            _document_array, _centroids, _centroid_id_to_internal_id, 
            _doc_id_to_internal_id, _internal_id_to_doc_id);
  }
};

}  // namespace thirdai::search