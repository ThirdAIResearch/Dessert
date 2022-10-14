#pragma once

#include <cereal/access.hpp>
#include <atomic>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace thirdai::hashtable {

/**
 * The goal of this class is to be a REALLY tiny hashtable.
 * Needless to say, you need to know the number of elements and
 * all hashes when you construct this table. By my estimate, current size is
 * 24 + sizeof(LABEL_T) * ((_hash_range + 1) * _num_tables + _num_elements *
 * _num_tables). As an example, if you are storing 200 elements in 32 tables
 * with a hash range of 64, this class will use less than 10 KB of memory.
 * LABEL_T should be an unsigned integer type, one of uint8_t, uint16_t, and
 * uint32_t. It should be the minimum size neccesary to fit the number of
 * elements in the table. Note that this class will give the first item id
 * 1, the second id 2, and so on.
 */
template <typename LABEL_T>
class TinyTable final {
 public:
  TinyTable(uint32_t num_tables, uint32_t range, LABEL_T num_elements,
            const std::vector<uint32_t>& hashes)
      : _hash_range(range),
        _num_elements(num_elements),
        _num_tables(num_tables),
        _index((_hash_range + 1 + _num_elements) * _num_tables) {
    for (uint32_t table = 0; table < num_tables; table++) {
      // Generate inverted index from hashes to vec_ids
      std::vector<std::vector<LABEL_T>> temp_buckets(_hash_range);
      for (uint64_t vec_id = 0; vec_id < num_elements; vec_id++) {
        uint64_t hash = hashes.at(vec_id * num_tables + table);
        temp_buckets.at(hash).push_back(vec_id);
      }

      // Populate bucket start and end offsets
      // See doc comment for _index to understand what is going on here
      const uint32_t table_offsets_start = table * (_hash_range + 1);
      _index.at(table_offsets_start) = 0;
      for (uint32_t bucket = 1; bucket <= _hash_range; bucket++) {
        _index.at(table_offsets_start + bucket) =
            _index.at(table_offsets_start + bucket - 1) +
            temp_buckets.at(bucket - 1).size();
      }

      // Populate hashes into table itself
      // See doc comment for _index to understand what is going on here
      uint64_t current_offset = _table_start + _num_elements * table;
      for (uint64_t bucket = 0; bucket < _hash_range; bucket++) {
        for (LABEL_T item : temp_buckets.at(bucket)) {
          _index.at(current_offset) = item;
          current_offset += 1;
        }
      }
    }
  }

  void queryByCount(const std::vector<uint32_t>& hashes, uint32_t hash_offset,
                    std::vector<uint32_t>& counts) const {
    for (uint32_t table = 0; table < _num_tables; table++) {
      uint32_t hash = hashes[hash_offset + table];
      LABEL_T start_offset = _index[(_hash_range + 1) * table + hash];
      LABEL_T end_offset = _index[(_hash_range + 1) * table + hash + 1];
      uint64_t table_offset =
          _table_start + static_cast<uint64_t>(table) * _num_elements;
      for (uint64_t offset = table_offset + start_offset;
           offset < table_offset + end_offset; offset++) {
        counts[_index[offset]]++;
      }
    }
  }

  constexpr uint32_t numTables() const { return _num_tables; }

  constexpr LABEL_T numElements() const { return _num_elements; }

  // Delete copy constructor and assignment
  TinyTable(const TinyTable&) = delete;
  TinyTable& operator=(const TinyTable&) = delete;

 private:
  TinyTable<LABEL_T>(){};

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hash_range, _num_elements, _num_tables, _table_start, _index);
  }

  // Techincally this is 16 + sizeof(LABEL_T) wasted bytes per table,
  // but it's fine for now
  uint32_t _hash_range;
  LABEL_T _num_elements;
  uint32_t _num_tables;
  uint64_t _table_start =
      static_cast<uint64_t>(_num_tables) * (_hash_range + 1);
  /** _index is a compact representation of a multitable hashtable. Each of the
   * hash table repetitions are conceptually split into two parts: a list of
   * offsets and a list of hashes. The hashes are the concatenated contents of
   * the buckets of the table with no space in between (so the hash list is
   * some permutation of 0 through _num_elements - 1). Thus we need a way to
   * tell where one bucket ends and the next begins. To do this, the ith entry
   * in the offset list is the index of the start of the ith hash bucket, and
   * the i+1th entry is the end of the ith hash bucket. Thus the offset list
   * for each table is of length _hash_range + 1 (the +1 comes from the fact
   * that we need both the beginning of the first bucket and the end of the
   * last, though technically we could get away without both of those and
   * just use if statements in the code, since they will always be 0 and
   * _num_elements + 1, respectively). The layout in memory of _index is simple:
   * the first _hash_range elements + 1 are offsets into the first table,
   * the second array second _hash_range + 1 elements are offsets into the
   * second table, and so on repeated _num_tables times. After this (starting
   * at _table_start), the next _num_elements are the elements in the first
   * table, the next _num_elements are the elements in the second table, and so
   * on for _num_tables times. Finally, we note that all of the offsets and ids
   * can be safely be stored as LABEL_Ts. Thus the total size of this vector
   * (including it's length, which is 8 bytes on a 64 bit machine) in bytes is
   * 8 + sizeof(LABEL_T) * _num_tables * (_hash_range + 1 + _num_elements).
   */
  std::vector<LABEL_T> _index;
};

}  // namespace thirdai::hashtable