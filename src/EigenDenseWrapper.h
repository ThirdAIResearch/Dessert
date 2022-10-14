#pragma once

// You should include this file instead of <Eigen/Dense>. For now,
// it gets rid of clang tidy errors, adds Cereal serialization for
// dense Eigen matrices, and makes sure all included code is under
// EIGEN_MPL2_ONLY.

// This is a hack to get clang tidy to work, but it means if we ever want to
// switch to clang for compilation we'll have to figure this all out.
#if defined(__clang__)
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_PARALLELIZE
#endif

// Only use Eigen MPL2 code
#define EIGEN_MPL2_ONLY

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <Eigen/Dense>

// See the following link for further reference
// https://stackoverflow.com/questions/22884216/serializing-eigenmatrix-using-cereal-library/22885856
namespace cereal {

// Derived is the derived type of PlainObjectBase, e.g., a Matrix or Array
template <class Archive, class Derived>
void save(Archive& ar, Eigen::PlainObjectBase<Derived> const& to_serialize) {
  // This will be the Matrix or Array type itself
  typedef Eigen::PlainObjectBase<Derived> EigenTypeToSerialize;
  // By default, this will be one of int, float, double, std::complex<float>,
  // std::complex<double>, long double, long long int, and bool
  typedef typename EigenTypeToSerialize::Scalar EigenScalarType;
  // Cereal version of the Eigen Type
  typedef BinaryData<EigenScalarType> CerealScalarType;

  // Since this is a static assert and not a SFINAE, we are removing the
  // possibility of eventually writing a serializer for types that
  // are not natievly serializable by Cereal. However, the error messages
  // will be much simpler. CerealType is the
  static_assert(
      traits::is_output_serializable<CerealScalarType, Archive>::value,
      "The type of the Eigen matrix or array you are trying to "
      "serialize must be serializable.");

  // Only serialize the rows and the columns if they are not dynamic (if
  // they are static the user will load them into an object of the same
  // type, which will contain information about number of rows and columns)
  if (EigenTypeToSerialize::RowsAtCompileTime == Eigen::Dynamic) {
    ar(to_serialize.rows());
  }
  if (EigenTypeToSerialize::ColsAtCompileTime == Eigen::Dynamic) {
    ar(to_serialize.cols());
  }

  // Since this is a PlainObjectBase the memory must be contiguous and we
  // can copy directly in, see
  ar(binary_data(to_serialize.data(),
                 to_serialize.size() * sizeof(EigenScalarType)));
}

// This method is very similar to the above load method, see comments there
template <class Archive, class Derived>
void load(Archive& ar, Eigen::PlainObjectBase<Derived>& to_fill) {
  typedef Eigen::PlainObjectBase<Derived> EigenTypeToDeserialize;
  typedef typename EigenTypeToDeserialize::Scalar EigenScalarType;
  typedef BinaryData<EigenScalarType> CerealScalarType;
  static_assert(traits::is_input_serializable<CerealScalarType, Archive>::value,
                "The type of the Eigen matrix or array you are trying to "
                "deserialize must be deserializable.");

  // Different deserialization logic depending on whether the rows and cols
  // are dynamic to match the similar logic above
  Eigen::Index rows;
  Eigen::Index cols;
  if (EigenTypeToDeserialize::RowsAtCompileTime == Eigen::Dynamic) {
    ar(rows);
  } else {
    rows = EigenTypeToDeserialize::RowsAtCompileTime;
  }
  if (EigenTypeToDeserialize::ColsAtCompileTime == Eigen::Dynamic) {
    ar(cols);
  } else {
    cols = EigenTypeToDeserialize::ColsAtCompileTime;
  }

  to_fill.resize(rows, cols);
  ar(binary_data(to_fill.data(), static_cast<std::size_t>(
                                     rows * cols * sizeof(EigenScalarType))));
}
}  // namespace cereal