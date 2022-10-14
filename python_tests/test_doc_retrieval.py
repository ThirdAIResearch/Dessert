# Add unit and release test markers for all tests in this file
import pytest

import dessert_py
from doc_retrieval_helpers import get_build_and_run_functions_random
import numpy as np
import time


def test_random_docs():
    index_func, query_func = get_build_and_run_functions_random()
    index = index_func()
    query_func(index)


@pytest.mark.unit
def test_random_docs_serialization():
    index_func, query_func = get_build_and_run_functions_random()
    index1 = index_func()
    index1.serialize_to_file("test.serialized")
    index2 = dessert_py.DocRetrieval.deserialize_from_file("test.serialized")
    query_results1 = query_func(index1)
    query_results2 = query_func(index2)
    for a, b in zip(query_results1, query_results2):
        assert a == b


def expect_error_on_construction(
    num_tables=1, dense_input_dimension=1, hashes_per_table=1, centroids=[[0]]
):
    with pytest.raises(Exception):
        dessert_py.DocRetrieval(
            centroids=centroids,
            hashes_per_table=hashes_per_table,
            num_tables=num_tables,
            dense_input_dimension=dense_input_dimension,
        )


def test_error_inputs():

    start = time.time()
    expect_error_on_construction(num_tables=0)
    expect_error_on_construction(num_tables=-7)
    expect_error_on_construction(dense_input_dimension=0)
    expect_error_on_construction(dense_input_dimension=-7)
    expect_error_on_construction(
        dense_input_dimension=100
    )  # Since the default centroids will still be dim=1
    expect_error_on_construction(hashes_per_table=0)
    expect_error_on_construction(hashes_per_table=-7)
    expect_error_on_construction(centroids=[])
    end = time.time()

    # We have a time assertion because should catch input errors quickly and
    # not e.g. build any big objects with invalid input
    assert end - start < 0.1


def test_add_doc_find_centroids_is_fast():
    """
    The idea of this test is the time to add a doc without the centroids shouldn't
    be that much more than the time to add a doc if we find the centroids with
    numpy and pass them in (this uses the assumption that finding the centroids is
    the slowest part). Note the query process uses the same code to find
    the closest centroids so this also will test the performance of that code.
    This could be flaky if suddenly the machine gets increased load in the middle
    of the test, but this seems unlikely.
    """

    num_centroids = 2**18  # 262144
    data_dim = 128
    words_per_doc = 256
    num_docs = 10
    # It would be nice if this were much smaller (this is 300%) but I don't want
    # a flaky test (it is this high because we can't fully optimize CPU
    # vectorization)
    max_slowdown_factor = 3.00
    centroids = np.random.rand(num_centroids, data_dim)
    centroids_transposed = centroids.transpose().copy()
    docs = np.random.rand(num_docs, words_per_doc, data_dim)

    doc_index_precomputed_centroids = dessert_py.DocRetrieval(
        centroids=centroids,
        hashes_per_table=8,
        num_tables=8,
        dense_input_dimension=data_dim,
    )
    doc_index_compute_centroids = dessert_py.DocRetrieval(
        centroids=centroids,
        hashes_per_table=8,
        num_tables=8,
        dense_input_dimension=data_dim,
    )

    numpy_start = time.time()
    centroid_ids_list = []
    for doc in docs:
        dot = doc.dot(centroids_transposed)
        centroid_ids_list.append(np.argmax(dot, axis=1))
    avg_numpy_time = (time.time() - numpy_start) / len(docs)

    precomputed_start = time.time()
    for i, (doc, centroid_ids) in enumerate(zip(docs, centroid_ids_list)):
        doc_index_precomputed_centroids.add_doc(
            doc_id=str(i),
            doc_embeddings=np.array(doc),
            doc_centroid_ids=centroid_ids.flatten(),
        )
    avg_precomputed_time = (time.time() - precomputed_start) / len(docs)

    with_compute_start = time.time()
    for i, doc in enumerate(docs):
        doc_index_compute_centroids.add_doc(doc_id=str(i), doc_embeddings=np.array(doc))
    avg_with_compute_time = (time.time() - with_compute_start) / len(docs)

    print(
        f"Average numpy time {avg_numpy_time}, average our time {avg_with_compute_time}"
    )
    assert (
        avg_with_compute_time - avg_precomputed_time
    ) / avg_numpy_time < max_slowdown_factor


# # TOOD(josh): Add the following tests:
# # 1. Test with lots of elements to ensure centroid reranking is reasonably fast
# # 2. Test with overall benchmarking test, probably on blade node,
# #    perhaps download msmarco index
