import dessert_py
import numpy as np
import random

# Helper method that returns a tuple of two functions. The first function
# takes no arguments and returns a document retrieval index with all generated
# documents added. The second function takes a document retrieval index
# and queries it with generated queries, asserting that the top result is
# as expected, and also returning all results.
# The general idea for this test is that each word is a normal distribution
# somehwhere in the vector space. A doc is made up of a vector from each
# of words_per_doc normal distributions. A ground truth query is made up of
# some words from a single doc's word distributions and some random words.
def get_build_and_run_functions_random(num_docs=100, num_queries=100):

    hashes_per_table = 7
    num_tables = 32
    data_dim = 100
    vocab_size = 10000
    words_per_doc = 200
    words_per_query_random = 5
    words_per_query_from_doc = 10
    words_per_query = words_per_query_random + words_per_query_from_doc
    between_word_std = 1
    within_word_std = 0.1

    np.random.seed(42)
    random.seed(42)

    # Generate word centers
    word_centers = np.random.normal(size=(vocab_size, data_dim), scale=between_word_std)

    # Generates docs
    doc_word_ids = [
        random.sample(range(vocab_size), words_per_doc) for _ in range(num_docs)
    ]
    doc_offsets = np.random.normal(
        size=(num_docs, words_per_doc, data_dim), scale=within_word_std
    )
    docs = []
    for i in range(num_docs):
        doc = []
        for j in range(words_per_doc):
            doc.append(doc_offsets[i][j] + word_centers[doc_word_ids[i][j]])
        docs.append(doc)

    # Generate queries. GT for query i is doc i
    query_random_word_ids = [
        random.sample(range(vocab_size), words_per_query_random)
        for _ in range(num_queries)
    ]
    query_same_word_ids = [
        ids[:words_per_query_from_doc] for ids in doc_word_ids[:num_queries]
    ]
    query_word_ids = [a + b for a, b in zip(query_same_word_ids, query_random_word_ids)]
    query_offsets = np.random.normal(
        size=(num_queries, words_per_query, data_dim), scale=within_word_std
    )
    queries = []
    for i in range(num_queries):
        query = []
        for j in range(words_per_query):
            query.append(query_offsets[i][j] + word_centers[query_word_ids[i][j]])
        queries.append(query)

    index_func = lambda: _build_index_random(
        docs, hashes_per_table, num_tables, data_dim, word_centers
    )
    query_func = lambda index: _do_queries_random(index, queries, num_docs)

    return index_func, query_func


def _build_index_random(docs, hashes_per_table, num_tables, data_dim, centroids):
    index = dessert_py.DocRetrieval(
        centroids=centroids,
        hashes_per_table=hashes_per_table,
        num_tables=num_tables,
        dense_input_dimension=data_dim,
    )
    for i, doc in enumerate(docs):
        index.add_doc(doc_id=str(i), doc_embeddings=np.array(doc))
    return index


def _do_queries_random(index, queries, num_docs):
    result = []
    for gt, query in enumerate(queries):
        query_result = index.query(query_embeddings=np.array(query), top_k=10)
        result += query_result
        assert int(query_result[0]) == gt
    return result
