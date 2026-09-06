"""Regression tests for the FAISS ``-1`` sentinel bug in ``dspy.retrievers.embeddings``.

FAISS ``IndexIVFPQ.search`` returns the sentinel id ``-1`` for slots it cannot fill when the
nprobe-limited probed pool is smaller than the requested candidate count. Before the fix,
``Embeddings`` forwarded those ``-1`` ids verbatim into ``dspy.Prediction.indices`` and into
reranking (where numpy negative-indexing silently aliased ``corpus[-1]``), producing invalid
indices and possibly duplicated ``corpus[-1]`` passages.

These tests use a small fake FAISS-like index (``_FakeFaissIndex``) that returns ``-1``
sentinels, so the sentinel-handling code paths run without requiring ``faiss-cpu`` to be
installed. The drop-in regression test from the bug report (``test_faiss_path_no_sentinel_indices``)
uses real ``faiss.IndexIVFPQ`` and is skipped via ``pytest.importorskip("faiss")`` when faiss
is absent.
"""

import numpy as np
import pytest

from dspy.retrievers.embeddings import Embeddings, EmbeddingsWithScores


class _FakeFaissIndex:
    """Mimics a FAISS IVF index whose ``search`` may return ``-1`` sentinel ids.

    FAISS returns -1 for slots it cannot fill (paired with +inf distance). We precompute
    the ids table per query and slice it to the requested ``num_candidates`` so the mock
    honors the same shape contract as ``faiss.IndexIVFPQ.search``.
    """

    def __init__(self, ids_table):
        self._ids_table = np.asarray(ids_table, dtype=np.int64)
        self._used = 0

    def search(self, query_embeddings, num_candidates):
        n_q = query_embeddings.shape[0]
        ids = self._ids_table[self._used : self._used + n_q, :num_candidates]
        self._used += n_q
        dists = np.where(ids >= 0, 0.0, np.inf).astype(np.float32)
        return dists, ids


def _make_retriever(cls, corpus, embedder, k, brute_force_threshold=None, normalize=False):
    """Construct a retriever on the brute-force path, leaving ``self.index`` free to be patched.

    Tests that need the FAISS path replace ``retriever.index`` with a ``_FakeFaissIndex`` so
    the sentinel handling code runs without requiring faiss-cpu to be installed.
    """
    if brute_force_threshold is None:
        brute_force_threshold = len(corpus) + 1
    return cls(
        corpus=corpus,
        embedder=embedder,
        k=k,
        brute_force_threshold=brute_force_threshold,
        normalize=normalize,
    )


def test_faiss_search_masks_sentinel_ids_with_mocked_index():
    """``_faiss_search`` must remap ``-1`` sentinels to ``0`` and return a mask marking them.

    This is the surgical test of the fix: it directly drives ``_faiss_search`` with a fake
    index that returns ``-1`` and verifies the (ids, mask) tuple has the documented shape.
    """
    corpus = [f"doc {i}" for i in range(6)]
    r = _make_retriever(Embeddings, corpus, lambda t: np.ones((len(t), 4), dtype="float32"), k=2)

    # Fake FAISS returns 3 real ids + 2 sentinel -1 ids for a single query.
    ids_table = np.array([[3, 5, 0, -1, -1]], dtype=np.int64)
    r.index = _FakeFaissIndex(ids_table)

    q_embeds = np.ones((1, 4), dtype="float32")
    ids, mask = r._faiss_search(q_embeds, 5)

    assert ids.shape == (1, 5)
    assert mask.shape == (1, 5)
    # Real ids passed through unchanged.
    assert ids[0, 0] == 3
    assert ids[0, 1] == 5
    assert ids[0, 2] == 0
    # Sentinel -1 remapped to 0 so the array stays rectangular.
    assert ids[0, 3] == 0
    assert ids[0, 4] == 0
    # Mask marks exactly the real candidates.
    assert mask.tolist() == [[True, True, True, False, False]]
    # No -1 anywhere in the returned ids.
    assert (ids >= 0).all()


def test_rerank_and_predict_drops_sentinel_candidates_via_mask():
    """When a sentinel-aliasing slot has the highest raw score, the mask must keep it out of top-k.

    Reproduces the active-corruption scenario: corpus[-1] is query-relevant and the FAISS
    return row uses -1 (aliased to corpus[-1] by numpy negative indexing). Without the fix,
    the aliased row scores max and dominates top-k, duplicating ``corpus[-1]``.
    """
    corpus = ["cat doc", "bird doc", "fish doc", "horse doc", "dog doc"]
    dim = 5

    def embedder(texts):
        a = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            if "dog" in t:
                a[i, 0] = 1.0
            elif "cat" in t:
                a[i, 1] = 1.0
            elif "bird" in t:
                a[i, 2] = 1.0
            elif "fish" in t:
                a[i, 3] = 1.0
            elif "horse" in t:
                a[i, 4] = 1.0
        return a

    r = _make_retriever(Embeddings, corpus, embedder, k=3, normalize=False)

    # Candidate ids exactly as FAISS would return: real id 4 (dog) at slot 0,
    # then four -1 sentinels (which numpy would alias to corpus[-1] = the dog doc).
    candidate_indices = np.array([[4, -1, -1, -1, -1]], dtype=np.int64)
    mask = candidate_indices >= 0  # [True, False, False, False, False]

    # Remap to 0 so candidate_embeddings stays rectangular (this is what _faiss_search does).
    candidate_indices = np.where(mask, candidate_indices, 0)

    q_embeds = embedder(["dog"])  # [1, 0, 0, 0, 0]
    results = r._rerank_and_predict(q_embeds, candidate_indices, mask)

    assert len(results) == 1
    passages, indices, scores = results[0]
    # Only the one real candidate survives (k=3 requested but only 1 real candidate probed).
    assert indices == [4]
    assert passages == ["dog doc"]
    assert scores == [pytest.approx(1.0)]
    # No -1, no duplicates, no out-of-range indices.
    assert all(0 <= i < len(corpus) for i in indices)
    assert len(indices) == len(set(indices))


def test_batch_forward_filters_sentinels_and_keeps_real_ranking():
    """End-to-end through ``_batch_forward``: mixed real + sentinel ids are filtered, real ones ranked.

    Guards the ≥k case: when at least k real candidates are probed, exactly k valid distinct
    results are returned in descending-score order.
    """
    corpus = ["cat doc", "dog doc", "bird doc", "fish doc", "horse doc"]
    dim = 5

    def embedder(texts):
        a = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            for j, key in enumerate(("cat", "dog", "bird", "fish", "horse")):
                if key in t:
                    a[i, j] = 1.0
        return a

    # k*10 = 30, but corpus has only 5 docs -> num_candidates clamps to 5.
    r = _make_retriever(Embeddings, corpus, embedder, k=3, normalize=False)
    # Fake FAISS returns 3 real ids (1, 0, 2) and 2 sentinels.
    ids_table = np.array([[1, -1, 0, -1, 2]], dtype=np.int64)
    r.index = _FakeFaissIndex(ids_table)

    results = r._batch_forward(["dog"])
    passages, indices, scores = results[0]

    # 3 real candidates probed, k=3 requested -> exactly 3 returned, ranked by score:
    # dog (1.0) > cat/bird (0.0). Order among ties is not asserted beyond being valid.
    assert len(indices) == 3
    assert len(passages) == 3
    assert len(scores) == 3
    # Highest-scoring real candidate (dog) ranks first.
    assert indices[0] == 1
    assert passages[0] == "dog doc"
    assert scores[0] == pytest.approx(1.0)
    # Remaining are the two tied real candidates (cat, bird) in any order.
    assert set(indices[1:]) == {0, 2}
    # No -1 sentinels, no duplicates, no out-of-range.
    assert all(0 <= i < len(corpus) for i in indices)
    assert len(indices) == len(set(indices))


def test_retriever_call_filters_sentinels_via_public_api():
    """End-to-end through the public ``retriever(query)`` API with a sentinel-producing fake index.

    This exercises ``Unbatchify`` + ``_batch_forward`` + ``_rerank_and_predict`` + the
    Prediction packing in ``forward``. Verifies ``dspy.Prediction.indices`` carries no -1
    and that the ``corpus[-1]`` duplication symptom (phantoms aliasing the last row) is gone.
    """
    corpus = [f"filler {i}" for i in range(8)]
    corpus[-1] = "the only dog document"
    dim = 6

    def embedder(texts):
        a = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            if "dog" in t:
                a[i, 0] = 1.0
            else:
                a[i, 1] = 1.0
        return a

    # k*10 = 50, corpus has 8 docs -> num_candidates = 8. Fake FAISS returns 8 ids,
    # 7 of which are sentinel -1 (so they would alias corpus[-1] = the dog doc without the fix).
    r = _make_retriever(Embeddings, corpus, embedder, k=5, normalize=False)
    ids_table = np.array([[7, -1, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    r.index = _FakeFaissIndex(ids_table)

    pred = r("dog")
    assert all(0 <= i < len(corpus) for i in pred.indices), f"sentinel leaked: {pred.indices}"
    # Only 1 real candidate was probed; result must drop the 4 sentinel slots that the bug
    # would have left in place (k=5, but only 1 valid candidate available).
    assert pred.indices == [7]
    assert pred.passages == ["the only dog document"]


def test_embeddings_with_scores_aligned_after_sentinel_filter():
    """``EmbeddingsWithScores`` must keep passages/indices/scores aligned after sentinel drops."""
    corpus = ["cat doc", "dog doc", "bird doc", "fish doc", "horse doc"]
    dim = 5

    def embedder(texts):
        a = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            for j, key in enumerate(("cat", "dog", "bird", "fish", "horse")):
                if key in t:
                    a[i, j] = 1.0
        return a

    # k=4 but only 3 real candidates probed (2 sentinels present).
    r = _make_retriever(EmbeddingsWithScores, corpus, embedder, k=4, normalize=False)
    ids_table = np.array([[1, 2, -1, 0, -1]], dtype=np.int64)
    r.index = _FakeFaissIndex(ids_table)

    pred = r("dog")

    # Three real candidates survive (k=4 requested but only 3 probed), ranked dog > {cat, bird}.
    assert len(pred.passages) == len(pred.indices) == len(pred.scores) == 3
    assert pred.indices[0] == 1
    assert pred.passages[0] == "dog doc"
    assert pred.scores[0] == pytest.approx(1.0)
    assert all(0 <= i < len(corpus) for i in pred.indices)
    # No duplicates, no -1.
    assert len(set(pred.indices)) == len(pred.indices)
    assert -1 not in pred.indices
    # Remaining two are the tied real candidates (cat, bird) in any order.
    assert set(pred.indices[1:]) == {0, 2}


def test_faiss_path_no_sentinel_indices():
    """Drop-in regression test from the bug report.

    Forces the FAISS path on a 300-doc corpus with ``k=25`` (k*10=250 > ~141 probed pool),
    which triggers FAISS ``IndexIVFPQ`` to return ``-1`` sentinels. Requires ``faiss-cpu``.
    """
    pytest.importorskip("faiss")
    np.random.seed(0)
    n, dim = 300, 64  # dim must be a multiple of _build_faiss's nbytes=32
    corpus = [f"doc {i} filler" for i in range(n)]
    corpus[-1] = "the only relevant document about dogs"

    class E:
        def __call__(self, texts):
            a = np.zeros((len(texts), dim), dtype="float32")
            for i, t in enumerate(texts):
                a[i, 0] = 1.0 if "dog" in t else 0.0
                a[i, 1] = 0.0 if "dog" in t else 1.0
            return a

    # Force the FAISS path on a small corpus by lowering brute_force_threshold.
    r = Embeddings(corpus=corpus, embedder=E(), k=25, brute_force_threshold=10)
    assert r.index is not None  # sanity: we're actually on the FAISS path

    pred = r("dog")
    assert all(i in range(len(corpus)) for i in pred.indices), f"sentinel -1 leaked into indices: {pred.indices}"
