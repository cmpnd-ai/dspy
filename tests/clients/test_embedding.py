import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import dspy
from dspy.clients.embedding import Embedder


# Mock response format similar to litellm's embedding response.
class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [{"embedding": emb} for emb in embeddings]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model = "mock_model"
        self.object = "list"


@pytest.fixture
def cache(tmp_path):
    original_cache = dspy.cache
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache")
    yield
    dspy.cache = original_cache


def test_litellm_embedding(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello", "world"]
    mock_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
    ]

    with patch("litellm.embedding") as mock_litellm:
        # Configure mock to return proper response format.
        mock_litellm.return_value = MockEmbeddingResponse(mock_embeddings)

        # Create embedding instance and call it.
        embedding = Embedder(model, caching=True)
        result = embedding(inputs)

        # Verify litellm was called with correct parameters.
        # Because we disable the litellm cache, it should be called with caching=False.
        mock_litellm.assert_called_once_with(model=model, input=inputs, caching=False)

        assert len(result) == len(inputs)
        np.testing.assert_allclose(result, mock_embeddings)

        # Second call should be cached.
        result = embedding(inputs)
        assert mock_litellm.call_count == 1
        np.testing.assert_allclose(result, mock_embeddings)

        # Disable cache should issue new calls.
        embedding = Embedder(model, caching=False)
        result = embedding(inputs)
        assert mock_litellm.call_count == 2
        np.testing.assert_allclose(result, mock_embeddings)


def test_callable_embedding(cache):
    inputs = ["hello", "world", "test"]

    expected_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
        [0.7, 0.8, 0.9],  # embedding for "test"
    ]

    class EmbeddingFn:
        def __init__(self):
            self.call_count = 0

        def __call__(self, texts):
            # Simple callable that returns random embeddings.
            self.call_count += 1
            return expected_embeddings

    embedding_fn = EmbeddingFn()

    # Create embedding instance with callable
    embedding = Embedder(embedding_fn)
    result = embedding(inputs)

    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)

    result = embedding(inputs)
    # The second call should be cached.
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)


def test_callable_numpy_embedding_persists_to_disk(cache, tmp_path):
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache_safe", restrict_pickle=True)

    inputs = ["hello", "world"]
    expected_embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    embedding_fn = MagicMock(return_value=expected_embeddings)
    embedding = Embedder(embedding_fn)

    result = embedding(inputs)
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)

    result = embedding(inputs)
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)

    dspy.cache.reset_memory_cache()

    result = embedding(inputs)
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)


def test_invalid_model_type():
    # Test that invalid model type raises ValueError
    with pytest.raises(ValueError):
        embedding = Embedder(123)  # Invalid model type
        embedding(["test"])


@pytest.mark.asyncio
async def test_async_embedding():
    model = "text-embedding-ada-002"
    inputs = ["hello", "world"]
    mock_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
    ]

    with patch("litellm.aembedding") as mock_litellm:
        # Configure mock to return proper response format.
        mock_litellm.return_value = MockEmbeddingResponse(mock_embeddings)

        # Create embedding instance and call it.
        embedding = Embedder(model, caching=False)
        result = await embedding.acall(inputs)

        # Verify litellm was called with correct parameters.
        mock_litellm.assert_called_once_with(model=model, input=inputs, caching=False)

        assert len(result) == len(inputs)
        np.testing.assert_allclose(result, mock_embeddings)


def test_call_caching_false_overrides_instance_true(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.embedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, caching=True)
        embedding(inputs)
        embedding(inputs, caching=False)
        assert mock_litellm.call_count == 2


def test_call_caching_true_overrides_instance_false(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.embedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, caching=False)
        embedding(inputs, caching=True)
        embedding(inputs, caching=True)
        assert mock_litellm.call_count == 1


@pytest.mark.asyncio
async def test_acall_caching_false_overrides_instance_true(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.aembedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, caching=True)
        await embedding.acall(inputs)
        await embedding.acall(inputs, caching=False)
        assert mock_litellm.call_count == 2


@pytest.mark.asyncio
async def test_acall_caching_true_overrides_instance_false(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.aembedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, caching=False)
        await embedding.acall(inputs, caching=True)
        await embedding.acall(inputs, caching=True)
        assert mock_litellm.call_count == 1


# --- Callable-model cache-collision fixtures and tests ---
#
# The following helper classes are intentionally defined at module top-level (not inside a
# test function) so that ``inspect.getsource`` can retrieve their source. ``FakeSTSource``
# mirrors ``SentenceTransformer``: ``encode`` is a *class-level* method, so
# ``inspect.getsource(instance.encode)`` returns byte-identical source across two distinct
# instances and triggers the ``<callable_source:...>`` branch of ``_transform_value``.
# ``FakeCallableModel`` defines ``__call__`` but is a callable *instance* (no ``__name__``),
# which triggers the ``<callable:lambda>`` fallback branch.


class FakeSTSource:
    """Mimics ``SentenceTransformer.encode`` as a class-level bound method (source branch)."""

    def __init__(self, offset=0):
        self.offset = offset
        self.call_count = 0
        self.last_kwargs = None

    def encode(self, texts, **kwargs):
        self.call_count += 1
        self.last_kwargs = dict(kwargs)
        return [[float(self.offset + i)] for i in range(len(texts))]


class FakeCallableModel:
    """A callable instance (defines ``__call__``); triggers the fallback branch."""

    def __init__(self, offset=0):
        self.offset = offset
        self.call_count = 0
        self.last_kwargs = None

    def __call__(self, texts, **kwargs):
        self.call_count += 1
        self.last_kwargs = dict(kwargs)
        return [[float(self.offset + i)] for i in range(len(texts))]


def test_transform_value_branches_behave_as_reported():
    """Confirm the two buggy branches (source + fallback) collapse distinct callables."""
    from dspy.clients.cache import _transform_value

    a, b = FakeSTSource(0), FakeSTSource(1)
    # Source branch: identical source across same-class bound methods.
    assert inspect.getsource(a.encode) == inspect.getsource(b.encode)
    assert _transform_value(a.encode) == _transform_value(b.encode)

    ia, ib = FakeCallableModel(0), FakeCallableModel(1)
    # Fallback branch: instances lack __name__ -> both collapse to "<callable:lambda>".
    assert _transform_value(ia) == _transform_value(ib) == "<callable:lambda>"


def test_model_id_disambiguates_source_branch(cache):
    """Documented pattern ``Embedder(model.encode)`` + ``model_id`` no longer collides."""
    inst_a = FakeSTSource(offset=0)
    inst_b = FakeSTSource(offset=1000)
    emb_a = Embedder(inst_a.encode, model_id="ckpt-a")
    emb_b = Embedder(inst_b.encode, model_id="ckpt-b")
    inputs = ["x", "y"]

    r_a = emb_a(inputs)
    r_b = emb_b(inputs)

    assert inst_a.call_count == 1
    assert inst_b.call_count == 1  # computed its own, did not reuse emb_a's cache
    assert not np.array_equal(r_a, r_b)
    np.testing.assert_allclose(r_a, [[0.0], [1.0]])
    np.testing.assert_allclose(r_b, [[1000.0], [1001.0]])


def test_model_id_disambiguates_fallback_branch_callable_instance(cache):
    """Distinct callable instances (fallback branch) no longer collide with ``model_id``."""
    inst_a = FakeCallableModel(offset=0)
    inst_b = FakeCallableModel(offset=1000)
    emb_a = Embedder(inst_a, model_id="instance-a")
    emb_b = Embedder(inst_b, model_id="instance-b")
    inputs = ["x", "y"]

    r_a = emb_a(inputs)
    r_b = emb_b(inputs)

    assert inst_a.call_count == 1
    assert inst_b.call_count == 1
    assert not np.array_equal(r_a, r_b)
    np.testing.assert_allclose(r_a, [[0.0], [1.0]])
    np.testing.assert_allclose(r_b, [[1000.0], [1001.0]])


def test_same_model_id_reuses_cache_across_instances(cache, tmp_path):
    """A stable ``model_id`` still reuses the cache for the same model across instances/sessions."""
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache_reuse")

    inst_a = FakeSTSource(offset=0)
    inst_b = FakeSTSource(offset=0)  # same vectors -> models intended to be "the same"
    emb_a = Embedder(inst_a.encode, model_id="same-ckpt")
    emb_b = Embedder(inst_b.encode, model_id="same-ckpt")
    inputs = ["x"]

    r_a = emb_a(inputs)
    assert inst_a.call_count == 1

    # Clear memory cache to simulate a fresh process reusing the on-disk cache.
    dspy.cache.reset_memory_cache()

    r_b = emb_b(inputs)
    # Same model_id + same input -> on-disk cache hit; inst_b never invoked.
    assert inst_b.call_count == 0
    np.testing.assert_allclose(r_b, r_a)


def test_model_id_prevents_cross_cache_collision(cache, tmp_path):
    """Simulate the cross-process scenario from the report (E3): distinct ``model_id`` values
    keep two same-class bound methods from sharing cached vectors through the on-disk cache."""
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache_xproc")

    inst_a = FakeSTSource(offset=0)
    emb_a = Embedder(inst_a.encode, model_id="checkpoint-A")
    inputs = ["hello", "world"]

    r_a = emb_a(inputs)
    assert inst_a.call_count == 1

    # Fresh process: clear memory cache, leaving only the shared on-disk cache.
    dspy.cache.reset_memory_cache()

    inst_b = FakeSTSource(offset=1000)
    emb_b = Embedder(inst_b.encode, model_id="checkpoint-B")
    r_b = emb_b(inputs)

    # inst_b computes its own vectors; it does NOT read inst_a's on-disk cache entry.
    assert inst_b.call_count == 1
    assert not np.array_equal(r_b, r_a)
    np.testing.assert_allclose(r_b, [[1000.0], [1001.0]])


def test_model_id_none_preserves_existing_cache_keys(cache):
    """``model_id=None`` (explicit or unset) must not add a key dimension, so existing cache
    entries are still reused (no silent global cache invalidation)."""
    inst = FakeSTSource(offset=0)
    emb_no_id = Embedder(inst.encode)  # unset
    emb_none = Embedder(inst.encode, model_id=None)  # explicit None
    inputs = ["hello"]

    r1 = emb_no_id(inputs)
    assert inst.call_count == 1

    r2 = emb_none(inputs)
    # Same key (no model_id field) -> cache hit, callable not invoked again.
    assert inst.call_count == 1
    np.testing.assert_allclose(r2, r1)


def test_model_id_not_forwarded_to_callable(cache):
    """``model_id`` folds into the cache key but is never passed to the callable model."""
    inst = FakeSTSource(offset=0)
    emb = Embedder(inst.encode, model_id="my-ckpt", caching=True)
    emb(["hello"])

    assert inst.call_count == 1
    assert inst.last_kwargs is not None
    assert "model_id" not in inst.last_kwargs
    assert "caching" not in inst.last_kwargs


def test_model_id_not_forwarded_to_litellm(cache):
    """``model_id`` is not forwarded to litellm for hosted models."""
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.embedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, model_id="my-ckpt", caching=True)
        result = embedding(inputs)

        mock_litellm.assert_called_once()
        assert "model_id" not in mock_litellm.call_args.kwargs
        np.testing.assert_allclose(result, [[0.1, 0.2, 0.3]])


def test_per_call_model_id_overrides_instance_model_id(cache):
    """A per-call ``model_id`` overrides the instance-level one for the cache key."""
    inst_a = FakeSTSource(offset=0)
    inst_b = FakeSTSource(offset=1000)
    emb = Embedder(inst_a.encode, model_id="ckpt")

    # First call uses the instance model_id "ckpt".
    emb(["x"])
    assert inst_a.call_count == 1

    # Second call with the same callable but a different per-call model_id must recompute
    # (distinct cache key), and may bind to a different underlying model.
    emb.model = inst_b.encode
    r2 = emb(["x"], model_id="ckpt-b")
    assert inst_b.call_count == 1
    np.testing.assert_allclose(r2, [[1000.0]])


@pytest.mark.asyncio
async def test_model_id_disambiguates_async(cache):
    """The async path also disambiguates distinct callable models via ``model_id``."""
    inst_a = FakeSTSource(offset=0)
    inst_b = FakeSTSource(offset=1000)
    emb_a = Embedder(inst_a.encode, model_id="ckpt-a")
    emb_b = Embedder(inst_b.encode, model_id="ckpt-b")
    inputs = ["x", "y"]

    r_a = await emb_a.acall(inputs)
    r_b = await emb_b.acall(inputs)

    assert inst_a.call_count == 1
    assert inst_b.call_count == 1
    assert not np.array_equal(r_a, r_b)
    np.testing.assert_allclose(r_a, [[0.0], [1.0]])
    np.testing.assert_allclose(r_b, [[1000.0], [1001.0]])
