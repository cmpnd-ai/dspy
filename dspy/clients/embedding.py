from __future__ import annotations

from typing import Any, Callable

from dspy.clients._litellm import get_litellm
from dspy.clients.cache import request_cache
from dspy.utils.lazy_import import require

np = require("numpy")


def _get_litellm():
    return get_litellm(feature="dspy.Embedder")


class Embedder:
    """DSPy embedding class.

    The class for computing embeddings for text inputs. This class provides a unified interface for both:

    1. Hosted embedding models (e.g. OpenAI's text-embedding-3-small) via litellm integration
    2. Custom embedding functions that you provide

    For hosted models, simply pass the model name as a string (e.g., "openai/text-embedding-3-small"). The class will use
    litellm to handle the API calls and caching.

    For custom embedding models, pass a callable function that:
    - Takes a list of strings as input.
    - Returns embeddings as either:
        - A 2D numpy array of float32 values
        - A 2D list of float32 values
    - Each row should represent one embedding vector

    Args:
        model: The embedding model to use. This can be either a string (representing the name of the hosted embedding
            model, must be an embedding model supported by litellm) or a callable that represents a custom embedding
            model.
        batch_size (int, optional): The default batch size for processing inputs in batches. Defaults to 200.
        caching (bool, optional): Whether to cache the embedding response when using a hosted model. Defaults to True.
        model_id (str | None, optional): A stable, user-supplied identifier for the underlying callable model, used only
            to disambiguate the embedding cache key (it is never forwarded to the model). Set this when caching is
            enabled and the callable ``model`` may collide in the cache with a *different* callable model that shares
            the same source (e.g. two instances/checkpoints of the same ``SentenceTransformer`` class, whose bound
            ``encode`` methods are byte-identical) or the same ``__name__`` (e.g. two callable instances or
            ``functools.partial`` wrappers, which would both collapse to ``"<callable:lambda>"``). A stable string
            (such as the checkpoint name) survives across processes, so the same ``model_id`` reuses the on-disk cache
            while distinct ``model_id`` values never share cached results. When unset (``None``), this argument is
            omitted from the cache key entirely, preserving existing cache keys. Defaults to None.
        **kwargs: Additional default keyword arguments to pass to the embedding model.

    Examples:
        Example 1: Using a hosted model.

        ```python
        import dspy

        embedder = dspy.Embedder("openai/text-embedding-3-small", batch_size=100)
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 1536)
        ```

        Example 2: Using any local embedding model, e.g. from https://huggingface.co/models?library=sentence-transformers.

        ```python
        # pip install sentence_transformers
        import dspy
        from sentence_transformers import SentenceTransformer

        # Load an extremely efficient local model for retrieval
        model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

        embedder = dspy.Embedder(model.encode)
        embeddings = embedder(["hello", "world"], batch_size=1)

        assert embeddings.shape == (2, 1024)
        ```

        Example 3: Using a custom function.

        ```python
        import dspy
        import numpy as np

        def my_embedder(texts):
            return np.random.rand(len(texts), 10)

        embedder = dspy.Embedder(my_embedder)
        embeddings = embedder(["hello", "world"], batch_size=1)

        assert embeddings.shape == (2, 10)
        ```

        Example 4: Avoiding cache collisions across distinct callable models of the same class.

        The cache key for a callable ``model`` is derived from the callable's source (or ``__name__``), which is
        identical across two instances/checkpoints of the same class (e.g. two ``SentenceTransformer`` checkpoints) and
        collapses to ``"<callable:lambda>"`` for callable instances / ``functools.partial``. With ``caching=True``
        (the default) and a shared cache namespace (e.g. the default on-disk ``~/.dspy_cache`` reused by every run),
        such callables would otherwise collide and return the wrong cached embeddings. Pass a stable, checkpoint-level
        ``model_id`` to disambiguate the cache key while still reusing the cache across processes for the same model:

        ```python
        import dspy
        from sentence_transformers import SentenceTransformer

        # Switching checkpoints across runs that share ~/.dspy_cache: pass the checkpoint
        # name as ``model_id`` so each checkpoint keeps its own cached vectors.
        embedder = dspy.Embedder(SentenceTransformer("paraphrase-MiniLM-L6-v2").encode, model_id="paraphrase-MiniLM-L6-v2")
        embeddings = embedder(["hello", "world"], batch_size=1)
        ```
    """

    def __init__(
        self,
        model: str | Callable,
        batch_size: int = 200,
        caching: bool = True,
        model_id: str | None = None,
        **kwargs: dict[str, Any],
    ):
        self.model = model
        self.batch_size = batch_size
        self.caching = caching
        self.model_id = model_id
        self.default_kwargs = kwargs

    def _preprocess(self, inputs, batch_size=None, caching=None, model_id=None, **kwargs):
        if isinstance(inputs, str):
            is_single_input = True
            inputs = [inputs]
        else:
            is_single_input = False

        if not all(isinstance(inp, str) for inp in inputs):
            raise ValueError("All inputs must be strings.")

        batch_size = batch_size or self.batch_size
        caching = caching if caching is not None else self.caching
        model_id = model_id if model_id is not None else self.model_id
        merged_kwargs = self.default_kwargs.copy()
        merged_kwargs.update(kwargs)

        input_batches = []
        for i in range(0, len(inputs), batch_size):
            input_batches.append(inputs[i : i + batch_size])

        return input_batches, caching, model_id, merged_kwargs, is_single_input

    def _postprocess(self, embeddings_list, is_single_input):
        embeddings = np.array(embeddings_list, dtype=np.float32)
        if is_single_input:
            return embeddings[0]
        else:
            return np.array(embeddings, dtype=np.float32)

    def __call__(
        self,
        inputs: str | list[str],
        batch_size: int | None = None,
        caching: bool | None = None,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Compute embeddings for the given inputs.

        Args:
            inputs: The inputs to compute embeddings for, can be a single string or a list of strings.
            batch_size (int, optional): The batch size for processing inputs. If None, defaults to the batch_size set
                during initialization.
            caching (bool, optional): Whether to cache the embedding response when using a hosted model. If None,
                defaults to the caching setting from initialization.
            model_id (str | None, optional): If provided, overrides the ``model_id`` set during initialization for this
                call only and is folded into the embedding cache key (it is never forwarded to the model). See
                ``Embedder`` for the full cache-safety rationale.
            kwargs: Additional keyword arguments to pass to the embedding model. These will override the default
                kwargs provided during initialization.

        Returns:
            numpy.ndarray: If the input is a single string, returns a 1D numpy array representing the embedding.
            If the input is a list of strings, returns a 2D numpy array of embeddings, one embedding per row.
        """
        input_batches, caching, model_id, kwargs, is_single_input = self._preprocess(
            inputs, batch_size, caching, **kwargs
        )

        compute_embeddings = _cached_compute_embeddings if caching else _compute_embeddings

        call_kwargs = {"caching": caching}
        if model_id is not None:
            call_kwargs["model_id"] = model_id

        embeddings_list = []

        for batch in input_batches:
            embeddings_list.extend(compute_embeddings(self.model, batch, **call_kwargs, **kwargs))
        return self._postprocess(embeddings_list, is_single_input)

    async def acall(self, inputs, batch_size=None, caching=None, **kwargs):
        input_batches, caching, model_id, kwargs, is_single_input = self._preprocess(
            inputs, batch_size, caching, **kwargs
        )

        acompute_embeddings = _cached_acompute_embeddings if caching else _acompute_embeddings

        call_kwargs = {"caching": caching}
        if model_id is not None:
            call_kwargs["model_id"] = model_id

        embeddings_list = []

        for batch in input_batches:
            embeddings_list.extend(await acompute_embeddings(self.model, batch, **call_kwargs, **kwargs))
        return self._postprocess(embeddings_list, is_single_input)


def _compute_embeddings(model, batch_inputs, caching=False, model_id=None, **kwargs):
    if isinstance(model, str):
        caching = caching and _get_litellm().cache is not None
        embedding_response = _get_litellm().embedding(model=model, input=batch_inputs, caching=caching, **kwargs)
        return [data["embedding"] for data in embedding_response.data]
    elif callable(model):
        return model(batch_inputs, **kwargs)
    else:
        raise ValueError(f"`model` in `dspy.Embedder` must be a string or a callable, but got {type(model)}.")


@request_cache(ignored_args_for_cache_key=["api_key", "api_base", "base_url"])
def _cached_compute_embeddings(model, batch_inputs, caching=True, model_id=None, **kwargs):
    return _compute_embeddings(model, batch_inputs, caching=caching, model_id=model_id, **kwargs)


async def _acompute_embeddings(model, batch_inputs, caching=False, model_id=None, **kwargs):
    if isinstance(model, str):
        caching = caching and _get_litellm().cache is not None
        embedding_response = await _get_litellm().aembedding(model=model, input=batch_inputs, caching=caching, **kwargs)
        return [data["embedding"] for data in embedding_response.data]
    elif callable(model):
        return model(batch_inputs, **kwargs)
    else:
        raise ValueError(f"`model` in `dspy.Embedder` must be a string or a callable, but got {type(model)}.")


@request_cache(ignored_args_for_cache_key=["api_key", "api_base", "base_url"])
async def _cached_acompute_embeddings(model, batch_inputs, caching=True, model_id=None, **kwargs):
    return await _acompute_embeddings(model, batch_inputs, caching=caching, model_id=model_id, **kwargs)
