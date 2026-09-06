"""Tests for ``dspy.retrievers.databricks_rm.DatabricksRM`` ``docs_uri_column_name``.

These tests are fully hermetic: ``databricks-sdk`` and ``mlflow`` are not
required. The Databricks Vector Search query boundary is mocked at
``DatabricksRM._query_via_databricks_sdk`` (SDK path) and
``dspy.retrievers.databricks_rm.requests.post`` (requests path), and the
``mlflow.models.set_retriever_schema`` import exercised by the agent-framework
path is stubbed via ``sys.modules``.

The central regression covered here is the ``docs_uri_column_name`` bug: the
column was never added to ``self.columns`` (the list of columns requested from
the Vector Search API), so ``forward`` raised ``KeyError`` whenever a user opted
into URI retrieval by setting ``docs_uri_column_name``.
"""

import sys
import types
from importlib.util import spec_from_loader
from unittest.mock import MagicMock, patch

import pytest

import dspy
import dspy.retrievers.databricks_rm as databricks_rm_module
from dspy.retrievers.databricks_rm import DatabricksRM

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


# The Databricks Vector Search API automatically appends a system-computed
# ``score`` column to every response (the retriever sorts on it unconditionally).
# User metadata columns (id, text, uri) are returned only when requested via
# the ``columns`` request parameter. ``_build_response`` mirrors that contract:
# it lists exactly the requested columns plus ``score`` in the manifest, and
# emits positional data rows to match.
def _build_response(requested_columns, rows):
    """Build a canned Vector Search ``Query an index`` response.

    Args:
        requested_columns: The column names the API was asked to return. The
            response manifest lists exactly these plus ``"score"`` (which the
            API appends automatically).
        rows: A list of dicts mapping column name -> value. Each row must
            include a ``"score"`` entry; values for columns that were not
            requested are ignored.
    """
    manifest_columns = [{"name": col} for col in requested_columns] + [{"name": "score"}]
    data_array = [[row.get(col) for col in requested_columns] + [row["score"]] for row in rows]
    return {"manifest": {"columns": manifest_columns}, "result": {"data_array": data_array}}


def _make_sdk_query(rows, capture=None):
    """Return a fake ``_query_via_databricks_sdk`` that mirrors requested cols."""

    def fake_query(*args, **kwargs):
        requested = list(kwargs.get("columns") or [])
        if capture is not None:
            capture["columns"] = requested
            capture["kwargs"] = kwargs
        return _build_response(requested, rows)

    return staticmethod(fake_query)


def _make_requests_post(rows, capture=None):
    """Return a fake ``requests.post`` that mirrors requested cols from payload."""

    def fake_post(url, json=None, headers=None, **kwargs):
        payload = json or {}
        requested = list(payload.get("columns") or [])
        if capture is not None:
            capture["columns"] = requested
            capture["url"] = url
            capture["payload"] = payload
        mock_response = MagicMock()
        mock_response.json.return_value = _build_response(requested, rows)
        return mock_response

    return fake_post


def _stub_mlflow_modules():
    """Build minimal ``mlflow`` / ``mlflow.models`` stubs for the agent path."""
    mlflow = types.ModuleType("mlflow")
    mlflow_models = types.ModuleType("mlflow.models")
    mlflow_models.set_retriever_schema = lambda **kwargs: None
    mlflow.models = mlflow_models
    mlflow.__spec__ = spec_from_loader("mlflow", loader=None)
    mlflow.__path__ = []
    mlflow_models.__spec__ = spec_from_loader("mlflow.models", loader=None)
    mlflow_models.__path__ = []
    return {"mlflow": mlflow, "mlflow.models": mlflow_models}


SAMPLE_ROWS = [
    {"id": 1, "text": "hello world", "uri": "https://example.com/1", "score": 0.9},
    {"id": 2, "text": "foo bar", "uri": "https://example.com/2", "score": 0.7},
]


@pytest.fixture
def mlflow_stub():
    with patch.dict(sys.modules, _stub_mlflow_modules(), clear=False):
        yield


# --------------------------------------------------------------------------- #
# self.columns construction (the core bug fix)
# --------------------------------------------------------------------------- #


def test_columns_include_docs_uri_column_when_set():
    """Setting docs_uri_column_name must request that column from the API."""
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", True):
        rm = DatabricksRM(
            databricks_index_name="dummy.index",
            docs_id_column_name="id",
            text_column_name="text",
            docs_uri_column_name="uri",
        )
    assert "uri" in rm.columns
    assert "id" in rm.columns
    assert "text" in rm.columns


def test_columns_exclude_none_when_docs_uri_column_not_set():
    """When docs_uri_column_name is None, None must not leak into self.columns."""
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", True):
        rm = DatabricksRM(
            databricks_index_name="dummy.index",
            docs_id_column_name="id",
            text_column_name="text",
        )
    assert None not in rm.columns
    assert set(rm.columns) == {"id", "text"}


def test_columns_include_extra_columns_and_uri():
    """User-supplied extra columns and the uri column are both requested."""
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", True):
        rm = DatabricksRM(
            databricks_index_name="dummy.index",
            docs_id_column_name="id",
            text_column_name="text",
            docs_uri_column_name="uri",
            columns=["extra", "date"],
        )
    assert {"id", "text", "uri", "extra", "date"} == set(rm.columns)
    assert None not in rm.columns


# --------------------------------------------------------------------------- #
# forward() -- standard Prediction path (SDK)
# --------------------------------------------------------------------------- #


def test_forward_standard_path_with_docs_uri_includes_uris():
    """Regression: forward no longer raises KeyError when docs_uri_column_name is set.

    Before the fix, the uri column was never requested, so ``doc["uri"]``
    raised KeyError. Now the column is requested and ``doc_uris`` is populated.
    """
    capture = {}
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", True):
        with patch.object(DatabricksRM, "_query_via_databricks_sdk", _make_sdk_query(SAMPLE_ROWS, capture)):
            rm = DatabricksRM(
                databricks_index_name="dummy.index",
                docs_id_column_name="id",
                text_column_name="text",
                docs_uri_column_name="uri",
                k=3,
            )
            result = rm.forward("query")

    # The uri column must have been requested from the API.
    assert "uri" in capture["columns"]
    assert isinstance(result, dspy.Prediction)
    assert result.doc_ids == [1, 2]
    assert result.docs == ["hello world", "foo bar"]
    assert result.doc_uris == ["https://example.com/1", "https://example.com/2"]


def test_forward_standard_path_without_docs_uri_returns_none_uris():
    """No regression: when docs_uri_column_name is unset, doc_uris is None."""
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", True):
        with patch.object(DatabricksRM, "_query_via_databricks_sdk", _make_sdk_query(SAMPLE_ROWS)):
            rm = DatabricksRM(
                databricks_index_name="dummy.index",
                docs_id_column_name="id",
                text_column_name="text",
                k=3,
            )
            result = rm.forward("query")

    assert isinstance(result, dspy.Prediction)
    assert result.doc_uris is None
    assert result.doc_ids == [1, 2]
    assert result.docs == ["hello world", "foo bar"]


# --------------------------------------------------------------------------- #
# forward() -- Mosaic Agent Framework path (SDK)
# --------------------------------------------------------------------------- #


def test_forward_agent_framework_path_with_docs_uri(mlflow_stub):
    """Regression: agent-framework path no longer raises KeyError for uri column."""
    capture = {}
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", True):
        with patch.object(DatabricksRM, "_query_via_databricks_sdk", _make_sdk_query(SAMPLE_ROWS, capture)):
            rm = DatabricksRM(
                databricks_index_name="dummy.index",
                docs_id_column_name="id",
                text_column_name="text",
                docs_uri_column_name="uri",
                k=3,
                use_with_databricks_agent_framework=True,
            )
            result = rm.forward("query")

    assert "uri" in capture["columns"]
    assert isinstance(result, list)
    assert len(result) == 2
    assert [doc["metadata"]["doc_uri"] for doc in result] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert [doc["metadata"]["doc_id"] for doc in result] == [1, 2]
    assert [doc["page_content"] for doc in result] == ["hello world", "foo bar"]
    assert all(doc["type"] == "Document" for doc in result)


# --------------------------------------------------------------------------- #
# forward() -- requests (non-SDK) path
# --------------------------------------------------------------------------- #


def test_forward_requests_path_with_docs_uri_requests_and_returns_uris():
    """The non-SDK requests path also requests & returns the uri column."""
    capture = {}
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", False):
        with patch(
            "dspy.retrievers.databricks_rm.requests.post",
            new=_make_requests_post(SAMPLE_ROWS, capture),
        ):
            rm = DatabricksRM(
                databricks_index_name="dummy.index",
                databricks_endpoint="https://example.com",
                databricks_token="tok",
                docs_id_column_name="id",
                text_column_name="text",
                docs_uri_column_name="uri",
                k=3,
            )
            result = rm.forward("query")

    assert "uri" in capture["payload"]["columns"]
    assert isinstance(result, dspy.Prediction)
    assert result.doc_uris == ["https://example.com/1", "https://example.com/2"]
    assert result.doc_ids == [1, 2]


# --------------------------------------------------------------------------- #
# Validation of the URI column in the index response
# --------------------------------------------------------------------------- #


def test_forward_raises_when_docs_uri_column_missing_from_index():
    """A configured docs_uri_column_name absent from the API gets a clear error.

    This exercises the validation added alongside the fix: previously the URI
    column was neither requested nor validated, and forward raised a bare
    KeyError. Now, if the index genuinely lacks the column, the user gets a
    descriptive exception naming the misconfigured column.
    """
    response = {
        "manifest": {"columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}]},
        "result": {"data_array": [[1, "x", 0.5]]},
    }
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", True):
        with patch.object(
            DatabricksRM,
            "_query_via_databricks_sdk",
            new=staticmethod(lambda *a, **k: response),
        ):
            rm = DatabricksRM(
                databricks_index_name="dummy.index",
                docs_id_column_name="id",
                text_column_name="text",
                docs_uri_column_name="uri",
            )
            with pytest.raises(Exception, match="docs_uri_column_name: 'uri' is not in the index columns"):
                rm.forward("query")


def test_forward_does_not_validate_uri_when_unset():
    """No uri validation runs when docs_uri_column_name is None."""
    response = {
        "manifest": {"columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}]},
        "result": {"data_array": [[1, "x", 0.5]]},
    }
    with patch.object(databricks_rm_module, "_databricks_sdk_installed", True):
        with patch.object(
            DatabricksRM,
            "_query_via_databricks_sdk",
            new=staticmethod(lambda *a, **k: response),
        ):
            rm = DatabricksRM(
                databricks_index_name="dummy.index",
                docs_id_column_name="id",
                text_column_name="text",
            )
            result = rm.forward("query")

    assert result.doc_uris is None
    assert result.doc_ids == [1]
