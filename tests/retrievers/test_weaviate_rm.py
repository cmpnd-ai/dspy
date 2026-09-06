import sys
import types

# The `weaviate` extra is optional and not installed in the default dev/test
# environment, yet dspy/retrievers/weaviate_rm.py imports `weaviate` and
# `weaviate.util` at module load time. Stub those modules before importing
# WeaviateRM so this test runs without the extra. Mirrors the sys.modules
# patching pattern used in tests/teleprompt/test_gepa.py.
if "weaviate" not in sys.modules:
    _weaviate_stub = types.ModuleType("weaviate")
    _weaviate_stub.WeaviateClient = type("WeaviateClient", (), {})
    _weaviate_stub.Client = type("Client", (), {})

    _weaviate_util_stub = types.ModuleType("weaviate.util")

    def _passthrough(value):
        return value

    _weaviate_util_stub.get_valid_uuid = _passthrough
    _weaviate_stub.util = _weaviate_util_stub
    sys.modules["weaviate"] = _weaviate_stub
    sys.modules["weaviate.util"] = _weaviate_util_stub

from dspy.retrievers.weaviate_rm import WeaviateRM


class _StubQuery:
    """Mimics the weaviate v4 `_QueryCollection`.

    Faithfully does NOT define `with_tenant`, matching the real library across the
    entire pinned range (`weaviate-client>=4.5.4,<4.22.0`): `with_tenant` lives on
    `Collection`, not on its `.query` namespace. This is exactly why the buggy
    `collection.query.with_tenant(...)` raised `AttributeError`; the fix scopes the
    collection via `collection.with_tenant(...).query.hybrid(...)` instead.
    """

    def __init__(self, collection):
        self._collection = collection

    def hybrid(self, query, limit, **kwargs):
        self._collection.calls.append(("hybrid", query, limit, self._collection.tenant, kwargs))
        return _StubResults(self._collection, limit)


class _StubResults:
    def __init__(self, collection, limit):
        tag = collection.tenant or "global"
        count = min(limit, collection.n_objects)
        self.objects = [types.SimpleNamespace(properties={collection.text_key: f"doc-{tag}-{i}"}) for i in range(count)]


class _StubCollection:
    """Mimics the weaviate v4 `Collection`: `with_tenant` returns a tenant-scoped
    collection; `.query` exposes hybrid search but (correctly) not `with_tenant`."""

    def __init__(self, name="Passages", text_key="content", tenant=None, n_objects=6):
        self.name = name
        self.text_key = text_key
        self.tenant = tenant
        self.n_objects = n_objects
        self.calls = []
        self.scoped_collections = []
        self.query = _StubQuery(self)

    def with_tenant(self, tenant):
        self.calls.append(("with_tenant", tenant))
        scoped = _StubCollection(self.name, self.text_key, tenant=tenant, n_objects=self.n_objects)
        self.scoped_collections.append(scoped)
        return scoped


class _StubV4Client:
    def __init__(self, collection):
        self.collections = types.SimpleNamespace(get=lambda name: collection)


class _StubGraphQLBuilder:
    def __init__(self, collection_name, text_key):
        self.collection_name = collection_name
        self.text_key = text_key
        self.tenant = None
        self.calls = []

    def with_tenant(self, tenant):
        self.tenant = tenant
        self.calls.append(("with_tenant", tenant))
        return self

    def with_hybrid(self, query):
        self.calls.append(("with_hybrid", query))
        return self

    def with_limit(self, k):
        self.calls.append(("with_limit", k))
        return self

    def do(self):
        tag = self.tenant or "global"
        return {"data": {"Get": {self.collection_name: [{self.text_key: f"doc-{tag}-{i}"} for i in range(3)]}}}


class _StubV3Client:
    def __init__(self, collection_name, text_key):
        self.collection_name = collection_name
        self.text_key = text_key
        self.builders = []
        outer = self

        class _Query:
            def get(self, name, keys):
                builder = _StubGraphQLBuilder(collection_name, text_key)
                outer.builders.append(builder)
                return builder

        self.query = _Query()


def _make_v4_rm(collection, tenant_id=None, k=3):
    client = _StubV4Client(collection)
    return WeaviateRM("Passages", weaviate_client=client, k=k, tenant_id=tenant_id)


def _make_v3_rm(collection_name="Passages", text_key="content", tenant_id=None):
    # The v3 `Client` branch is unreachable via WeaviateRM.__init__ (it assumes
    # `weaviate_client.collections.get(...)` exists, a v4-only attribute), so build
    # the instance via __new__ and set the same attributes __init__ would, targeting
    # the v3 `Client` code path. This isolates the v3 forward() regression test.
    client = _StubV3Client(collection_name, text_key)
    rm = WeaviateRM.__new__(WeaviateRM)
    rm._weaviate_collection_name = collection_name
    rm._weaviate_client = client
    rm._weaviate_collection = None
    rm._weaviate_collection_text_key = text_key
    rm._tenant_id = tenant_id
    rm._client_type = "Client"
    rm.k = 3
    rm.stage = "test"
    rm.callbacks = []
    return rm, client


def test_v4_query_namespace_has_no_with_tenant():
    """The query object returned by `collection.query` must NOT expose `with_tenant`.
    This documents the v4 API shape the fix relies on; guards against re-introducing
    the bug by calling `with_tenant` on the query namespace."""
    collection = _StubCollection()
    assert hasattr(collection, "with_tenant")
    assert hasattr(collection.query, "hybrid")
    assert not hasattr(collection.query, "with_tenant")


def test_v4_tenant_retrieval_scopes_collection_to_tenant():
    """Regression for the reported bug: with a tenant, forward() must scope the
    COLLECTION via with_tenant (not call with_tenant on the query namespace)."""
    collection = _StubCollection()
    rm = _make_v4_rm(collection, tenant_id="Tenant1", k=3)

    passages = rm.forward("hello world")

    # forward returned tenant-scoped passages without raising AttributeError
    assert [p.long_text for p in passages] == ["doc-Tenant1-0", "doc-Tenant1-1", "doc-Tenant1-2"]

    # with_tenant was invoked on the COLLECTION itself...
    assert collection.calls == [("with_tenant", "Tenant1")]
    assert len(collection.scoped_collections) == 1
    scoped = collection.scoped_collections[0]
    assert scoped.tenant == "Tenant1"

    # ...and hybrid was executed against that tenant-scoped collection with the tenant set
    assert scoped.calls == [("hybrid", "hello world", 3, "Tenant1", {})]

    # the unscoped collection's query namespace was never asked for with_tenant
    assert not hasattr(collection.query, "with_tenant")


def test_v4_no_tenant_does_not_scope_collection():
    """Without a tenant, forward() must run hybrid directly on the unscoped collection
    and never call with_tenant. Guards the no-tenant path against the refactor."""
    collection = _StubCollection()
    rm = _make_v4_rm(collection, tenant_id=None, k=3)

    passages = rm.forward("hello")

    assert passages[0].long_text == "doc-global-0"
    assert collection.scoped_collections == []
    assert collection.calls == [("hybrid", "hello", 3, None, {})]


def test_v4_extra_kwargs_flow_to_hybrid_but_tenant_id_is_popped():
    """tenant_id must be popped from kwargs (consumed by forward to scope the
    collection) and never forwarded to hybrid(); other kwargs must flow through."""
    collection = _StubCollection()
    rm = _make_v4_rm(collection, tenant_id="Tenant1", k=3)

    rm.forward("hello", tenant_id="Tenant1", query_properties=["title"], alpha=0.5)

    forwarded_kwargs = collection.scoped_collections[0].calls[0][4]
    assert forwarded_kwargs == {"query_properties": ["title"], "alpha": 0.5}
    assert "tenant_id" not in forwarded_kwargs


def test_v3_tenant_retrieval_uses_graphql_builder_with_tenant():
    """Regression guard for the v3 `Client` path: with_tenant belongs on the GraphQL
    Get builder (where it exists in v3), not on the collection. Ensures the v4 fix
    does not change the v3 idiom."""
    rm, client = _make_v3_rm(tenant_id=None)

    passages = rm.forward("hello", tenant_id="Tenant1")

    assert [p.long_text for p in passages] == ["doc-Tenant1-0", "doc-Tenant1-1", "doc-Tenant1-2"]
    assert len(client.builders) == 1
    builder = client.builders[0]
    assert ("with_tenant", "Tenant1") in builder.calls
    assert builder.tenant == "Tenant1"
    assert ("with_hybrid", "hello") in builder.calls
    assert ("with_limit", 3) in builder.calls
