"""Milvus schema creation for patch-level ColPali indexing."""

from __future__ import annotations

from indexer.shared.errors import DependencyUnavailableError, IndexingRuntimeError

try:
    from pymilvus import DataType
except ImportError:  # pragma: no cover - exercised in runtime environments without pymilvus.
    DataType = None

VECTOR_DIMENSION = 128


def ensure_collection(
    client: object,
    collection_name: str,
    recreate_collection: bool,
    index_type: str,
    metric_type: str,
    nlist: int,
) -> None:
    """Create or recreate the Milvus collection used by the pipeline."""

    if DataType is None:
        raise DependencyUnavailableError(
            "pymilvus is required for Milvus schema management."
        )

    if client.has_collection(collection_name=collection_name):
        if recreate_collection:
            client.drop_collection(collection_name=collection_name)
        else:
            return

    try:
        schema = client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
            description="ColPali patch-level index for PDF pages",
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="doc_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="domain", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="page_number", datatype=DataType.INT64)
        schema.add_field(field_name="patch_id", datatype=DataType.INT64)
        schema.add_field(field_name="page_uid", datatype=DataType.VARCHAR, max_length=768)
        schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIMENSION,
        )
        schema.add_field(
            field_name="source_sha256", datatype=DataType.VARCHAR, max_length=64
        )
        schema.add_field(field_name="page_width", datatype=DataType.INT64)
        schema.add_field(field_name="page_height", datatype=DataType.INT64)
        schema.add_field(field_name="indexed_at", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="run_id", datatype=DataType.VARCHAR, max_length=64)

        index_params = client.prepare_index_params()
        if index_type == "FLAT":
            index_params.add_index(
                field_name="embedding",
                index_type=index_type,
                metric_type=metric_type,
                params={},
            )
        else:
            index_params.add_index(
                field_name="embedding",
                index_type=index_type,
                metric_type=metric_type,
                params={"nlist": nlist},
            )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
    except Exception as error:  # pragma: no cover - depends on pymilvus runtime.
        raise IndexingRuntimeError(
            f"Failed to create Milvus collection '{collection_name}': {error}"
        ) from error
