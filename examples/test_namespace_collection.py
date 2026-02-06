import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pymilvus import DataType, MilvusClient


def main() -> None:
    fmt = "\n=== {:30} ===\n"
    dim = 8
    namespace_count = 100_000
    vectors_path = "namespace_vectors.npy"
    collection_name = "namespace_collection_test"
    milvus_client = MilvusClient("http://localhost:19530")

    if milvus_client.has_collection(collection_name, timeout=5):
        milvus_client.drop_collection(collection_name)

    schema = milvus_client.create_schema(enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("num", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    milvus_client.create_collection(
        collection_name,
        schema=schema,
        enable_dynamic_field=True,
        num_shards=1,
    )

    print(fmt.format("Start inserting partition data"))
    if os.path.exists(vectors_path):
        vectors = np.load(vectors_path)
        if vectors.shape != (namespace_count, dim):
            raise ValueError(
                f"vectors shape mismatch: expected {(namespace_count, dim)}, got {vectors.shape}"
            )
    else:
        rng = np.random.default_rng(seed=19530)
        vectors = rng.random((namespace_count, dim))
        np.save(vectors_path, vectors)
    partition_names = [f"partition_{idx + 1}" for idx in range(namespace_count)]
    partition_workers = min(64, (os.cpu_count() or 4) * 4)
    print(fmt.format(f"Start creating partitions (workers={partition_workers})"))
    with ThreadPoolExecutor(max_workers=partition_workers) as executor:
        futures = [
            executor.submit(milvus_client.create_partition, collection_name, name)
            for name in partition_names
        ]
        for future in as_completed(futures):
            future.result()
    print(fmt.format("Creating partitions done"))

    partition_batches = []
    for idx, partition_name in enumerate(partition_names):
        entity_id = idx + 1
        rows = [
            {
                "id": entity_id,
                "num": entity_id,
                "vector": vectors[idx],
            }
        ]
        milvus_client.insert(
            collection_name,
            rows,
            progress_bar=False,
            partition_name=partition_name,
        )
        partition_batches.append(
            {
                "partition": partition_name,
                "entity_id": entity_id,
                "expected": 1,
            }
        )
    print(fmt.format("Inserting partition data done"))

    milvus_client.flush(collection_name)
    print(fmt.format("Start creating index"))
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        "vector",
        index_type="HNSW",
        index_name="hnsw_index",
        metric_type="L2",
        M=16,
        efConstruction=100,
    )
    milvus_client.create_index(collection_name, index_params)
    print(fmt.format("Creating index done"))
    milvus_client.load_collection(collection_name)
    print(fmt.format("Loading collection done"))

    print(fmt.format("Start query by partition"))
    query_costs = []
    total_query_start = time.perf_counter()
    for batch_info in partition_batches:
        expr = f"id == {batch_info['entity_id']}"
        expected = batch_info["expected"]
        query_start = time.perf_counter()
        query_results = milvus_client.query(
            collection_name,
            filter=expr,
            output_fields=["id", "num"],
            limit=expected,
            partition_names=[batch_info["partition"]],
        )
        query_cost_ms = (time.perf_counter() - query_start) * 1000
        query_costs.append(query_cost_ms)
        if len(query_results) != expected:
            raise ValueError(
                f"partition={batch_info['partition']}, rows={len(query_results)}, expected={expected}"
            )

    total_query_cost = time.perf_counter() - total_query_start
    avg_query_cost_ms = sum(query_costs) / len(query_costs)
    print(
        f"query_count={len(query_costs)}, avg_cost_ms={avg_query_cost_ms:.2f}, "
        f"total_cost_s={total_query_cost:.3f}"
    )


if __name__ == "__main__":
    main()
