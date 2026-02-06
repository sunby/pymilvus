"""
单线程insert测试，测量insert耗时
"""

import time
import argparse
import numpy as np
from pymilvus import MilvusClient, DataType


def create_collection_if_not_exists(client: MilvusClient, collection_name: str, dim: int = 128):
    """创建collection（如果不存在）"""
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists, skipping creation...")
    else:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"Collection '{collection_name}' created successfully")


def main():
    parser = argparse.ArgumentParser(description="单线程insert测试")
    parser.add_argument("--uri", type=str, default="http://localhost:19530",
                        help="Milvus服务地址 (default: http://localhost:19530)")
    parser.add_argument("--collection", type=str, default="test_single_insert",
                        help="Collection名称 (default: test_single_insert)")
    parser.add_argument("--num-partitions", type=int, default=100,
                        help="要写入的partition数量 (default: 100)")
    parser.add_argument("--num-records", type=int, default=100,
                        help="每个partition写入的记录数 (default: 100)")
    parser.add_argument("--dim", type=int, default=128,
                        help="向量维度 (default: 128)")
    parser.add_argument("--report-interval", type=int, default=10,
                        help="每多少个partition打印一次进度 (default: 10)")

    args = parser.parse_args()

    print(f"Connecting to Milvus at {args.uri}...")
    client = MilvusClient(uri=args.uri)

    create_collection_if_not_exists(client, args.collection, dim=args.dim)

    # 创建partitions
    print(f"\nCreating {args.num_partitions} partitions...")
    for i in range(args.num_partitions):
        partition_name = f"partition_{i:06d}"
        if partition_name not in client.list_partitions(args.collection):
            client.create_partition(args.collection, partition_name)

    # 单线程insert
    print(f"\nInserting {args.num_records} records into each partition (single thread)...")

    insert_times = []
    start_time = time.time()

    for i in range(args.num_partitions):
        partition_name = f"partition_{i:06d}"

        # 生成数据
        data = [
            {"id": j, "vector": np.random.random(args.dim).tolist()}
            for j in range(args.num_records)
        ]

        # 计时insert
        t0 = time.time()
        client.insert(args.collection, data, partition_name=partition_name)
        t1 = time.time()

        insert_times.append(t1 - t0)

        # 打印进度
        if (i + 1) % args.report_interval == 0 or i == args.num_partitions - 1:
            elapsed = time.time() - start_time
            avg_insert = sum(insert_times) / len(insert_times) * 1000
            rate = (i + 1) / elapsed
            print(f"Progress: {i+1}/{args.num_partitions} | Rate: {rate:.2f} partitions/s | Insert avg: {avg_insert:.2f}ms | Elapsed: {elapsed:.1f}s")

    # 汇总
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total partitions: {args.num_partitions}")
    print(f"Records per partition: {args.num_records}")
    print(f"Total records: {args.num_partitions * args.num_records}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Rate: {args.num_partitions / elapsed:.2f} partitions/s")
    print(f"\n--- Insert Timing ---")
    print(f"Count: {len(insert_times)}")
    print(f"Avg: {sum(insert_times)/len(insert_times)*1000:.2f}ms")
    print(f"Min: {min(insert_times)*1000:.2f}ms")
    print(f"Max: {max(insert_times)*1000:.2f}ms")
    print(f"Total: {sum(insert_times):.2f}s")


if __name__ == "__main__":
    main()
