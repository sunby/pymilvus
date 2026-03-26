"""
向 create_100k_partitions.py 创建的 collection 继续写入数据的示例程序
- 默认执行 10 轮写入
- 每轮向每个 partition 写入 500 条数据
- 每轮结束后执行 flush
- 全部写入完成后创建索引
"""

import os
import time
import argparse
import multiprocessing as mp
from threading import Lock
from typing import Optional

import numpy as np

from pymilvus import MilvusClient


_WORKER_CLIENT = None
_WORKER_RNG = None


def init_worker(uri: str):
    """为每个进程初始化独立的 Milvus 连接和随机数生成器"""
    global _WORKER_CLIENT, _WORKER_RNG
    _WORKER_CLIENT = MilvusClient(uri=uri)
    _WORKER_RNG = np.random.default_rng(seed=time.time_ns() ^ os.getpid())


def get_worker_client() -> MilvusClient:
    if _WORKER_CLIENT is None:
        raise RuntimeError("Worker client is not initialized")
    return _WORKER_CLIENT


class TimingStats:
    """主进程中的耗时统计"""

    def __init__(self):
        self.lock = Lock()
        self.insert_times = []
        self.flush_times = []

    def add_insert(self, elapsed: float):
        with self.lock:
            self.insert_times.append(elapsed)

    def add_flush(self, elapsed: float):
        with self.lock:
            self.flush_times.append(elapsed)

    def summary(self) -> dict:
        with self.lock:
            def calc_stats(times):
                if not times:
                    return {"count": 0, "avg": 0, "min": 0, "max": 0, "total": 0}
                return {
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times),
                }

            return {
                "insert": calc_stats(self.insert_times),
                "flush": calc_stats(self.flush_times),
            }


class ProgressTracker:
    """主进程中的进度追踪器"""

    def __init__(self, total: int, report_interval: int = 1000, records_per_partition: int = 0):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.report_interval = report_interval
        self.records_per_partition = records_per_partition
        self.start_time = time.time()
        self.lock = Lock()
        self.timing = TimingStats()

    def update(self, success: bool = True):
        with self.lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1

            processed = self.completed + self.failed
            should_report = (processed % self.report_interval == 0) or (processed == self.total)
            if not should_report:
                return

            elapsed = time.time() - self.start_time
            rate = processed / elapsed if elapsed > 0 else 0
            msg = (
                f"Progress: {processed}/{self.total} ({processed * 100 / self.total:.1f}%) "
                f"| Success: {self.completed} | Failed: {self.failed} "
                f"| Rate: {rate:.1f} partition-rounds/s"
            )
            if self.records_per_partition > 0:
                msg += f" ({rate * self.records_per_partition:.1f} records/s)"
            msg += f" | Elapsed: {elapsed:.1f}s"
            print(msg)


def chunk_list(lst: list, num_chunks: int) -> list:
    """将列表切分成多个块"""
    if not lst:
        return []
    if num_chunks <= 0:
        return [lst]
    chunk_size = (len(lst) + num_chunks - 1) // num_chunks
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_vector_dim(client: MilvusClient, collection_name: str, vector_field: str) -> int:
    """从 collection schema 中提取向量维度"""
    schema = client.describe_collection(collection_name)
    for field in schema.get("fields", []):
        if field.get("name") != vector_field:
            continue
        dim = field.get("params", {}).get("dim")
        if dim is None:
            raise ValueError(f"Field '{vector_field}' does not expose a dim parameter")
        return int(dim)
    raise ValueError(f"Vector field '{vector_field}' not found in collection '{collection_name}'")


def insert_chunk_round_task(
    collection_name: str,
    partition_items: list,
    round_idx: int,
    total_rounds: int,
    records_per_round: int,
    dim: int,
    id_base: int,
) -> tuple:
    """一个 worker 进程负责一批 partition 的单轮写入"""
    client = get_worker_client()
    results = []
    insert_times = []

    for partition_idx, partition_name in partition_items:
        try:
            round_offset = partition_idx * total_rounds * records_per_round + round_idx * records_per_round
            ids = range(id_base + round_offset, id_base + round_offset + records_per_round)
            vectors = _WORKER_RNG.random((records_per_round, dim)).tolist()
            data = [{"id": row_id, "vector": vector} for row_id, vector in zip(ids, vectors)]

            t0 = time.time()
            client.insert(collection_name, data, partition_name=partition_name)
            insert_times.append(time.time() - t0)
            results.append((partition_name, True, None))
        except Exception as exc:
            results.append((partition_name, False, str(exc)))

    return results, insert_times


def insert_chunk_round_worker_task(args) -> tuple:
    """multiprocessing.Pool 的包装任务"""
    return insert_chunk_round_task(*args)


def drop_vector_indexes(client: MilvusClient, collection_name: str, vector_field: str):
    """删除向量字段上已有的索引，以便在全部写入完成后重建"""
    existing_indexes = client.list_indexes(collection_name, field_name=vector_field)
    if not existing_indexes:
        print(f"No existing indexes found on field '{vector_field}', skip drop.")
        return

    print(f"Dropping existing indexes on field '{vector_field}': {existing_indexes}")
    for existing_index in existing_indexes:
        client.drop_index(collection_name, existing_index)


def create_vector_index(
    client: MilvusClient,
    collection_name: str,
    vector_field: str,
    index_type: str,
    metric_type: str,
    nlist: int,
    index_name: str,
):
    """在全部写入结束后创建向量索引"""
    print(
        f"Creating index on field '{vector_field}' "
        f"(index_type={index_type}, metric_type={metric_type}, nlist={nlist}, index_name={index_name})..."
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=vector_field,
        index_type=index_type,
        metric_type=metric_type,
        index_name=index_name,
        params={"nlist": nlist},
    )

    t0 = time.time()
    client.create_index(collection_name, index_params)
    elapsed = time.time() - t0
    print(f"Index creation finished in {elapsed:.2f}s")


def insert_rounds_to_existing_collection(
    uri: str,
    collection_name: str,
    num_rounds: int = 10,
    records_per_round: int = 500,
    max_workers: int = 16,
    vector_field: str = "vector",
    report_interval: int = 1000,
    id_base: Optional[int] = None,
    index_type: str = "IVF_FLAT",
    metric_type: str = "L2",
    nlist: int = 128,
    index_name: str = "vector_ivf_flat_after_insert",
):
    """对已有的多 partition collection 执行多轮插入"""
    print(f"Connecting to Milvus at {uri}...")
    client = MilvusClient(uri=uri)

    if not client.has_collection(collection_name):
        raise ValueError(
            f"Collection '{collection_name}' does not exist. "
            "Create it first with examples/create_100k_partitions.py"
        )

    partition_names = [name for name in client.list_partitions(collection_name) if name != "_default"]
    if not partition_names:
        raise ValueError(
            f"Collection '{collection_name}' has no non-default partitions. "
            "Create partitions first with examples/create_100k_partitions.py"
        )

    dim = get_vector_dim(client, collection_name, vector_field)
    if id_base is None:
        id_base = time.time_ns()

    drop_vector_indexes(client, collection_name, vector_field)

    partition_items = list(enumerate(partition_names))
    chunks = chunk_list(partition_items, max_workers)
    total_ops = len(partition_names) * num_rounds
    tracker = ProgressTracker(total_ops, report_interval=report_interval, records_per_partition=records_per_round)
    failed_partitions = []

    print("\n" + "=" * 60)
    print("ROUND INSERT CONFIGURATION")
    print("=" * 60)
    print(f"Collection: {collection_name}")
    print(f"Partitions: {len(partition_names)}")
    print(f"Rounds: {num_rounds}")
    print(f"Records per round per partition: {records_per_round}")
    print(f"Total records per partition: {num_rounds * records_per_round}")
    print(f"Total records: {len(partition_names) * num_rounds * records_per_round}")
    print(f"Vector field: {vector_field}")
    print(f"Vector dim: {dim}")
    print(f"Workers: {min(max_workers, len(chunks))}")
    print(f"ID base: {id_base}")
    print("=" * 60)

    start_time = time.time()
    ctx = mp.get_context("spawn")

    with ctx.Pool(
        processes=min(max_workers, len(chunks)),
        initializer=init_worker,
        initargs=(uri,),
    ) as pool:
        for round_idx in range(num_rounds):
            round_start = time.time()
            print(f"\nStarting round {round_idx + 1}/{num_rounds}...")

            worker_args = [
                (collection_name, chunk, round_idx, num_rounds, records_per_round, dim, id_base)
                for chunk in chunks
            ]

            for results, insert_times in pool.imap_unordered(
                insert_chunk_round_worker_task,
                worker_args,
            ):
                for insert_time in insert_times:
                    tracker.timing.add_insert(insert_time)
                for partition_name, success, error in results:
                    tracker.update(success=success)
                    if not success:
                        failed_partitions.append((round_idx + 1, partition_name, error))

            print(f"Round {round_idx + 1}/{num_rounds} writes completed, starting flush...")
            t0 = time.time()
            client.flush(collection_name)
            flush_time = time.time() - t0
            tracker.timing.add_flush(flush_time)
            round_time = time.time() - round_start
            print(
                f"Round {round_idx + 1}/{num_rounds} done "
                f"| Round time: {round_time:.2f}s | Flush time: {flush_time:.2f}s"
            )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("INSERT SUMMARY")
    print("=" * 60)
    print(f"Partitions: {len(partition_names)}")
    print(f"Rounds: {num_rounds}")
    print(f"Records per round per partition: {records_per_round}")
    print(f"Insert operations completed: {tracker.completed}")
    print(f"Insert operations failed: {tracker.failed}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average rate: {tracker.completed / elapsed:.2f} partition-rounds/s")

    stats = tracker.timing.summary()
    insert_stats = stats["insert"]
    flush_stats = stats["flush"]
    print("\n--- Timing Stats ---")
    print(
        f"Insert: count={insert_stats['count']}, avg={insert_stats['avg'] * 1000:.2f}ms, "
        f"min={insert_stats['min'] * 1000:.2f}ms, max={insert_stats['max'] * 1000:.2f}ms, "
        f"total={insert_stats['total']:.2f}s"
    )
    print(
        f"Flush: count={flush_stats['count']}, avg={flush_stats['avg'] * 1000:.2f}ms, "
        f"min={flush_stats['min'] * 1000:.2f}ms, max={flush_stats['max'] * 1000:.2f}ms, "
        f"total={flush_stats['total']:.2f}s"
    )

    if failed_partitions:
        print("\nFailed partition writes (showing first 10):")
        for round_number, partition_name, error in failed_partitions[:10]:
            print(f"  - round={round_number}, partition={partition_name}: {error}")

    create_vector_index(
        client=client,
        collection_name=collection_name,
        vector_field=vector_field,
        index_type=index_type,
        metric_type=metric_type,
        nlist=nlist,
        index_name=index_name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="向 create_100k_partitions.py 创建的 collection 继续分轮写入数据"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default="http://10.15.1.205:19530",
        help="Milvus 服务地址 (default: http://10.15.1.205:19530)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="test_100k_partitions",
        help="Collection 名称 (default: test_100k_partitions)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="并发 worker 数 (default: 16)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="写入轮数 (default: 10)",
    )
    parser.add_argument(
        "--records-per-round",
        type=int,
        default=400,
        help="每轮每个 partition 写入的记录数 (default: 500)",
    )
    parser.add_argument(
        "--vector-field",
        type=str,
        default="vector",
        help="向量字段名 (default: vector)",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=1000,
        help="每完成多少个 partition-round 打印一次进度 (default: 1000)",
    )
    parser.add_argument(
        "--id-base",
        type=int,
        default=None,
        help="主键起始值，默认自动使用 time.time_ns()",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="IVF_FLAT",
        help="索引类型 (default: IVF_FLAT)",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default="L2",
        help="索引距离类型 (default: L2)",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=128,
        help="IVF nlist 参数 (default: 128)",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="vector_ivf_flat_after_insert",
        help="创建索引时使用的 index_name (default: vector_ivf_flat_after_insert)",
    )
    args = parser.parse_args()

    insert_rounds_to_existing_collection(
        uri=args.uri,
        collection_name=args.collection,
        num_rounds=args.num_rounds,
        records_per_round=args.records_per_round,
        max_workers=args.workers,
        vector_field=args.vector_field,
        report_interval=args.report_interval,
        id_base=args.id_base,
        index_type=args.index_type,
        metric_type=args.metric_type,
        nlist=args.nlist,
        index_name=args.index_name,
    )


if __name__ == "__main__":
    main()
