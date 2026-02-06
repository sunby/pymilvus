"""
并发创建100k partition的示例程序，并向每个partition写入数据
"""

import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
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


class TimingStats:
    """线程安全的耗时统计"""
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
                    "total": sum(times)
                }
            return {
                "insert": calc_stats(self.insert_times),
                "flush": calc_stats(self.flush_times)
            }


class ProgressTracker:
    """线程安全的进度追踪器"""
    def __init__(self, total: int, report_interval: int = 1000, window_size: int = 1000, records_per_partition: int = 0):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.lock = Lock()
        self.report_interval = report_interval
        self.start_time = time.time()
        self.records_per_partition = records_per_partition
        # 滑动窗口用于计算实时速率
        self.window_size = window_size
        self.timestamps = []  # 记录最近window_size个完成时间
        # 耗时统计
        self.timing = TimingStats()

    def update(self, success: bool = True):
        with self.lock:
            now = time.time()
            if success:
                self.completed += 1
            else:
                self.failed += 1

            # 更新滑动窗口
            self.timestamps.append(now)
            if len(self.timestamps) > self.window_size:
                self.timestamps.pop(0)

            processed = self.completed + self.failed
            if processed == self.total:
                elapsed = time.time() - self.start_time
                avg_rate = processed / elapsed if elapsed > 0 else 0

                # 计算实时速率（基于滑动窗口）
                if len(self.timestamps) >= 2:
                    window_elapsed = self.timestamps[-1] - self.timestamps[0]
                    realtime_rate = (len(self.timestamps) - 1) / window_elapsed if window_elapsed > 0 else 0
                else:
                    realtime_rate = 0

                # 构建输出
                msg = (f"Progress: {processed}/{self.total} ({processed*100/self.total:.1f}%) "
                       f"| Success: {self.completed} | Failed: {self.failed} "
                       f"| AvgRate: {avg_rate:.1f} partitions/s")
                if self.records_per_partition > 0:
                    records_rate = avg_rate * self.records_per_partition
                    msg += f" ({records_rate:.1f} records/s)"
                msg += f" | RealtimeRate: {realtime_rate:.1f} partitions/s | Elapsed: {elapsed:.1f}s"

                # 添加timing stats
                stats = self.timing.summary()
                ins = stats["insert"]
                if ins["count"] > 0:
                    msg += f"\n  -> Insert: avg={ins['avg']*1000:.1f}ms"
                print(msg)


def create_partition_task(client: MilvusClient, collection_name: str,
                          partition_name: str, tracker: ProgressTracker) -> tuple:
    """创建单个partition的任务"""
    try:
        client.create_partition(collection_name, partition_name)
        tracker.update(success=True)
        return (partition_name, True, None)
    except Exception as e:
        tracker.update(success=False)
        return (partition_name, False, str(e))


def insert_data_task(client: MilvusClient, collection_name: str,
                     partition_name: str, num_records: int, dim: int,
                     flush_interval: int, tracker: ProgressTracker) -> tuple:
    """向单个partition写入数据的任务"""
    try:
        for i in range(0, num_records, flush_interval):
            batch_size = min(flush_interval, num_records - i)
            data = [
                {"id": j, "vector": np.random.random(dim).tolist()}
                for j in range(i, i + batch_size)
            ]
            client.insert(collection_name, data, partition_name=partition_name)
            client.flush(collection_name)
        tracker.update(success=True)
        return (partition_name, True, None)
    except Exception as e:
        tracker.update(success=False)
        return (partition_name, False, str(e))


def chunk_list(lst: list, num_chunks: int) -> list:
    """将列表分成num_chunks块"""
    if num_chunks <= 0:
        return [lst]
    chunk_size = (len(lst) + num_chunks - 1) // num_chunks
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def insert_chunk_task_one_round(client: MilvusClient, collection_name: str,
                                partition_names: list, round_idx: int,
                                records_per_round: int, dim: int,
                                tracker: ProgressTracker) -> list:
    """一轮insert：向一批partition各写入records_per_round条数据"""
    results = []
    id_offset = round_idx * records_per_round
    for partition_name in partition_names:
        try:
            data = [
                {"id": id_offset + j, "vector": np.random.random(dim).tolist()}
                for j in range(records_per_round)
            ]
            t0 = time.time()
            client.insert(collection_name, data, partition_name=partition_name)
            t1 = time.time()
            tracker.timing.add_insert(t1 - t0)
            results.append((partition_name, True, None))
        except Exception as e:
            results.append((partition_name, False, str(e)))
    return results


def insert_data_concurrent(
    uri: str,
    collection_name: str,
    partition_names: list,
    num_rounds: int = 10,
    records_per_round: int = 10,
    dim: int = 128,
    max_workers: int = 50,
    report_interval: int = 100
):
    """
    并发向多个partition写入数据（每个worker负责一批partition），分多轮执行，每轮结束后flush

    Args:
        uri: Milvus服务地址
        collection_name: collection名称
        partition_names: partition名称列表
        num_rounds: 总轮数
        records_per_round: 每轮每个partition写入的记录数
        dim: 向量维度
        max_workers: 最大并发线程数
        report_interval: 每完成多少个partition打印一次进度
    """
    total_records = num_rounds * records_per_round
    # 将partitions分成max_workers块，每个worker负责一块
    chunks = chunk_list(partition_names, max_workers)
    print(f"\nInserting {total_records} records per partition ({num_rounds} rounds x {records_per_round} records, flush after each round)...")
    print(f"Split {len(partition_names)} partitions into {len(chunks)} chunks, ~{len(chunks[0]) if chunks else 0} partitions per worker")

    # tracker统计的是 partition*round 的完成数
    total_ops = len(partition_names) * num_rounds
    tracker = ProgressTracker(total_ops, report_interval=report_interval, records_per_partition=records_per_round)
    failed_partitions = []

    start_time = time.time()
    flush_client = MilvusClient(uri=uri)

    # 每个worker维护一个持久连接
    worker_clients = [MilvusClient(uri=uri) for _ in range(len(chunks))]

    for round_idx in range(num_rounds):
        round_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def worker_task(args):
                chunk_idx, chunk = args
                return insert_chunk_task_one_round(
                    worker_clients[chunk_idx], collection_name, chunk,
                    round_idx, records_per_round, dim, tracker
                )

            futures = [executor.submit(worker_task, (i, chunk)) for i, chunk in enumerate(chunks)]

            for future in as_completed(futures):
                results = future.result()
                for partition_name, success, error in results:
                    if success:
                        tracker.update(success=True)
                    else:
                        tracker.update(success=False)
                        failed_partitions.append((partition_name, error))

        # 每轮结束后flush
        print(f"Round {round_idx + 1}/{num_rounds} writes completed, starting flush...")
        t0 = time.time()
        flush_client.flush(collection_name)
        flush_time = time.time() - t0
        round_time = time.time() - round_start
        print(f"Round {round_idx + 1}/{num_rounds} done | Round time: {round_time:.2f}s | Flush time: {flush_time:.2f}s")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("DATA INSERTION SUMMARY")
    print("=" * 60)
    print(f"Total partitions: {len(partition_names)}")
    print(f"Rounds: {num_rounds}")
    print(f"Records per round per partition: {records_per_round}")
    print(f"Total records per partition: {total_records}")
    print(f"Total records: {len(partition_names) * total_records}")
    print(f"Insert operations completed: {tracker.completed}")
    print(f"Insert operations failed: {tracker.failed}")
    print(f"Total time: {elapsed:.2f} seconds")
    ops_rate = tracker.completed / elapsed if elapsed > 0 else 0
    records_rate = ops_rate * records_per_round
    print(f"Average rate: {ops_rate:.2f} inserts/s ({records_rate:.2f} records/s)")

    # 打印insert耗时统计
    stats = tracker.timing.summary()
    print(f"\n--- Timing Stats ---")
    ins = stats["insert"]
    print(f"Insert: count={ins['count']}, avg={ins['avg']*1000:.2f}ms, min={ins['min']*1000:.2f}ms, max={ins['max']*1000:.2f}ms, total={ins['total']:.2f}s")

    if failed_partitions:
        print(f"\nFailed partitions (showing first 10):")
        for name, error in failed_partitions[:10]:
            print(f"  - {name}: {error}")

    return tracker.completed, tracker.failed


def create_partitions_concurrent(
    uri: str,
    collection_name: str,
    num_partitions: int = 100000,
    max_workers: int = 50,
    partition_prefix: str = "partition_",
    num_rounds: int = 10,
    records_per_round: int = 10,
    dim: int = 128,
    report_interval: int = 100
):
    """
    并发创建多个partition并写入数据

    Args:
        uri: Milvus服务地址
        collection_name: collection名称
        num_partitions: 要创建的partition数量
        max_workers: 最大并发线程数
        partition_prefix: partition名称前缀
        num_rounds: 总轮数
        records_per_round: 每轮每个partition写入的记录数
        dim: 向量维度
        report_interval: 每完成多少个partition打印一次进度
    """
    print(f"Connecting to Milvus at {uri}...")
    client = MilvusClient(uri=uri)

    # 创建collection
    create_collection_if_not_exists(client, collection_name, dim=dim)

    # 获取已存在的partition列表
    existing_partitions = set(client.list_partitions(collection_name))
    print(f"Existing partitions: {len(existing_partitions)}")

    # 如果已经有100k partition，跳过创建
    if len(existing_partitions) >= num_partitions:
        print(f"Already have {len(existing_partitions)} partitions (>= {num_partitions}), skipping partition creation...")
        # 直接进行数据插入
        all_partitions = [p for p in existing_partitions if p != "_default"]
        if all_partitions:
            insert_data_concurrent(
                uri=uri,
                collection_name=collection_name,
                partition_names=all_partitions,
                num_rounds=num_rounds,
                records_per_round=records_per_round,
                dim=dim,
                max_workers=max_workers,
                report_interval=report_interval
            )
        return

    # 生成要创建的partition名称列表
    partitions_to_create = []
    for i in range(num_partitions):
        name = f"{partition_prefix}{i:06d}"
        if name not in existing_partitions:
            partitions_to_create.append(name)

    if not partitions_to_create:
        print("All partitions already exist!")
        return

    print(f"Creating {len(partitions_to_create)} partitions with {max_workers} workers...")

    tracker = ProgressTracker(len(partitions_to_create), report_interval=1000)
    failed_partitions = []

    start_time = time.time()

    # 使用线程池并发创建partition
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 每个线程创建自己的client连接
        def worker_task(partition_name):
            # 每个任务使用独立的client连接
            worker_client = MilvusClient(uri=uri)
            return create_partition_task(worker_client, collection_name, partition_name, tracker)

        futures = {executor.submit(worker_task, name): name for name in partitions_to_create}

        for future in as_completed(futures):
            partition_name, success, error = future.result()
            if not success:
                failed_partitions.append((partition_name, error))

    elapsed = time.time() - start_time

    # 打印结果
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total partitions requested: {num_partitions}")
    print(f"Partitions created: {tracker.completed}")
    print(f"Partitions failed: {tracker.failed}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average rate: {tracker.completed / elapsed:.2f} partitions/second")

    if failed_partitions:
        print(f"\nFailed partitions (showing first 10):")
        for name, error in failed_partitions[:10]:
            print(f"  - {name}: {error}")

    # 验证结果
    final_partitions = client.list_partitions(collection_name)
    print(f"\nFinal partition count in collection: {len(final_partitions)}")

    # 向所有partition写入数据
    all_partitions = [p for p in final_partitions if p != "_default"]
    if all_partitions:
        insert_data_concurrent(
            uri=uri,
            collection_name=collection_name,
            partition_names=all_partitions,
            num_rounds=num_rounds,
            records_per_round=records_per_round,
            dim=dim,
            max_workers=max_workers,
            report_interval=report_interval
        )


def main():
    parser = argparse.ArgumentParser(description="并发创建100k partition并写入数据")
    parser.add_argument("--uri", type=str, default="http://localhost:19530",
                        help="Milvus服务地址 (default: http://localhost:19530)")
    parser.add_argument("--collection", type=str, default="test_100k_partitions",
                        help="Collection名称 (default: test_100k_partitions)")
    parser.add_argument("--num-partitions", type=int, default=100000,
                        help="要创建的partition数量 (default: 100000)")
    parser.add_argument("--workers", type=int, default=50,
                        help="并发线程数 (default: 50)")
    parser.add_argument("--prefix", type=str, default="partition_",
                        help="Partition名称前缀 (default: partition_)")
    parser.add_argument("--num-rounds", type=int, default=10,
                        help="总轮数 (default: 10)")
    parser.add_argument("--records-per-round", type=int, default=10,
                        help="每轮每个partition写入的记录数 (default: 10)")
    parser.add_argument("--dim", type=int, default=128,
                        help="向量维度 (default: 128)")
    parser.add_argument("--report-interval", type=int, default=100,
                        help="每完成多少个partition打印一次进度 (default: 100)")

    args = parser.parse_args()

    create_partitions_concurrent(
        uri=args.uri,
        collection_name=args.collection,
        num_partitions=args.num_partitions,
        max_workers=args.workers,
        partition_prefix=args.prefix,
        num_rounds=args.num_rounds,
        records_per_round=args.records_per_round,
        dim=args.dim,
        report_interval=args.report_interval
    )


if __name__ == "__main__":
    main()
