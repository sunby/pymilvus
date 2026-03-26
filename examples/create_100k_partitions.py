"""
并发创建100k partition的示例程序，并向每个partition写入数据
"""

import time
import os
import argparse
import logging
import multiprocessing as mp
import numpy as np
from threading import Lock
from pymilvus import MilvusClient, DataType


_WORKER_CLIENT = None
_WORKER_RNG = None
WORKER_LOG_INTERVAL = 100


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def init_worker(uri: str):
    """为每个进程初始化独立的Milvus连接和随机数生成器"""
    global _WORKER_CLIENT, _WORKER_RNG
    _WORKER_CLIENT = MilvusClient(uri=uri)
    _WORKER_RNG = np.random.default_rng(seed=time.time_ns() ^ os.getpid())


def get_worker_client() -> MilvusClient:
    """获取当前worker进程中的Milvus连接"""
    if _WORKER_CLIENT is None:
        raise RuntimeError("Worker client is not initialized")
    return _WORKER_CLIENT


def create_collection_if_not_exists(client: MilvusClient, collection_name: str, dim: int = 128):
    """创建collection（如果不存在）"""
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists, skipping creation...")
    else:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
        )
        print(f"Collection '{collection_name}' created successfully")


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
                    "total": sum(times)
                }
            return {
                "insert": calc_stats(self.insert_times),
                "flush": calc_stats(self.flush_times)
            }


class ProgressTracker:
    """主进程中的进度追踪器"""
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
            should_report = (processed % self.report_interval == 0) or (processed == self.total)
            if should_report:
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


def create_partition_chunk_task(collection_name: str, partition_names: list) -> tuple:
    """在一个worker进程中顺序创建一批partition"""
    client = get_worker_client()
    success_count = 0
    failed_partitions = []
    total = len(partition_names)

    for idx, partition_name in enumerate(partition_names, start=1):
        try:
            client.create_partition(collection_name, partition_name)
            success_count += 1
        except Exception as e:
            logging.warning(
                "Failed to create partition '%s' in collection '%s': %s",
                partition_name,
                collection_name,
                e,
            )
            failed_partitions.append((partition_name, str(e)))
        if idx % WORKER_LOG_INTERVAL == 0 or idx == total:
            logging.info(
                "Worker pid=%s create progress: %s/%s partitions in collection '%s'",
                os.getpid(),
                idx,
                total,
                collection_name,
            )
    return success_count, failed_partitions


def create_partition_worker_task(args) -> list:
    """multiprocessing.Pool 的包装任务"""
    return create_partition_chunk_task(*args)


def chunk_list(lst: list, num_chunks: int) -> list:
    """将列表分成num_chunks块"""
    if num_chunks <= 0:
        return [lst]
    chunk_size = (len(lst) + num_chunks - 1) // num_chunks
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def insert_chunk_task_one_round(collection_name: str, partition_names: list,
                                round_idx: int, records_per_round: int,
                                dim: int) -> tuple:
    """一轮insert：在一个worker进程中向一批partition写入数据"""
    client = get_worker_client()
    results = []
    insert_times = []
    id_offset = round_idx * records_per_round
    total = len(partition_names)
    for idx, partition_name in enumerate(partition_names, start=1):
        try:
            vectors = _WORKER_RNG.random((records_per_round, dim)).tolist()
            data = [
                {"id": id_offset + j, "vector": vector}
                for j, vector in enumerate(vectors)
            ]
            t0 = time.time()
            client.insert(collection_name, data, partition_name=partition_name)
            insert_times.append(time.time() - t0)
            results.append((partition_name, True, None))
        except Exception as e:
            logging.warning(
                "Failed to insert into partition '%s' in collection '%s' (round=%s): %s",
                partition_name,
                collection_name,
                round_idx + 1,
                e,
            )
            results.append((partition_name, False, str(e)))
        if idx % WORKER_LOG_INTERVAL == 0 or idx == total:
            logging.info(
                "Worker pid=%s insert progress: round=%s %s/%s partitions in collection '%s'",
                os.getpid(),
                round_idx + 1,
                idx,
                total,
                collection_name,
            )
    return results, insert_times


def insert_chunk_worker_task(args) -> tuple:
    """multiprocessing.Pool 的包装任务"""
    return insert_chunk_task_one_round(*args)


def insert_data_concurrent(
    uri: str,
    collection_name: str,
    partition_names: list,
    num_rounds: int = 10,
    records_per_round: int = 10,
    dim: int = 128,
    max_workers: int = 16,
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
        max_workers: 最大并发worker数
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

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=min(max_workers, len(chunks)),
        initializer=init_worker,
        initargs=(uri,),
    ) as pool:
        for round_idx in range(num_rounds):
            round_start = time.time()
            worker_args = [
                (collection_name, chunk, round_idx, records_per_round, dim)
                for chunk in chunks
            ]
            for results, insert_times in pool.imap_unordered(insert_chunk_worker_task, worker_args):
                for elapsed_insert in insert_times:
                    tracker.timing.add_insert(elapsed_insert)
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
            tracker.timing.add_flush(flush_time)
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
    flush = stats["flush"]
    print(f"Flush: count={flush['count']}, avg={flush['avg']*1000:.2f}ms, min={flush['min']*1000:.2f}ms, max={flush['max']*1000:.2f}ms, total={flush['total']:.2f}s")

    if failed_partitions:
        print(f"\nFailed partitions (showing first 10):")
        for name, error in failed_partitions[:10]:
            print(f"  - {name}: {error}")

    return tracker.completed, tracker.failed


def create_partitions_concurrent(
    uri: str,
    collection_name: str,
    num_partitions: int = 100000,
    max_workers: int = 16,
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
        max_workers: 最大并发worker数
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

    failed_partitions = []
    created_count = 0

    start_time = time.time()
    chunks = chunk_list(partitions_to_create, max_workers)
    print(f"Split creation work into {len(chunks)} worker chunks, ~{len(chunks[0]) if chunks else 0} partitions per worker")

    # 使用进程池并发创建partition，避免Python层线程竞争GIL
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=min(max_workers, len(chunks)),
        initializer=init_worker,
        initargs=(uri,),
    ) as pool:
        worker_args = [
            (collection_name, chunk)
            for chunk in chunks
        ]

        for success_count, failed_chunk in pool.imap_unordered(create_partition_worker_task, worker_args):
            created_count += success_count
            failed_partitions.extend(failed_chunk)

    elapsed = time.time() - start_time

    # 打印结果
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total partitions requested: {num_partitions}")
    print(f"Partitions created: {created_count}")
    print(f"Partitions failed: {len(failed_partitions)}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average rate: {created_count / elapsed:.2f} partitions/second")

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
    parser.add_argument("--uri", type=str, default="http://10.15.1.205:19530",
                        help="Milvus服务地址 (default: http://10.15.1.205:19530)")
    parser.add_argument("--collection", type=str, default="test_100k_partitions",
                        help="Collection名称 (default: test_100k_partitions)")
    parser.add_argument("--num-partitions", type=int, default=100000,
                        help="要创建的partition数量 (default: 100000)")
    parser.add_argument("--workers", type=int, default=16,
                        help="并发worker数 (default: 16)")
    parser.add_argument("--prefix", type=str, default="partition_",
                        help="Partition名称前缀 (default: partition_)")
    parser.add_argument("--num-rounds", type=int, default=10,
                        help="总轮数 (default: 10)")
    parser.add_argument("--records-per-round", type=int, default=10,
                        help="每轮每个partition写入的记录数 (default: 10)")
    parser.add_argument("--dim", type=int, default=128,
                        help="向量维度 (default: 128)")
    parser.add_argument("--report-interval", type=int, default=500,
                        help="每完成多少个partition打印一次进度 (default: 500)")

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
