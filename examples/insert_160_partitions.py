"""
向 create_100k_partitions.py 已创建的 collection 中，并发持续写入数据，不手动flush
"""

import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pymilvus import MilvusClient


class Stats:
    """线程安全的全局统计，仅用于最终汇总"""
    def __init__(self):
        self.lock = Lock()
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.insert_times = []

    def record(self, success: bool, elapsed: float = 0):
        with self.lock:
            if success:
                self.completed += 1
                self.insert_times.append(elapsed)
            else:
                self.failed += 1

    def summary(self):
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total inserts: {self.completed}")
        print(f"Failed: {self.failed}")
        print(f"Total time: {elapsed:.2f}s")
        if self.completed > 0:
            rate = self.completed / elapsed if elapsed > 0 else 0
            print(f"Throughput: {rate:.1f} inserts/s")
            times = self.insert_times
            print(f"Insert latency: avg={np.mean(times)*1000:.1f}ms, "
                  f"min={np.min(times)*1000:.1f}ms, max={np.max(times)*1000:.1f}ms")


def worker_loop(worker_id: int, client: MilvusClient, collection_name: str,
                partition_names: list, records_per_insert: int,
                dim: int, report_every: int, stats: Stats):
    """每个worker持续循环向自己负责的partition写入数据"""
    id_counter = 0
    local_inserted = 0
    local_failed = 0
    local_start = time.time()
    while True:
        for pname in partition_names:
            try:
                data = [
                    {"id": id_counter + j, "vector": np.random.random(dim).tolist()}
                    for j in range(records_per_insert)
                ]
                t0 = time.time()
                client.insert(collection_name, data, partition_name=pname)
                elapsed = time.time() - t0
                stats.record(success=True, elapsed=elapsed)
                id_counter += records_per_insert
                local_inserted += records_per_insert
            except Exception as e:
                stats.record(success=False)
                local_failed += 1
                print(f"[Worker-{worker_id}] Error inserting to {pname}: {e}")

            if local_inserted > 0 and local_inserted % report_every == 0:
                wall = time.time() - local_start
                rate = local_inserted / wall if wall > 0 else 0
                print(f"[Worker-{worker_id}] inserted: {local_inserted} | "
                      f"failed: {local_failed} | rate: {rate:.1f} records/s | "
                      f"elapsed: {wall:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="并发向160个partition持续写入数据（不手动flush）")
    parser.add_argument("--uri", type=str, default="http://localhost:19530")
    parser.add_argument("--collection", type=str, default="test_100k_partitions",
                        help="Collection名称，需与create_100k_partitions.py一致")
    parser.add_argument("--num-partitions", type=int, default=200,
                        help="要写入的partition数量 (default: 200)")
    parser.add_argument("--workers", type=int, default=200)
    parser.add_argument("--prefix", type=str, default="partition_",
                        help="Partition名称前缀，需与create_100k_partitions.py一致")
    parser.add_argument("--records-per-insert", type=int, default=1000,
                        help="每次insert的记录数 (default: 1000)")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--report-every", type=int, default=10000,
                        help="每个worker每写入多少条记录打印一次日志 (default: 10000)")

    args = parser.parse_args()

    print(f"Connecting to Milvus at {args.uri}...")
    client = MilvusClient(uri=args.uri)

    if not client.has_collection(args.collection):
        print(f"ERROR: Collection '{args.collection}' does not exist. "
              f"Please run create_100k_partitions.py first.")
        return

    # 选取前num_partitions个partition（与create_100k_partitions.py的命名格式一致: partition_000000）
    existing = set(client.list_partitions(args.collection))
    partition_names = []
    for i in range(args.num_partitions):
        name = f"{args.prefix}{i:06d}"
        if name in existing:
            partition_names.append(name)

    if not partition_names:
        print(f"ERROR: No matching partitions found with prefix '{args.prefix}'. "
              f"Please run create_100k_partitions.py first.")
        return

    print(f"Found {len(partition_names)} partitions to write")

    # 将partition分配给各worker
    chunks = [[] for _ in range(args.workers)]
    for i, pname in enumerate(partition_names):
        chunks[i % args.workers].append(pname)

    worker_clients = [MilvusClient(uri=args.uri) for _ in range(args.workers)]
    stats = Stats()

    print(f"\nStarting {args.workers} workers, {args.records_per_insert} records per insert, "
          f"no flush, Ctrl+C to stop\n")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for i in range(args.workers):
                f = executor.submit(
                    worker_loop, i, worker_clients[i], args.collection,
                    chunks[i], args.records_per_insert, args.dim,
                    args.report_every, stats
                )
                futures.append(f)
            # 等待（实际会被Ctrl+C中断）
            for f in futures:
                f.result()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        stats.summary()


if __name__ == "__main__":
    main()
