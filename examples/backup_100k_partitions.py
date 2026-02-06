"""
备份100k partitions collection的程序
将数据保存到 ~/Downloads/milvus_backup/ 目录下
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pymilvus import MilvusClient


class ProgressTracker:
    """线程安全的进度追踪器"""
    def __init__(self, total: int, report_interval: int = 1000):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.total_records = 0
        self.lock = Lock()
        self.report_interval = report_interval
        self.start_time = time.time()

    def update(self, success: bool = True, records: int = 0):
        with self.lock:
            if success:
                self.completed += 1
                self.total_records += records
            else:
                self.failed += 1

            processed = self.completed + self.failed
            if processed % self.report_interval == 0 or processed == self.total:
                elapsed = time.time() - self.start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"Progress: {processed}/{self.total} ({processed*100/self.total:.1f}%) "
                      f"| Success: {self.completed} | Failed: {self.failed} "
                      f"| Records: {self.total_records} | Rate: {rate:.1f}/s | Elapsed: {elapsed:.1f}s")


def backup_partition(client: MilvusClient, collection_name: str, partition_name: str,
                     backup_dir: Path, tracker: ProgressTracker) -> tuple:
    """备份单个partition的数据"""
    try:
        # 查询partition中的所有数据
        results = client.query(
            collection_name=collection_name,
            partition_names=[partition_name],
            filter="",
            output_fields=["*"],
            limit=100000  # 假设每个partition最多100000条记录
        )

        if results:
            # 将vector转换为list以便JSON序列化
            for record in results:
                if "vector" in record and isinstance(record["vector"], np.ndarray):
                    record["vector"] = record["vector"].tolist()

            # 保存到文件
            partition_file = backup_dir / f"{partition_name}.json"
            with open(partition_file, "w") as f:
                json.dump(results, f)

            tracker.update(success=True, records=len(results))
            return (partition_name, True, len(results), None)
        else:
            tracker.update(success=True, records=0)
            return (partition_name, True, 0, None)

    except Exception as e:
        tracker.update(success=False)
        return (partition_name, False, 0, str(e))


def backup_collection(
    uri: str,
    collection_name: str,
    backup_dir: str = None,
    max_workers: int = 20
):
    """
    备份整个collection的数据

    Args:
        uri: Milvus服务地址
        collection_name: collection名称
        backup_dir: 备份目录路径
        max_workers: 最大并发线程数
    """
    # 设置备份目录
    if backup_dir is None:
        backup_dir = Path.home() / "Downloads" / "milvus_backup" / collection_name
    else:
        backup_dir = Path(backup_dir)

    # 创建备份目录
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"Backup directory: {backup_dir}")

    # 连接Milvus
    print(f"Connecting to Milvus at {uri}...")
    client = MilvusClient(uri=uri)

    # 检查collection是否存在
    if not client.has_collection(collection_name):
        print(f"Collection '{collection_name}' does not exist!")
        return

    # 获取collection信息
    collection_info = client.describe_collection(collection_name)
    print(f"Collection info: {collection_info}")

    # 保存collection schema信息
    schema_file = backup_dir / "_schema.json"
    with open(schema_file, "w") as f:
        json.dump(collection_info, f, indent=2, default=str)
    print(f"Schema saved to {schema_file}")

    # 获取所有partition
    partitions = client.list_partitions(collection_name)
    print(f"Total partitions: {len(partitions)}")

    # 过滤掉_default partition（如果为空）
    partitions_to_backup = [p for p in partitions if p != "_default"]

    if not partitions_to_backup:
        print("No partitions to backup!")
        return

    print(f"Backing up {len(partitions_to_backup)} partitions with {max_workers} workers...")

    tracker = ProgressTracker(len(partitions_to_backup), report_interval=1000)
    failed_partitions = []

    start_time = time.time()

    # 使用线程池并发备份
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        def worker_task(partition_name):
            worker_client = MilvusClient(uri=uri)
            return backup_partition(worker_client, collection_name, partition_name, backup_dir, tracker)

        futures = {executor.submit(worker_task, name): name for name in partitions_to_backup}

        for future in as_completed(futures):
            partition_name, success, records, error = future.result()
            if not success:
                failed_partitions.append((partition_name, error))

    elapsed = time.time() - start_time

    # 保存备份元数据
    metadata = {
        "collection_name": collection_name,
        "total_partitions": len(partitions_to_backup),
        "backed_up_partitions": tracker.completed,
        "failed_partitions": tracker.failed,
        "total_records": tracker.total_records,
        "backup_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed
    }
    metadata_file = backup_dir / "_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # 打印结果
    print("\n" + "=" * 60)
    print("BACKUP SUMMARY")
    print("=" * 60)
    print(f"Collection: {collection_name}")
    print(f"Backup directory: {backup_dir}")
    print(f"Total partitions: {len(partitions_to_backup)}")
    print(f"Partitions backed up: {tracker.completed}")
    print(f"Partitions failed: {tracker.failed}")
    print(f"Total records: {tracker.total_records}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average rate: {tracker.completed / elapsed:.2f} partitions/second")

    if failed_partitions:
        print(f"\nFailed partitions (showing first 10):")
        for name, error in failed_partitions[:10]:
            print(f"  - {name}: {error}")

        # 保存失败列表
        failed_file = backup_dir / "_failed.json"
        with open(failed_file, "w") as f:
            json.dump(failed_partitions, f, indent=2)

    print(f"\nBackup completed! Files saved to: {backup_dir}")


def main():
    parser = argparse.ArgumentParser(description="备份100k partitions collection")
    parser.add_argument("--uri", type=str, default="http://localhost:19530",
                        help="Milvus服务地址 (default: http://localhost:19530)")
    parser.add_argument("--collection", type=str, default="test_100k_partitions",
                        help="Collection名称 (default: test_100k_partitions)")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="备份目录路径 (default: ~/Downloads/milvus_backup/<collection_name>)")
    parser.add_argument("--workers", type=int, default=20,
                        help="并发线程数 (default: 20)")

    args = parser.parse_args()

    backup_collection(
        uri=args.uri,
        collection_name=args.collection,
        backup_dir=args.backup_dir,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
