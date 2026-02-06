"""
并发给100k partitions执行bulk import的示例程序
- 使用RemoteBulkWriter准备parquet文件（每个文件10条记录）
- 每个partition执行10次bulk import
- 并发执行，打印进度日志
"""

import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List

from pymilvus import MilvusClient, DataType, Collection, connections
from pymilvus.bulk_writer import (
    RemoteBulkWriter,
    BulkFileType,
    bulk_import,
    get_import_progress,
)


# 默认配置
MINIO_ADDRESS = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
DEFAULT_BUCKET_NAME = "a-bucket"
REMOTE_DATA_PATH = "bulk_import_100k"


class ProgressTracker:
    """线程安全的进度追踪器"""
    def __init__(self, total: int, report_interval: int = 100):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.lock = Lock()
        self.report_interval = report_interval
        self.start_time = time.time()

    def update(self, success: bool = True, job_id: str = "", partition: str = ""):
        with self.lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1

            processed = self.completed + self.failed

            # 每隔report_interval或达到100%时打印进度
            should_report = (processed % self.report_interval == 0) or (processed == self.total)

            if should_report:
                elapsed = time.time() - self.start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[Progress] {processed}/{self.total} ({processed*100/self.total:.1f}%) "
                      f"| Success: {self.completed} | Failed: {self.failed} "
                      f"| Rate: {rate:.1f} jobs/s | Elapsed: {elapsed:.1f}s")


def generate_files_with_bulk_writer(
    schema,
    num_files: int,
    records_per_file: int,
    dim: int,
    minio_address: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    remote_path: str
) -> List[str]:
    """使用RemoteBulkWriter生成并上传parquet文件到MinIO"""
    print(f"\nGenerating {num_files} parquet files with {records_per_file} records each using BulkWriter...")

    remote_files = []

    for i in range(num_files):
        file_path = f"{remote_path}/data_{i:04d}"

        with RemoteBulkWriter(
            schema=schema,
            remote_path=file_path,
            connect_param=RemoteBulkWriter.S3ConnectParam(
                endpoint=minio_address,
                access_key=access_key,
                secret_key=secret_key,
                bucket_name=bucket_name,
            ),
            file_type=BulkFileType.PARQUET,
        ) as writer:
            for j in range(records_per_file):
                writer.append_row({
                    "id": np.int64(j),
                    "vector": np.random.random(dim).astype(np.float32).tolist(),
                })
            writer.commit()
            batch_files = writer.batch_files

            # batch_files是List[List[str]]，取第一个batch的第一个文件
            if batch_files and batch_files[0]:
                remote_files.append(batch_files[0][0])

        if (i + 1) % 10 == 0 or i == num_files - 1:
            print(f"  Generated and uploaded {i + 1}/{num_files} files")

    print(f"All {num_files} files generated and uploaded to MinIO")
    return remote_files


def get_collection_schema(collection_name: str):
    """获取collection的schema"""
    collection = Collection(name=collection_name)
    return collection.schema


def submit_bulk_import_job(
    url: str,
    collection_name: str,
    partition_name: str,
    remote_file: str,
    api_key: str = "",
    timeout: int = 60
) -> str:
    """提交单个bulk import任务，返回job_id"""
    resp = bulk_import(
        url=url,
        collection_name=collection_name,
        partition_name=partition_name,
        files=[[remote_file]],
        api_key=api_key,
        timeout=timeout
    )
    result = resp.json()
    if result.get("code") != 0:
        raise Exception(f"Bulk import failed: {result.get('message')}")
    return result["data"]["jobId"]


def wait_for_job_completion(
    url: str,
    job_id: str,
    api_key: str = "",
    timeout: float = 300,
    poll_interval: float = 1.0
) -> bool:
    """等待bulk import任务完成"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        resp = get_import_progress(url=url, job_id=job_id, api_key=api_key)
        result = resp.json()

        if result.get("code") != 0:
            return False

        state = result["data"]["state"]

        if state == "Completed":
            return True
        elif state == "Failed":
            reason = result["data"].get("reason", "Unknown")
            print(f"  Job {job_id} failed: {reason}")
            return False

        time.sleep(poll_interval)

    print(f"  Job {job_id} timed out after {timeout}s")
    return False


def bulk_import_task(
    url: str,
    collection_name: str,
    partition_name: str,
    remote_file: str,
    api_key: str,
    progress: ProgressTracker,
    timeout: float = 300,
    request_timeout: int = 60
) -> tuple:
    """执行单个bulk import任务（提交+等待完成）"""
    try:
        # 提交任务
        job_id = submit_bulk_import_job(
            url=url,
            collection_name=collection_name,
            partition_name=partition_name,
            remote_file=remote_file,
            api_key=api_key,
            timeout=request_timeout
        )

        # 等待完成
        success = wait_for_job_completion(
            url=url,
            job_id=job_id,
            api_key=api_key,
            timeout=timeout
        )

        progress.update(success=success, job_id=job_id, partition=partition_name)
        return (partition_name, remote_file, success, None)

    except Exception as e:
        progress.update(success=False, partition=partition_name)
        return (partition_name, remote_file, False, str(e))


def bulk_import_concurrent(
    uri: str,
    collection_name: str,
    partition_names: List[str],
    remote_files: List[str],
    imports_per_partition: int = 10,
    max_workers: int = 50,
    report_interval: int = 100,
    api_key: str = "",
    job_timeout: float = 300,
    request_timeout: int = 60
):
    """
    并发执行bulk import

    Args:
        uri: Milvus服务地址
        collection_name: collection名称
        partition_names: partition名称列表
        remote_files: MinIO上的文件路径列表
        imports_per_partition: 每个partition执行的import次数
        max_workers: 最大并发数
        report_interval: 进度报告间隔
        api_key: API密钥
        job_timeout: 单个任务超时时间
    """
    total_jobs = len(partition_names) * imports_per_partition
    print(f"\n{'='*60}")
    print(f"BULK IMPORT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Partitions: {len(partition_names)}")
    print(f"Imports per partition: {imports_per_partition}")
    print(f"Total jobs: {total_jobs}")
    print(f"Max workers: {max_workers}")
    print(f"Available files: {len(remote_files)}")
    print(f"{'='*60}\n")

    progress = ProgressTracker(total_jobs, report_interval=report_interval)
    failed_jobs = []

    start_time = time.time()

    # 构建任务列表：每个partition执行imports_per_partition次import
    tasks = []
    for partition_name in partition_names:
        for i in range(imports_per_partition):
            # 轮流使用文件
            file_idx = i % len(remote_files)
            tasks.append((partition_name, remote_files[file_idx]))

    print(f"Starting bulk import with {max_workers} concurrent workers...")
    print(f"Total tasks: {len(tasks)}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for partition_name, remote_file in tasks:
            future = executor.submit(
                bulk_import_task,
                url=uri,
                collection_name=collection_name,
                partition_name=partition_name,
                remote_file=remote_file,
                api_key=api_key,
                progress=progress,
                timeout=job_timeout,
                request_timeout=request_timeout
            )
            futures[future] = (partition_name, remote_file)

        for future in as_completed(futures):
            partition_name, remote_file, success, error = future.result()
            if not success:
                failed_jobs.append((partition_name, remote_file, error))

    elapsed = time.time() - start_time

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"BULK IMPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Total jobs: {total_jobs}")
    print(f"Completed: {progress.completed}")
    print(f"Failed: {progress.failed}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average rate: {total_jobs / elapsed:.2f} jobs/second")

    if failed_jobs:
        print(f"\nFailed jobs (showing first 10):")
        for partition, file, error in failed_jobs[:10]:
            print(f"  - {partition} ({file}): {error}")

    return progress.completed, progress.failed


def main():
    parser = argparse.ArgumentParser(description="并发给100k partitions执行bulk import")
    parser.add_argument("--uri", type=str, default="http://localhost:19530",
                        help="Milvus服务地址 (default: http://localhost:19530)")
    parser.add_argument("--collection", type=str, default="test_100k_partitions",
                        help="Collection名称 (default: test_100k_partitions)")
    parser.add_argument("--num-partitions", type=int, default=100000,
                        help="要处理的partition数量 (default: 100000)")
    parser.add_argument("--imports-per-partition", type=int, default=10,
                        help="每个partition执行的import次数 (default: 10)")
    parser.add_argument("--num-files", type=int, default=10,
                        help="生成的parquet文件数量 (default: 10)")
    parser.add_argument("--records-per-file", type=int, default=10,
                        help="每个文件的记录数 (default: 10)")
    parser.add_argument("--dim", type=int, default=128,
                        help="向量维度 (default: 128)")
    parser.add_argument("--workers", type=int, default=50,
                        help="并发线程数 (default: 50)")
    parser.add_argument("--report-interval", type=int, default=100,
                        help="进度报告间隔 (default: 100)")
    parser.add_argument("--minio-address", type=str, default=MINIO_ADDRESS,
                        help=f"MinIO地址 (default: {MINIO_ADDRESS})")
    parser.add_argument("--minio-access-key", type=str, default=MINIO_ACCESS_KEY,
                        help="MinIO access key")
    parser.add_argument("--minio-secret-key", type=str, default=MINIO_SECRET_KEY,
                        help="MinIO secret key")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET_NAME,
                        help=f"MinIO bucket名称 (default: {DEFAULT_BUCKET_NAME})")
    parser.add_argument("--api-key", type=str, default="",
                        help="Milvus API key (如果需要)")
    parser.add_argument("--job-timeout", type=float, default=300,
                        help="单个bulk import任务超时时间(秒) (default: 300)")
    parser.add_argument("--request-timeout", type=int, default=60,
                        help="HTTP请求超时时间(秒) (default: 60)")
    parser.add_argument("--skip-file-gen", action="store_true",
                        help="跳过文件生成（假设文件已存在）")
    parser.add_argument("--remote-path", type=str, default=REMOTE_DATA_PATH,
                        help=f"MinIO远程路径 (default: {REMOTE_DATA_PATH})")

    args = parser.parse_args()

    print(f"Connecting to Milvus at {args.uri}...")
    client = MilvusClient(uri=args.uri)

    # 建立ORM连接（用于获取schema）
    # 解析uri获取host和port
    uri = args.uri.replace("http://", "").replace("https://", "")
    host, port = uri.split(":") if ":" in uri else (uri, "19530")
    connections.connect(host=host, port=port)

    # 检查collection是否存在
    if not client.has_collection(args.collection):
        print(f"Collection '{args.collection}' not found. Please create it first.")
        return

    # 获取partition列表
    existing_partitions = client.list_partitions(args.collection)
    partition_names = [p for p in existing_partitions if p != "_default"]

    if len(partition_names) == 0:
        print("No partitions found. Please create partitions first using create_100k_partitions.py")
        return

    # 限制partition数量
    if len(partition_names) > args.num_partitions:
        partition_names = partition_names[:args.num_partitions]

    print(f"Found {len(partition_names)} partitions to process")

    # 生成文件
    if not args.skip_file_gen:
        # 获取collection schema
        schema = get_collection_schema(args.collection)

        remote_files = generate_files_with_bulk_writer(
            schema=schema,
            num_files=args.num_files,
            records_per_file=args.records_per_file,
            dim=args.dim,
            minio_address=args.minio_address,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            bucket_name=args.bucket,
            remote_path=args.remote_path
        )
    else:
        # 假设文件已存在
        remote_files = [f"{args.remote_path}/data_{i:04d}/1.parquet" for i in range(args.num_files)]
        print(f"Skipping file generation, using existing files: {remote_files}")

    # 执行bulk import
    bulk_import_concurrent(
        uri=args.uri,
        collection_name=args.collection,
        partition_names=partition_names,
        remote_files=remote_files,
        imports_per_partition=args.imports_per_partition,
        max_workers=args.workers,
        report_interval=args.report_interval,
        api_key=args.api_key,
        job_timeout=args.job_timeout,
        request_timeout=args.request_timeout
    )


if __name__ == "__main__":
    main()
