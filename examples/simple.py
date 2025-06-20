import time
from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
)

from pymilvus import __version__
print(f"pymilvus version: {__version__}")

from pymilvus.milvus_client.index import IndexParams

fmt = "\n=== {:30} ===\n"
collection_name = "processed_with_user_data_with_scalar_v2"
token = "kelper_test:gengDANDAN0110"
milvus_client = MilvusClient("https://in01-3d2e3cd69c4eab7.ali-cn-hangzhou.vectordb.zilliz.com.cn:19530", token=token)

def read_embeddings_from_file(filename="embeddings.txt"):
    """
    Read embeddings from file, one embedding vector per line
    """
    embeddings = []
    with open(filename, 'r') as f:
        for line in f:
            # Convert each line string back to a list of numerical values
            embedding = [float(x) for x in line.strip().split()]
            embeddings.append(embedding)
    print(f"Read {len(embeddings)} embeddings from {filename}")
    return embeddings

# Read embeddings from file
embeddings = read_embeddings_from_file("embeddings.txt")
# print(f"embeddings[0]: {embeddings[0]}")
# print(f"embeddings[0] length: {len(embeddings[0])}")

# Define a function to perform search operations
def do_search(client, collection_name, query_vector):
    """
    Execute a single search operation
    """
    try:
        client.search(
            collection_name=collection_name,
            filter="position in ['北京市'] and age>18 and registration_days < 1000",
            data=[query_vector],
            limit=10,
            output_fields=["auto_id"]
            # search_params={"params": {"index_algo": "quantbf"}}
        )
        # client.query(
        #     collection_name=collection_name,
        #     filter="position in ['北京市'] and age>18 and registration_days < 1000",
        #     # data=[query_vector],
        #     limit=10,
        #     output_fields=["*"]
        #     # search_params={"params": {"index_algo": "quantbf"}}
        # )
        return True
    except Exception as e:
        print(f"Search error")
        return False

# Execute QPS test
def test_search_qps(concurrency=60, test_duration=180):
    """
    Execute search QPS test using multi-threaded concurrent search
    Parameters:
        concurrency: Number of concurrent threads
        test_duration: Test duration (seconds)
    """
    import threading
    import random
    from queue import Queue
    
    print(fmt.format("Starting concurrent QPS test"))
    
    
    # Variables for statistics
    successful_queries = 0
    total_queries = 0
    lock = threading.Lock()
    
    # Create a queue to store results
    result_queue = Queue()
    
    # Collect all thread statistics
    all_thread_stats = []
    
    def worker():
        nonlocal successful_queries, total_queries
        thread_start = time.time()
        
        # Initialize thread-local statistics
        thread_stats = {
            'successful_queries': 0,
            'total_queries': 0,
            'total_response_time': 0,
            'max_response_time': 0,
            'min_response_time': float('inf')
        }
        
        while time.time() - thread_start < test_duration:
            # Randomly select an embedding as query vector
            query_vector = embeddings[random.randint(0, len(embeddings)-1)]
            
            query_start = time.time()
            
            # Execute search
            success = do_search(milvus_client, collection_name, query_vector)
            
            query_end = time.time()
            response_time = query_end - query_start
            
            # Update thread-local statistics (no lock needed)
            if success:
                thread_stats['successful_queries'] += 1
            thread_stats['total_queries'] += 1
            thread_stats['total_response_time'] += response_time
            thread_stats['max_response_time'] = max(thread_stats['max_response_time'], response_time)
            thread_stats['min_response_time'] = min(thread_stats['min_response_time'], response_time)
        
        # Add thread statistics to global list
        with lock:
            all_thread_stats.append(thread_stats)
            successful_queries += thread_stats['successful_queries']
            total_queries += thread_stats['total_queries']
    
    # Create and start threads
    threads = []
    start_time = time.time()
    
    for _ in range(concurrency):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    end_time = time.time()
    actual_duration = end_time - start_time
    actual_qps = total_queries / actual_duration
    success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
    
    # Calculate response time statistics
    if sum(thread_stats['total_response_time'] for thread_stats in all_thread_stats) > 0:
        avg_response_time = sum(thread_stats['total_response_time'] for thread_stats in all_thread_stats) / total_queries
    else:
        avg_response_time = 0
    
    print(f"Test completed:")
    print(f"Total queries: {total_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Actual QPS: {actual_qps:.2f}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Actual duration: {actual_duration:.2f} seconds")
    print(f"Response time statistics:")
    print(f"  Average response time: {avg_response_time:.3f} seconds")
    print(f"  Maximum response time: {max(thread_stats['max_response_time'] for thread_stats in all_thread_stats):.3f} seconds")
    print(f"  Minimum response time: {min(thread_stats['min_response_time'] for thread_stats in all_thread_stats):.3f} seconds")



def test_insert_qps(concurrency=20, test_duration=180):
    """
    Execute insert QPS test using multi-threaded concurrent insert
    """
    import threading
    import random

    print(fmt.format("Starting concurrent Insert QPS test"))

    successful_inserts = 0
    total_inserts = 0
    lock = threading.Lock()
    all_thread_stats = []

    def do_insert():
        try:
            query_vector = embeddings[random.randint(0, len(embeddings) - 1)]
            rows = [
                {
                    "embedding": query_vector,
                    "position": "北京市",
                    "age": 25,
                    "registration_days": 999
                }
            ]
            milvus_client.insert(collection_name, rows)
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Insert error: {e}")
            return False

    def worker():
        nonlocal successful_inserts, total_inserts
        thread_start = time.time()

        thread_stats = {
            'successful_queries': 0,
            'total_queries': 0,
            'total_response_time': 0,
            'max_response_time': 0,
            'min_response_time': float('inf')
        }

        while time.time() - thread_start < test_duration:
            insert_start = time.time()
            success = do_insert()
            insert_end = time.time()
            response_time = insert_end - insert_start

            if success:
                thread_stats['successful_queries'] += 1
            thread_stats['total_queries'] += 1
            thread_stats['total_response_time'] += response_time
            thread_stats['max_response_time'] = max(thread_stats['max_response_time'], response_time)
            thread_stats['min_response_time'] = min(thread_stats['min_response_time'], response_time)

        with lock:
            all_thread_stats.append(thread_stats)
            successful_inserts += thread_stats['successful_queries']
            total_inserts += thread_stats['total_queries']

    threads = []
    start_time = time.time()

    for _ in range(concurrency):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    actual_duration = end_time - start_time
    
    if actual_duration > 0:
        actual_qps = total_inserts / actual_duration
    else:
        actual_qps = 0

    success_rate = (successful_inserts / total_inserts) * 100 if total_inserts > 0 else 0

    if total_inserts > 0:
        avg_response_time = sum(thread_stats['total_response_time'] for thread_stats in all_thread_stats) / total_inserts
    else:
        avg_response_time = 0

    print(f"\nInsert Test completed:")
    print(f"Total inserts: {total_inserts}")
    print(f"Successful inserts: {successful_inserts}")
    print(f"Actual insert QPS: {actual_qps:.2f}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Actual duration: {actual_duration:.2f} seconds")
    print(f"Response time statistics:")
    print(f"  Average response time: {avg_response_time:.3f} seconds")

    if all_thread_stats and total_inserts > 0:
        print(f"  Maximum response time: {max(thread_stats['max_response_time'] for thread_stats in all_thread_stats):.3f} seconds")
        min_response_time = min(thread_stats['min_response_time'] for thread_stats in all_thread_stats if thread_stats['min_response_time'] != float('inf'))
        print(f"  Minimum response time: {min_response_time:.3f} seconds")

# Run QPS test
if __name__ == "__main__":
    import threading

    # Create threads for each test function
    # You can adjust concurrency and duration for each test
    insert_thread = threading.Thread(target=test_insert_qps)
    search_thread = threading.Thread(target=test_search_qps)

    # Start the threads
    insert_thread.start()
    search_thread.start()

    # Wait for both threads to complete
    insert_thread.join()
    search_thread.join()

    print("\nBoth insert and search QPS tests are complete.")