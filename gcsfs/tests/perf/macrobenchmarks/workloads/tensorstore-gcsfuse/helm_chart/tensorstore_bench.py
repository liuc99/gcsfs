#!/usr/bin/env python3
"""
TensorStore + GCSFuse Benchmark Script.

Tests writing and reading multi-dimensional arrays using TensorStore over
a GCSFuse mounted file system path.
"""

import argparse
import os
import sys
import time
import shutil
import numpy as np
import tensorstore as ts


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="TensorStore + GCSFuse Read/Write Benchmark")
    parser.add_argument(
        "--mount-path",
        type=str,
        default="/gcs/checkpoint",
        help="Target directory path (e.g., GCSFuse mount point)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="tensorstore_bench.zarr",
        help="Name of the array dataset folder",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="1000,1000,100",
        help="Shape of the array as comma-separated integers (e.g., 1000,1000,100)",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="100,100,100",
        help="Chunk shape as comma-separated integers (e.g., 100,100,100)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Numpy data type (e.g., float32, float64, int32)",
    )
    parser.add_argument(
        "--driver",
        type=str,
        default="zarr",
        choices=["zarr", "zarr3", "n5", "gcs"],
        help="TensorStore driver for multi-dimensional array storage (use 'gcs' for native GCS kvstore with zarr)",
    )
    parser.add_argument(
        "--kvstore-driver",
        type=str,
        default=None,
        choices=["file", "gcs", "auto"],
        help="TensorStore kvstore driver ('file' for filesystem/GCSFuse, 'gcs' for native GCS kvstore)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of read/write iterations",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify read data matches written data",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        default=False,
        help="Skip writing and only benchmark reading an existing dataset",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("TENSORSTORE_NUM_WORKERS", "1")),
        help="Number of concurrent worker processes (default: 1)",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ.get("NODE_RANK", os.environ.get("JOB_COMPLETION_INDEX", "0"))),
        help="Rank of the current node (0-indexed)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=int(os.environ.get("NUM_NODES", os.environ.get("NNODES", "1"))),
        help="Total number of nodes participating in the benchmark",
    )
    parser.add_argument(
        "--per-worker-shape",
        action="store_true",
        default=os.environ.get("TENSORSTORE_PER_WORKER_SHAPE", "false").lower() in ("true", "1"),
        help="Treat --shape as the per-worker shape rather than partitioning global shape across workers",
    )
    return parser.parse_args(args)


def run_worker(worker_id, global_worker_id, total_global_workers, node_rank, num_nodes, shape, chunks, dtype, array_driver, kvstore_driver, args):
    dataset_name = f"{args.dataset_name}_rank_{global_worker_id}" if total_global_workers > 1 else args.dataset_name
    target_dir = os.path.join(args.mount_path, dataset_name)
    num_elements = int(np.prod(shape))
    size_bytes = num_elements * dtype.itemsize
    size_mb = size_bytes / (1024 * 1024)

    if kvstore_driver == "gcs":
        path_str = target_dir
        if path_str.startswith("gs://"):
            path_str = path_str[5:]
        elif path_str.startswith("/gcs/"):
            path_str = path_str[5:]
        path_str = path_str.strip("/")
        bucket, _, object_path = path_str.partition("/")
        kvstore_spec = {"driver": "gcs", "bucket": bucket, "path": object_path}
    else:
        kvstore_spec = {"driver": "file", "path": target_dir}

    if kvstore_driver == "file":
        os.makedirs(args.mount_path, exist_ok=True)

    results = {}
    for i in range(args.iterations):
        if not args.read_only:
            if kvstore_driver == "file" and os.path.exists(target_dir):
                try:
                    shutil.rmtree(target_dir)
                except Exception as e:
                    print(f"[Node {node_rank} | Worker {worker_id} (Global {global_worker_id})] Warning: Failed to clean up {target_dir}: {e}")

            buf_size_elements = 4 * 1024 * 1024
            random_buf = np.random.default_rng().random(buf_size_elements, dtype=dtype)
            total_elements = int(np.prod(shape))
            repeats = (total_elements + buf_size_elements - 1) // buf_size_elements
            data_to_write = np.tile(random_buf, repeats)[:total_elements].reshape(shape)

            ts_spec = {
                "driver": array_driver,
                "kvstore": kvstore_spec,
                "metadata": {
                    "dtype": f"<{dtype.str[1:]}" if dtype.byteorder == "=" else dtype.str,
                    "shape": shape,
                    "chunks": chunks,
                    "compressor": None,
                },
                "create": True,
                "delete_existing": True,
            }
            ts_context = ts.Context({
                "file_io_concurrency": {"limit": 8},
                "data_copy_concurrency": {"limit": 32},
            })

            w_start = time.perf_counter()
            dataset = ts.open(ts_spec, context=ts_context).result()
            write_future = dataset.write(data_to_write)
            write_future.result()
            w_time = time.perf_counter() - w_start
            print(f"[Node {node_rank} | Worker {worker_id} (Global {global_worker_id})] Write finished in {w_time:.4f} sec | Throughput: {size_mb/w_time:.2f} MB/s")
            results["write_time"] = w_time
        else:
            data_to_write = None
            ts_context = ts.Context({
                "file_io_concurrency": {"limit": 8},
                "data_copy_concurrency": {"limit": 32},
            })

        read_spec = {"driver": array_driver, "kvstore": kvstore_spec, "open": True}
        r_start = time.perf_counter()
        read_dataset = ts.open(read_spec, context=ts_context).result()
        read_future = read_dataset.read()
        read_data = read_future.result()
        r_time = time.perf_counter() - r_start
        print(f"[Node {node_rank} | Worker {worker_id} (Global {global_worker_id})] Read finished in {r_time:.4f} sec | Throughput: {size_mb/r_time:.2f} MB/s")
        results["read_time"] = r_time

        if args.verify and data_to_write is not None:
            if not np.array_equal(data_to_write, read_data):
                print(f"[Node {node_rank} | Worker {worker_id} (Global {global_worker_id})] FAILURE: Read data mismatch!", file=sys.stderr)
                sys.exit(1)

        slice_shape = [min(dim, chunk) for dim, chunk in zip(shape, chunks)]
        s_start = time.perf_counter()
        slice_dataset = read_dataset[tuple(slice(0, s) for s in slice_shape)]
        _ = slice_dataset.read().result()
        s_time = time.perf_counter() - s_start
        results["slice_time"] = s_time

    return results


def main():
    import concurrent.futures
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()

    shape = [int(x) for x in args.shape.split(",")]
    chunks = [int(x) for x in args.chunks.split(",")]
    dtype = np.dtype(args.dtype)

    array_driver = args.driver
    kvstore_driver = args.kvstore_driver

    if args.driver == "gcs":
        array_driver = "zarr"
        kvstore_driver = "gcs"
    elif kvstore_driver is None:
        if args.mount_path.startswith("gs://"):
            kvstore_driver = "gcs"
        else:
            kvstore_driver = "file"

    num_workers = max(1, args.num_workers)
    node_rank = max(0, args.node_rank)
    num_nodes = max(1, args.num_nodes)
    total_global_workers = num_nodes * num_workers
    target_dir = os.path.join(args.mount_path, args.dataset_name)

    partition_dim = 1
    if args.per_worker_shape:
        worker_shape = shape
        node_shape = list(shape)
        node_shape[0] = shape[0] * num_workers
    elif num_workers > 1:
        node_shape = shape
        worker_shape = list(shape)
        if shape[0] % num_workers == 0 and (shape[0] // num_workers) >= chunks[0]:
            worker_shape[0] = shape[0] // num_workers
            partition_dim = 0
        else:
            worker_shape[1] = max(1, shape[1] // num_workers)
            partition_dim = 1
    else:
        node_shape = shape
        worker_shape = shape

    worker_elements = int(np.prod(worker_shape))
    worker_size_mb = (worker_elements * dtype.itemsize) / (1024 * 1024)
    node_size_mb = worker_size_mb * num_workers
    total_cluster_size_mb = node_size_mb * num_nodes

    print(f"==================================================")
    print(f" TensorStore + GCSFuse Benchmark")
    print(f"==================================================")
    print(f" Mount Path          : {args.mount_path}")
    print(f" Target Dir          : {target_dir}")
    print(f" Node Shape          : {node_shape}")
    print(f" Worker Shape        : {worker_shape}")
    print(f" Chunk Shape         : {chunks}")
    print(f" Data Type           : {dtype.name}")
    print(f" Node Rank           : {node_rank} / {num_nodes}")
    print(f" Local Workers/Node  : {num_workers}")
    print(f" Total Global Workers: {total_global_workers}")
    print(f" Per-Worker Data Size: {worker_size_mb:.2f} MB")
    print(f" Node Data Size      : {node_size_mb:.2f} MB ({node_size_mb/1024:.2f} GB)")
    print(f" Cluster Data Size   : {total_cluster_size_mb:.2f} MB ({total_cluster_size_mb/1024:.2f} GB)")
    print(f"==================================================")

    if num_workers == 1:
        global_worker_id = node_rank * num_workers
        all_results = [
            run_worker(
                0, global_worker_id, total_global_workers, node_rank, num_nodes, worker_shape, chunks, dtype, array_driver, kvstore_driver, args
            )
        ]
    else:
        start_gid = node_rank * num_workers
        end_gid = start_gid + num_workers - 1
        print(f"[Node {node_rank}] Launching {num_workers} concurrent worker processes (Global IDs {start_gid}..{end_gid})...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_worker, w, node_rank * num_workers + w, total_global_workers, node_rank, num_nodes, worker_shape, chunks, dtype, array_driver, kvstore_driver, args
                )
                for w in range(num_workers)
            ]
            all_results = [f.result() for f in futures]

    if not args.read_only:
        max_write_time = max(r["write_time"] for r in all_results)
        node_write_tp = node_size_mb / max_write_time
        print(f"\n[BENCHMARK] [Node {node_rank}] Node Write finished in {max_write_time:.4f} sec | Size: {node_size_mb:.2f} MB ({node_size_mb/1024:.2f} GB) | Throughput: {node_write_tp:.2f} MB/s")

    max_read_time = max(r["read_time"] for r in all_results)
    node_read_tp = node_size_mb / max_read_time
    print(f"[BENCHMARK] [Node {node_rank}] Node Read finished in {max_read_time:.4f} sec | Size: {node_size_mb:.2f} MB ({node_size_mb/1024:.2f} GB) | Throughput: {node_read_tp:.2f} MB/s")

    print("\n==================================================")
    print(f" TensorStore + GCSFuse Benchmark (Node {node_rank}) Completed Successfully")
    print("==================================================")
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    import atexit
    atexit.register(sys.stdout.flush)
    atexit.register(sys.stderr.flush)
    main()
