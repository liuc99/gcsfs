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
        "--bucket",
        type=str,
        default=None,
        help="GCS bucket name for native GCS kvstore",
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
    return parser.parse_args(args)


def main():
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

    target_dir = os.path.join(args.mount_path, args.dataset_name)
    print(f"==================================================")
    print(f" TensorStore + GCSFuse Benchmark")
    print(f"==================================================")
    print(f" Mount Path   : {args.mount_path}")
    print(f" Target Dir   : {target_dir}")
    print(f" Array Shape  : {shape}")
    print(f" Chunk Shape  : {chunks}")
    print(f" Data Type    : {dtype.name}")
    print(f" Array Driver : {array_driver}")
    print(f" KVStore      : {kvstore_driver}")
    print(f" Iterations   : {args.iterations}")
    print(f"==================================================")

    # Calculate dataset size in MB
    num_elements = int(np.prod(shape))
    size_bytes = num_elements * dtype.itemsize
    size_mb = size_bytes / (1024 * 1024)
    print(f" Total Array Size: {size_mb:.2f} MB ({size_bytes} bytes)")

    # Construct KVStore spec
    if kvstore_driver == "gcs":
        bucket_name = args.bucket or os.environ.get("CHECKPOINT_BUCKET") or os.environ.get("GCSFUSE_CHECKPOINT_BUCKET")
        path_str = target_dir
        if path_str.startswith("gs://"):
            path_str = path_str[5:]
        elif path_str.startswith("/gcs/"):
            path_str = path_str[5:]
        path_str = path_str.strip("/")

        if bucket_name:
            parts = path_str.split("/", 1)
            if len(parts) > 1 and parts[0] in (bucket_name, "checkpoints", "dataset"):
                object_path = parts[1]
            else:
                object_path = path_str
        else:
            bucket_name, _, object_path = path_str.partition("/")

        kvstore_spec = {
            "driver": "gcs",
            "bucket": bucket_name,
            "path": object_path,
        }
    else:
        kvstore_spec = {
            "driver": "file",
            "path": target_dir,
        }

    # Ensure target directory parent exists for local filesystem
    if kvstore_driver == "file":
        os.makedirs(args.mount_path, exist_ok=True)

    for i in range(args.iterations):
        print(f"\n--- Iteration {i+1}/{args.iterations} ---")
        
        # Clean up existing directory if present on local filesystem
        if kvstore_driver == "file" and os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up existing path {target_dir}: {e}")

        # Generate data
        print("Generating random numpy array...")
        data_to_write = np.random.randn(*shape).astype(dtype)

        # 1. Write Benchmark
        print(f"Writing via TensorStore ({array_driver} on {kvstore_driver})...")
        ts_spec = {
            "driver": array_driver,
            "kvstore": kvstore_spec,
            "metadata": {
                "dtype": f"<{dtype.str[1:]}" if dtype.byteorder == "=" else dtype.str,
                "shape": shape,
                "chunks": chunks,
            },
            "create": True,
            "delete_existing": True,
        }

        start_time = time.perf_counter()
        dataset = ts.open(ts_spec).result()
        write_future = dataset.write(data_to_write)
        write_future.result()  # Wait for completion
        write_time = time.perf_counter() - start_time
        write_throughput = size_mb / write_time

        print(f"[BENCHMARK] Write finished in {write_time:.4f} sec | Size: {size_bytes} bytes ({size_mb:.2f} MB / {size_mb/1024:.2f} GB) | Throughput: {write_throughput:.2f} MB/s")

        # 2. Read Benchmark
        print(f"Reading back via TensorStore ({array_driver} on {kvstore_driver})...")
        read_spec = {
            "driver": array_driver,
            "kvstore": kvstore_spec,
            "open": True,
        }

        start_time = time.perf_counter()
        read_dataset = ts.open(read_spec).result()
        read_future = read_dataset.read()
        read_data = read_future.result()
        read_time = time.perf_counter() - start_time
        read_throughput = size_mb / read_time

        print(f"[BENCHMARK] Read finished in {read_time:.4f} sec | Size: {size_bytes} bytes ({size_mb:.2f} MB / {size_mb/1024:.2f} GB) | Throughput: {read_throughput:.2f} MB/s")

        # 3. Verification
        if args.verify:
            print("Verifying data integrity...")
            if np.array_equal(data_to_write, read_data):
                print(" SUCCESS: Read data matches written data exactly.")
            else:
                print(" FAILURE: Read data does NOT match written data!", file=sys.stderr)
                sys.exit(1)

        # 4. Partial Read / Slice Benchmark
        slice_shape = [min(dim, chunk) for dim, chunk in zip(shape, chunks)]
        slice_elements = int(np.prod(slice_shape))
        slice_bytes = slice_elements * dtype.itemsize
        slice_mb = slice_bytes / (1024 * 1024)
        print(f"Benchmarking slice read ({slice_shape})...")
        
        start_time = time.perf_counter()
        slice_dataset = read_dataset[tuple(slice(0, s) for s in slice_shape)]
        slice_data = slice_dataset.read().result()
        slice_time = time.perf_counter() - start_time
        slice_throughput = slice_mb / slice_time
        print(f"[BENCHMARK] Slice Read finished in {slice_time:.4f} sec | Size: {slice_bytes} bytes ({slice_mb:.2f} MB) | Throughput: {slice_throughput:.2f} MB/s")

    print("\n==================================================")
    print(" TensorStore + GCSFuse Benchmark Completed Successfully")
    print("==================================================")
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    import atexit
    atexit.register(sys.stdout.flush)
    atexit.register(sys.stderr.flush)
    main()
