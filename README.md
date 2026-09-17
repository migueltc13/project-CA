# CHORUS: Checkpoint Orchestration with Rate-limiting and Unified Service-policies <br> *Advanced Computing Project*

## Grade: 18/20 ⭐

## Assignment

> **Advisors:**
Ricardo Macedo ([d12010@di.uminho.pt](mailto:d12010@di.uminho.pt));
João Paulo ([jtpaulo@di.uminho.pt](mailto:jtpaulo@di.uminho.pt))
>
> Large Language Models (LLMs) training is a complex workload that can take
several weeks to complete. The complexity and long execution time are inherent
to their size, as they can easily reach billions of parameters that need to be
refined across training. Upon a failure (e.g., one or more nodes used in the
training process become unavailable), if these parameters are not saved, the
training process needs to be restarted from scratch. To address this challenge,
the model's state (i.e., activations, weights) is periodically checkpointed to
persistent storage, such as the Parallel File System (PFS) in HPC environments.
However, storing multiple checkpoints in a shared storage system can lead to
performance degradation, especially when the storage system is being used by
multiple jobs at the same time.
>
> The goal of this work is to propose a solution that improves checkpointing
performance of LLM training (without compromising correctness), while reducing
the I/O pressure at the PFS. The project should consider different fault models
(*i.e.*, failure of multiple nodes) and leverage the resources of all nodes
participating in the LLM training.
>
> **References**
> - Accelerating Deep Learning Training Through Transparent Storage Tiering.
    Dantas, M., Leitão, D., Cui, P., Macedo, R., Liu, X., Xu, W., & Paulo, J.
    In 22nd IEEE/ACM International Symposium on Cluster, Cloud and Internet
    Computing (CCGrid). IEEE, 2022.
> - Mohan, Jayashree, Amar Phanishayee, and Vijay Chidambaram. "{CheckFreq}:
    Frequent,{Fine-Grained}{DNN} Checkpointing." In 19th USENIX Conference on
    File and Storage Technologies (FAST 21), pp. 203-216. 2021.

## Overview

This project explores checkpoint migration for distributed training workloads
in HPC environments. Instead of letting every worker write checkpoints directly
to the Parallel File System (PFS), workers save checkpoints locally and notify
a Migrater service. The Migrater coordinates with an Orchestrator, which decides
when each worker may flush data to the PFS and at what rate.

The main goal is to reduce PFS contention while preserving checkpoint
correctness.

**Report:** [docs/report/report.pdf](docs/report/report.pdf)

## Architecture

The system is composed of four main parts:

- **Training / checkpoint extension** (`src/train/`, `src/torch_ext/`): saves
    checkpoints locally and notifies the Migrater.
- **Migrater** (`src/migrater/`): receives checkpoint notifications, copies local
    checkpoints to the PFS, and enforces rate limits with a token-bucket copy
    loop.
- **Orchestrator** (`src/orchestrator/`): receives worker heartbeats and assigns
    transfer actions/rates according to a scheduling policy.
- **Benchmarks** (`benchmarks/`, `scripts/`): compare direct PFS checkpointing
    against orchestrated checkpoint migration.

The communication protocol is defined in
[src/protocol/cluster.proto](src/protocol/cluster.proto) and generated with
[scripts/generate-protocols.sh](scripts/generate-protocols.sh).

## Repository structure

```sh
src/
  migrater/        # Migrater service, token bucket copy, progress tracking
  orchestrator/    # Scheduler policies and orchestrator server
  protocol/        # gRPC protocol definitions and generated bindings
  torch_ext/       # PyTorch checkpoint helper
  train/           # Minimal training workload using checkpoint migration
  utils/           # Shared helpers

benchmarks/
  e2e/             # End-to-end benchmark workers, collection and plotting
  token_bucket/    # Token bucket microbenchmark

scripts/           # SLURM submission and setup scripts
docs/              # HOWTOs, methodology, report and slides
tests/             # Policy and token bucket test scripts
```

## Requirements

On Deucalion, load Python with:

```bash
module load "Python/3.12.3-GCCcore-13.3.0"
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.12+ is expected. The main dependencies include PyTorch, Transformers,
gRPC, Pandas and Matplotlib.

Alternatively, use the provided setup script for Deucalion ARM partitions:

```bash
sbatch scripts/setup.sh
```

## Running the services manually

Open three terminals from the project root and activate the virtual environment in each one.

Start the orchestrator:

```bash
python3 -m src.orchestrator.server \
  --port 50052 \
  --policy uniform-fair-share \
  --pfs-bw 500MB
```

Start one migrater:

```bash
python3 -m src.migrater.server \
  --orchestrator-addr localhost \
  --orchestrator-port 50052
```

Run the training client:

```bash
python3 -m src.train.train \
  --checkpoint-pfs-dir /tmp/pfs \
  --checkpoint-local-dir /tmp/local \
  --total-steps 50 \
  --checkpoint-interval 10
```

## Benchmarking

### Baseline

The baseline simulates workers writing checkpoints directly to the PFS:

```bash
./scripts/submit-baseline.sh \
  --n-workers 4 \
  --n-checkpoints 3 \
  --pfs-dir "$PWD/checkpoints" \
  --results-dir "$PWD/results"
```

### Orchestrated

The orchestrated benchmark runs the Orchestrator plus worker Migraters:

```bash
./scripts/submit-orchestrated.sh \
  --policy age-priority \
  --pfs-bw 2GB \
  --n-workers 4 \
  --n-checkpoints 3 \
  --pfs-dir "$PWD/checkpoints" \
  --results-dir "$PWD/results"
```

Common policies can be swept with:

```bash
POLICIES=(uniform-fair-share active-fair-share age-priority epoch-priority)

for POLICY in "${POLICIES[@]}"; do
  ./scripts/submit-orchestrated.sh \
    --policy "$POLICY" \
    --pfs-bw 2GB \
    --n-workers 4 \
    --pfs-dir "$PWD/checkpoints" \
    --results-dir "$PWD/results"
done
```

### Collecting and plotting results

Aggregate benchmark outputs:

```bash
python3 -m benchmarks.e2e.collect \
  --results-dir "$PWD/results" \
  --output-csv "$PWD/results/summary.csv" \
  --detail-csv "$PWD/results/detail.csv"
```

Generate summary plots:

```bash
python3 -m benchmarks.e2e.plot \
  --results-dir "$PWD/results" \
  --output-dir "$PWD/results/plots"
```

Generate accumulated bandwidth plots from migrater logs:

```bash
python3 benchmarks/e2e/plot_accumulated_bw.py \
  --orch-bw 2GB \
  --output-dir "$PWD/results/plots" \
  --prefix age_priority \
  worker1.log worker2.log worker3.log worker4.log
```

### Token bucket microbenchmark

```bash
RESULTS_DIR="benchmarks/token_bucket/results"

python3 -m benchmarks.token_bucket.benchmark \
  --src checkpoints/random_small.bin \
  --local-dir /tmp/token_bucket_local \
  --pfs-dir checkpoints/token_bucket_pfs \
  --results-file "$RESULTS_DIR/results.jsonl" \
  --metrics-dir "$RESULTS_DIR/per_run" \
  --dry-run  # Remove --dry-run to actually run the benchmark
```

Plot token bucket results:

```bash
python3 -m benchmarks.token_bucket.plot \
  --results-file "$RESULTS_DIR/results.jsonl" \
  --output-dir "$RESULTS_DIR/plots"
```

### End-to-end

See [benchmarks/e2e/HOWTO.md](benchmarks/e2e/HOWTO.md) for detailed
instructions on running the end-to-end benchmark.

## Documentation

More detailed usage notes are in:

- [docs/report/report.pdf](docs/report/report.pdf)
- [docs/HOWTO.md](docs/HOWTO.md)
- [benchmarks/e2e/HOWTO.md](benchmarks/e2e/HOWTO.md)
- [docs/methodology.md](docs/methodology.md)
- [docs/implemented_policies.md](docs/implemented_policies.md)
