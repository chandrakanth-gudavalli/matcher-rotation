#!/usr/bin/env bash
# Run sweeps sequentially on one GPU: queue.sh <gpu> <python> <matcher>...
gpu=$1; py=$2; shift 2
cd "$(dirname "$0")/.."
for m in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu "$py" bench/run_sweep.py "$m" > "outputs/sweep_$m.log" 2>&1
  echo "$m exit $?" >> "outputs/queue_gpu$gpu.status"
done
echo DONE >> "outputs/queue_gpu$gpu.status"
