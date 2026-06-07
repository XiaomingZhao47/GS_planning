#!/bin/bash
# Multi-seed sweep launcher per docs/adr/0001-multi-seed-evaluation-design.md.
# 5 seeds (0-4) x 5 methods (uniform, random, volumetric, q3_iter, q3_refined).
# Each cell: plan + train + render + metrics. Total est. ~3 hr on the 5090.

set -uo pipefail
ROOT=/home/xiaoming/GS_planning_handoff
DATA=$ROOT/data/wreck_exp1
PY=/home/xiaoming/miniconda3/envs/gs_planning/bin/python
LOG=$DATA/multiseed_sweep.log

cd $ROOT

METHODS="uniform random volumetric q3_iter q3_refined"
SEEDS="0 1 2 3 4"

echo "=== multi-seed sweep start $(date -Is) ===" | tee -a $LOG
echo "seeds: $SEEDS  methods: $METHODS" | tee -a $LOG

for s in $SEEDS; do
  echo "" | tee -a $LOG
  echo "============ seed=$s plan+train start $(date -Is) ============" | tee -a $LOG
  if $PY -m gazebo_sortie.exp1_run --seed $s --methods $METHODS --budget 18 >> $LOG 2>&1; then
    echo "seed=$s plan+train OK" | tee -a $LOG
  else
    echo "seed=$s plan+train FAILED (rc=$?)" | tee -a $LOG
    continue
  fi
  echo "============ seed=$s eval start $(date -Is) ============" | tee -a $LOG
  if $PY -m gazebo_sortie.exp_multimetric --seed $s >> $LOG 2>&1; then
    echo "seed=$s eval OK" | tee -a $LOG
  else
    echo "seed=$s eval FAILED (rc=$?)" | tee -a $LOG
  fi
done

echo "" | tee -a $LOG
echo "=== aggregating $(date -Is) ===" | tee -a $LOG
if $PY -m gazebo_sortie.aggregate_multiseed --expected-seeds 5 >> $LOG 2>&1; then
  echo "aggregate OK" | tee -a $LOG
else
  echo "aggregate FAILED (rc=$?)" | tee -a $LOG
fi

echo "=== sweep complete $(date -Is) ===" | tee -a $LOG
