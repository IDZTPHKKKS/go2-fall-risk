#!/bin/zsh
# Finish the 600-episode pilot (seeds 100000-100599) with 2 low-priority workers.
# Run as: caffeinate -i sim/run_pilot.sh (keeps the Mac awake; lid must stay open).
# New episodes go to new files, so the finished pilot_0..2 files are never reopened.
cd "$(dirname "$0")"
OUT=../data/pilot
lane() {  # file, then (start count) pairs, run one after another
  local out=$1; shift
  while (( $# )); do
    taskpolicy -b python3 generate.py --start $1 --count $2 --out $OUT/$out.h5 >> $OUT/$out.log 2>&1
    shift 2
  done
}
lane pilot_3 100037 163 100439 61 &
lane pilot_4 100239 161 100500 100 &
wait
echo "done $(date)" >> $OUT/pilot_3.log
