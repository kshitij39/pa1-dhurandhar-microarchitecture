#!/usr/bin/env bash
# mpki.sh — run a command under `perf stat` and print the L1 D-cache MPKI
# (misses per kilo-instruction) directly, instead of doing the division by hand.
#
# Usage:
#   sudo ./mpki.sh <command> [args...]
#
# Example:
#   sudo ./mpki.sh ./bin/matmul optimized 1024 1024 1024
#
# On hybrid CPUs (P-core/E-core, e.g. Alder Lake) `perf` reports separate
# cpu_core/... and cpu_atom/... counters for the same event. This script
# prefers the cpu_core counters (the ones with real coverage %), and falls
# back to plain event names on non-hybrid machines.

set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 1
fi

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

perf stat -e instructions,L1-dcache-load-misses "$@" 2>"$LOG"

# Show perf's normal report unchanged, exactly as if you'd run it directly.
cat "$LOG" >&2

# $1: preferred (hybrid, cpu_core) grep pattern
# $2: fallback grep pattern (non-hybrid systems)
extract() {
    local line
    line=$(grep -E "$1" "$LOG" | head -n1)
    if [ -z "$line" ]; then
        line=$(grep -E "$2" "$LOG" | grep -Ev 'cpu_atom' | head -n1)
    fi
    [ -z "$line" ] && return 1
    [[ "$line" == *"<not supported>"* ]] && return 1
    # leading numeric token, stripping thousands-separator commas
    echo "$line" | grep -oE '^[[:space:]]*[0-9,]+' | tr -d ' ,'
}

instr=$(extract 'cpu_core/instructions/'              '(^|[[:space:]])instructions([[:space:]]|$)')
miss=$(extract  'cpu_core/L1-dcache-load-misses/'      'L1-dcache-load-misses')

if [ -z "${instr:-}" ] || [ -z "${miss:-}" ]; then
    echo "mpki.sh: could not find both 'instructions' and 'L1-dcache-load-misses' counts in perf output" >&2
    exit 1
fi

awk -v i="$instr" -v m="$miss" '
    BEGIN {
        printf "\n--- MPKI ---\n"
        printf "instructions          : %s\n", i
        printf "L1-dcache-load-misses : %s\n", m
        printf "L1 D-cache MPKI       : %.4f\n", (m / i) * 1000
    }
'
