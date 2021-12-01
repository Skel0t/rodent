#!/bin/bash

BENCH_BINARY_PATH=../../build/bin/

# Benchmarks that should be run
BENCHMARKS=(denoise512 denoise1k sres512 sres1k)

# Platforms and Ierations per platform that should be run
PLATFORMS=(cuda cublas cublaslt cpu oneapi)
ITERATIONS=(600 600 600 100 100)
WARMUP=(100 100 100 10 10)
DEVICES=(0 0 0 0 0)     # UNUSED, just for output

# Set up output directory
mkdir -p benchmarks
CONSOLE_OUT=benchmarks/bench_results.txt
NULL_OUT=/dev/null

echo "Benchmarking time:" > $CONSOLE_OUT
echo "Benchmarking time:"

# Time and where it is spent on GPU and CPU
for ((i=0;i < ${#PLATFORMS[@]}; i++))
do
    echo "Benchmarking time on ${PLATFORMS[$i]}..."
    echo "--------------------------------------------------------------------------------" >> $CONSOLE_OUT
    echo "Benchmarking device ${DEVICES[$i]} on platform ${PLATFORMS[$i]} with ${ITERATIONS[$i]}/${WARMUP[$i]} iterations/warmup" >> $CONSOLE_OUT
    echo "--------------------------------------------------------------------------------" >> $CONSOLE_OUT

    for ((j=0;j < ${#BENCHMARKS[@]}; j++))
    do
        echo -ne "${BENCHMARKS[$j]}:\n" >> $CONSOLE_OUT
        $BENCH_BINARY_PATH/bench_convolution --bench ${BENCHMARKS[$j]} --iterations ${ITERATIONS[$i]} --warmup ${WARMUP[$i]} --backend ${PLATFORMS[$i]} >> $CONSOLE_OUT
        echo -ne "\n" >> $CONSOLE_OUT
        sleep 10
    done

    echo "Benchmarking time on ${PLATFORMS[$i]} done."
done

# Occupancy is defined as the ratio of active warps on an SM to the maximum number of active warps supported by the SM.
# https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm

echo "Benchmarking other metrics:" >> $CONSOLE_OUT
echo "Benchmarking other metrics:"

# Platforms and Ierations per platform that should be run
# Redefined as running nvprof reruns the kernel several times, and thus doing many iterations can take too long
PLATFORMS=(cuda cublas cublaslt cpu oneapi)
ITERATIONS=(50 50 50 50 50)
WARMUP=(50 50 50 10 10)
DEVICES=(0 0 0 0 0)

# GPU metrics with nvprof
for ((i=0;i < ${#PLATFORMS[@]} - 2; i++))
do
    echo "Benchmarking metrics on ${PLATFORMS[$i]}..."
    echo "--------------------------------------------------------------------------------" >> $CONSOLE_OUT
    echo "Benchmarking device ${DEVICES[$i]} on platform ${PLATFORMS[$i]} with ${ITERATIONS[$i]}/${WARMUP[$i]} iterations/warmup" >> $CONSOLE_OUT
    echo "--------------------------------------------------------------------------------" >> $CONSOLE_OUT
    sleep 15

    for ((j=0;j < ${#BENCHMARKS[@]}; j++))
    do
        LOGFILE=benchmarks/${PLATFORMS[$i]}_${BENCHMARKS[$j]}.txt
        echo -ne "${BENCHMARKS[$j]}:\n" > $NULL_OUT
        nvprof --csv --log-file $LOGFILE --metrics achieved_occupancy,tex_cache_hit_rate,l2_tex_hit_rate --replay-mode application $BENCH_BINARY_PATH/bench_convolution --bench ${BENCHMARKS[$j]} --iterations ${ITERATIONS[$i]} --warmup ${WARMUP[$i]} --backend ${PLATFORMS[$i]} > $NULL_OUT
        sleep 20
    done
    echo -ne "\n" > $NULL_OUT

    echo "Benchmarking metrics on ${PLATFORMS[$i]} done."
done

# CPU performance counters with perf
for ((i=3;i < ${#PLATFORMS[@]}; i++))
do
    echo "Benchmarking metrics on ${PLATFORMS[$i]}..."
    echo "--------------------------------------------------------------------------------" >> $CONSOLE_OUT
    echo "Benchmarking device ${DEVICES[$i]} on platform ${PLATFORMS[$i]} with ${ITERATIONS[$i]}/${WARMUP[$i]} iterations/warmup" >> $CONSOLE_OUT
    echo "--------------------------------------------------------------------------------" >> $CONSOLE_OUT

    for ((j=0;j < ${#BENCHMARKS[@]}; j++))
    do
        echo -ne "${BENCHMARKS[$j]}:\n" >> $CONSOLE_OUT
        $BENCH_BINARY_PATH/bench_convolution --bench ${BENCHMARKS[$j]} --iterations ${ITERATIONS[$i]} --warmup ${WARMUP[$i]} --backend ${PLATFORMS[$i]} > $NULL_OUT &
        sleep 10    # Wait till everything is read in and program is really running
        perf stat -t $! -d 2>> $CONSOLE_OUT
    done
    echo -ne "\n" >> $CONSOLE_OUT
    sleep 10

    echo "Benchmarking metrics on ${PLATFORMS[$i]} done."
done

echo "--------------------------------------------------------------------------------" >> $CONSOLE_OUT
