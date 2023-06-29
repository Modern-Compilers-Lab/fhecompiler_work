#!/bin/bash

repeat=30

for ((i = 0; i < repeat; i++))
do
    ./clang-O0.build/main_opt  >> exec_times_opt.txt
done

for ((i = 0; i < repeat; i++))
do
    ./clang-O0.build/main_noopt  >> exec_times_noopt.txt
done