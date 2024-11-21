#! /bin/bash

set -xe

clang -O3 -Wall -Wextra main.c -o main -lm

# clang-format -i main.c
# clang-format -i mat.h
# clang-format -i nn.h

./main
