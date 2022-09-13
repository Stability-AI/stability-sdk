#!/bin/bash

# 1. ensure api-interfaces is populated w latest artifacts
#git pull --recurse-submodules
git submodule update --remote --init --recursive

# 2. overwrite local artifacts for this language
SRC=./api-interfaces/gooseai/generation
TGT=./src/stability-sdk/gooseai/generation
cp $SRC/generation_pb2_grpc.py $TGT/generation_pb2_grpc.py
cp $SRC/generation_pb2.py $TGT/generation_pb2.py

### for backwards compatibility ###

# 3. update generation.proto

# 4. update js/ts stubs