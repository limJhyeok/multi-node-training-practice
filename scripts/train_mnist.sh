#!/bin/bash

# 사용자 입력을 받아 변수 설정
read -p "Enter number of processes per node (nproc_per_node): " NPROC_PER_NODE
read -p "Enter total number of nodes (nnodes): " NNODES
read -p "Enter node rank (node_rank): " NODE_RANK
read -p "Enter master node address (master_addr): " MASTER_ADDR
read -p "Enter master port (master_port): " MASTER_PORT

# 변수 출력
echo "Running train_mnist.py with the following settings:"
echo "  - nproc_per_node: $NPROC_PER_NODE"
echo "  - nnodes: $NNODES"
echo "  - node_rank: $NODE_RANK"
echo "  - master_addr: $MASTER_ADDR"
echo "  - master_port: $MASTER_PORT"

# 실행
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_mnist.py
