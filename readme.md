# Multi-Node Training Practice

<p align="left">
    <a href="readme.ko.md">한국어</a>&nbsp ｜ &nbspEnglish&nbsp
</p>

## Overview
`multi-node-training-practice` is a repository for practicing multi-node (server) distributed training using PyTorch.
This repository covers basic message passing (`message_passing.py`) and MNIST training (`train_mnist.py`) in a multi-node environment.

## Requirements

- Python 3.10+
- PyTorch (with distributed support)
- CUDA (Required for GPU usage)

### Checking Network Connectivity

Ensuring seamless communication between nodes is essential for multi-node distributed training.

Verify connectivity using the `ping` command:
```sh
ping <public IP of another node>
# Example: ping 8.8.8.8
```

### Establishing SSH Connection Between Nodes
To set up a multi-node training environment, each node must be able to exchange data seamlessly by sharing SSH public keys.

1. Generate an SSH key on each node:
```sh
ssh-keygen -t rsa -b 4096
```

2. Register the public key in the `~/.ssh/authorized_keys` file of the other node:
```sh
ssh-copy-id user@<public IP of another node>
# Example: ssh-copy-id root@123.123.123.123
```

If `ssh-copy-id` is unavailable, manually add the public key (`~/.ssh/id_rsa.pub`) of the other node to the `~/.ssh/authorized_keys` file.

## Running the Scripts

### 1. Message Passing Test
This test verifies that communication between nodes is functioning properly.

Node 0 sends `torch.tensor([42], dtype=torch.float32)` to the other nodes, which then receive and print the message.

#### Usage:
```sh
bash ./scripts/msg_pass.sh
```

Executing `msg_pass.sh` inside the `scripts` folder will prompt the user for necessary configuration values and then run `torchrun` with the provided inputs.

#### Example Output:
```
Enter number of processes per node (nproc_per_node): 1
Enter total number of nodes (nnodes): 2
Enter node rank (node_rank): 0
Enter master node address (master_addr): 123.123.123.123
Enter master port (master_port): 1234
Running message_passing.py with the following settings:
  - nproc_per_node: 1
  - nnodes: 2
  - node_rank: 0
  - master_addr: 123.123.123.123
  - master_port: 1234
```

Input values:
- **nproc_per_node**: Number of processes per node
- **nnodes**: Total number of nodes
- **node_rank**: Rank of the current node (starting from 0)
- **master_addr**: IP address of the master node
- **master_port**: Port number used by the master node

### 2. Multi-Node MNIST Training
This example builds a simple deep learning model and trains a model on the MNIST dataset in a multi-node environment.
Each node generates a `snapshot.pt` file upon training completion.

#### Usage:
```sh
bash ./scripts/train_mnist.sh
```

#### Example Output:
```
Enter number of processes per node (nproc_per_node): 1
Enter total number of nodes (nnodes): 2
Enter node rank (node_rank): 0
Enter master node address (master_addr): 123.123.123.123
Enter master port (master_port): 1234
Running train_mnist.py with the following settings:
  - nproc_per_node: 1
  - nnodes: 2
  - node_rank: 0
  - master_addr: 123.123.123.123
  - master_port: 1234
```

## Notes
- `message_passing.py` tests message passing between two nodes using PyTorch's `torch.distributed`.
- `train_mnist.py` trains a model on the MNIST dataset in a multi-node environment using `DistributedDataParallel (DDP)`.
- The `scripts/` folder contains shell scripts to facilitate execution.
- The `NCCL` backend requires GPUs, while `Gloo` works on CPUs.

## References
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html?utm_source=chatgpt.com)
