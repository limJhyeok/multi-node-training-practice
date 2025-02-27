# Multi-Node Training Practice

<p align="left">
    &nbsp한국어&nbsp ｜ <a href="readme.md">English</a>&nbsp
</p>

## Overview
`multi-node-training-practice`는 PyTorch를 활용하여 다중 노드(서버) 분산 학습을 실습하는 저장소입니다.
기본적인 메시지 전달(`message_passing.py`)과 MNIST 학습(`train_mnist.py`)을 다중 노드 환경에서 실행하는 방법을 다룹니다.

## Requirements

- Python 3.10+
- PyTorch (with distributed support)
- CUDA (if using GPUs)

### Ping 확인

다중 노드 분산 학습을 위해서는 서로간의 통신이 보장이 되어야합니다.

ping을 통해 서로 통신을 할 수 있는 상태인지 먼저 확인해주세요.
```sh
ping <다른 노드의 Public IP>
# 예시: ping 8.8.8.8
```

### 각 노드간 SSH 연결
다중 노드 환경을 구축하려면 서로 데이터를 원활하게 주고받을 수 있도록 SSH 공개 키를 공유해야 합니다.

1. 각 노드에서 SSH Key를 생성합니다.
```sh
ssh-keygen -t rsa -b 4096
```
2. 다른 노드의 ~/.ssh/authorized_keys에 공개 키를 등록합니다.
```sh
ssh-copy-id user@<다른 노드의 Public IP>
# 예시: ssh-copy-id root@123.123.123.123
```
만약 `ssh-copy-id`를 통해서 공개 키를 등록하기 어렵다면 각 노드들의 `~/.ssh/authorized_keys`의 다른 노드의 public key(`~/.ssh/id_rsa.pub`)를 적어주면 됩니다

## Installation

```sh
pip install torch torchvision
```

## Running the Scripts

### 1. Message Passing Test
각 노드 간 통신이 정상적으로 이루어지는지 확인하는 테스트입니다.

0번 노드가 torch.tensor([42], dtype=torch.float32)를 다른 노드에 전달하고, 수신한 노드는 해당 값을 출력합니다.

사용방법:
```sh
bash ./scripts/msg_passing.sh
```
scripts 폴더 안에 msg_passing.sh를 실행시키면 아래와 같이 필요한 셋팅 값들을 물어보고 사용자가 입력을 하면 입력한 값을 토대로 `torchrun` 명령어를 실행시킵니다.

출력 예시:
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
- number of processes per node (**nproc_per_node**): 각 노드에서 실행할 프로세스 수
- total number of nodes (**nnods**): 총 노드 개수
- node rank (**node_rank**): 현재 노드의 랭크 (0부터 시작)
- master node address (**master_addr**): 마스터 노드의 IP 주소
- master port (**master_port**): 마스터 노드에서 사용할 포트 번호

### 2. Multi-Node MNIST Training
간단한 딥러닝 모델을 구축하여 다중 노드 환경에서 MNIST 데이터셋을 학습하는 예제입니다.

각 노드는 학습이 끝나면 snapshot.pt 파일을 생성합니다.

**실행방법:**
```sh
bash ./scripts/train_mnist.sh
```
**실행 예시:**
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

## 참고사항
- `message_passing.py`는 PyTorch의 `torch.distributed`를 이용하여 두 개의 노드 간 메시지 전달을 테스트합니다.
- `train_mnist.py`는 `DistributedDataParallel(DDP)`을 사용하여 다중 노드 환경에서 MNIST 학습을 수행합니다.
- `scripts/` 폴더에는 실행을 돕기 위한 Shell Script가 포함되어 있습니다.
- `NCCL` backend를 사용할 경우 GPU가 필요하며, `Gloo`는 CPU 기반에서도 동작합니다.

## References
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html?utm_source=chatgpt.com)


