import torch
import torch.distributed as dist
import os

def init_process(rank, world_size):
    """Initialize distributed process"""
    dist.init_process_group(backend = "gloo", rank=rank, world_size=world_size)

def message_passing(rank, world_size):
    """Simple message passing"""
    tensor = torch.zeros(1)

    if rank == 0:
        msg = torch.tensor([42], dtype = torch.float32) 
        for i in range(1, world_size):
            dist.send(msg, dst=i) 
        print(f"Rank {rank} sent message: {msg.item()}")
    else:
        # 다른 rank들은 메시지를 수신
        dist.recv(tensor, src=0)
        print(f"Rank {rank} received message: {tensor.item()}")

if __name__ == "__main__":
    rank = int(os.environ["RANK"])  
    world_size = int(os.environ["WORLD_SIZE"])  

    init_process(rank, world_size)
    message_passing(rank, world_size)

    dist.destroy_process_group()  
