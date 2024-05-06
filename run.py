import torch
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
from trainers.torch_trainer import TorchTrainer
from trainers.slang_trainer import SlangTrainer

def main(trainer_type):
    torch.cuda.empty_cache()
    if trainer_type == "slang":
        model = SlangTrainer()
    elif trainer_type == "torch":
        model = TorchTrainer()
    else:
        raise ValueError("Invalid trainer type. Please provide 'slang' or 'torch'.")
    model.train(iters=1000)
    model.render(saveimg=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and render a model.')
    parser.add_argument('trainer_type', choices=['slang', 'torch'],
                        help='Type of trainer to use: "slang" or "torch"')
    args = parser.parse_args()
    main(args.trainer_type)
    
    
    