import torch
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
from trainers.torch_trainer import TorchTrainer
from trainers.slang_trainer import SlangTrainer
from trainers.torchhash_trainer import TorchHashTrainer
from trainers.slanghash_trainer import SlangHashTrainer
import torch.cuda.profiler as profiler

def main(trainer_type):
    torch.cuda.empty_cache()
    if trainer_type == "slang":
        model = SlangTrainer()
    elif trainer_type == "torch":
        model = TorchTrainer()
    elif trainer_type == "slanghash":
        model = SlangHashTrainer()
    elif trainer_type == "torchhash":
        model = TorchHashTrainer()
    else:
        raise ValueError("Invalid trainer type. Please provide 'slang' or 'torch'.")
    model.train(iters=50)
    model.render(saveimg=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and render a model.')
    parser.add_argument('trainer_type', choices=['slang', 'torch', 'slanghash', 'torchhash'],
                        help='Type of trainer to use: "slang", "torch","slanghash","torchhash"')
    args = parser.parse_args()
    main(args.trainer_type)
    
    
    