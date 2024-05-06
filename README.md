# Repo  


```bash
InstantNeRF/
├── data/                      # Datasets
│   └── dataset.npz/           
├── models/                    # MLP models
│   ├── slang_mlp    
│   │   └── ...                # slang inline-MLP files
│   └── troch_mlp
│       └──mlp.py              # Pytorch MLP
├── result/                    # Directory to store rendered images
├── trainers/                  
│   ├── slang_trainer.py       # train with slang_mlp
│   └── torch_trainer.py       # train with torch_mlp
├── utils/                     # some helper functions
│   ├── data_loader.py         # load & pre-process data
│   ├── encoder.py             
│   ├── rendering_utils.py    
│   └── save_image.py
└── run.py                      # main calls either slang_trainer or torch_trainer
```

# How to run

```bash
python run.py torch             # use PyTorch MLP
python run.py slang             # use Slang MLP
```

# TODO
## David
[ ] implement hash encoding
[ ] add multiple images 
