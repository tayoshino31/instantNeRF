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
├── result/                    
├── trainers/                  
│   ├── slang_trainer.py       # train with slang_mlp
│   ├── torch_trainer.py       # train with torch_mlp
│   ├── slanghash_trainer.py   # train with slang_mlp with hash encoding 
│   └── torchhash_trainer.py   # train with torch_mlp with hash encoding 
├── utils/                     
│   ├── hashencoder            
│   ├── data_loader.py         # load & pre-process data
│   ├── encoder.py             
│   ├── rendering_utils.py    
│   └── save_image.py
└── run.py                      # main calls one of the trainer
```

# How to run

```bash
python run.py torch             # use PyTorch MLP
python run.py slang             # use Slang MLP
python run.py torchhash         # use PyTorch MLP with hash encoding 
python run.py slanghash         # use Slang MLP with hash encoding 
```

# TODO
## David
[ ] implement hash encoding
[ ] add multiple images 
