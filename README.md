# Repo  


```bash
InstantNeRF/
├── data/                      # Datasets
│   └── dataset.npz/           
├── models/                    # MLP models
│   ├── slang_mlp    
│   │   └── ...    
│   └── troch_mlp
│       └──mlp.py       
├── result/                    # Directory to store rendered imags
├── trainers/                  # Training Class
│   ├── slang_trainer.py       
│   └── torch_trainer.py    
├── utils/                     # some helper functions
│   ├── data_loader.py               
│   ├── encoder.py            
│   ├── rendering_utils.py    
│   └── save_image.py
└── run.py                      # main function
```

# TODO
## David
[ ] implement hash encoding
[ ] add multiple images 