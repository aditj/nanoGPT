
source venv/bin/activate
torchrun --standalone --nproc_per_node=8 train.py config/train_12_full_24.py
torchrun --standalone --nproc_per_node=8 train.py config/train_4_full_12_half_16.py
torchrun --standalone --nproc_per_node=8 train.py config/train_12_half_12.py
torchrun --standalone --nproc_per_node=8 train.py config/train_9_half_3_none_9.py
torchrun --standalone --nproc_per_node=8 train.py config/train_6_half_6_none_6.py
