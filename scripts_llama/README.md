## LLAMA scripts

This directory contains python scripts to train and evaluate LLAMA + LORA models

#### Training

Usage
```
sub=democrats
python train.py --db_path data/${sub}.db --gpus -1 --bs 4 --acc_grad_batches 32 --out_dir out/models/subs/lora/${sub}_llama --do_eval 1 --num_classes 2 --eval_freq 3 --subreddits_list None --epochs 40
```

#### Evaluation

Usage
```
python eval.py --db_path data/askaliberal.db --gpus 1 --bs 64 --out_dir influence_risk_analysis/out/models/subs/lora/askaliberal_llama --num_classes 2 --subreddits_list None --checkpoint_path influence_risk_analysis/out/models/subs/lora/askaliberal_llama/ckpt/askaliberal_epoch=7-val_loss=0.32.ckpt --out_file_suffix AskALiberal
```