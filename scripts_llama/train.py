from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    LlamaForSequenceClassification,
)
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from sklearn.metrics import classification_report
import pdb
from pytorch_lightning.loggers import CSVLogger
import argparse
from pathlib import Path
from transformers import RobertaTokenizer
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
import json
from model import ClassificationModel
from data import ClassificationDataModule
from utils import EvalCallback, create_arg_parser


def read_subreddit_list(path):
    with open(path) as f:
        subreddits = [cvt_subreddit_to_token(e.lower()) for e in json.load(f)]

# python -m pdb scripts_llama/train.py --db_path ./religion_humanism.db --gpus 1 --bs 1 --acc_grad_batches 16 --out_dir out/models/subs/religion_humanism --do_eval 1 --num_classes 2 --eval_freq 3 --subreddits_list None --epochs 3
if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    print(args)

    db_path, out_dir = Path(args.db_path), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.pretrained_model_name
    post_title_col, post_text_col, comment_col = (
        args.post_title_col,
        args.post_text_col,
        args.comment_col,
    )
    label_col, post_id_col, subreddit_col, bs = (
        args.label_col,
        args.post_id_col,
        args.subreddit_col,
        args.bs,
    )
    primary_col = args.primary_col

    subreddits = (
        []
        if args.subreddits_list == "None"
        else read_subreddit_list(args.subreddits_list)
    )

    # tok = LlamaTokenizer.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    dm = ClassificationDataModule(
        db_path,
        tok,
        primary_col=primary_col,
        title_col=post_title_col,
        post_title_col=post_title_col,
        subreddit_col=subreddit_col,
        post_text_col=post_text_col,
        comment_col=comment_col,
        label_col=label_col,
        batch_size=bs,
        id_col=post_id_col,
        task=args.task,
        add_subreddit_tokens=True if len(subreddits) > 0 else False,
        instruction_finetune=bool(args.instruction_finetune),
        num_freq_subs=args.num_freq_subs,
        freq_subs_dropout=args.freq_subs_dropout,
    )

    if args.checkpoint_path:
        model = ClassificationModel.load_from_checkpoint(
            args.checkpoint_path,
            num_classes=args.num_classes,
            comment_only=False if args.comment_only == 0 else True,
            model_name=model_name,
            subreddit_tokens=subreddits,
            tok=tok,
        )
    else:
        model = ClassificationModel(
            model_name,
            args.num_classes,
            comment_only=False if args.comment_only == 0 else True,
            lr=args.lr,
            subreddit_tokens=subreddits,
            tok=tok,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir / "ckpt",
        save_top_k=1,
        monitor="val_loss",
        filename="{epoch}-{val_loss:.2f}",
    )
    cbs = [checkpoint_callback]
    if (args.do_eval != 0) and (args.task == "classification"):
        cbs.append(EvalCallback(dm, eval_freq=args.eval_freq))
    logger = CSVLogger(out_dir / "logs", name="persuasion_clf")
    torch.set_float32_matmul_precision('medium')

    # dm.setup(None)
    # train(model, dm.train_dataloader(), model.configure_optimizers(), epochs=1)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu" if args.gpus == 0 else "gpu",
        devices=1 if args.gpus == 0 else args.gpus,
        callbacks=cbs,
        accumulate_grad_batches=args.acc_grad_batches,
        logger=logger,
        precision='bf16-mixed',
        strategy='ddp_find_unused_parameters_true'
    )
    
    trainer.fit(model, dm)

    if args.do_eval != 0:
        model = model.cuda()
        dm.setup(None)

        print("Validation prediction")
        res = model.predict_dl(dm.val_dataloader(), "cuda" if args.gpus != 0 else "cpu")
        res.to_csv(out_dir / "val_preds.csv", index=False)
        print(classification_report(res["y"], res["prediction"]))
        print()

        print("Test prediction")
        res = model.predict_dl(
            dm.test_dataloader(), "cuda" if args.gpus != 0 else "cpu"
        )
        res.to_csv(out_dir / "test_preds.csv", index=False)
        print(classification_report(res["y"], res["prediction"]))
        print()
