from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
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
from data import DBDataset, SingleRecordDataset
from utils import EvalCallback, create_arg_parser

if __name__ == "__main__":
    parser = create_arg_parser()
    parser.add_argument("--table_name", type=str, default="test")
    parser.add_argument("--out_file_suffix", type=str, default="")
    parser.add_argument("--single_record_json", type=str, default=None)
    args = parser.parse_args()
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

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    
    model = ClassificationModel.load_from_checkpoint(
        args.checkpoint_path,
        num_classes=args.num_classes,
        comment_only=False if args.comment_only == 0 else True,
        model_name=model_name,
        subreddit_tokens=subreddits,
        tok=tok,
        map_location="cuda" if args.gpus != 0 else "cpu"
    )
    model = model.cuda() if args.gpus != 0 else model.cpu()

    print("Test prediction")
    if args.single_record_json:
        with open(args.single_record_json) as f:
            srj = json.load(f)
        ds = SingleRecordDataset(srj['subreddit'], srj['post_title'], srj['self_text'], srj['comment'])
    else:
        ds = DBDataset(
            "sqlite:///" + str(db_path),
            args.table_name,
            primary_col,
            title_col=post_title_col,
            post_text_col=post_text_col,
            comment_col=comment_col,
            label_col=label_col,
            id_col=post_id_col,
            subreddit_col=subreddit_col,
            add_subreddit_token=True if len(subreddits) > 0 else False,
            instruction_finetune=bool(args.instruction_finetune),
            num_freq_subs=args.num_freq_subs,
            freq_subs_dropout=args.freq_subs_dropout,
        )
    dl = DataLoader(
        ds,
        batch_size=args.bs,
        collate_fn=partial(ds.collate, tok=tok),
        shuffle=False,
    )
    res = model.predict_dl(dl, "cuda" if args.gpus != 0 else "cpu")
    suffix = args.out_file_suffix
    res.to_csv(out_dir / f"test_preds_{suffix}.csv", index=False)
    print(classification_report(res["y"], res["prediction"]))
    print()
