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


class EvalCallback(pl.callbacks.Callback):
    def __init__(self, data_module, eval_freq=3):
        self.dm = data_module
        self.epochs = 0
        self.eval_freq = eval_freq

    def on_train_epoch_end(self, trainer, model):
        if self.epochs % self.eval_freq == 0:
            # print('Training prediction')
            # res = model.predict_dl(self.dm.train_dataloader(), 'cuda')
            # print(classification_report(res['y'], res['prediction']))
            # print()

            print("Validation prediction")
            res = model.predict_dl(self.dm.val_dataloader(), "cuda")
            print(classification_report(res["y"], res["prediction"]))
            print()

            print("Test prediction")
            res = model.predict_dl(self.dm.test_dataloader(), "cuda")
            print(classification_report(res["y"], res["prediction"]))
            print()
        self.epochs += 1


def cvt_subreddit_to_token(s):
    return f"<r/{s}>".lower()


def add_token(model, tokenizer, token):
    tokenizer.add_tokens(token)
    model.resize_token_embeddings(len(tokenizer))


def add_tokens(model, tokenizer, tokens):
    [add_token(model, tokenizer, token) for token in tokens]


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./out/")
    parser.add_argument("--pretrained_model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--primary_col", type=str, default="id")
    parser.add_argument("--post_title_col", type=str, default="p_title")
    parser.add_argument("--post_text_col", type=str, default="p_selftext")
    parser.add_argument("--comment_col", type=str, default="c_body")
    parser.add_argument("--label_col", type=str, default="c_polarity")
    parser.add_argument("--subreddit_col", type=str, default="p_subreddit")
    parser.add_argument("--post_id_col", type=str, default="id")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--acc_grad_batches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument(
        "--do_eval", type=int, default=0
    )  # 0 for no eval. Anything other than 0 to turn run eval.
    parser.add_argument(
        "--comment_only", type=int, default=0
    )  # 0 if post title, text and comment should be used.
    # Anything other than 0 to just use the commment.
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--subreddits_list", type=str, default="subreddit_list.json"
    )  # if subreddit tokens should not be added, set this to the string "None"
    parser.add_argument("--eval_freq", type=int, default=3)
    parser.add_argument("--instruction_finetune", type=int, default=0)  # 0 for False. 1 for True
    parser.add_argument("--num_freq_subs", type=int, default=5)
    parser.add_argument("--freq_subs_dropout", type=float, default=0.3)
    return parser
