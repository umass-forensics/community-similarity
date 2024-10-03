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
from utils import cvt_subreddit_to_token
from collections import defaultdict
import random

sub_map = {
    "catholicism": 0,
    "christianity": 1,
    "exmormon": 2,
    "islam": 3,
    "atheism": 4,
    "buddhism": 5,
    "judaism": 6,
    "truechristian": 7,
    "exmuslim": 8,
    "mormon": 9,
    "orthodoxchristianity": 10,
    "religion": 11,
    "openchristian": 12,
    "hinduism": 13,
    "humanism": 14,
    "progressive_islam": 15,
    "islamicstudies": 16,
}

INST_FT_TEMPLATE = """<s>[INST] <<SYS>>
You are an AI trained on data from Reddit. You will be given a Post Title, Post Description, and a Comment from the subreddit {0}. Your job is to predict whether the comment will get upvoted or downvoted by the members of {0}.
<</SYS>>
Post Title: {1}

Comment: {3}

Post Description: {2}
[/INST]"""


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        db_path,
        tokenizer,
        primary_col="id",
        title_col="op_title",
        post_text_col="op_text",
        post_title_col="p_title",
        comment_col="comment",
        label_col: str = "delta",
        id_col="post_id",
        batch_size=32,
        task="classification",
        subreddit_col="p_subreddit",
        add_subreddit_tokens=False,
        instruction_finetune=False,
        num_freq_subs=5,
        freq_subs_dropout=0.3,
    ):
        super().__init__()
        self.db_path = db_path
        self.primary_col = primary_col
        self.bs = batch_size
        self.tok = tokenizer
        self.title_col, self.label_col = title_col, label_col
        self.post_text_col, self.comment_col = post_text_col, comment_col
        self.subreddit_col = subreddit_col
        self.id_col = id_col
        self.task = task
        self.add_subreddit_tokens = add_subreddit_tokens
        self.instruction_finetune = instruction_finetune
        self.num_freq_subs = num_freq_subs
        self.freq_subs_dropout = freq_subs_dropout

    def setup(self, stage: str):
        self.train = DBDataset(
            "sqlite:///" + str(self.db_path),
            "train",
            self.primary_col,
            title_col=self.title_col,
            post_text_col=self.post_text_col,
            comment_col=self.comment_col,
            label_col=self.label_col,
            id_col=self.id_col,
            subreddit_col=self.subreddit_col,
            add_subreddit_token=self.add_subreddit_tokens,
            instruction_finetune=self.instruction_finetune,
            num_freq_subs=self.num_freq_subs,
            freq_subs_dropout=self.freq_subs_dropout,
        )
        self.val = DBDataset(
            "sqlite:///" + str(self.db_path),
            "val",
            self.primary_col,
            title_col=self.title_col,
            post_text_col=self.post_text_col,
            comment_col=self.comment_col,
            label_col=self.label_col,
            id_col=self.id_col,
            subreddit_col=self.subreddit_col,
            add_subreddit_token=self.add_subreddit_tokens,
            instruction_finetune=self.instruction_finetune,
            num_freq_subs=self.num_freq_subs,
            freq_subs_dropout=0.0,
        )
        self.test = DBDataset(
            "sqlite:///" + str(self.db_path),
            "test",
            self.primary_col,
            title_col=self.title_col,
            post_text_col=self.post_text_col,
            comment_col=self.comment_col,
            label_col=self.label_col,
            id_col=self.id_col,
            subreddit_col=self.subreddit_col,
            add_subreddit_token=self.add_subreddit_tokens,
            instruction_finetune=self.instruction_finetune,
            num_freq_subs=self.num_freq_subs,
            freq_subs_dropout=0.0,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.bs,
            collate_fn=partial(self.train.collate, tok=self.tok),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.bs,
            collate_fn=partial(self.val.collate, tok=self.tok),
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.bs,
            collate_fn=partial(self.test.collate, tok=self.tok),
            shuffle=True,
        )


class DBDataset(Dataset):
    def __init__(
        self,
        connection_string,
        table_name,
        primary_col_name,
        title_col: str = "p_title",
        post_text_col="p_selftext",
        comment_col="c_body",
        label_col: str = "c_polarity",
        id_col="link_id",
        subreddit_col="p_subreddit",
        author_col="c_author",
        year_col="year",
        month_col="month",
        add_subreddit_token=False,
        max_length=610,
        instruction_finetune=False,
        num_freq_subs=5,
        freq_subs_dropout=0.3,
    ):
        self.engine = create_engine(connection_string)
        self.table_name = table_name
        self.primary_col_name = primary_col_name
        metadata = MetaData(bind=self.engine)
        self.table = Table(table_name, metadata, autoload=True)

        self.connection = self.engine.connect()
        Session = sessionmaker(self.engine)
        session = Session()
        self.len = session.query(self.table).count()
        session.close()
        self.title_col, self.label_col = title_col, label_col
        self.post_text_col, self.comment_col = post_text_col, comment_col
        self.id_col, self.subreddit_col = id_col, subreddit_col
        self.add_subreddit_token = add_subreddit_token
        self.max_length = max_length
        self.instruction_finetune = instruction_finetune
        self.author_col = author_col
        self.year_col, self.month_col = year_col, month_col
        self.num_freq_subs = num_freq_subs
        self.freq_subs_dropout = freq_subs_dropout  # set subs = [] in this ratio

    def __getitem__(self, idx):
        query = self.table.select().where(self.table.c[self.primary_col_name] == idx)
        result = self.connection.execute(query).fetchone()
        if result:
            # comment is in the middle so that it doesn't get truncated because of a large self text
            data = self.form_post_str(result)
            return (
                data,
                result[self.id_col],
                result[self.label_col],
            )
        else:
            raise IndexError()

    def form_post_str(self, result):
        keys = result.keys()
        author = result[self.author_col] if self.author_col in keys else ''
        sub = result[self.subreddit_col] if self.subreddit_col in keys else self.subreddit_col
        title = result[self.title_col] if self.title_col in keys else result[self.comment_col]
        text = result[self.post_text_col] if self.post_text_col in keys else ''
        comment = result[self.comment_col]
        year = str(result[self.year_col])
        month = str(result[self.month_col])
        if self.instruction_finetune:
            template = ""
        else:
            template = "<sub>{0}</sub><post>{1}</post><comment>{2}</comment><text>{3}</text>"
        
        return template.format(sub, title, comment, text)

    def __len__(self):
        return self.len

    def close(self):
        self.connection.close()

    def collate(self, batch, tok):
        y = torch.tensor(
            [(sub_map[e[-1].lower()] if type(e[-1]) == str else e[-1]) for e in batch],
            dtype=torch.long,
        )
        ids = [e[-2] for e in batch]
        batch_data = [e[0] for e in batch]

        X = tok(
            batch_data,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        res = {
            "X":X,
            "y": y,
            "id": ids,
        }

        return res


class SingleRecordDataset(Dataset):
    def __init__(
        self,
        subreddit,
        post_title,
        self_text,
        comment,
        max_length=610
    ):
        self.subreddit, self.post_title = subreddit, post_title
        self.self_text, self.comment = self_text, comment
        self.max_length = max_length
        

    def __getitem__(self, idx):
        template = "<sub>{0}</sub><post>{1}</post><comment>{2}</comment><text>{3}</text>"
        return (template.format(self.subreddit, self.post_title, self.comment, self.self_text), 1, 0)

    def __len__(self):
        return 1

    def collate(self, batch, tok):
        y = torch.tensor(
            [(sub_map[e[-1].lower()] if type(e[-1]) == str else e[-1]) for e in batch],
            dtype=torch.long,
        )
        ids = [e[-2] for e in batch]
        batch_data = [e[0] for e in batch]

        X = tok(
            batch_data,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        res = {
            "X":X,
            "y": y,
            "id": ids,
        }

        return res

    
    
    