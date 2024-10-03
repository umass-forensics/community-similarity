from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
)
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from peft import LoraConfig, TaskType, get_peft_model
from collections import OrderedDict

def freeze_model_params(model):
    for name, param in model.named_parameters():
        if "model" in name: # and (('model.layers.31' in name) == False)
            param.requires_grad = False
    return model


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_classes,
        tok,
        lr=1e-5,
        comment_only=False,
        subreddit_tokens=[],
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.lr = lr
        self.comment_only = comment_only
        self.tok = tok
        self.subreddit_tokens = subreddit_tokens
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_classes, torch_dtype='auto',
            cache_dir='/work/anon/influence_risk_analysis/hf_cache'
        )
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, 
            r=4, lora_alpha=32, lora_dropout=0.1, #  layers_to_transform=list(range(28, 32))
        )
        self.model = get_peft_model(self.model, peft_config)
        print(self.model.print_trainable_parameters())
        # self.model = freeze_model_params(self.model)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = self.lr
        self.comment_only = self.comment_only
        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"\n\nTotal parameters = {pytorch_total_params}\n\n")
        # add_tokens(self.model, tok, subreddit_tokens)

    def forward(self, batch):
        out = self.model(**batch["X"]).logits
        return out

    def predict_proba(self, batch):
        out = self.forward(batch)
        out = torch.nn.functional.softmax(out, dim=1)
        return out

    def predict(self, batch):
        out = self.predict_proba(batch)
        out = out.max(dim=1)
        return out.indices, out.values

    def predict_dl(self, dl, device):
        with torch.no_grad():
            res = []
            for batch in tqdm(dl):
                for col in ["X"]:
                    batch[col] = {k: v.to(device) for k, v in batch[col].items()}
                batch["prediction"], batch["prob"] = self.predict(batch)
                cur_res = []
                for cur_id, pred, prob, y in zip(
                    batch["id"], batch["prediction"], batch["prob"], batch["y"]
                ):
                    cur_res.append(
                        {
                            "y": y.cpu().item(),
                            "prediction": pred.cpu().item(),
                            "prob": prob.cpu().item(),
                            "id": cur_id,
                        }
                    )
                res.extend(cur_res)
        return pd.DataFrame(res)

    def forward_and_calc_loss(self, batch):
        y = batch["y"]
        out = self.forward(batch)
        loss = self.loss_fn(out, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_and_calc_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_and_calc_loss(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-4)
        return optimizer


class EvalCallback(pl.callbacks.Callback):
    def __init__(self, data_module, eval_freq=3):
        self.dm = data_module
        self.epochs = 0
        self.eval_freq = eval_freq

    def on_train_epoch_end(self, trainer, model):
        if self.epochs % self.eval_freq == 0:
            print("Training prediction")
            res = model.predict_dl(self.dm.train_dataloader(), "cuda")
            print(classification_report(res["y"], res["prediction"]))
            print()

            print("Validation prediction")
            res = model.predict_dl(self.dm.val_dataloader(), "cuda")
            print(classification_report(res["y"], res["prediction"]))
            print()

            print("Test prediction")
            res = model.predict_dl(self.dm.test_dataloader(), "cuda")
            print(classification_report(res["y"], res["prediction"]))
            print()
        self.epochs += 1