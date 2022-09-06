
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from torch.optim import AdamW

class LayoutLMv3ForMLM(pl.LightningModule):
    def __init__(self, hparams):
        super(LayoutLMv3ForMLM, self).__init__()
        self.save_hyperparameters(hparams)

        self.config = AutoConfig.from_pretrained(self.hparams.model_name)
        self.model = AutoModel.from_config(self.config)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.decoder = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False)
        self.bias = nn.Parameter(torch.zeros(self.config.vocab_size))
        self.decoder.bias = self.bias
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        outputs = self.model(**batch)
        outputs = self.dense(outputs[0])
        outputs = self.transform_act_fn(outputs)
        outputs = self.LayerNorm(outputs)
        outputs = self.decoder(outputs)
        logits = outputs[:, 0:512]
        return logits
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)
    
    def training_step(self, batch, _):
        inputs = {k: batch[k] for k in ["input_ids", "bbox", "pixel_values", "attention_mask"]}
        logits = self.forward(inputs)
        loss = self._cal_loss(logits, batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss.detach(), on_step=True, prog_bar=True)

    def validation_step(self, batch, _):
        inputs = {k: batch[k] for k in ["input_ids", "bbox", "pixel_values", "attention_mask"]}
        logits = self.forward(inputs)
        loss = self._cal_loss(logits, batch)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss.detach(),on_epoch=True,  prog_bar=True)

    def predict_step(self, batch, _):
        inputs = {k: batch[k] for k in ["input_ids", "bbox", "pixel_values", "attention_mask"]}
        return self.forward(batch)
    
    def _cal_loss(self, logits, batch):
        t = []
        for i in range(len(batch["mask_position"])):
            if len(batch["mask_position"][i]) == 0:
                continue
            t.append(logits[i][batch["mask_position"][i]])
        logits = torch.cat(t)
        labels = torch.cat(batch["mask_label"])
        loss = self.criterion(logits, labels)
        return loss        
