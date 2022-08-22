import torch
from torch import nn
from transformers import LayoutLMv3Model
from transformers import AutoConfig, AutoModel






## model head for masked language model 
class HeadForMLM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.transform_act_fn = nn.GELU()
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
    self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    self.decoder.bias = self.bias

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.transform_act_fn(hidden_states)
    hidden_states = self.LayerNorm(hidden_states)
    hidden_states = self.decoder(hidden_states)
    return hidden_states



## layoutLMv3 + head for MLM 
##config : config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base")
class LayoutLMv3ForMLM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.model = AutoModel.from_config(config)
    self.HeadForMLM = HeadForMLM(config)
  
  def forward(self, input):
    if input["input_ids"].shape[1] > 512:
      print("over lengths")
    outputs = self.model(**input)
    outputs = self.HeadForMLM(outputs[0])
    logits = outputs[:, 0:512]
    return logits