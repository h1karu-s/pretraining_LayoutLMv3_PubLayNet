
from PIL import Image
import os 
from utils import utils
import torch
import sys
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer
import matplotlib.pyplot as plt
# 一個上の階層をpathに追加
sys.path.append('../')
from model import mlm_layoutLMv3 
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel
from transformers import RobertaModel
import random
from transformers import get_constant_schedule_with_warmup



#hyper parameter
num_epochs = 4
batch_size = 16
lr = 1e-4



if not torch.cuda.is_available():
    raise ValueError("GPU is not available.")



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_ids = list(range(torch.cuda.device_count()))





#vocab_list作成
tokenizer = LayoutLMv3Tokenizer("../model/tokenizer_vocab/vocab.json", "../model/tokenizer_vocab/merges.txt")
ids = range(tokenizer.vocab_size)
vocab = tokenizer.convert_ids_to_tokens(ids)

#LayoutLMv3 model config 読み取り
config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base")
model = mlm_layoutLMv3.LayoutLMv3ForMLM(config)

#roberta_pretrainingモデルを読み取り
Roberta_model = RobertaModel.from_pretrained("roberta-base")


## embedidng 層の重みをRobertaの重みで初期化
weight_size = model.state_dict()["model.embeddings.word_embeddings.weight"].shape
for i in range(weight_size[0]):
  model.state_dict()["model.embeddings.word_embeddings.weight"][i] = \
  Roberta_model.state_dict()["embeddings.word_embeddings.weight"][i]

#optimizer 
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.98))

#cross entropy
loss_fn = torch.nn.CrossEntropyLoss()


#modelをGPUへ
model = torch.nn.DataParallel(model, device_ids = device_ids)
model = model.to(f'cuda:{model.device_ids[0]}')

torch.save(model.module.state_dict(), '../model/init_layoutLMv3.params')       
        


image_file_path = "../../datasets/pdfs/images/ex"
image_names = os.listdir(image_file_path)

file_names = []
for n in image_names:
    name = os.path.splitext(n)[0]
    file_names.append(name)

file_path = "../../datasets/pdfs/train/"

words, bboxes = utils.extraction_text_from_pdf(file_path, file_names)

tokenizer = LayoutLMv3Tokenizer("../model/tokenizer_vocab/vocab.json", "../model/tokenizer_vocab/merges.txt")
enc = enc = tokenizer(text=words, boxes = bboxes, add_special_tokens=False)

pixel_values = []
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
for file_name in file_names[:100]:
    image = Image.open(f"../../datasets/pdfs/images/train/{file_name}.png")
    pixel_value = feature_extractor(image, return_tensors="pt")["pixel_values"]
    pixel_values.append(pixel_value)

#512に分割 + <s> , </s>, attention mask padding をする
tokens, bboxes, pixel, attention_masks = utils.subset_tokens_from_document(enc.input_ids, enc.bbox, pixel_values, vocab, max_len=512)



dataset = []
for i in range(len(tokens)):
    dataset.append({"input_ids": torch.tensor(tokens[i]),"bbox": torch.tensor(bboxes[i]), 
    "attention_mask": torch.tensor(attention_masks[i]), "pixel_values": pixel[i].squeeze()})

data_loader = torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle=True)


iter_per_epoch = len(data_loader)
num_training_steps = iter_per_epoch * num_epochs
num_warmup_steps = round((num_training_steps) * 0.048)
scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)




model.train()


loss_list = []
for epcoh in range(num_epochs):
    for inputs in data_loader:
        inputs["input_ids"], maske_positions, labels = utils.batch_create_span_mask(inputs["input_ids"], 0.3, 153, vocab, 3, random)
        inputs = {k: v.to(f'cuda:{model.device_ids[0]}') for k, v in inputs.items()}
        logits = model.forward(inputs)

        t = []
        for i in range(len(maske_positions)):
            if len(maske_positions[i]) == 0:
                continue
            t.append(logits[i][maske_positions[i]])
        logits = torch.cat(t)

        labels = torch.cat(labels)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

        print(loss.item())       
        loss_list.append(loss.item())
   
        

plt.xlabel("iter", fontsize=14)
plt.ylabel("loss", fontsize=14)        
plt.plot(range(len(loss_list)), loss_list)
plt.savefig("../model/params/loss.png")

# torch.save(model.module.state_dict(), '../model/params/pretrained_layoutLMv3.params')              

      
torch.save(
    {
        "epoch": num_epochs,
        "batch_size": batch_size,
        "loss_list": loss_list,
        "model_state_dict": model.module.to("cpu").state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "../model/params/pretreined_layoutLMv3.params",
)       


