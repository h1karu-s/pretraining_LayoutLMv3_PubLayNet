
from PIL import Image
import os 
from utils import utils
import torch
import sys
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer


print("start!", flush=True)


#vocab_list作成
tokenizer = LayoutLMv3Tokenizer("../model/tokenizer_vocab/vocab.json", "../model/tokenizer_vocab/merges.txt")
ids = range(tokenizer.vocab_size)
vocab = tokenizer.convert_ids_to_tokens(ids)

print("create vocab", flush=True)

print("get file_name", flush=True)

image_file_path = "../../datasets/pdfs/images/train"
image_names = os.listdir(image_file_path)

file_names = []
for n in image_names:
    name = os.path.splitext(n)[0]
    file_names.append(name)

print("start! extraction words and bboxes from pdf.", flush=True)

#image fileと対応するpdfから単語とbboxを抜き取る
file_path = "../../datasets/pdfs/train/"
words, bboxes = utils.extraction_text_from_pdf(file_path, file_names)

print("finish! extraction words and bboxes from pdf. ", flush=True)

print("start! tokenize" , flush=True)
#bpe tokenizer 

tokenizer = LayoutLMv3Tokenizer("../model/tokenizer_vocab/vocab.json", "../model/tokenizer_vocab/merges.txt")



enc_input_ids = []
enc_bboxes = []
for i in range(len(words)):
    try:
        enc = tokenizer(text=words[i], boxes = bboxes[i], add_special_tokens=False)
        enc_input_ids.append(enc["input_ids"])
        enc_bboxes.append(enc["bbox"])
    except:
        print(i, file_names.pop(i), flush=True)
    

print("finish! tokenzie ", flush=True)

print("start! extraciton pixel value from image ", flush=True)

#image からpixel_valueを抜き取る
pixel_values = []
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
for file_name in file_names:
    image = Image.open(f"../../datasets/pdfs/images/train/{file_name}.png")
    pixel_value = feature_extractor(image, return_tensors="pt")["pixel_values"]
    pixel_values.append(pixel_value)

print("finish! extraction pixelvalues", flush=True)

print("start! create subset tokens", flush=True)
#512に分割 + <s> , </s>, attention mask padding をする
tokens, bboxes, pixel, attention_masks = utils.subset_tokens_from_document(enc_input_ids, enc_bboxes, pixel_values, vocab, max_len=512)

print("finish! create subset tokens", flush=True)

print("start! create dataset", flush=True)
dataset = []
for i in range(len(tokens)):
    dataset.append({"input_ids": torch.tensor(tokens[i]),"bbox": torch.tensor(bboxes[i]), 
    "attention_mask": torch.tensor(attention_masks[i]), "pixel_values": pixel[i].squeeze()})

print("saving......! ../../datasets/pdfs/tensor_dataset.pt", flush=True)

torch.save(dataset, "../../datasets/pdfs/tensor_dataset.pt")

print("all process finished!!!!!", flush=True)