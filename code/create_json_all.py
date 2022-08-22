
from PIL import Image
import os 
from utils import utils
import torch
import sys
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer
import time
import json
from json import JSONEncoder
import numpy

start = time.time()
print("start!", flush=True)


#vocab_list作成
tokenizer = LayoutLMv3Tokenizer("../model/tokenizer_vocab/vocab.json", "../model/tokenizer_vocab/merges.txt")
ids = range(tokenizer.vocab_size)
vocab = tokenizer.convert_ids_to_tokens(ids)

print("create vocab", time.time() - start, flush=True)

print("get file_name", flush=True)

image_file_path = "../../datasets/pdfs/images/train"
image_names = os.listdir(image_file_path)

image_names = image_names

file_names = []
for n in image_names:
    name = os.path.splitext(n)[0]
    file_names.append(name)

print("start! extraction words and bboxes from pdf.",  time.time() - start, flush=True)

#image fileと対応するpdfから単語とbboxを抜き取る
file_path = "../../datasets/pdfs/train/"
words, bboxes = utils.extraction_text_from_pdf(file_path, file_names)

print("finish! extraction words and bboxes from pdf. ", time.time() - start, flush=True)

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
    

print("finish! tokenzie ", time.time() - start, flush=True)

print("start! extraciton pixel value from image ", time.time() - start, flush=True)

#image からpixel_valueを抜き取る
pixel_values = []
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
count = 0
num = 1
for file_name in file_names:
    image = Image.open(f"../../datasets/pdfs/images/train/{file_name}.png")
    pixel_value = feature_extractor(image)["pixel_values"]
    pixel_values.append(pixel_value)
    count +=1
    if count >1000:
        print(num*count)
        count = 0
        num += 1

print("finish! extraction pixelvalues", time.time() - start, flush=True)

print("start! create subset tokens", time.time() - start, flush=True)
#512に分割 + <s> , </s>, attention mask padding をする
tokens, bboxes, doc_ids = utils.subset_tokens_from_document_light(enc_input_ids, enc_bboxes, vocab, max_len=512)

print("finish! create subset tokens", time.time() - start, flush=True)

print("start! create dataset", time.time() - start, flush=True)

dataset = []
for i in range(len(tokens)):
    dataset.append({"input_ids": tokens[i], "bbox": bboxes[i], 
     "pixel_values": pixel_values[doc_ids[i]][0]})

print("saving......! ../../datasets/pdfs/tensor_dataset.pt", time.time() - start, flush=True)



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


file_path="../../datasets/pdfs/processing/encode_dataset.json"
with open(file_path, 'w') as f:
    json.dump(dataset, f, indent=4, cls=NumpyArrayEncoder)


print("all process finished!!!!!", time.time() - start, flush=True)