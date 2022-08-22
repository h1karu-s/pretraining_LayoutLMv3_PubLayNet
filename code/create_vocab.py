from transformers import LayoutLMv3FeatureExtractor
from PIL import Image
import fitz
import os 
from transformers import AutoTokenizer

file_names = os.listdir("../../datasets/pdfs/train")


training_corpus = []
for file_name in file_names:
    try:
        doc = fitz.open(f'../../datasets/pdfs/train/{file_name}')
        for page in doc:
            training_corpus.append(page.get_text().lower())
    except Exception:
        print("error!", file_name)

print("created training_corpus and starting leaning tokenizer!")

old_tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

print("finish leaning tokenizer")

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 50265)

tokenizer.save_pretrained("../model/tokenizer_vocab")