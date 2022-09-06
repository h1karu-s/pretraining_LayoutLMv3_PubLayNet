# -*-coding:utf-8-*-

import argparse
import os 

import fitz
from transformers import AutoTokenizer


def main(args):
    file_names = os.listdir(args.input_dir)
    training_corpus = []
    for file_name in file_names:
        try:
            doc = fitz.open(f"{args.input_dir}{file_name}")
            for page in doc:
                training_corpus.append(page.get_text().lower())
        except Exception as e:
            print(file_name, e)
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, args.vocab_size)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()
    main(args)