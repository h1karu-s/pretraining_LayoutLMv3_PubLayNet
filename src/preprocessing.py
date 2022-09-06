# -*-coding:utf-8-*-

import argparse
import os
from PIL import Image
import pickle

from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer

from utils import utils


def main(args):
    print(args, flush=True)
    tokenizer = LayoutLMv3Tokenizer(f"{args.tokenizer_vocab_dir}vocab.json", f"{args.tokenizer_vocab_dir}merges.txt")
    ids = range(tokenizer.vocab_size)
    vocab = tokenizer.convert_ids_to_tokens(ids)

    image_names = os.listdir(args.image_file_dir)
    if hasattr(args, 'datasize'):
        image_names = image_names[:args.datasize]
    file_names = []
    for n in image_names:
        name = os.path.splitext(n)[0]
        file_names.append(name)

    print("start! extraction words and bboxes from pdf.", flush=True)
    words, bboxes = utils.extraction_text_from_pdf(args.pdf_file_dir, file_names)

    enc_input_ids = []
    enc_bboxes = []
    for i in range(len(words)):
        try:
            enc = tokenizer(text=words[i], boxes = bboxes[i], add_special_tokens=False)
            enc_input_ids.append(enc["input_ids"])
            enc_bboxes.append(enc["bbox"])
        except Exception as e:
            print(file_names.pop(i), e, flush=True)
    
    pixel_values = []
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    for file_name in file_names:
        image = Image.open(f"{args.image_file_dir}{file_name}.png")
        pixel_value = feature_extractor(image)["pixel_values"]
        pixel_values.append(pixel_value)

    print("start! create subset tokens", flush=True)
    tokens, bboxes, doc_ids = utils.subset_tokens_from_document_light(enc_input_ids, enc_bboxes, vocab, max_len=512)

    dataset = []
    for i in range(len(tokens)):
        dataset.append({"input_ids": tokens[i], "bbox": bboxes[i], 
        "pixel_values": pixel_values[doc_ids[i]][0]})
    
    with open(f"{args.output_dir}{args.output_filename}", 'wb') as f:
        pickle.dump(dataset, f, protocol=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_vocab_dir", type=str)
    parser.add_argument("--image_file_dir", type=str, required=True)
    parser.add_argument("--pdf_file_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--datasize", type=int)
    args = parser.parse_args()
    main(args)