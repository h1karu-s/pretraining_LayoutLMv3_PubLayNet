# -*-coding:utf-8-*-

import argparse
import os

import fitz


def main(args):
    file_names = []
    pdf_names = os.listdir(args.input_dir)
    for n in pdf_names:
        name = os.path.splitext(n)[0]
        file_names.append(name)

    for file_name in file_names:
        try:
            doc = fitz.open(f"{args.input_dir}{file_name}.pdf")
            count = doc.page_count
            if count > 1:
                print("page over 1: ",count)
            page = doc.load_page(0)
            pix = page.get_pixmap()
            pix.save(f"{args.output_dir}{file_name}.png")
        except Exception as e:
            print(file_name, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)