output_filename=wpa_3000_split

mkdir -p ./data/preprocessing_shared/${output_filename}

python ./src/preprocessing_for_wpa.py \
    --tokenizer_vocab_dir ./data/vocab/tokenizer_vocab/ \
    --image_file_dir ../datasets/pdfs/images/train/ \
    --pdf_file_dir ../datasets/pdfs/train/ \
    --output_dir ./data/preprocessing_shared/ \
    --output_filename ${output_filename} \
    --split_size 3000