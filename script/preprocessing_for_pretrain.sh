output_filename=10000_split

mkdir -p ./data/preprocessing_shared/${output_filename}

python ./src/preprocessing_pretrain_2.py \
    --tokenizer_vocab_dir ./data/vocab/tokenizer_vocab/ \
    --image_file_dir ../datasets/pdfs/images/train/ \
    --pdf_file_dir ../datasets/pdfs/train/ \
    --output_dir ./data/preprocessing_shared/ \
    --output_filename ${output_filename} \
    --split_size 10000