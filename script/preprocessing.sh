mkdir -p ./data/preprocessing_shared/

python ./src/preprocessing.py \
    --tokenizer_vocab_dir ./data/vocab/tokenizer_vocab/ \
    --image_file_dir ../datasets/pdfs/images/train/ \
    --pdf_file_dir ../datasets/pdfs/train/ \
    --output_dir ./data/preprocessing_shared/ \
    --output_filename encoded_dataset.pkl \
    --datasize 50000