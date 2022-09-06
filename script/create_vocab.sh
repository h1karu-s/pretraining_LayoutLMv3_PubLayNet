mkdir -p ./data/vocab/

VOCAB_DIR_NAME=tokenizer_vocab

python ./src/create_vocab.py \
    --input_dir ../datasets/pdfs/train/ \
    --output_dir ./data/vocab/${VOCAB_DIR_NAME} \
    --model_name microsoft/layoutlmv3-base \
    --vocab_size 50265