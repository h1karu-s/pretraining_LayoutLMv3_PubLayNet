mkdir -p ./data/train/model_1000/

python ./src/pretrain_2.py \
    --input_file ./data/preprocessing_shared/encoded_dataset_10000.pkl \
    --tokenizer_vocab_dir ./data/vocab/tokenizer_vocab/ \
    --output_model_dir ./data/train/model_10000_1e-4/ \
    --output_file_name pretrained_layoutLMv3_1.params\
    --model_name microsoft/layoutlmv3-base \
    --ratio_train 0.9 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 10