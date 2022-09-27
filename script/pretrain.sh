output_dir=pretrain_lr_1e-4_dtasize_18_batch32

max_epochs=20

uper_range=`expr $max_epochs - 1`


mkdir -p ./data/train/${output_dir}/

for i in `seq 0 ${uper_range}`
do
    mkdir -p ./data/train/${output_dir}/epoch_${i}/
done


python3 ./src/pretrain_3.py \
    --input_file ./data/preprocessing_shared/wpa_10000/ \
    --tokenizer_vocab_dir ./data/vocab/tokenizer_vocab/ \
    --output_model_dir /cl/work2/hikaru-si/development/exp_005/data/train/${output_dir}/ \
    --output_file_name pretrained_layoutLMv3_1.params\
    --model_name microsoft/layoutlmv3-base \
    --ratio_train 0.9 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_epochs ${max_epochs} \
    --model_params ./data/train/pretrain_lr_1e-4_dtasize_18_batch32/epoch_15/checkpoint.cpt