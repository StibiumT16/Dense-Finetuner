data_type="nq320k"
model_type='bert'
run_name='sbert'
model_name_or_path="sentence-transformers/msmarco-bert-base-dot-v5" 

CUDA_VISIBLE_DEVICES=1 python ./star/my_train.py \
    --do_train \
    --model_type $model_type \
    --max_query_length 32 \
    --max_doc_length 512 \
    --preprocess_dir ./data/${data_type}_${run_name}/preprocess \
    --hardneg_path ./data/${data_type}_${run_name}/warmup_retrieve/hard.json \
    --init_path $model_name_or_path \
    --output_dir ./data/${data_type}_${run_name}/star_train/models \
    --logging_dir ./data/${data_type}_${run_name}/star_train/log \
    --optimizer_str adamw \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 92 \
    --my_gradient_checkpointing \
    --fp16 \
    --overwrite_output_dir \
    --use_mean