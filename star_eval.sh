data_type="nq320k"
backbone="t5"
model="gtrt5"
model_name_or_path="sentence-transformers/gtr-t5-base"

CUDA_VISIBLE_DEVICES=0 python ./star/infer.py \
    --data_type $data_type \
    --max_doc_length 512 \
    --max_query_length 64 \
    --mode dev \
    --model_type $backbone \
    --run_name $model \
    --model_name_or_path $model_name_or_path\
    --eval_batch_size 256 \
    --use_mean \
    --use_cos \
    --faiss_gpus 0

python ./msmarco_eval.py ./data/${data_type}_${model}/preprocess/dev-qrel.tsv ./data/${data_type}_${model}/evaluate/star/dev.rank.tsv