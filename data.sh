# docid 目前只能全数字
# preprocessed docs：
# corpus.tsv
# qrels.train.tsv query.train.tsv
# qrels.dev.tsv query.dev.tsv

python process/data_process.py \
    --data_type nq320k \
    --run_name sbert \
    --model_name_or_path sentence-transformers/msmarco-bert-base-dot-v5