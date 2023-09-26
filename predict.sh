model_path=$1

python3 run_wav2vec_clf.py \
    --target_column="emotion" \
    --pooling_mode="mean" \
    --model_name_or_path=$model_path \
    --output_dir=output/ \
    --cache_dir=cache/ \
    --train_file=data/train.csv \
    --validation_file=data/dev.csv \
    --test_file=data/test.csv \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --num_train_epochs=40.0 \
    --evaluation_strategy="steps"\
    --save_steps=100 \
    --eval_steps=100 \
    --logging_steps=100 \
    --save_total_limit=2 \
    --do_predict \
    --fp16 \
    --freeze_feature_extractor 
    # --model_mode="wav2vec2" \ 