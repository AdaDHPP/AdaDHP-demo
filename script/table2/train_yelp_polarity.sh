for task in yelp_polarity
do
  for lr in 0.0009
  do
    for seed in 0 1 2
    do
      CUDA_VISIBLE_DEVICES=2 python train.py \
      --peft_type ADADHP \
      --seed ${seed} \
      --learning_rate ${lr} \
      --task_name ${task} \
      --dataset_config_name en \
      --model_name_or_path google-t5/t5-base \
      --do_train \
      --do_eval \
      --do_predict \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --max_seq_length 256 \
      --save_strategy epoch \
      --save_steps 1000 \
      --evaluation_strategy epoch \
      --num_train_epochs 10 \
      --warmup_steps 500 \
      --load_best_model_at_end \
      --save_total_limit 1 \
      --tinit 3600 \
      --tfinal 33500 \
      --deltaT 2000 \
      --target_num 96 \
      --output_dir adadhp_t5-base/${task}_lr${lr}_seed${seed} \
      --save_safetensors False \
      --overwrite_output_dir
    done
  done
done
