for task in cola
do
  for lr in 0.007
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
      --per_device_train_batch_size 100 \
      --per_device_eval_batch_size 100 \
      --max_seq_length 128 \
      --save_strategy epoch \
      --save_steps 1000 \
      --evaluation_strategy epoch \
      --num_train_epochs 20 \
      --warmup_steps 500 \
      --load_best_model_at_end \
      --save_total_limit 1 \
      --tinit 100 \
      --tfinal 800 \
      --deltaT 10 \
      --target_num 96 \
      --output_dir adadhp_t5-base/${task}_lr${lr}_seed${seed} \
      --save_safetensors False \
      --overwrite_output_dir
    done
  done
done
