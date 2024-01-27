## Training LLaMA-VQA (LLaMA + Flipped-VQA)

### siq2

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 2 train.py --model 7B \
--max_seq_len 128 --batch_size 8 --epochs 1 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset siq2 \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/siq2_newtest --accum_iter 2 --vaq --qav --sub
```



torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 2 train.py --model 7B \
--max_seq_len 650 --batch_size 1 --epochs 1 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 60 --dataset siq2 \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/siq2_newtest_2 --accum_iter 2 --vaq --qav --sub


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 2 eval.py --model 7B \
--max_seq_len 650 --batch_size 1 --epochs 1 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 60 --dataset siq2 \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/siq2_newtest --accum_iter 2 --vaq --qav --sub --resume checkpoint/siq2/sub/checkpoint_best.pth


# Finetune siq2 one gpu's, use transcripts sampling
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 2 train.py --model 7B \
--max_seq_len 650 --batch_size 1 --epochs 1 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset siq2 \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/siq2_newtest_2 --accum_iter 2 --vaq --qav --sub --speaking_turns_path=data/SIQ2/turn_information.pkl --speaking_turns

DUMMY ignore:
nohup sh -c 'python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ftsiq2_turns_transcripts_nosampling --lr=5e-5 --schedule=linear_with_warmup --load=models/frozenbilm.pth --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size=8 --batch_size_val=8 --max_tokens=256 --epochs=60 --siq_speaking_turns_path=data/SIQ2/turn_information.pkl --use_speaking_turns_sampling False --features_dim 768 --siq_features_path=datasets/SIQ2/clipvitl14_vid_subs_768.pth' > ftsiq2_turns_transcripts_nosampling/output_print.log 2>&1 &