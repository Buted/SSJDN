#!/usr/bin/env bash
# -*- coding:utf-8 -*-
cuda="0"
lr=1e-5
batch=32
opt="adam"
early=60
re_start=20
loss_alpha=2
bert_layers=4
keep_event=0
dropout=0.5
contiguous=False
seq_lambda=15
seed=42
data="ACE"
dir_prefix="rand-SSJDN"
dir_postfix="seed"$seed"-"$data"-wcjdn-seq"$seq_lambda
finetune="models/"$dir_prefix"-"$dir_postfix"/best_model.pt"
dir=$dir_prefix"-"$dir_postfix
hps="{'loss_alpha': $loss_alpha, 'bert_layers': $bert_layers}"
bert="bert-large-uncased"

mkdir -p models/$dir
cp scripts/train.sh models/$dir
cp -r enet models/$dir
PYTHONIOENCODING=utf8 python3 -m enet.run.ee.runner \
       --train $data"/train.json" \
       --test $data"/test.json" \
       --dev $data"/dev.json" \
       --keep_event $keep_event \
       --earlystop $early \
       --restart $re_start \
       --optimizer $opt \
       --lr $lr \
       --batch $batch \
       --epochs 100 \
       --device "cuda:$cuda" \
       --out "models/$dir" \
       --l2decay 1e-8 \
       --dropout $dropout \
       --contiguous $contiguous \
       --seq_lambda $seq_lambda \
       --seed $seed \
       --hps "$hps" \
       --bert "$bert" \
>& models/$dir/log &
