#!/bin/bash

# Paths and files
VIRTUALENV=~/.virtualenvs/d4.3/bin # it should be a path to a virtualenv in which Fairseq is installed
VOCAB=training/dict.40K.txt
VALID_PREF=corpora/dev/dev.spm 
DEST_DIR=training/data

# Training parameters
seed=222
architecture=mbart_large
task=translation_from_pretrained_bart
criterion=label_smoothed_cross_entropy
lr_scheduler=polynomial_decay
legacy_ddp=c10d
thresholdtgt=0
thresholdsrc=0
workers=70
max_tokens=5120
label_smoothing=0.2
max_epoch=1
optimiser=adam
adam_betas='(0.9,0.98)'
adam_eps=1e-06
weight_decay=0.0
update_frequency=4
warmup_updates=2500
total_num_update=40000
learning_rate=3e-05
dropout=0.3
attention_dropout=0.1
log_interval=2
log_format=simple
save_interval=1
languages=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

# Variables to control training
I=1
SIZE=30000
MAX=30000001
CURRENT=$(( I * SIZE ))

# Activate virtual environment
source ${VIRTUALENV}/activate

while [[ $CURRENT -lt $MAX ]]
do
    # SPANISH TO ENGLISH =======================================
    # Subsample SIZE elements
    cat corpora/en-es/train_en-es.en.spm | head -${CURRENT} | tail -${SIZE} > training/train_en-es.spm.en_XX
    cat corpora/en-es/train_en-es.es.spm | head -${CURRENT} | tail -${SIZE} > training/train_en-es.spm.es_XX

    rm -fr training/data
    ${VIRTUALENV}/fairseq-preprocess \
        --source-lang es_XX \
        --target-lang en_XX \
        --trainpref training/train_en-es.spm \
        --validpref ${VALID_PREF} \
        --destdir ${DEST_DIR} \
        --thresholdtgt ${thresholdtgt} \
        --thresholdsrc ${thresholdsrc} \
        --srcdict ${VOCAB} \
        --tgtdict ${VOCAB} \
        --workers ${workers}

    # In the first pass, load the pre-trained monolingual model. Afterwards, load the last finetuned model.
    if [[ $I -gt 1 ]]
    then
    	model_to_load=checkpoints/checkpoint_last.pt
    else
	model_to_load=training/model.pt
    fi
    ${VIRTUALENV}/fairseq-train training/data \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --arch ${architecture} \
        --layernorm-embedding \
        --task ${task} \
        --source-lang es_XX \
        --target-lang en_XX \
        --criterion ${criterion} \
        --label-smoothing ${label_smoothing} \
        --optimizer ${optimiser} \
        --adam-eps ${adam_eps} \
        --adam-betas ${adam_betas} \
        --lr-scheduler ${lr_scheduler} \
        --lr ${learning_rate} \
        --warmup-updates ${warmup_updates} \
        --total-num-update ${total_num_update} \
        --dropout ${dropout} \
        --attention-dropout ${attention_dropout} \
        --weight-decay ${weight_decay} \
        --max-tokens ${max_tokens} \
        --update-freq ${update_frequency} \
        --save-interval ${save_interval} \
        --max-epoch ${max_epoch} \
        --seed ${seed} \
        --log-format ${log_format} \
        --log-interval ${log_interval} \
        --restore-file ${model_to_load} \
        --reset-optimizer \
        --reset-meters \
        --reset-dataloader \
        --reset-lr-scheduler \
        --langs ${languages} \
        --ddp-backend ${legacy_ddp} \
        --fp16

    mv -f checkpoints/checkpoint1.pt checkpoints/checkpoint.es-en.pt
    
    # ENGLISH TO DUTCH =======================================
    # Subsample SIZE elements
    cat corpora/en-es/train_en-nl.en.spm | head -${CURRENT} | tail -${SIZE} > training/train_en-nl.spm.en_XX
    cat corpora/en-es/train_en-nl.nl.spm | head -${CURRENT} | tail -${SIZE} > training/train_en-nl.spm.nl_XX

    rm -fr training/data
    ${VIRTUALENV}/fairseq-preprocess  \
        --source-lang en_XX \
        --target-lang nl_XX \
        --trainpref training/train_en-nl.spm \
        --validpref ${VALID_PREF} \
        --destdir ${DEST_DIR} \
        --thresholdtgt ${thresholdtgt} \
        --thresholdsrc ${thresholdsrc} \
        --srcdict ${VOCAB} \
        --tgtdict ${VOCAB} \
        --workers ${workers}

    ${VIRTUALENV}/fairseq-train training/data \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --arch ${architecture} \
        --layernorm-embedding \
        --task ${task} \
        --source-lang es_XX \
        --target-lang nl_XX \
        --criterion ${criterion} \
        --label-smoothing ${label_smoothing} \
        --optimizer ${optimiser} \
        --adam-eps ${adam_eps} \
        --adam-betas ${adam_betas} \
        --lr-scheduler ${lr_scheduler} \
        --lr ${learning_rate} \
        --warmup-updates ${warmup_updates} \
        --total-num-update ${total_num_update} \
        --dropout ${dropout} \
        --attention-dropout ${attention_dropout} \
        --weight-decay ${weight_decay} \
        --max-tokens ${max_tokens} \
        --update-freq ${update_frequency} \
        --save-interval ${save_interval} \
        --max-epoch ${max_epoch} \
        --seed ${seed} \
        --log-format ${log_format} \
        --log-interval ${log_interval} \
        --restore-file ${model_to_load} \
        --reset-optimizer \
        --reset-meters \
        --reset-dataloader \
        --reset-lr-scheduler \
        --langs ${languages} \
        --ddp-backend ${legacy_ddp} \
        --fp16

    mv -f checkpoints/checkpoint1.pt checkpoints/checkpoint.en-nl.pt
    I=$(( I + 1 ))
    CURRENT=$(( I * SIZE ))

    # ENGLISH TO SPANISH =======================================
    # Subsample SIZE elements
    cat corpora/en-es/train_en-nl
    cat corpora/en-es/train_en-es.en.spm | head -${CURRENT} | tail -${SIZE} > training/train_en-es.spm.en_XX
    cat corpora/en-es/train_en-es.es.spm | head -${CURRENT} | tail -${SIZE} > training/train_en-es.spm.es_XX

    rm -fr training/data
    ${VIRTUALENV}/fairseq-preprocess \
        --source-lang en_XX \
        --target-lang es_XX \
        --trainpref training/train_en-es.spm \
        --validpref ${VALID_PREF} \
        --destdir ${DEST_DIR} \
        --thresholdtgt ${thresholdtgt} \
        --thresholdsrc ${thresholdsrc} \
        --srcdict ${VOCAB} \
        --tgtdict ${VOCAB} \
        --workers ${workers}

    ${VIRTUALENV}/fairseq-train training/data \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --arch ${architecture} \
        --layernorm-embedding \
        --task ${task} \
        --source-lang en_XX \
        --target-lang es_XX \
        --criterion ${criterion} \
        --label-smoothing ${label_smoothing} \
        --optimizer ${optimiser} \
        --adam-eps ${adam_eps} \
        --adam-betas ${adam_betas} \
        --lr-scheduler ${lr_scheduler} \
        --lr ${learning_rate} \
        --warmup-updates ${warmup_updates} \
        --total-num-update ${total_num_update} \
        --dropout ${dropout} \
        --attention-dropout ${attention_dropout} \
        --weight-decay ${weight_decay} \
        --max-tokens ${max_tokens} \
        --update-freq ${update_frequency} \
        --save-interval ${save_interval} \
        --max-epoch ${max_epoch} \
        --seed ${seed} \
        --log-format ${log_format} \
        --log-interval ${log_interval} \
        --restore-file ${model_to_load} \
        --reset-optimizer \
        --reset-meters \
        --reset-dataloader \
        --reset-lr-scheduler \
        --langs ${languages} \
        --ddp-backend ${legacy_ddp} \
        --fp16

    mv -f checkpoints/checkpoint1.pt checkpoints/checkpoint.en-es.pt

    # DUTCH TO ENGLISH =======================================
    # Subsample SIZE elements
    cat corpora/en-es/train_en-nl.en.spm | head -${CURRENT} | tail -${SIZE} > training/train_en-nl.spm.en_XX
    cat corpora/en-es/train_en-nl.nl.spm | head -${CURRENT} | tail -${SIZE} > training/train_.en-nl.spm.nl_XX

    rm -fr training/data
    ${VIRTUALENV}/fairseq-preprocess \
        --source-lang nl_XX \
        --target-lang en_XX  \
        --trainpref training/train_en-nl.spm \
        --validpref ${VALID_PREF} \
        --destdir ${DEST_DIR} \
        --thresholdtgt ${thresholdtgt} \
        --thresholdsrc ${thresholdsrc} \
        --srcdict ${VOCAB} \
        --tgtdict ${VOCAB} \
        --workers ${workers}

    ${VIRTUALENV}/fairseq-train training/data \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --arch ${architecture} \
        --layernorm-embedding \
        --task ${task} \
        --source-lang nl_XX \
        --target-lang en_XX \
        --criterion ${criterion} \
        --label-smoothing ${label_smoothing} \
        --optimizer ${optimiser} \
        --adam-eps ${adam_eps} \
        --adam-betas ${adam_betas} \
        --lr-scheduler ${lr_scheduler} \
        --lr ${learning_rate} \
        --warmup-updates ${warmup_updates} \
        --total-num-update ${total_num_update} \
        --dropout ${dropout} \
        --attention-dropout ${attention_dropout} \
        --weight-decay ${weight_decay} \
        --max-tokens ${max_tokens} \
        --update-freq ${update_frequency} \
        --save-interval ${save_interval} \
        --max-epoch ${max_epoch} \
        --seed ${seed} \
        --log-format ${log_format} \
        --log-interval ${log_interval} \
        --restore-file ${model_to_load} \
        --reset-optimizer \
        --reset-meters \
        --reset-dataloader \
        --reset-lr-scheduler \
        --langs ${languages} \
        --ddp-backend ${legacy_ddp} \
        --fp16

    mv -f checkpoints/checkpoint1.pt checkpoints/checkpoint.nl-en.pt

    I=$(( I + 1 ))
    CURRENT=$(( I * SIZE ))
done
