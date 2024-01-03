#!/bin/bash

# Paths and files
VIRTUALENV=~/.virtualenvs/d4.3/bin # it should be a path to a virtualenv in which Fairseq is installed
ORIG_VOCAB=training/dict.40K.txt
VALID_PREF=corpora/dev/dev.spm 
DEST_DIR=evaluation
OUTPUTS=outputs

# Parameters
checkpoint=checkpoints/checkpoint_last.pt
task=translation_from_pretrained_bart
batch_size=32
thresholdtgt=0
thresholdsrc=0
workers=70
languages=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

# Move dictionary used for fine-tuning to the evaluation folder
mkdir -p ${DEST_DIR} ${OUTPUTS}
cp ${ORIG_VOCAB} ${DEST_DIR}
filename=$(echo ${ORIG_VOCAB} | awk -F'/' '{print $2}')
VOCAB=${DEST_DIR}/${filename}

# Activate virtual environment
source ${VIRTUALENV}/activate

echo "Using last the weights from the 'checkpoint_last' file"

# SPANISH TO ENGLISH =======================================
echo "es -> en"

# Prepare inputs
cp corpora/en-es/test_en-es.en.spm evaluation/test_en-es.spm.en_XX
cp corpora/en-es/test_en-es.es.spm evaluation/test_en-es.spm.es_XX

${VIRTUALENV}/fairseq-preprocess \
    --source-lang es_XX \
    --target-lang en_XX \
    --testpref evaluation/test_en-es.spm \
    --destdir ${DEST_DIR} \
    --thresholdtgt ${thresholdtgt} \
    --thresholdsrc ${thresholdsrc} \
    --srcdict ${VOCAB} \
    --tgtdict ${VOCAB} \
    --workers ${workers} \
    > /dev/null 2>&1

${VIRTUALENV}/fairseq-generate ${DEST_DIR} \
    --path ${checkpoint} \
    --task ${task} \
    --gen-subset test -s es_XX -t en_XX \
    --sacrebleu \
    --remove-bpe 'sentencepiece' \
    --batch-size ${batch_size} \
    --langs ${languages} \
    > ${OUTPUTS}/es_en.ft_last

# Extract the generated output (hypothesis) and the ground truth (reference)
cat ${OUTPUTS}/es_en.ft_last | grep -P "^H" | sort -V | cut -f 3- | sed 's/\[en_XX\]//g' > ${OUTPUTS}/es_en.ft_last.hyp
cat ${OUTPUTS}/es_en.ft_last | grep -P "^T" | sort -V | cut -f 2- | sed 's/\[en_XX\]//g' > ${OUTPUTS}/es_en.ft_last.ref

# Compute the sacrebleu metric
sacrebleu ${OUTPUTS}/es_en.ft_last.ref < ${OUTPUTS}/es_en.ft_last.hyp

# ENGLISH TO DUTCH =======================================
echo "en -> nl"

# Prepare inputs
cp corpora/en-nl/test_en-nl.en.spm evaluation/test_en-nl.spm.en_XX
cp corpora/en-nl/test_en-nl.nl.spm evaluation/test_en-nl.spm.nl_XX

${VIRTUALENV}/fairseq-preprocess \
    --source-lang en_XX \
    --target-lang nl_XX \
    --testpref evaluation/test_en-nl.spm \
    --destdir ${DEST_DIR} \
    --thresholdtgt ${thresholdtgt} \
    --thresholdsrc ${thresholdsrc} \
    --srcdict ${VOCAB} \
    --tgtdict ${VOCAB} \
    --workers ${workers} \
    > /dev/null 2>&1

${VIRTUALENV}/fairseq-generate ${DEST_DIR} \
    --path ${checkpoint} \
    --task ${task} \
    --gen-subset test -s en_XX -t nl_XX \
    --sacrebleu \
    --remove-bpe 'sentencepiece' \
    --batch-size ${batch_size} \
    --langs ${languages} \
    > ${OUTPUTS}/en_nl.ft_last

# Extract the generated output (hypothesis) and the ground truth (reference)
cat ${OUTPUTS}/en_nl.ft_last | grep -P "^H" | sort -V | cut -f 3- | sed 's/\[nl_XX\]//g' > ${OUTPUTS}/en_nl.ft_last.hyp
cat ${OUTPUTS}/en_nl.ft_last | grep -P "^T" | sort -V | cut -f 2- | sed 's/\[nl_XX\]//g' > ${OUTPUTS}/en_nl.ft_last.ref

# Compute the sacrebleu metric
sacrebleu ${OUTPUTS}/en_nl.ft_last.ref < ${OUTPUTS}/en_nl.ft_last.hyp

# ENGLISH TO SPANISH =======================================
echo "en -> es"

${VIRTUALENV}/fairseq-preprocess \
    --source-lang en_XX \
    --target-lang es_XX \
    --testpref evaluation/test_en-es.spm \
    --destdir ${DEST_DIR} \
    --thresholdtgt ${thresholdtgt} \
    --thresholdsrc ${thresholdsrc} \
    --srcdict ${VOCAB} \
    --tgtdict ${VOCAB} \
    --workers ${workers} \
    > /dev/null 2>&1

${VIRTUALENV}/fairseq-generate ${DEST_DIR} \
    --path ${checkpoint} \
    --task ${task} \
    --gen-subset test -s en_XX -t es_XX \
    --sacrebleu \
    --remove-bpe 'sentencepiece' \
    --batch-size ${batch_size} \
    --langs ${languages} \
    > ${OUTPUTS}/en_es.ft_last

# Extract the generated output (hypothesis) and the ground truth (reference)
cat ${OUTPUTS}/en_es.ft_last | grep -P "^H" | sort -V | cut -f 3- | sed 's/\[es_XX\]//g' > ${OUTPUTS}/en_es.ft_last.hyp
cat ${OUTPUTS}/en_es.ft_last | grep -P "^T" | sort -V | cut -f 2- | sed 's/\[es_XX\]//g' > ${OUTPUTS}/en_es.ft_last.ref

# Compute the sacrebleu metric
sacrebleu ${OUTPUTS}/en_es.ft_last.ref < ${OUTPUTS}/en_es.ft_last.hyp

# DUTCH TO ENGLISH =======================================
echo "nl -> en"

${VIRTUALENV}/fairseq-preprocess \
    --source-lang nl_XX \
    --target-lang en_XX \
    --testpref evaluation/test_en-nl.spm \
    --destdir ${DEST_DIR} \
    --thresholdtgt ${thresholdtgt} \
    --thresholdsrc ${thresholdsrc} \
    --srcdict ${VOCAB} \
    --tgtdict ${VOCAB} \
    --workers ${workers} \
    > /dev/null 2>&1

${VIRTUALENV}/fairseq-generate ${DEST_DIR} \
    --path ${checkpoint} \
    --task ${task} \
    --gen-subset test -s en_XX -t nl_XX \
    --sacrebleu \
    --remove-bpe 'sentencepiece' \
    --batch-size ${batch_size} \
    --langs ${languages} \
    > ${OUTPUTS}/nl_en.ft_last

# Extract the generated output (hypothesis) and the ground truth (reference)
cat ${OUTPUTS}/nl_en.ft_last | grep -P "^H" | sort -V | cut -f 3- | sed 's/\[en_XX\]//g' > ${OUTPUTS}/nl_en.ft_last.hyp
cat ${OUTPUTS}/nl_en.ft_last | grep -P "^T" | sort -V | cut -f 2- | sed 's/\[en_XX\]//g' > ${OUTPUTS}/nl_en.ft_last.ref

# Compute the sacrebleu metric
sacrebleu ${OUTPUTS}/nl_en.ft_last.ref < ${OUTPUTS}/nl_en.ft_last.hyp