#!/bin/bash

TEST_SIZE=2000

# Download the parallel corpora
if [ ! -f en-es.txt.zip ]; then
    wget https://opus.nlpl.eu/download.php?f=ParaCrawl/v7.1/moses/en-es.txt.zip
fi
if [ ! -f en-es.txt.zip ]; then
    wget https://opus.nlpl.eu/download.php?f=ParaCrawl/v7.1/moses/en-nl.txt.zip
fi

# Unzip the parallel corpora files
if [ ! -f corpora/en-es/ParaCrawl.en-es.en ]; then
   mkdir -p corpora/en-es
   unzip -j "en-es.txt.zip" "ParaCrawl.en-es.en" -d "corpora/en-es"
   unzip -j "en-es.txt.zip" "ParaCrawl.en-es.es" -d "corpora/en-es"
fi
if [ ! -f corpora/en-nl/ParaCrawl.en-nl.en ]; then
   mkdir -p corpora/en-nl
   unzip -j "en-nl.txt.zip" "ParaCrawl.en-nl.en" -d "corpora/en-nl"
   unzip -j "en-nl.txt.zip" "ParaCrawl.en-nl.nl" -d "corpora/en-nl"
fi


# Create a file to gather the corpora
mkdir -p training/
touch training/corpus.all
: > training/corpus.all

# Split each corpus into training and test sets
languages=("en-es" "en-nl")
for langs in ${languages[*]}; do
    for i in {1..2}; do
        lang="$(cut -d'-' -f$i <<<"$langs")"
        awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' corpora/$langs/ParaCrawl.$langs.$lang | sort -n | cut -c8- > corpora/$langs/ParaCrawl.$langs.$lang_shuf
        head -n $TEST_SIZE corpora/$langs/ParaCrawl.$langs.$lang_shuf > corpora/$langs/test_$langs.$lang
        tail -n +$TEST_SIZE corpora/$langs/ParaCrawl.$langs.$lang_shuf > corpora/$langs/train_$langs.$lang
        cat corpora/$langs/train_$langs.$lang >> training/corpus.all
    done
done

# Combine and sort the corpora (removing duplicated sentences)
mkdir -p temp/
sort -u -T temp/ training/corpus.all

# Download the mBART files (including the model and the SentencePiece tokenizer)
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
tar -xzvf mbart.cc25.v2.tar.gz
rm mbart.cc25.v2.tar.gz

# Generate the vocabulary from the multilingual parallel corpus created before
spm_encode --model=mbart.cc25.v2/sentence.bpe.model --generate_vocabulary < training/corpus.all > training/dictionary.all

# Get the most frequent tokens (40,000)
head -40000 training/dictionary.all | sed -e 's/\t/ /' > training/dict.40K.txt

# Tokenise the original parallel corpora 
languages=("en-es" "en-nl")
for langs in ${languages[*]}; do
    for i in {1..2}; do
    	lang="$(cut -d'-' -f$i <<<"$langs")"
	spm_encode --model=mbart.cc25.v2/sentence.bpe.model --vocabulary=training/dict.40K.txt < corpora/$langs/train_$langs.$lang > corpora/$langs/train_$langs.$lang.spm
	spm_encode --model=mbart.cc25.v2/sentence.bpe.model --vocabulary=training/dict.40K.txt < corpora/$langs/test_$langs.$lang > corpora/$langs/test_$langs.$lang.spm
    done
done

# Necessary files for the finetuning
mkdir -p corpora/dev
echo . > corpora/dev/dev.spm 
languages=("en" "es" "nl")
for lang in ${languages[*]}; do
    echo . > corpora/dev/dev.spm.${lang}_XX
done
