# Code for Deliverable 4.3

## Description

This project provides the preparation and finetuning of an [mBART model](https://arxiv.org/abs/2001.08210 "Link to the ArXiV paper") for the English-to-Spanish, Spanish-to-English, English-to-Dutch and Dutch-to-English language pairs.

## Installation

Open a terminal and execute the following:

```
chmod +x setup.sh
./setup.sh
```

## Preparing the data

The `prepare_data.sh` script already downloads all the necessary files and creates the required folders for the project. Execute the following:

```
chmod +x prepare_data.sh
./prepare_data.sh
```

The script downloads the mBART model, the tokeniser and the original vocabulary file. Concerning the data, the ParaCrawl 7.1 dataset is used, downloading the "en-es" and "en-nl" parallel corpora. These files are tokenised and a reduced vocabulary of 40,000 tokens is extracted.

## Training

Run the following commands to start training the mBART model.

```
chmod +x finetuning.sh
./finetuning.sh
```

## Evaluation

There are three scripts to evaluate the trained model. The first one, `evaluate.sh`, takes the last saved model and evaluates it on the available language pairs. Run the following to execute it:

```
chmod +x evaluate.sh
./evaluate.sh
```

`evaluate2.sh` uses the model saved after training with a specific language pair to evaluate that language pair. Run:

```
chmod +x evaluate2.sh
./evaluate2.sh
```

Finally, `evaluate3.sh` uses an average checkpoint created using the four available models (one per language pair). First, the average checkpoint must be created. Then, the evaluation script can be run. Execute the following:

```
chmod +x average_checkpoints.sh
chmod +x evaluate3.sh
./average_checkpoints.sh
./evaluate3.sh
```