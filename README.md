# Extending a Parser to Distant Domains

This repository contains the code used to generate the results described in [Extending a Parser to Distant Domains Using a Few Dozen Partially Annotated Examples](TODO) from ACL 2018, forked from the [Minimal Span Parser](https://github.com/mitchellstern/minimal-span-parser) repository.
A more user friendly implementation of the parser is available in PyTorch through [AllenNLP](https://github.com/allenai/allennlp) and a demo here http://demo.allennlp.org/constituency-parsing.
## Requirements and Setup

* Python 3.5 or higher.
* [DyNet](https://github.com/clab/dynet). We recommend installing DyNet from source with MKL support for significantly faster run time.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation.
* Pre-trained models. Before starting, run `unzip models/model_dev=94.48.zip` and `unzip zipped/no_elmo_model_dev=92.34.zip` in the `models/` directory to extract the pre-trained models.

## data

This directory contains the train and development partial annotations made on text from the bio-chemistry and geometry domains.

1) biochem-train.txt - Annotations on text from the biochem domain. Referred to in the submission as BIOCHEMTRAIN.
2) biochem-dev.txt - Annotations on text from the biochem domain. Referred to in the submission as BIOCHEMDEV.
3) geo-train.txt - Annotations on the text from the geometry domain. Referred to in the submission as GEOTRAIN.
4) geo-dev.txt - Annotations on the text from the geometry domain. Referred to in the submission as GEODEV.
5) geo-additional-annotations.txt - 3 additional annotated sentences mentioned in the submission.


## predictions

This directory contains the model's predictions under the various training conditions.

1) geo-predicted-parses-before-training.txt - Parses predicted by RSP for sentences in GEODEV trained on WSJTEST, but not fine-tuned on GEOTRAIN.
2) geo-predicted-parses-after-training.txt - Parses predicted by RSP for sentences in GEODEV trained on WSJTEST, and fine-tuned on GEOTRAIN.
3) geo-predicted-parses-after-finetuning-on-additional-annotations.txt - Parses predicted by RSP for sentences in GEODEV trained on WSJTEST, and fine-tuned on GEOTRAIN and the 3 additional annotations in geo-additional-annotations.txt.
4) biochem-predicted-parses-before-finetuning.txt - Parses predicted by RSP for sentences in BIOCHEMDEV trained on WSJTEST, but not fine-tuned on BIOCHEMTRAIN.
5) biochem-predicted-parses-after-finetuning.txt - Parses predicted by RSP for sentences in BIOCHEMDEV trained on WSJTEST, and fine-tuned on BIOCHEMTRAIN.

## models

This directory contains pre-trained models.

1) model_dev=94.48.zip - Compressed model trained on the WSJ corpus that gets 94.48 F1 on the development set and 94.28 F1 on the test set.
2) no_elmo_model_dev=92.34 - Compressed model train on the WSJ corpus without ELMo word vectors that gets 92.34 F1 on the development set.

## Experiments

### Training

A new model can be trained using the command `python3 src/main.py train ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--experiment-directory` | Path to the directory for this experiment | N/A
`--no-elmo` | Whether to not use ELMo word embeddings | False
`--path-to-python` | Path to a Python installation with AllenNLP | "python3"
`--numpy-seed` | NumPy random seed | Random
`--word-embedding-dim` | Dimension of the learned word embeddings | 100
`--lstm-layers` | Number of bidirectional LSTM layers | 2
`--lstm-dim` | Hidden dimension of each LSTM within each layer | 250
`--dropout` | Dropout rate for LSTMs | 0.4
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--batch-size` | Number of examples per training update | 10
`--epochs` | Number of training epochs | No limit
`--checks-per-epoch` | Number of development evaluations per epoch | 4

The directory specified via `--experiment-directory` must exist and contain train and dev parse trees in files named `train_trees.txt` and `dev_trees.txt` respectively. If additional trees are to be used for fine-tuning, these must be placed in a file named `additional_trees.txt`.

Any of the DyNet command line options can also be specified.

## Evaluation

A saved model can be evaluated on a test corpus using the command `python3 src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--trees-path` | Path to test trees | N/A
`--elmo-embeddings-path` | Path to ELMo embeddings | N/A

As above, any of the DyNet command line options can also be specified.

As an example, after extracting the pre-trained model, you can evaluate it on the test set using the following command:

```
python3 src/main.py test --model-path-base models/model_dev=94.48 --trees-path data/test.trees --elmo-embeddings-path data/test.hdf5
```


## Parsing New Sentences
The `parse` method of the parser can be used to parse new sentences. In particular, `parser.parse(sentence, elmo_embeddings)` will return a the predicted tree.

See the `run_test` function in `src/main.py` for an example of how a parser can be loaded from disk and used to parse sentences.

## Citation

If you use this software for research, please cite our paper as follows:

```
TODO
```
