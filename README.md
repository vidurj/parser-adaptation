# Extending a Parser to Distant Domains

This repository contains the code used to generate the results described in [Extending a Parser to Distant Domains Using a Few Dozen Partially Annotated Examples](http://arxiv.org/abs/1805.06556) from ACL 2018, forked from the [Minimal Span Parser](https://github.com/mitchellstern/minimal-span-parser) repository.
A more user friendly implementation of the parser is available in PyTorch through [AllenNLP](https://github.com/allenai/allennlp), and a demo at http://demo.allennlp.org/constituency-parsing.

## Setup

* Run `pip install -r requirements.txt`.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation.
* Pre-trained models. Before starting, run `unzip models/model_dev=94.48.zip` and `unzip zipped/no_elmo_model_dev=92.34.zip` in the `models/` directory to extract the pre-trained models.

### Pro Tip
* [DyNet](https://github.com/clab/dynet). Installing DyNet from source with MKL support will result in significantly faster run time.




## Experiments

### Training

#### Full Parses
A new model can be trained using the command `python3 src/main.py train-on-parses ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--experiment-directory` | Path to the directory for this experiment | N/A
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

#### Partial Annotations
A model can trained on fine-tuned on partial annotations using the command `python3 src/main.py train-on-partial-annotations ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--experiment-directory` | Path to the directory for this experiment | N/A

The directory specified via `--experiment-directory` must exist and contain partial annotations in a file named `partial_annotations.txt`, and a pre-trained model via files `model.data` and `model.meta`. If additional trees are to be used for fine-tuning, these must be placed in a file named `additional_trees.txt`.
Examples of partial annotations can be found in `data/biochem-train.txt`.


### Evaluation

#### Full Parses

A saved model can be evaluated on a partially annotated test corpus using the command `python3 src/main.py test-on-parses ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--input-file` | Path to parses to evaluate on | N/A
`--model-path` | Path to saved model | N/A
`--experiment-directory` | Path to the directory for this experiment | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`



#### Partial Annotations

 saved model can be evaluated on a test corpus using the command `python3 src/main.py test-on-partial-annotations ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--input-file` | Path to partially bracketed sentences to evaluate on | N/A
`--model-path` | Path to saved model | N/A
`--experiment-directory` | Path to the directory for this experiment | N/A

Examples of partial annotations can be found in `data/biochem-dev.txt`.

## Parsing New Sentences
The `parse` method of the parser can be used to parse new sentences. In particular, `parser.parse(sentence, elmo_embeddings)` will return a the predicted tree.

See the `run_test` function in `src/main.py` for an example of how a parser can be loaded from disk and used to parse sentences.



## Directories

### models

This directory contains pre-trained models.

1) model_dev=94.59.zip - Compressed model trained on the WSJ corpus that gets 94.59 F1 on the development set and 94.34 F1 on the test set.
2) no_elmo_model_dev=92.34 - Compressed model train on the WSJ corpus without ELMo word vectors that gets 92.34 F1 on the development set.


### data

This directory contains the train and development partial annotations made on text from the bio-chemistry and geometry domains.

1) biochem-train.txt - Annotations on text from the biochem domain. Referred to in the submission as BIOCHEMTRAIN.
2) biochem-dev.txt - Annotations on text from the biochem domain. Referred to in the submission as BIOCHEMDEV.
3) geo-train.txt - Annotations on the text from the geometry domain. Referred to in the submission as GEOTRAIN.
4) geo-dev.txt - Annotations on the text from the geometry domain. Referred to in the submission as GEODEV.
5) geo-additional-annotations.txt - 3 additional annotated sentences mentioned in the submission.


### predictions

This directory contains the model's predictions under the various training conditions.

1) geo-predicted-parses-before-training.txt - Parses predicted by RSP for sentences in GEODEV trained on WSJTEST, but not fine-tuned on GEOTRAIN.
2) geo-predicted-parses-after-training.txt - Parses predicted by RSP for sentences in GEODEV trained on WSJTEST, and fine-tuned on GEOTRAIN.
3) geo-predicted-parses-after-finetuning-on-additional-annotations.txt - Parses predicted by RSP for sentences in GEODEV trained on WSJTEST, and fine-tuned on GEOTRAIN and the 3 additional annotations in geo-additional-annotations.txt.
4) biochem-predicted-parses-before-finetuning.txt - Parses predicted by RSP for sentences in BIOCHEMDEV trained on WSJTEST, but not fine-tuned on BIOCHEMTRAIN.
5) biochem-predicted-parses-after-finetuning.txt - Parses predicted by RSP for sentences in BIOCHEMDEV trained on WSJTEST, and fine-tuned on BIOCHEMTRAIN.

## Contact

For questions, contact the first author of the paper [Extending a Parser to Distant Domains Using a Few Dozen Partially Annotated Examples](http://arxiv.org/abs/1805.06556).

## Citation

If you use this software for research, please cite our paper as follows:

```
@InProceedings{Joshi-2018-ParserExtension,
  author    = {Joshi, Vidur and Peters, Matthew and Hopkins, Mark},
  title     = {Extending a Parser to Distant Domains Using a Few Dozen Partially Annotated Examples},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2018},
  address   = {Melbourne, Australia},
  publisher = {Association for Computational Linguistics},
}
```
