# SimpleRE

The code is divided into two parts, DialogRE and TACRED, by datasets.

# Requirements
PyTorch == 1.4
CUDA == 10.1
Apex
We perform our experiments on Q8000 GPU. Please use the same hardware and software environment if possible to ensure the same results.


# DialogRE

This dataset can be downloaded at: https://github.com/nlpdata/dialogre
Download and unzip BERT from https://github.com/google-research/bert, and set up the environment variable for BERT by export BERT_BASE_DIR=/PATH/TO/BERT/DIR in every run_simplere.sh.

(1) Please copy the *.json files into DialogRE/data
(2) Train the SimpleRE model
```sh
$ cd SimpleRE
$ bash run_simplere.sh
```


# TACRED

TACRED URL: https://nlp.stanford.edu/projects/tacred/
TACRED-Revisit URL: https://github.com/DFKI-NLP/tacrev/
(1) Please download the TACRED and TACRED-Revisit and copy them into SimpleRE/tacred_data respectively.
    TACRED URL: https://nlp.stanford.edu/projects/tacred/
    TACRED-Revisit URL: https://github.com/DFKI-NLP/tacrev/

(2) Train the SimpleRE model
```sh
$ cd SimpleRE
$ bash run_simplere.sh
$ bash test_simplere.sh
```


