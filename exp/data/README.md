# Data Directory

This directory contains selected public benchmark splits used by the experiments. Large corpora, retrieval indexes, model caches, and some third-party datasets are not bundled.

## Included Splits

```text
data/
├── nq/nq-test-contriever.json
├── hotpotqa/test_qa_pairs.json
└── SQuAD/validation-00000-of-00001.parquet
```

## Download-Only Splits

The following files are expected by the scripts but are not included in this repository:

| Expected local file | Source |
| --- | --- |
| `data/webqa/wq-test-contriever.json` | Hugging Face: https://huggingface.co/datasets/stanfordnlp/web_questions |
| `data/TriviaQA/unfiltered-web-dev.json` | TriviaQA unfiltered v1.0: https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz |

You can also prepare these files through the repository script:

```bash
cd exp
python cross_query_se/scripts/download_datasets.py webqa triviaqa
```

For reference, the original Stanford WebQuestions release is available at:

```text
http://nlp.stanford.edu/static/software/sempre/release-emnlp2013/lib/data/webquestions/dataset_11/webquestions.examples.train.json.bz2
http://nlp.stanford.edu/static/software/sempre/release-emnlp2013/lib/data/webquestions/dataset_11/webquestions.examples.test.json.bz2
```

## Retrieval Corpus And Indexes

Open-domain experiments expect a local Wikipedia passage corpus:

```text
data/21MWiki/psgs_w100.tsv
data/21MWiki_bge/faiss_index_emb
data/21MWiki_bge/corpus_embeddings.npy
```

Prepare the corpus from the source used in your experiment environment, then build indexes locally:

```bash
bash cross_query_se/scripts/run_build_index.sh
```

These files can be large and are not redistributed here.
