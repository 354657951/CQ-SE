# Data Directory

This directory is intentionally empty in the public repository. Do not commit downloaded datasets, Wikipedia corpora, embeddings, FAISS indexes, or model caches.

## Benchmark QA Splits

From `exp/`, run:

```bash
python cross_query_se/scripts/download_datasets.py
```

The script downloads public benchmark splits and converts them to the expected local layout:

```text
data/
├── nq/nq-test-contriever.json
├── webqa/wq-test-contriever.json
├── TriviaQA/unfiltered-web-dev.json
├── hotpotqa/test_qa_pairs.json
└── SQuAD/validation-00000-of-00001.parquet
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
