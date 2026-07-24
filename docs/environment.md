# Runtime Environment

This file records the runtime environment and model dependencies used by the
CQ-SE pipeline. It is intended as a practical reference for users who want to set
up the project on a GPU instance.

## Scope

The environment below supports the main CQ-SE components:

- Qwen2.5-7B-Instruct and Qwen2.5-72B-Instruct served locally through vLLM;
- BGE-large-en-v1.5 dense retrieval;
- DeBERTa-v2-xlarge-MNLI semantic clustering;
- stage-by-stage CQ-SE execution on QA benchmarks;
- 21MWiki passage embeddings, depending on available storage and runtime budget.

Compatible versions may also work. The versions below are provided as a known
reference configuration.

## Python And CUDA

| Component | Version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8 |

## Python Dependencies

| Package | Role in the project | Version note |
| --- | --- | --- |
| `torch` | GPU tensor runtime used by retrieval, NLI, and local model execution. | `2.8.0+cu128` |
| `vllm` | Local high-throughput generation backend for Qwen models. | `0.11.0` |
| `transformers` | Tokenizers and model loading for Qwen and DeBERTa. | `4.57.1` |
| `accelerate` | Multi-GPU model loading support for large-model baseline stages. | Compatible with the Transformers stack |
| `sentence-transformers` | Loads BGE-large-en-v1.5 and encodes queries/passages. | `5.6.0` |
| `FlagEmbedding` | BGE-related embedding utilities used by project checks/tools. | Required |
| `faiss-cpu` | Builds and reads dense vector indexes where FAISS is used. | FAISS `1.14.3` |
| `numpy` | Embedding arrays, matrix operations, and saved `.npy` files. | `2.2.6` |
| `scipy` | Statistical analysis and supporting numerical routines. | `1.15.3` |
| `scikit-learn` | Metrics such as AUROC and supporting evaluation utilities. | `1.7.2` |
| `pandas` | Tabular result processing and analysis. | `2.3.3` |
| `pyarrow` | Dataset/table IO support used with benchmark data processing. | Required |
| `datasets` | Downloads or loads Hugging Face benchmark datasets. | Required |
| `matplotlib` | Figure generation for analysis scripts. | Required for plots |
| `seaborn` | Statistical plot styling for analysis scripts. | Required for plots |
| `tqdm` | Progress bars for indexing, generation, and filtering scripts. | Required |
| `regex` | Text normalization in QA evaluation utilities. | Required |
| `python-dotenv` | Allows scripts to read local `.env` files when users choose to create one. | Required |
| `openai` | Optional OpenAI-compatible client for API-based query perturbation generation. | Required only for API mode |

## Models

| Role | Model |
| --- | --- |
| Generator / answer model, 7B scale | `Qwen/Qwen2.5-7B-Instruct` |
| Generator / answer model, 72B scale | `Qwen/Qwen2.5-72B-Instruct` |
| Dense retriever encoder | `BAAI/bge-large-en-v1.5` |
| NLI semantic clustering | `microsoft/deberta-v2-xlarge-mnli` |

## Runtime Variables

| Variable | Purpose |
| --- | --- |
| `HF_HOME` | Optional Hugging Face cache root for downloaded models and datasets. |
| `LEMMA_MAAS_BASE_URL` | Optional OpenAI-compatible endpoint for API-based query perturbation generation. |
| `LEMMA_MAAS_API_KEY` | Optional API key for the same endpoint. Do not commit real keys. |

Users can either rely on the Hugging Face cache or point the scripts to local
model directories when using a local model mirror or pre-downloaded weights.
