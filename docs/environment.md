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

## Hardware Profiles

| Profile | Setting |
| --- | --- |
| 7B scale | 1 x 80GB-class GPU. NVIDIA RTX 6000D reports 85,651 MiB GPU memory through `nvidia-smi`. |
| 72B scale | Multi-GPU 80GB-class setup. The 72B vLLM scripts use tensor parallelism with `tensor_parallel_size=4`; full baseline scripts assume an 8-GPU node so generation, retrieval, NLI, and 72B model sharding have enough device capacity. |
| Data disk | Large data disk recommended for 21MWiki, embeddings, outputs, and model caches. A 350 GB data disk is sufficient for a compact local setup; full-scale artifacts require more storage. |
| OS image | Ubuntu 22.04 LTS family |

## Python And CUDA

| Component | Version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8 |

## Core Python Packages

| Package | Version |
| --- | --- |
| vLLM | 0.11.0 |
| transformers | 4.57.1 |
| sentence-transformers | 5.6.0 |
| scikit-learn | 1.7.2 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| FAISS | 1.14.3 |
| FlagEmbedding | project dependency |

## Models

| Role | Model |
| --- | --- |
| Generator / answer model, 7B scale | `Qwen/Qwen2.5-7B-Instruct` |
| Generator / answer model, 72B scale | `Qwen/Qwen2.5-72B-Instruct` |
| Dense retriever encoder | `BAAI/bge-large-en-v1.5` |
| NLI semantic clustering | `microsoft/deberta-v2-xlarge-mnli` |

Users can either rely on the Hugging Face cache or point the scripts to local
model directories when using a local model mirror or pre-downloaded weights.
