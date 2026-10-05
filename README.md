# Complete LLM Fine-Tuning

A practical, end-to-end study repository focused on modern large language model training, adaptation, alignment, quantization, and deployment. This project brings together notebooks, training configs, and reference materials covering the full life cycle of LLM engineering—from understanding why fine-tuning is difficult to using production-grade frameworks like Hugging Face, Unsloth, Axolotl, and LlamaFactory.

The goal of this repository is to provide a hands-on path for learning how to:

- train or adapt language models on custom data
- apply parameter-efficient fine-tuning methods like LoRA and QLoRA
- align models with human preferences using DPO, RLHF, ORPO, and GRPO
- improve efficiency through quantization and distillation
- work with multimodal and embedding-based tasks
- deploy models locally with Ollama and other inference workflows

## Why this repository exists

Large language models are no longer limited to research labs. In practice, the real value comes from adapting them for a domain, a task, or a user experience. This repository is designed to help you understand the key ideas behind that process and practice them using real tools and examples.

It combines:

- concept-heavy learning notebooks
- real model training workflows
- preference-tuning and alignment methods
- optimization techniques for memory and speed
- deployment-focused patterns for local and cloud inference

## What is covered

This repository spans the major areas of modern LLM development:

- LLM foundations and training intuition
- classical and modern fine-tuning strategies
- instruction tuning and domain adaptation
- LoRA / QLoRA / PEFT patterns
- preference optimization (DPO, ORPO, RLHF, GRPO)
- knowledge distillation
- model quantization (GPTQ, AWQ, GGUF, QAT)
- multimodal fine-tuning
- embedding fine-tuning and retrieval use cases
- LLM inference and deployment workflows
- practical frameworks such as Hugging Face, Unsloth, Axolotl, and LlamaFactory

## Repository structure

The directories in this repo are organized as modular learning tracks. Each folder typically contains a notebook or setup notes for a specific technique or concept.

| Folder | Focus |
| --- | --- |
| LLM Fine-Tuning-01 | Foundation module for the overall course and setup |
| LLM Fine-Tuning-02 | Continued foundational training content |
| LLM Fine-Tuning-04 | Additional early-stage LLM fine-tuning concepts |
| LLM Fine-Tuning-05-Why-Finetuning-Hard-in-LSTM | Explains why fine-tuning is challenging in recurrent sequences and LSTM-based systems |
| LLM Fine-Tuning-08-Huggingface | Hugging Face fundamentals and model workflows |
| LLM Fine-Tuning-09-Bert-finetuning | BERT fine-tuning for downstream tasks |
| LLM Fine-Tuning-10-11-knowledge-distillation | Knowledge distillation principles and LLM-oriented compression strategies |
| LLM Fine-Tuning-12-13-LLM-Quantization | GPTQ, AWQ, GGUF, QAT, and practical quantization patterns |
| LLM Fine-Tuning-14-Train-LLMs-on-Your-PDF-Text-Data -Domain-Specific-Fine-Tuning-with-HuggingFace | Training domain-specific models on custom text/PDF data |
| LLM Fine-Tuning-15-Instruction Fine-Tuning Explained -Domain-Specific Fine-Tuning with Hugging Face | Instruction tuning for domain-specific LLM behavior |
| LLM Fine-Tuning-16-Preference-based-training | Preference optimization and alignment workflows |
| LLM Fine-Tuning-17-Llama-Factory | LlamaFactory-based fine-tuning and efficient model training |
| LLM Fine-Tuning-18-unsloth | Unsloth fast finetuning workflows |
| LLM Fine-Tuning-19-Axolotl | Axolotl training setup, configs, and Docker-based examples |
| LLM Fine-Tuning-20-GPT-Finetuning | OpenAI API and GPT model finetuning workflows |
| LLM Fine-Tuning-21-GEMINI-Finetuning | Google Gemini fine-tuning examples |
| LLM Fine-Tuning-22-Finetune-Any-SLM | Finetuning small language models (SLMs) |
| LLM Fine-Tuning-23-Multimodal-LLM-Finetuning | Vision-language model tuning and multimodal dataset patterns |
| LLM Fine-Tuning-24-Embedding-and-Embedding-Finetuning | Embeddings and embedding customization |
| LLM Fine-Tuning-25-LoRA | LoRA and QLoRA practical workflows |
| LLM Fine-Tuning-26-RLHF | Reinforcement learning from human feedback basics |
| LLM Fine-Tuning-27-GRPO | Group Relative Policy Optimization methods |
| LLM Fine-Tuning-28-DPO | Direct Preference Optimization examples |
| LLM Fine-Tuning-29-ORPO | Odds Ratio Preference Optimization workflows |
| LLM Fine-Tuning-unsloth-vs-hf | Comparison between Unsloth and standard Hugging Face training approaches |
| LLM Finetuning-Crash-Course | Consolidated fast-track notebook covering the full small-course experience |

## Recommended learning path

For a structured study path, follow these stages:

1. Start with core concepts and model basics
   - LLM Fine-Tuning-05
   - LLM Fine-Tuning-08
   - LLM Finetuning-Crash-Course

2. Learn fine-tuning foundations
   - LLM Fine-Tuning-09
   - LLM Fine-Tuning-14
   - LLM Fine-Tuning-15

3. Understand efficiency and adaptation techniques
   - LLM Fine-Tuning-17
   - LLM Fine-Tuning-18
   - LLM Fine-Tuning-19
   - LLM Fine-Tuning-25

4. Study alignment and preference-based training
   - LLM Fine-Tuning-16
   - LLM Fine-Tuning-26
   - LLM Fine-Tuning-27
   - LLM Fine-Tuning-28
   - LLM Fine-Tuning-29

5. Move to optimization and deployment
   - LLM Fine-Tuning-12-13-LLM-Quantization
   - LLM Fine-Tuning-20
   - LLM Fine-Tuning-21
   - LLM Fine-Tuning-22
   - LLM Fine-Tuning-23
   - LLM Fine-Tuning-24

## Core topics in detail

### 1. Fine-tuning fundamentals

This repo begins from the idea that not every model should be trained from scratch. Fine-tuning allows a pretrained model to become specialized for a domain, task, or style. Examples include:

- domain-specific chatbots
- document-grounded assistants
- customer support models
- research or legal summarization models
- instruction-following models in a narrow domain

The repository emphasizes both the conceptual and practical side of this process.

### 2. Instruction tuning and domain adaptation

When the model is trained on domain-specific prompts and completions, it learns to answer in context and follow task-oriented instructions. The notebooks under the instruction tuning tracks show how to adapt general-purpose models to a narrower, more useful behavior.

### 3. Parameter-efficient fine-tuning (PEFT)

Training large models end-to-end can be prohibitively expensive. PEFT methods like LoRA and QLoRA modify a small set of extra parameters while keeping the main model weights mostly frozen. This dramatically lowers:

- GPU memory use
- training time
- storage requirements

The repository includes several practical examples for this.

### 4. Preference optimization and alignment

Modern LLM alignment is not only about next-token prediction. It also focuses on modeling human preferences, ranking, and quality. Methods in this repo include:

- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- RLHF (Reinforcement Learning from Human Feedback)
- GRPO (Group Relative Policy Optimization)

These methods teach a model to prefer higher-quality responses over lower-quality ones.

### 5. Quantization and compression

Large models are expensive to deploy. Quantization shrinks model size and memory footprint while keeping the model useful for inference. The repo covers:

- GPTQ
- AWQ
- GGUF / GGML
- QAT
- practical conversion workflows for local deployment

### 6. Embeddings and multimodal models

The repository also includes workflows that go beyond text-only models:

- embedding generation for semantic search and retrieval
- embedding fine-tuning for better similarity tasks
- vision-language fine-tuning
- multimodal data-prep and adaptation pipelines

### 7. Local deployment and inference

Many notebooks and guides focus on the transition from training to using the model in practice:

- local Ollama deployment
- Hugging Face model export
- inference with trained adapters
- conversion to GGUF for efficient local serving

## Tools and frameworks featured

This repository is built around the most common modern ecosystem pieces for LLM work.

- Hugging Face Transformers
- Datasets
- TRL (Transformer Reinforcement Learning)
- PEFT
- Accelerate
- Bitsandbytes
- SentencePiece
- Unsloth
- LlamaFactory
- Axolotl
- Ollama
- llama.cpp
- OpenAI fine-tuning interfaces
- Gemini model workflows

## Suggested environment

Most notebooks in this repo are best run in an environment with:

- Python 3.10+ (3.11 is usually a good target)
- a CUDA-capable NVIDIA GPU for training and fine-tuning tasks
- at least 16 GB RAM; more is strongly recommended for larger models
- Conda or a Python virtual environment
- Jupyter Notebook / JupyterLab / VS Code notebook support

## Typical setup

A standard environment for this repository often includes:

```bash
conda create -n llmft python=3.11
conda activate llmft

pip install -U pip
pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece
pip install jupyter notebook
```

Depending on the notebook, you may also need:

```bash
pip install unsloth
pip install axolotl
pip install llama-cpp-python
pip install flash-attn
```

Some folders contain specific instructions or config files that are more specialized than the base setup above.

## How to use this repository

1. Choose a topic based on your experience level.
2. Open the corresponding notebook in Jupyter or VS Code.
3. Install the required libraries and verify the runtime environment.
4. Read the notebook step by step rather than attempting a full end-to-end training run immediately.
5. Use smaller models or smaller datasets first when experimenting.
6. Move to larger fine-tuning and alignment tasks only after understanding the training loop.

## Practical advice for learning

- Start with smaller models before trying larger, production-scale fine-tuning.
- Understand the dataset format before training—prompt, chosen, rejected, and instructions may vary widely.
- Check GPU availability and memory limits before running training notebooks.
- Treat quantization, PEFT, and distillation as part of a real-world deployment workflow, not just academic topics.
- Prefer a single well-understood notebook or example before trying multiple frameworks at once.

## Repository theme

This repository is an applied learning archive for modern LLM work. It is especially relevant for students, practitioners, and engineers who want to understand:

- what fine-tuning actually changes in a model
- how preference optimization improves response quality
- when PEFT is preferable to full fine-tuning
- how to optimize models for local and low-resource deployment
- how to bring the theory of LLM training into working systems

## Final note

This project is intentionally broad. The value is not in a single notebook or one framework alone, but in seeing how the LLM ecosystem fits together: base models, data, adapters, preference tuning, memory optimization, and local serving.

If you are approaching this repository as a learning path, begin with the fundamentals, then move through PEFT, alignment, quantization, and deployment. That progression reflects how real-world LLM engineering is done.

## Quick index

- Base model understanding and foundations: LLM Fine-Tuning-05, LLM Finetuning-Crash-Course
- Fine-tuning on custom data: LLM Fine-Tuning-14, LLM Fine-Tuning-15
- Hugging Face workflow: LLM Fine-Tuning-08
- LoRA / QLoRA: LLM Fine-Tuning-25
- Preference tuning: LLM Fine-Tuning-16, LLM Fine-Tuning-28, LLM Fine-Tuning-29
- RLHF / GRPO: LLM Fine-Tuning-26, LLM Fine-Tuning-27
- Quantization and deployment: LLM Fine-Tuning-12-13-LLM-Quantization
- Multimodal and embeddings: LLM Fine-Tuning-23, LLM Fine-Tuning-24
- Framework-based training: LLM Fine-Tuning-17, LLM Fine-Tuning-18, LLM Fine-Tuning-19

This repository is best used as a structured set of learning modules rather than as a single monolithic project. Each folder contains a focused topic and the notebooks can often be explored independently.
