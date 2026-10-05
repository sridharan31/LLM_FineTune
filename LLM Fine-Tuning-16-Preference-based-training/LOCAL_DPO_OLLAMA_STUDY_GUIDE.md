# Local DPO Fine-Tuning and Ollama: Study Guide

This guide explains `Preference_Aligned_Training_DPO_ollama.ipynb` from the perspective of a frontend developer. It covers what Python environments are, why packages live under `Anaconda3\envs\llm_engineering\Lib`, what each notebook stage does, where files are saved, and how the trained model reaches Ollama.

## 1. What This Project Does

The workflow is:

1. Read preference examples from a CSV file.
2. Download a pretrained Qwen model from Hugging Face.
3. Train a small LoRA adapter with DPO (Direct Preference Optimization).
4. Merge the adapter into a copy of the original model.
5. Convert that merged model to GGUF format using `llama.cpp`.
6. Import the GGUF file into Ollama and run it locally.

Ollama does not train the model in this project. Python, PyTorch, Transformers, PEFT, and TRL do the training. `llama.cpp` converts the trained result into a format that Ollama can import, and Ollama serves it for local use.

A simplified data flow:

```text
pharma_preference_data.csv
             |
             v
Qwen base model + DPO training (Python / TRL / PEFT)
             |
             v
LoRA adapter: runs/qwen2.5-1.5b-dpo/adapter
             |
             | merge adapter into a fresh copy of base model
             v
Merged Hugging Face model: runs/qwen2.5-1.5b-dpo/ollama-model-TIMESTAMP/
             |
             | convert_hf_to_gguf.py from llama.cpp
             v
GGUF export: runs/qwen2.5-1.5b-dpo/ollama-model-TIMESTAMP.f16.gguf
             |
             | ollama create
             v
Local Ollama model: pharma-qwen-dpo
```

## 2. Python Packages and the `Lib` Folder

The notebook is running Python from this Conda environment:

```text
C:\Users\sridh\anaconda3\envs\llm_engineering\python.exe
```

Python packages for that environment are installed below:

```text
C:\Users\sridh\anaconda3\envs\llm_engineering\Lib\site-packages\
```

`Lib` is Python's standard Windows folder for its libraries. `site-packages` is where third-party packages installed into this environment are placed. For example, `transformers` may be found in a path like:

```text
C:\Users\sridh\anaconda3\envs\llm_engineering\Lib\site-packages\transformers\
```

This is expected. It does not mean packages were installed into the notebook project or into Node.js. Jupyter runs a Python kernel, and that kernel uses this Python environment to find imports.

### Frontend / Node.js comparison

| Frontend / Node.js | This notebook / Python |
| --- | --- |
| `node` runs JavaScript | `python.exe` runs Python |
| `npm install` installs packages | `%pip install` or `python -m pip install` installs packages |
| `package.json` records dependencies | `requirements.txt` or `pyproject.toml` commonly records dependencies |
| project `node_modules/` stores packages | a Python environment's `Lib/site-packages/` stores packages |
| `import React from "react"` loads a package | `from transformers import ...` loads a package |
| a Node process runs the app | a Jupyter Python kernel runs notebook cells |

Node projects commonly install dependencies into a project-local `node_modules` directory. Python packages are often installed into a virtual environment or Conda environment instead. The important point is that both systems isolate dependencies; they just use different conventions and tools.

### Why use `%pip` in a notebook?

`%pip install ...` is an IPython/Jupyter command that installs into the Python interpreter backing the active notebook kernel. This avoids accidentally installing into a different Python that happens to be first on the terminal's PATH.

In a terminal, the equivalent is to call pip through the intended Python interpreter:

```cmd
"C:\Users\sridh\anaconda3\envs\llm_engineering\python.exe" -m pip install transformers datasets peft trl accelerate bitsandbytes sentencepiece
```

Check which Python a notebook is using with:

```python
import sys
print(sys.executable)
```

Check installed packages from that notebook with:

```python
%pip show transformers trl peft sentencepiece
```

Do not use `npm install` for these Python libraries. Ollama itself is a separate Windows application/CLI, and `llama.cpp` is a separate source-code checkout with its own Python conversion requirements.

## 3. What Each Package Does

- `torch` (PyTorch): tensor math, neural network layers, automatic gradients, GPU access, and model training.
- `transformers`: downloads and loads the tokenizer and Qwen model from Hugging Face; defines model and training utilities.
- `datasets`: reads the CSV or downloads an optional dataset from Hugging Face.
- `peft`: adds and saves LoRA adapters, reloads adapters, and merges them into base model weights.
- `trl`: supplies `DPOTrainer` and `DPOConfig` for Direct Preference Optimization.
- `accelerate`: helps Transformers/TRL place and run models on available hardware.
- `bitsandbytes`: provides low-bit quantization tools used for lower-memory CUDA training. The notebook uses it for 4-bit QLoRA when CUDA is available.
- `sentencepiece`: tokenizer support required by some model/conversion paths, including the observed GGUF conversion path here.
- `huggingface_hub`: downloads models and files from Hugging Face; it is also a dependency used by Transformers and Datasets.
- `llama.cpp`: provides `convert_hf_to_gguf.py`, which converts the merged Hugging Face model folder to GGUF.
- Ollama: a separate local model runtime. Its `ollama` command imports the GGUF and serves the model.

`%pip install -U ...` means install or upgrade those packages in the current notebook environment. Package upgrades can change APIs, which is why this notebook checks which DPO configuration options the installed TRL version accepts.

## 4. Preference Data: What DPO Learns From

The default dataset is:

```text
pharma_preference_data.csv
```

Each row needs three fields:

- `prompt`: the question or conversation context.
- `chosen`: the preferred answer.
- `rejected`: a less preferred answer to the same prompt.

DPO trains the model to score the chosen response above the rejected response for that prompt. It is not ordinary next-token training on a single ideal answer. The local CSV is intentionally tiny for demonstration; a handful of rows is not enough to create a dependable domain expert or medical assistant.

The notebook checks that these columns exist and that values are nonempty before training. It can optionally read a Hugging Face dataset if `DATA_SOURCE` is changed from `"local"` to `"huggingface"`.

## 5. Base Model Download and Loading

The notebook uses:

```python
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

`AutoTokenizer.from_pretrained(BASE_MODEL)` downloads the tokenizer files. `AutoModelForCausalLM.from_pretrained(BASE_MODEL, ...)` downloads and loads the model weights. Hugging Face normally caches these downloads rather than downloading them again every run.

The usual Windows cache is beneath:

```text
%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct
```

The notebook prints the cache folder. Environment variables such as `HF_HOME` or `HF_HUB_CACHE` can move it elsewhere. This Hugging Face cache is separate from the project's `runs` output folder and separate from Ollama's own model storage.

The notebook checks `torch.cuda.is_available()`:

- If CUDA is available, `USE_4BIT` is enabled and the model is loaded with 4-bit NF4 quantization to reduce GPU memory requirements.
- Without CUDA, it falls back to full-precision CPU loading and standard LoRA. This can be extremely slow and needs substantial system RAM.

The `tokenizer` converts text to token IDs the model can process and converts generated token IDs back to text. The `model` is the pretrained neural network. They are separate objects and both are needed for normal text workflows.

## 6. LoRA and DPO Training

Full fine-tuning changes every model weight. LoRA instead adds small trainable matrices to selected model layers. Most pretrained weights remain frozen, so the training result is much smaller than a full model checkpoint.

In this notebook:

- `LoraConfig` chooses which attention/MLP layers receive adapters and sets adapter size and dropout.
- `get_peft_model(model, lora_config)` attaches the trainable adapter to the base model.
- `DPOConfig` sets optimizer/training parameters such as learning rate, batch size, epochs, beta, and sequence length.
- `DPOTrainer` reads `prompt`, `chosen`, and `rejected`, calculates the DPO objective, and updates the adapter parameters.

The installed TRL version may not support every setting listed in the notebook. For example, TRL 1.14.1 did not accept `max_prompt_length`; the notebook detects that and leaves that option at the library default.

When training finishes, the notebook saves the adapter and tokenizer to:

```text
runs\qwen2.5-1.5b-dpo\adapter\
```

It also writes checkpoints under the Trainer output directory. A folder named `checkpoint-N` is an intermediate training checkpoint, not the Ollama model.

## 7. What the Output Folders Mean

Paths are relative to the notebook's working directory. The notebook prints absolute paths to help locate them.

### Hugging Face cache

```text
%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\
```

Downloaded original model and tokenizer files. This is a download cache, not the trained export.

### Adapter

```text
runs\qwen2.5-1.5b-dpo\adapter\
```

The small LoRA parameter files and adapter configuration. It is not a complete standalone model. It needs the original base model to run or merge.

### Training checkpoints

```text
runs\qwen2.5-1.5b-dpo\checkpoint-N\
```

Intermediate Trainer save points. These support resuming training and are not automatically the final Ollama import.

### Merged Hugging Face model

```text
runs\qwen2.5-1.5b-dpo\ollama-model-YYYYMMDD-HHMMSS-ffffff\
```

A full standalone model produced by merging the adapter into a fresh copy of Qwen. It contains `config.json`, tokenizer files, and one or more `model-xxxxx-of-xxxxx.safetensors` weight shards. These files are large, so the model is split into shards.

A fresh timestamped directory is used each time to avoid overwriting a file Windows or another process has open. An incomplete folder can be left behind if a save was interrupted; the notebook looks for an export containing both `config.json` and Safetensors files.

### GGUF export

```text
runs\qwen2.5-1.5b-dpo\ollama-model-YYYYMMDD-HHMMSS-ffffff.f16.gguf
```

A converted copy of the merged model. `.f16.gguf` indicates GGUF container format with mostly 16-bit floating-point weights. It uses several gigabytes of disk space. GGUF is the format passed to Ollama in this workflow.

### Ollama's installed model

After `ollama create`, Ollama manages its own model storage, normally under:

```text
%USERPROFILE%\.ollama\models\
```

Do not manually edit Ollama's internal blobs. Use `ollama list`, `ollama run`, and `ollama rm` to manage installed model names.

## 8. How the Merge Works

The adapter cannot be imported as the whole model because it only stores changes to selected layers. The merge cell:

1. Reloads the original base model from the Hugging Face cache, on CPU for the merge step.
2. Loads the trained adapter from `runs/qwen2.5-1.5b-dpo/adapter`.
3. Calls `merge_and_unload()`, which combines adapter deltas with the corresponding base weights. Conceptually, `W_merged = W_base + delta_W`.
4. Saves the resulting full model in Safetensors format under a new timestamped `ollama-model-*` folder.

The merge does not overwrite the original Hugging Face cache. It creates a separate export. Merging may need enough RAM to hold the full model; using CPU avoids depending on GPU memory, but does not eliminate system RAM requirements.

If the kernel restarts after training, the saved adapter allows the merge to continue without a `trainer` variable. If training never reached the adapter-save step, training must be run first.

## 9. Why GGUF Is Needed for Ollama Here

The merged model's `config.json` identifies its architecture as `Qwen2ForCausalLM`. The Ollama Safetensors importer reported that this architecture was unsupported in the user's installed Ollama path. The model can still be converted through `llama.cpp`:

1. `convert_hf_to_gguf.py` reads the merged Hugging Face model folder.
2. It writes a `.f16.gguf` file under the `runs` folder.
3. The notebook generates a `Modelfile` whose `FROM` points to that GGUF file.
4. `ollama create pharma-qwen-dpo -f <Modelfile>` imports it into Ollama's managed storage.
5. `ollama run pharma-qwen-dpo` loads it for local inference.

The `Modelfile` contains model-source and behavior/runtime settings. Its `SYSTEM` text guides responses, and `PARAMETER temperature 0.3` adjusts sampling. These instructions do not fine-tune weights; the DPO training happened earlier in Python.

### `llama.cpp` setup in Command Prompt

The user is using Windows Command Prompt (`cmd.exe`). In that terminal, environment variables use `%USERPROFILE%` syntax:

```cmd
git clone https://github.com/ggml-org/llama.cpp "%USERPROFILE%\llama.cpp"
"C:\Users\sridh\anaconda3\envs\llm_engineering\python.exe" -m pip install -r "%USERPROFILE%\llama.cpp\requirements.txt"
```

The notebook expects the converter at:

```text
C:\Users\sridh\llama.cpp\convert_hf_to_gguf.py
```

If `C:\Users\sridh\llama.cpp` already exists, do not clone into it again. Update it with Git or use the existing directory. The converter runs through the same Python executable as the active notebook kernel so its dependencies can be found.

## 10. Recommended Notebook Run Order

For a fresh run, execute the notebook from top to bottom:

1. Install Python dependencies with the `%pip` cell, then restart the notebook kernel if Jupyter requests it.
2. Run the setup/imports cell. Confirm it prints the expected base model and output path.
3. Load and validate the preference CSV.
4. Load the tokenizer and base model, then attach LoRA.
5. Create the DPO training configuration.
6. Run DPO training. Wait for it to finish and confirm it prints the adapter save path.
7. Save/merge the adapter. Confirm it prints a complete timestamped merged folder.
8. Convert the merged folder to GGUF and import it into Ollama. The cell can find a complete timestamped export if the setup cell reset the `MERGED_MODEL_DIR` variable.
9. Run the inference smoke test, or use Command Prompt: `ollama run pharma-qwen-dpo`.

For a model that has already been trained and merged, you do not need to retrain just to redo the conversion/import. Run the setup cell if you need its imports/constants, then run the GGUF/Ollama cell. It searches the output directory for the latest complete merged export.

## 11. Common Errors and What They Mean

- `ModuleNotFoundError: sentencepiece`: install it into the active notebook environment with `%pip install sentencepiece`, then retry the failed conversion cell.
- `NameError: MERGED_MODEL_DIR is not defined`: the kernel was restarted or setup cell was skipped. The current conversion cell initializes its own paths; otherwise run the setup/imports cell first.
- `No complete merged model export was found`: no timestamped merge folder contains both `config.json` and Safetensors weights. Run the merge successfully before conversion.
- Ollama says `unsupported ... architecture`: do not try to import the Safetensors folder directly; use the notebook's GGUF conversion route.
- `ollama create` fails: read its captured stderr. Check that Ollama is installed/running and that the generated Modelfile points to an existing `.gguf` file.
- `SafetensorError` says a file is being used by another process: write to a fresh timestamped merge folder and close programs that may be reading the previous folder. The notebook now creates a fresh export folder per merge.
- CUDA/4-bit errors: confirm PyTorch detects CUDA in the active environment. If not, the notebook will use CPU mode; CPU training/merging may be very slow and memory-heavy.

## 12. Expectations and Safety

This is a learning workflow, not a validated pharmaceutical or clinical model. The sample CSV contains only a few preference pairs, and a short training run is mainly useful for seeing the pipeline execute. Meaningful preference alignment requires substantially more diverse, high-quality, reviewed examples plus held-out evaluation.

Do not use this model to diagnose, prescribe, or make treatment decisions. Medical responses require review by qualified professionals and reliable sources.
