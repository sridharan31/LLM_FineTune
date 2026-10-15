# Fine-Tuning with Unsloth: Study Guide

This guide covers the Unsloth lesson. It brings together the two handwritten-notes PDFs in this folder (`handwritten-notes.pdf` and `unsloth-handwritten-notes.pdf`) and explains `unsloth_practical.ipynb` cell by cell. It covers what Unsloth is, why it is faster, how each piece of the code works, what the numbers mean, and how to export the trained model.

**Where to run the notebook:** Unsloth needs an NVIDIA GPU (CUDA). The notebook stops at its first GPU check if there is none. Run it on **Google Colab or Kaggle with a T4 GPU**. A Windows laptop with an AMD GPU (the one used for the DPO notebook in lesson 16) can't run it as written.

---

## 1. What Is Unsloth?

Unsloth is an open-source library that makes LLM fine-tuning **faster and lighter on GPU memory**. It runs the full pipeline:

```text
 1. Load model  ->  2. Quantize  ->  3. Train  ->  4. Inference
 (HF Hub)           (4/8/16-bit)     (LoRA,       (2x faster
                                      QLoRA,       generation)
                                      full, RL)
                                                       |
 7. Export  <-  6. Save  <-  5. Evaluate  <------------+
 (GGUF, Ollama,  (adapter or
  vLLM, Hub)      merged model)
```

The main claims (from the notes, quoting Unsloth's own published figures):

| Claim | Value |
| --- | --- |
| Training speed | 2x to 3x faster than standard Hugging Face training |
| GPU memory (VRAM) | 50% to 80% less (up to 80% for RL workloads) |
| Accuracy | No loss from the optimizations themselves: the math is exact |
| Hardware | Trains models on free GPUs (Colab / Kaggle) |

What that means in practice:

| Task | Standard Hugging Face | With Unsloth |
| --- | --- | --- |
| Generic large job | 10 hours, 40 GB VRAM | ~5 hours, 12-16 GB VRAM |
| Llama 3.1 (8B) | 24 GB VRAM, 2 hours | ~7-8 GB VRAM, ~1 hour |

> **"Exact math" needs one clarification.** Unsloth's kernels give the same results as the standard Hugging Face code, so the *speed-ups* lose no accuracy. Loading a model in **4-bit** (QLoRA) still rounds the weights, and that is a separate, small accuracy cost of quantization. It isn't caused by Unsloth.

---

## 2. Where Unsloth Sits in the Software Stack

Unsloth does not replace Hugging Face. It works *on top of* Transformers, PEFT and TRL, and swaps their slow parts for faster GPU code.

```text
+--------------------------------------------------------------+
|  Your code (the notebook)                                    |
+--------------------------------------------------------------+
|  Unsloth  (FastLanguageModel)                                |
|   - patches HF model classes with fast kernels               |
|   - injects LoRA adapters efficiently                        |
+--------------------------------------------------------------+
|  Hugging Face                                                |
|   Transformers (model classes) | PEFT (LoRA) | TRL (trainer) |
+--------------------------------------------------------------+
|  PyTorch  (tensors, autograd, optimizers)                    |
+--------------------------------------------------------------+
|  CUDA (NVIDIA C/C++)   |   Triton (Python-like GPU language) |
+--------------------------------------------------------------+
|  NVIDIA GPU                                                  |
+--------------------------------------------------------------+
```

The two PDFs draw this stack in different orders. The second PDF's order, shown above, is the accurate one. You write Unsloth calls, Unsloth drives Hugging Face classes, and those classes run on PyTorch, which calls CUDA. Unsloth also skips some layers and calls its own Triton kernels directly.

### Who does what

The notebook's Hinglish cells (cells 35 and 42) make one key point: **Unsloth is not the trainer.**

| Job | Done by |
| --- | --- |
| Load the model fast, quantize it, add LoRA, speed up the kernels, enable long context | **Unsloth** (`FastLanguageModel`) |
| Loop over the dataset, compute the loss, run backward, step the optimizer, accumulate gradients, log, save checkpoints | **TRL** (`SFTTrainer`) |
| Define the model architecture (`LlamaForCausalLM`, `MistralForCausalLM`, ...) | **Transformers** |
| Store model weights, config and tokenizer files | **Hugging Face Hub** |

The model is the same Hugging Face model, with the same `config.json`, tokenizer and weights. **The model does not change. Unsloth changes how it runs.**

### Why use `SFTTrainer` instead of the plain Hugging Face `Trainer`?

| HF `Trainer` | TRL `SFTTrainer` |
| --- | --- |
| General-purpose | Built for instruction tuning |
| Can't pack sequences | Packs sequences (see section 4.4) |
| Chat and Alpaca formats take extra code | Reads text and chat datasets directly |
| LoRA needs manual setup | Works with LoRA out of the box |

### What is the Hugging Face Hub?

It's an online registry for models, datasets and tokenizers. It borrows Git ideas (repos, commits, Git-LFS-style storage for large files) but runs on Hugging Face's own servers, not GitHub. A model repo is just files: `config.json`, `*.safetensors` weights, and tokenizer files. The model's code lives in the Transformers library.

---

## 3. CUDA and Triton

| | CUDA | Triton |
| --- | --- | --- |
| What it is | NVIDIA's low-level C/C++ platform for GPU programming | A Python-like language (DSL) for writing GPU kernels |
| Who made it | NVIDIA | OpenAI |
| How it runs | Compiled for NVIDIA GPUs | Compiled into optimized GPU code (on NVIDIA, the same kind of code CUDA produces) |
| Pros | Full control, massively parallel, very fast | Much easier to write, close to hand-tuned CUDA speed |

A **kernel** is one function that runs on the GPU, across thousands of threads in parallel. Unsloth's speed comes mainly from replacing PyTorch's general-purpose kernels with its own kernels written for LLM layers.

---

## 4. Why Is Unsloth Fast and Memory-Efficient?

These are deep internal optimizations, not configuration tricks.

### 4.1 Custom CUDA and Triton kernels

Standard PyTorch runs each operation of a transformer layer as a separate general-purpose kernel. Unsloth has its own kernels for the operations LLMs use most: RoPE, RMSNorm, attention, SwiGLU and cross-entropy loss.

### 4.2 Fused attention and MLP operations

The GPU's processing units are fast, but its main memory (HBM) is slow by comparison. Every separate kernel reads its input from memory and writes its output back. **Fusing** joins several steps into one kernel, so the intermediate results stay in fast on-chip memory.

```text
WITHOUT fusion (3 kernels, 6 trips to GPU memory)

  memory -> [ gate_proj ] -> memory -> [ SiLU ] -> memory -> [ multiply ] -> memory

WITH fusion (1 kernel, 2 trips to GPU memory)

  memory -> [ gate_proj + SiLU + multiply ] -> memory
```

Fewer trips to memory mean faster training, and fewer stored intermediate tensors mean less memory used.

### 4.3 Manual backward pass instead of generic autograd

PyTorch autograd records every operation in the forward pass and replays the graph backward. That works for any model, but it stores many intermediate tensors. For the main LLM layers, Unsloth hand-derives the gradient formulas and writes them as its own backward functions (custom `torch.autograd.Function`s). This uses fewer operations and stores less, and still gives the same gradients.

### 4.4 Automatic sequence packing

GPUs process fixed-size rows. Short examples padded out to `max_seq_length` waste most of the compute on padding tokens.

```text
WITHOUT packing (max_seq_length = 12, P = padding)

  row 1: [ s1 s1 s1 P  P  P  P  P  P  P  P  P ]   <- 75% wasted
  row 2: [ s2 s2 s2 s2 P  P  P  P  P  P  P  P ]
  row 3: [ s3 s3 P  P  P  P  P  P  P  P  P  P ]

WITH packing (E = end-of-sequence token)

  row 1: [ s1 s1 s1 E s2 s2 s2 s2 E s3 s3 E ]     <- almost no waste
```

The notebook turns this on with `packing=True`.

### 4.5 Smart gradient checkpointing

The backward pass normally needs every layer's activations, all kept in memory. **Gradient checkpointing** keeps only some of them and recomputes the rest during the backward pass. That trades a little extra compute for much less memory. Unsloth's version (`use_gradient_checkpointing="unsloth"`) also moves the saved activations to CPU RAM, so very long contexts fit.

### 4.6 Flash-Attention compatibility

Standard attention builds a full `seq_len x seq_len` score matrix, so its memory grows with the **square** of the context length. Flash Attention computes attention in tiles and never stores that whole matrix. This, plus checkpointing and RoPE scaling, is what makes long-context training possible.

---

## 5. Top Features

### 5.1 Long-context training (the most impressive feature)

Unsloth can train on contexts of 20K to 300K+ tokens. Approximate maximum context for **Llama 3.1 (8B)** with QLoRA (from the notes):

| GPU memory | Unsloth max context | Standard Hugging Face |
| ---: | ---: | ---: |
| 8 GB | ~3,000 tokens | - |
| 12 GB | ~21,000 tokens | Out of memory (OOM) |
| 16 GB | ~40,000 tokens | - |
| 24 GB | ~78,000 tokens | - |
| 80 GB | ~340,000 tokens | ~28,000 tokens |

This works because of the memory-efficient kernels, smart checkpointing, and **RoPE scaling**. RoPE (rotary position embedding) encodes each token's position. Scaling it lets a model work on sequences longer than it was pretrained on.

### 5.2 The full feature list

1. Long-context training
2. 50-80% less VRAM
3. Reinforcement learning (GRPO, GSPO) on little VRAM
4. Custom CUDA and Triton kernels
5. A complete pipeline, with export to many tools
6. Exact math (the speed-ups don't approximate anything)
7. Runs on free GPUs (Colab / Kaggle)
8. Support for many GPUs and operating systems

### 5.3 Supported model types

- Text-to-text (text generation, chat models): Llama, Mistral, Qwen, Gemma, Phi, DeepSeek
- Multimodal (text + image + audio)
- Vision-language and image-to-text (OCR): e.g. Qwen-VL
- Text-to-speech (TTS): e.g. **Orpheus**
- Speech-to-text (STT): e.g. **Whisper**
- Classic encoder models like BERT

> **Corrections to the handwritten notes:** both PDFs list Whisper under *Text-to-Speech*. Whisper is a **speech-to-text** model: it transcribes audio into text. Orpheus is the TTS example. "SST" in the notes should read **STT**.

### 5.4 Training methods

| Method | What it trains |
| --- | --- |
| Full fine-tuning | Every weight |
| LoRA (16-bit) | Small adapter matrices; base weights frozen in 16-bit |
| QLoRA (4-bit / 8-bit) | Small adapters; base weights frozen **and quantized** |
| FP8 training | Lower-precision training on supported GPUs |
| Reinforcement learning | Reward modeling, PPO, **GRPO**, **GSPO**, **DPO**, **ORPO**, KTO |

Unsloth is especially useful for RL. Methods like GRPO normally need large servers, and Unsloth's memory savings (up to 80%) let them run on one GPU or on Colab. Lessons 16 (DPO), 27 (GRPO) and 29 (ORPO) in this repo cover these methods.

---

## 6. Background: LoRA and QLoRA in One Page

The notebook trains with **QLoRA**: a 4-bit frozen base model plus small trainable LoRA adapters.

### LoRA

Instead of changing a large weight matrix `W` (size `d x k`), LoRA learns two thin matrices, `A` (`r x k`) and `B` (`d x r`), and adds their product:

```text
                 frozen                 trainable
             +-----------+        +-----+   +-----------------+
 W_new  =    |     W     |  +  a/r|  B  | x |        A        |
             |  (d x k)  |        |(d x r)  |     (r x k)     |
             +-----------+        |     |   +-----------------+
                                  +-----+
   r = rank (32 in the notebook), much smaller than d and k
   a = lora_alpha (32 in the notebook) -> scale a/r = 1.0
```

For one `2048 x 2048` matrix in TinyLlama:

- Full fine-tuning updates 2048 x 2048 = **4,194,304** weights.
- LoRA with r=32 updates 32 x (2048 + 2048) = **131,072** weights, about 3%.

### QLoRA

QLoRA stores the frozen `W` in **4-bit NF4** format, about 0.5 bytes per weight instead of 2. The LoRA matrices `A` and `B` stay in 16/32-bit and are the only part that trains.

| Format | Bytes per weight | TinyLlama 1.1B weights | Llama 3.1 8B weights |
| --- | ---: | ---: | ---: |
| FP32 | 4 | ~4.4 GB | ~32 GB |
| FP16 / BF16 | 2 | ~2.2 GB | ~16 GB |
| 4-bit (NF4) | ~0.5 | ~0.6 GB | ~4.5 GB |

### Where LoRA is attached: the target modules

Each transformer layer (TinyLlama has 22) contains an attention block and an MLP block. The notebook attaches LoRA to all seven linear layers in both blocks:

```text
            input hidden state (2048)
                     |
   +-----------------v------------------+
   |  ATTENTION                         |
   |   q_proj  "what am I looking for?" |  <- LoRA
   |   k_proj  "what do I contain?"     |  <- LoRA
   |   v_proj  "what info do I carry?"  |  <- LoRA
   |   softmax(QK^T) V                  |
   |   o_proj  combine heads            |  <- LoRA
   +-----------------+------------------+
                     |
   +-----------------v------------------+
   |  MLP (SwiGLU)                      |
   |   gate_proj  controls info flow    |  <- LoRA
   |   up_proj    2048 -> 5632          |  <- LoRA
   |   down_proj  5632 -> 2048          |  <- LoRA
   +-----------------+------------------+
                     |
               next layer (x22)
```

Adapting both attention **and** MLP gives better quality than attention alone. Lesson 16's DPO notebook adapts only q/k/v/o.

---

## 7. Notebook Walkthrough (`unsloth_practical.ipynb`)

### Overall flow

```text
 [3]  pip install torch, unsloth, transformers 4.56.2, trl 0.22.2
  |
 [4]  config: seed, TF32, max_seq_length=4096, 4-bit
  |
 [7]  FastLanguageModel.from_pretrained("unsloth/tinyllama-bnb-4bit")
  |        -> 4-bit frozen base model + tokenizer
 [11] FastLanguageModel.get_peft_model(r=32, 7 target modules)
  |        -> base + trainable LoRA adapters
 [12-19] inspect: trainable params, device, dtype, VRAM
  |
 [24] alpaca-cleaned -> 1,500 random rows -> one "text" field + EOS
  |
 [27-28] reset GPU stats, record start time and CPU RAM
  |
 [33] SFTTrainer(packing=True, adamw_8bit).train()
  |
 [34] print time, peak VRAM, CPU RAM
  |
 [36] FastLanguageModel.for_inference(model) -> generate
  |
 [38] model.save_pretrained("lora_model")  -> adapter files only
```

### 7.1 Install (cell 3)

```python
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
```

| Package | Role |
| --- | --- |
| `torch` | Deep learning engine: tensors, autograd, GPU kernels |
| `torchvision`, `torchaudio` | Vision and audio helpers, needed indirectly |
| `xformers` | Memory-efficient attention kernels |
| `--index-url .../cu128` | Downloads PyTorch builds made for **CUDA 12.8** |
| `transformers==4.56.2`, `trl==0.22.2` | Pinned versions that work with this Unsloth release |
| `--no-deps` | Installs TRL without letting it upgrade or downgrade other packages |

Pinning versions matters: Unsloth patches Transformers and TRL internals, so a mismatched version can break it.

### 7.2 Configuration (cell 4)

```python
SEED = 3407
random.seed(SEED)                 # Python shuffles
np.random.seed(SEED)              # NumPy preprocessing
torch.manual_seed(SEED)           # CPU weight init, CPU randomness
torch.cuda.manual_seed_all(SEED)  # GPU randomness: dropout, CUDA ops

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

max_seq_length = 4096
dtype = None
load_in_4bit = True
```

- **Seeds:** without a seed, every run shuffles and initializes differently. With one, runs repeat the same way. 3407 is a common choice from the paper *"torch.manual_seed(3407) is all you need"*. GPU kernels can still vary slightly between runs.
- **TF32:** lets FP32 matrix multiplies use the faster TensorFloat-32 format, with almost no quality loss. It only works on **Ampere or newer** GPUs (A100, L4, RTX 30xx+). On Colab's T4 it does nothing.
- **`max_seq_length = 4096`:** TinyLlama was pretrained with 2048 tokens. Unsloth applies RoPE scaling automatically to reach 4096.
- **`dtype = None`:** auto-detect. Unsloth picks BF16 on GPUs that support it (A100, L4) and FP16 otherwise (T4).
- **`load_in_4bit = True`:** QLoRA-style 4-bit base weights.

### 7.3 GPU check and imports (cells 5-6)

```python
assert torch.cuda.is_available(), "Please enable GPU runtime (Colab -> Runtime -> GPU)"

from unsloth import FastLanguageModel   # import unsloth FIRST
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
```

Import `unsloth` **before** `transformers` and `trl`, so it can patch them before they are used.

### 7.4 Load the base model (cell 7)

```python
BASE_MODEL_NAME = "unsloth/tinyllama-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=max_seq_length,   # long context via RoPE scaling
    dtype=dtype,                     # auto FP16 / BF16
    load_in_4bit=load_in_4bit,       # 4-bit frozen weights
)
```

`unsloth/tinyllama-bnb-4bit` is TinyLlama 1.1B **already quantized** to 4-bit with bitsandbytes and uploaded by Unsloth. Compared with quantizing the original model on load:

```text
 Plain HF:  download 2.2 GB FP16  ->  quantize to 4-bit on load  ->  ~0.6 GB in VRAM
 Unsloth:   download ~0.7 GB pre-quantized  ------------------->  ~0.6 GB in VRAM
```

The download is smaller and loading is faster. Cell 23 shows the result: about **826 MiB** of GPU memory in use after loading and adding LoRA, with an **0.99 GB peak**.

### 7.5 Add the LoRA adapters (cell 11)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=3407,
)
```

| Parameter | Value | Meaning | Why |
| --- | --- | --- | --- |
| `r` | 32 | Rank: the inner size of the `A` and `B` matrices | Larger = more capacity and more VRAM. 8/16/32/64 are common |
| `target_modules` | 7 layers | Linear layers that get LoRA | Covers attention + MLP for the best quality |
| `lora_alpha` | 32 | Scaling; the adapter output is multiplied by `alpha / r` | `alpha = r` gives scale 1.0. `2*r` is also common |
| `lora_dropout` | 0.0 | Dropout on the adapter input | Unsloth's fast path is optimized for 0 |
| `bias` | `"none"` | Don't train bias terms | Little gain, extra parameters |
| `use_gradient_checkpointing` | `False` | Recompute activations to save VRAM | Not needed for 1.1B. For 7B+ or long context, use `"unsloth"` |
| `random_state` | 3407 | Seed for the adapter's initial values | Reproducible results |

**How many trainable parameters?** Calculated from TinyLlama's shapes (hidden 2048, MLP 5632, k/v projections 256 wide because of grouped-query attention, 22 layers):

```text
per layer, r = 32:
  q_proj    32 x (2048 + 2048) = 131,072
  k_proj    32 x (2048 +  256) =  73,728
  v_proj    32 x (2048 +  256) =  73,728
  o_proj    32 x (2048 + 2048) = 131,072
  gate_proj 32 x (2048 + 5632) = 245,760
  up_proj   32 x (2048 + 5632) = 245,760
  down_proj 32 x (5632 + 2048) = 245,760
                                ---------
                                1,146,880  x 22 layers = 25,231,360  (~25.2M)
```

That is about **2.2% of TinyLlama's 1.1B parameters**. The other ~98% stay frozen.

### 7.6 Inspect the model (cells 12-19)

| Cell | Code | What it shows |
| --- | --- | --- |
| 12 | `model.print_trainable_parameters()` | Trainable vs total parameters and the percentage |
| 13 | loop over `named_parameters()` where `requires_grad` | The name of every LoRA matrix, e.g. `...q_proj.lora_A.default.weight` |
| 14 | count `numel()` by hand | Trainable / total / percent (see the caveat below) |
| 15 | `next(model.parameters()).device` | `cuda:0`: the model is on the GPU |
| 16 | `next(model.parameters()).dtype` | Data type of the first parameter |
| 17 | `isinstance(model, PeftModel)` | `True`: the model is now wrapped by PEFT |
| 18 | `model.peft_config` | The saved LoRA settings |
| 19 | `torch.cuda.memory_summary()` | VRAM report (the output is pasted in cell 23) |

> **Caveat for cell 14:** bitsandbytes stores two 4-bit weights in each byte, so `numel()` on a 4-bit weight counts about **half** the real number of weights. The hand-computed total and percentage come out wrong. Trust `print_trainable_parameters()`, which adjusts for this.

Cells 20-22 are commented-out experiments: a manual forward and backward pass to check that only LoRA weights get gradients, plus a quick generation test.

### 7.7 Prepare the dataset (cell 24)

```python
EOS_TOKEN = tokenizer.eos_token

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_data(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN)
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.shuffle(seed=3407).select(range(1500))
dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)
```

What happens to one row:

```text
 raw row (3 columns)                         formatted row (1 column)
 +-------------+---------------------+       +-----------------------------------+
 | instruction | "Give three tips..."|       | text: "Below is an instruction... |
 | input       | ""                  |  -->  |  ### Instruction:                 |
 | output      | "1. Eat well..."    |       |  Give three tips...               |
 +-------------+---------------------+       |  ### Input:                       |
                                             |                                   |
                                             |  ### Response:                    |
                                             |  1. Eat well...</s>"              |
                                             +-----------------------------------+
```

- **Alpaca format:** a fixed template with Instruction, Input and Response sections. The model learns to write the Response part.
- **`EOS_TOKEN` is required.** It teaches the model where an answer ends. Without it, the model never learns to stop and generates until `max_new_tokens`.
- **`shuffle(seed=3407).select(range(1500))`:** a reproducible random 1,500-row subset of ~52K rows, to keep the demo short.
- **`batched=True`:** `format_data` gets lists of rows, not one row at a time, which is faster.
- **`remove_columns`:** drops `instruction` / `input` / `output` and keeps only `text`.

### 7.8 Measure resources (cells 27-32)

```python
torch.cuda.empty_cache()               # free cached, unused GPU memory
torch.cuda.reset_peak_memory_stats()   # measure only THIS training run

process = psutil.Process()             # this notebook's Python process
train_start_time = time.time()         # seconds since 1 Jan 1970 (Unix time)
cpu_ram_before = process.memory_info().rss / 1024**3   # GB of CPU RAM in use
```

`time.time()` returns a large number like `1767225600.776` because it counts seconds since 1 January 1970 (UTC). Only the **difference** between two calls matters.

### 7.9 Train (cell 33)

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    packing=True,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 2 x 4 = 8
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        optim="adamw_8bit",
        logging_steps=10,
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer.train()
```

**Gradient accumulation** lets a small GPU act like it has a bigger batch:

```text
 step: [batch of 2] -> backward (keep gradients)
       [batch of 2] -> backward (add up)
       [batch of 2] -> backward (add up)
       [batch of 2] -> backward (add up) -> optimizer.step() -> zero gradients
       \_________________ effective batch of 8 __________________/
```

| Setting | Meaning |
| --- | --- |
| `per_device_train_batch_size=2` | 2 (packed) sequences per forward pass |
| `gradient_accumulation_steps=4` | One weight update every 4 forward passes |
| `num_train_epochs=1` | One pass over the 1,500 examples |
| `learning_rate=2e-5` | Size of each update (see the note below) |
| `warmup_ratio=0.1` | The learning rate ramps up from 0 over the first 10% of steps, which avoids unstable early updates |
| `optim="adamw_8bit"` | AdamW keeps two extra values per trainable weight; storing them in 8-bit saves memory |
| `logging_steps=10` | Print the loss every 10 steps |
| `report_to="none"` | Don't send metrics to Weights & Biases |

> **Learning rate note:** `2e-5` is a typical rate for *full* fine-tuning. For LoRA, Unsloth's own examples use **`2e-4`**, ten times higher, because the adapters start at zero and need larger steps. With 2e-5 and only one short epoch, the loss may barely move.
>
> **Version note:** in TRL 0.22, `dataset_text_field` and `packing` normally go in `SFTConfig`, and `tokenizer=` is called `processing_class=`. Unsloth's patched `SFTTrainer` still accepts the older form used here. If you upgrade and get a `TypeError`, move those settings into `SFTConfig`.

### 7.10 Training statistics (cell 34)

```python
training_time_sec = round(train_end_time - train_start_time, 2)
peak_gpu_vram_gb  = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
cpu_ram_used_gb   = round(cpu_ram_after - cpu_ram_before, 3)
```

- The time includes building the trainer and tokenizing the data, not just `train()`.
- `max_memory_reserved` counts memory PyTorch **reserved** (including its cache), so it is a bit higher than the memory tensors actually use (`max_memory_allocated`).

These are the numbers to compare against a plain Hugging Face run (section 9).

### 7.11 Inference (cell 36)

```python
FastLanguageModel.for_inference(model)   # switch to Unsloth's 2x faster generation mode

prompt = alpaca_prompt.format(
    "Continue the Fibonacci sequence",   # instruction
    "1, 1, 2, 3, 5, 8",                  # input
    "",                                  # response left EMPTY for the model to fill in
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():                    # no gradients needed for generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        use_cache=True,                  # reuse past keys/values (KV cache)
        do_sample=False,                 # greedy = same output every time
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

- **`for_inference`** switches Unsloth to its fast generation kernels. Call `FastLanguageModel.for_training(model)` before training again.
- **Use the same template as in training**, with the Response left empty. The model learned to continue right after `### Response:`.
- **KV cache (`use_cache=True`):** each new token reuses the attention keys and values already computed for earlier tokens instead of recomputing them.
- **`do_sample=False`:** always picks the most likely token. Good for demos and testing. Use sampling with a `temperature` for more varied text.

The commented-out cell 37 measures tokens per second. `torch.cuda.synchronize()` is needed there because GPU work runs asynchronously. Without it, the timer stops before the GPU has finished.

### 7.12 Save the adapter (cell 38)

```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

This saves **only the LoRA adapter**, not the base model:

```text
lora_model/
  adapter_config.json          <- base model name, r, alpha, target modules
  adapter_model.safetensors    <- the ~25M trained LoRA weights (~100 MB)
  tokenizer.json, tokenizer_config.json, special_tokens_map.json, ...
```

To use it later, Unsloth reads the base model's name from `adapter_config.json` and loads both:

```python
model, tokenizer = FastLanguageModel.from_pretrained("lora_model", max_seq_length=4096, load_in_4bit=True)
FastLanguageModel.for_inference(model)
```

---

## 8. Exporting: Merge, GGUF, Ollama

The notebook stops after saving the adapter. Inference and deployment tools (llama.cpp, Ollama, vLLM, SGLang, Open WebUI) don't train models; they need **one merged model file or folder**. Unsloth has one-line helpers for this:

```text
                    lora_model/ (adapter only)
                              |
       +----------------------+-----------------------+
       |                      |                       |
 save_pretrained_merged  save_pretrained_gguf    push_to_hub_merged /
 (merged_16bit)          (q4_k_m, q8_0, f16)     push_to_hub_gguf
       |                      |                       |
 HF folder (safetensors) model.gguf              Hugging Face Hub
       |                      |
 vLLM, SGLang, HF        llama.cpp, Ollama,
 Transformers            Open WebUI (via Ollama)
```

```python
# 1. Merge the adapter into the base model -> standard Hugging Face folder (vLLM, SGLang)
model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")

# 2. Merge + convert to GGUF in one step -> llama.cpp / Ollama
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")

# 3. Upload to the Hugging Face Hub (needs a write token)
# model.push_to_hub_merged("your-username/tinyllama-alpaca", tokenizer, save_method="merged_16bit", token="hf_...")
# model.push_to_hub_gguf("your-username/tinyllama-alpaca-gguf", tokenizer, quantization_method="q4_k_m", token="hf_...")
```

Then import the GGUF file into Ollama with a `Modelfile`:

```text
FROM ./gguf_model/<file>.gguf
PARAMETER temperature 0.2
```

```bash
ollama create tinyllama-alpaca -f Modelfile
ollama run tinyllama-alpaca
```

The GGUF file has no prompt template set, so prompts must use the same Alpaca template the model was trained on. Otherwise answers will be poor. You can add a `TEMPLATE` line to the `Modelfile` that applies it automatically.

**The same steps without Unsloth:** lesson 16's [Preference_Aligned_Training_DPO_ollama.ipynb](../LLM%20Fine-Tuning-16-Preference-based-training/Preference_Aligned_Training_DPO_ollama.ipynb) merges with `PeftModel.from_pretrained(base, adapter).merge_and_unload()`, converts with llama.cpp's `convert_hf_to_gguf.py`, and runs `ollama create`. `save_pretrained_gguf` does all three in one call.

| `quantization_method` | Size vs F16 | Quality |
| --- | --- | --- |
| `f16` | 100% | Same as the merged model |
| `q8_0` | ~53% | Almost no loss |
| `q4_k_m` | ~30% | Good; the usual default for local use |

---

## 9. Benchmark: Hugging Face vs Unsloth

The notes and cells 1, 39 and 40 set up a comparison on TinyLlama:

| | Hugging Face baseline | Unsloth |
| --- | --- | --- |
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `unsloth/tinyllama-bnb-4bit` |
| Loading | Transformers + bitsandbytes 4-bit on load | Pre-quantized 4-bit |
| LoRA | PEFT | Unsloth-optimized PEFT |
| Kernels | Standard PyTorch | Triton kernels, fused operations |
| Backward pass | PyTorch autograd | Hand-written backward |
| Attention | Standard | Faster attention |
| Packing | Manual | Automatic |

How to run a fair comparison:

1. Use the **same base model** in both runs. The baseline above uses the *Chat* model while Unsloth's is the *base* model, so for a strict test use `TinyLlama/TinyLlama_v1.1` or the same chat model in both.
2. Keep `r`, `target_modules`, the dataset, the batch size, the steps and `max_seq_length` the same.
3. Record time, peak VRAM and loss with the code from cells 27, 28 and 34.
4. Restart the runtime between runs so GPU memory starts clean.

---

## 10. When NOT to Use Unsloth

| Situation | Better choice |
| --- | --- |
| Large distributed training across many machines | DeepSpeed, FSDP, Megatron, Axolotl (lesson 19) |
| You want a no-code UI | LLaMA-Factory (lesson 17) |
| Classical ML (not LLMs) | scikit-learn, XGBoost, plain PyTorch |
| No NVIDIA GPU (e.g. AMD on Windows) | Plain HF + PEFT on CPU or DirectML, as in lesson 16 |

---

## 11. What You Can Build Next

| Area | Example models | Method |
| --- | --- | --- |
| Text-to-text chat / domain assistants | Llama, Mistral, Qwen, Gemma, Phi, DeepSeek | SFT with LoRA / QLoRA |
| Text-to-speech | Orpheus | SFT on audio tokens |
| Speech-to-text | Whisper (large) | SFT |
| Vision | Qwen-VL (e.g. Qwen3-VL) | Vision SFT |
| Reasoning | Qwen, Llama | **GRPO** (RL with reward functions) |
| Alignment | Any chat model | **DPO / ORPO** (preference pairs) |

---

## 12. Common Mistakes

| Mistake | Symptom | Fix |
| --- | --- | --- |
| No GPU runtime | `AssertionError` in cell 5 | Colab: Runtime -> Change runtime type -> T4 GPU |
| `unsloth` imported after `transformers` / `trl` | Warning; runs without the speed-ups | Import `unsloth` first, then restart the runtime |
| Missing `EOS_TOKEN` | Model never stops generating | Add `+ tokenizer.eos_token` to every training text |
| Different prompt format at inference time | Off-topic or rambling answers | Use the exact training template, with an empty Response |
| Forgot `for_inference(model)` | Slow generation | Call it before `generate()` |
| Learning rate too low for LoRA | Loss barely drops | Try `2e-4` |
| Mismatched package versions | Import errors or `TypeError` | Keep the pinned versions from cell 3 |
| Out of memory with long context | CUDA OOM | `use_gradient_checkpointing="unsloth"`, lower the batch size, raise gradient accumulation |
| Shipping only `lora_model/` | Ollama / vLLM can't load it | Merge or export first (section 8) |

---

## 13. Self-Check Questions

1. Which jobs does Unsloth do, and which does `SFTTrainer` do?
2. Why does fusing kernels speed up training even though the math is the same?
3. What does sequence packing save, and how does the model tell where one packed example ends?
4. With `r=32` and `lora_alpha=32`, what is the LoRA scaling factor?
5. Why does cell 14's hand count of total parameters disagree with `print_trainable_parameters()`?
6. What is the effective batch size in cell 33, and how is it reached on a small GPU?
7. Why must `EOS_TOKEN` be added to every training example?
8. What is in `lora_model/` after cell 38, and why can't Ollama load it directly?
9. Is "exact math, no accuracy loss" true for a QLoRA run? What part loses accuracy?
10. Which model in the notes is speech-to-text and which is text-to-speech?

<details>
<summary>Answers</summary>

1. Unsloth loads the model, quantizes it, adds LoRA, and supplies fast kernels and long-context support. `SFTTrainer` runs the training loop: data batches, loss, backward, optimizer steps, gradient accumulation, logging and checkpoints.
2. GPU memory transfers, not arithmetic, are the bottleneck. Fusion keeps intermediate results on-chip, so there are fewer reads and writes to slow GPU memory.
3. It saves the compute wasted on padding tokens. The EOS token between examples marks where each one ends.
4. `alpha / r = 32 / 32 = 1.0`.
5. bitsandbytes packs two 4-bit weights per byte, so `numel()` counts about half of each quantized weight's values.
6. 2 x 4 = 8. Gradients from 4 small batches are added up before each optimizer step.
7. So the model learns where an answer ends. Otherwise it keeps generating until `max_new_tokens`.
8. Only the LoRA adapter weights, its config and the tokenizer. Ollama needs one complete model (GGUF), so the adapter must first be merged into the base model and converted.
9. The kernels are exact, but 4-bit quantization of the frozen base weights is lossy. That loss comes from QLoRA, not from Unsloth.
10. Whisper is speech-to-text. Orpheus is text-to-speech.

</details>

---

## 14. Quick Reference

```python
from unsloth import FastLanguageModel                      # 1. import first

model, tokenizer = FastLanguageModel.from_pretrained(      # 2. load (4-bit)
    "unsloth/tinyllama-bnb-4bit", max_seq_length=4096, load_in_4bit=True)

model = FastLanguageModel.get_peft_model(model, r=32,      # 3. add LoRA
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32, lora_dropout=0, use_gradient_checkpointing="unsloth")

trainer = SFTTrainer(model=model, train_dataset=dataset,   # 4. train with TRL
    args=SFTConfig(dataset_text_field="text", packing=True, learning_rate=2e-4, ...))
trainer.train()

FastLanguageModel.for_inference(model)                     # 5. fast inference
model.save_pretrained("lora_model")                        # 6. save the adapter
model.save_pretrained_gguf("gguf", tokenizer, quantization_method="q4_k_m")  # 7. export
```
