# AI Security Lab - Usage Guide & FAQ

> **Lab Root:** `/opt/ai-security-lab`
> **Repos:** `/opt/ai-security-lab/repos/`
> **Python Venvs:** `/opt/ai-security-lab/venvs/`

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Jupyter Notebook Environment](#2-jupyter-notebook-environment)
3. [Ollama (Local LLM Runtime)](#3-ollama)
4. [Garak (LLM Vulnerability Scanner)](#4-garak)
5. [PyRIT (AI Red-Teaming Toolkit)](#5-pyrit)
6. [PromptFoo (LLM Evaluation & Red-Teaming)](#6-promptfoo)
7. [LM Eval Harness (Benchmark Evaluations)](#7-lm-eval-harness)
8. [Giskard (AI Testing Framework)](#8-giskard)
9. [CleverHans (Adversarial ML)](#9-cleverhans)
10. [Lab Environments](#10-lab-environments)
    - [AI Red-Teaming Playground Labs](#101-ai-red-teaming-playground-labs)
    - [Damn Vulnerable LLM Agent](#102-damn-vulnerable-llm-agent)
    - [Vulnerable LLMs](#103-vulnerable-llms)
    - [FinBot CTF](#104-finbot-ctf)
    - [MAESTRO](#105-maestro)
    - [Vulnerable MCP Servers Lab](#106-vulnerable-mcp-servers-lab)
11. [General FAQ](#11-general-faq)

---

## 1. Overview & Architecture

This lab installs each Python tool into its **own virtual environment** under `/opt/ai-security-lab/venvs/` to prevent dependency conflicts. Node.js tools are installed globally or inside their repo directories.

**Activation pattern for any Python tool:**

```bash
source /opt/ai-security-lab/venvs/<tool-name>/bin/activate
# ... use the tool ...
deactivate
```

**Quick reference table:**

| Tool | Type | Activate / Run |
|------|------|----------------|
| JupyterLab | Python venv | `source /opt/ai-security-lab/venvs/jupyter/bin/activate && jupyter lab` |
| Ollama | System binary | `ollama` |
| Garak | Python venv | `source /opt/ai-security-lab/venvs/garak/bin/activate` |
| PyRIT | Python venv | `source /opt/ai-security-lab/venvs/pyrit/bin/activate` |
| PromptFoo | Global npm | `promptfoo` |
| LM Eval Harness | Python venv | `source /opt/ai-security-lab/venvs/lm-eval-harness/bin/activate` |
| Giskard | Python venv | `source /opt/ai-security-lab/venvs/giskard/bin/activate` |
| CleverHans | Python venv | `source /opt/ai-security-lab/venvs/cleverhans/bin/activate` |

---

## 2. Jupyter Notebook Environment

JupyterLab is the recommended way to work with the Python-based tools interactively. The setup script installs JupyterLab in its own venv and **registers every tool venv as a separate Jupyter kernel**, so you can switch between tools without leaving the notebook interface.

### 2.1 Starting JupyterLab

```bash
source /opt/ai-security-lab/venvs/jupyter/bin/activate

# Launch with the lab notebooks directory
jupyter lab --notebook-dir=/opt/ai-security-lab/notebooks

# Or launch on a specific port / allow remote access
jupyter lab --notebook-dir=/opt/ai-security-lab/notebooks \
    --ip=0.0.0.0 --port=8888 --no-browser
```

**Access:** `http://localhost:8888` (token shown in terminal output).

### 2.2 Available Kernels

The setup script registers these kernels automatically:

| Kernel Name | Tool | Use For |
|-------------|------|---------|
| Python (Garak) | Garak | LLM vulnerability scanning scripts |
| Python (PyRIT) | PyRIT | AI red-teaming attacks & orchestrations |
| Python (Giskard) | Giskard | AI testing, bias detection, RAG evaluation |
| Python (LM Eval Harness) | LM Eval Harness | Custom benchmark scripting |
| Python (CleverHans) | CleverHans | Adversarial ML attacks on image models |
| Python (Damn Vulnerable LLM Agent) | DVLA | Experimenting with the vulnerable agent |

**Switching kernels:** Click the kernel name in the top-right corner of any notebook, or use `Kernel > Change Kernel` from the menu bar. Each kernel runs in its own isolated venv with only that tool's dependencies.

### 2.3 Starter Notebook

The setup script creates `00_lab_overview.ipynb` in `/opt/ai-security-lab/notebooks/`. It contains:

- Ollama connectivity check
- Import verification cells for each tool
- A sample cell that chats with a local Ollama model

Open it after launch to verify your installation.

### 2.4 Creating Your Own Notebooks

```bash
# Notebooks directory
ls /opt/ai-security-lab/notebooks/

# Create a new notebook from the JupyterLab launcher (+ button)
# or copy an existing one:
cp /opt/ai-security-lab/notebooks/00_lab_overview.ipynb \
   /opt/ai-security-lab/notebooks/my_garak_scan.ipynb
```

When creating a new notebook, select the kernel matching the tool you want to use from the launcher or the kernel picker.

### 2.5 Running JupyterLab as a systemd Service

For persistent access (e.g., headless server):

```bash
sudo tee /etc/systemd/system/jupyterlab.service << 'EOF'
[Unit]
Description=JupyterLab AI Security Lab
After=network.target ollama.service

[Service]
Type=simple
User=YOUR_USERNAME
Environment="PATH=/opt/ai-security-lab/venvs/jupyter/bin:/usr/local/bin:/usr/bin"
ExecStart=/opt/ai-security-lab/venvs/jupyter/bin/jupyter lab \
    --notebook-dir=/opt/ai-security-lab/notebooks \
    --ip=127.0.0.1 --port=8888 --no-browser
Restart=on-failure
WorkingDirectory=/opt/ai-security-lab/notebooks

[Install]
WantedBy=multi-user.target
EOF

# Replace YOUR_USERNAME, then:
sudo systemctl daemon-reload
sudo systemctl enable --now jupyterlab
```

### 2.6 Installing Additional Packages Inside a Kernel

From within a notebook cell, install packages into the active kernel's venv:

```python
# This installs into whichever kernel is currently active
import sys
!{sys.executable} -m pip install some-package
```

> **Do not use** bare `!pip install` as that may install into the Jupyter venv instead of the tool venv.

### 2.7 FAQ - Jupyter

**Q: JupyterLab shows "No module named X" even though the tool is installed.**
A: You are likely using the wrong kernel. Check the kernel name in the top-right corner. Switch to the correct tool kernel via `Kernel > Change Kernel`.

**Q: A kernel fails to start or shows "Dead kernel".**
A: The underlying venv may be broken. Rebuild it:
```bash
# Remove and recreate (example: garak)
rm -rf /opt/ai-security-lab/venvs/garak
python3.11 -m venv /opt/ai-security-lab/venvs/garak
source /opt/ai-security-lab/venvs/garak/bin/activate
pip install --upgrade pip setuptools wheel
pip install garak ipykernel
python -m ipykernel install --name garak --display-name "Python (Garak)" \
    --prefix /opt/ai-security-lab/venvs/jupyter
deactivate
```

**Q: I don't see all the kernels in JupyterLab.**
A: Kernels are registered under the Jupyter venv prefix. Verify with:
```bash
source /opt/ai-security-lab/venvs/jupyter/bin/activate
jupyter kernelspec list
```
If a kernel is missing, re-register it:
```bash
source /opt/ai-security-lab/venvs/<tool>/bin/activate
pip install ipykernel
python -m ipykernel install --name <tool> --display-name "Python (<Tool>)" \
    --prefix /opt/ai-security-lab/venvs/jupyter
```

**Q: Jupyter runs out of memory.**
A: Each active kernel is a separate Python process. Close unused notebooks (`File > Close and Shut Down Notebook`) or shut down idle kernels via the sidebar's running-kernels panel. Avoid loading large LLMs (e.g., HuggingFace 7B models) directly in a notebook if Ollama is already loaded.

**Q: How do I set a password instead of using tokens?**
A:
```bash
source /opt/ai-security-lab/venvs/jupyter/bin/activate
jupyter lab password
# Enter a password, then restart JupyterLab
```

**Q: Async/await code fails with `RuntimeError: This event loop is already running`.**
A: Jupyter runs its own event loop. Install and apply `nest_asyncio` at the top of your notebook:
```python
import nest_asyncio
nest_asyncio.apply()
```
This is commonly needed for PyRIT, Giskard, and other async-heavy tools.

**Q: How do I access JupyterLab remotely (e.g., SSH into the VM)?**
A: Option 1 - SSH tunnel (recommended):
```bash
# On your local machine:
ssh -L 8888:localhost:8888 user@vm-ip
# Then open http://localhost:8888 in your browser
```
Option 2 - Bind to all interfaces (less secure):
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
# Access via http://vm-ip:8888
```
Always set a password if binding to `0.0.0.0`.

**Q: Can I use Jupyter Notebook (classic) instead of JupyterLab?**
A: Yes, both are installed:
```bash
source /opt/ai-security-lab/venvs/jupyter/bin/activate
jupyter notebook --notebook-dir=/opt/ai-security-lab/notebooks
```

---

## 3. Ollama

Ollama is the local LLM runtime that many other tools in this lab depend on. Start it first.

### 3.1 Starting Ollama

```bash
# Start as a systemd service (recommended)
sudo systemctl start ollama
sudo systemctl enable ollama     # auto-start on boot

# Or start manually in foreground
ollama serve
```

### 3.2 Pulling Models

```bash
ollama pull llama3.2:1b          # Small, fast (good for testing)
ollama pull mistral-nemo         # Reliable for DVLA lab
ollama pull qwen3:0.6b           # Lightweight
```

### 3.3 Common Commands

```bash
ollama list                      # Show downloaded models
ollama ps                        # Show running models (memory usage)
ollama run mistral-nemo          # Interactive chat session
ollama stop mistral-nemo         # Unload model from memory
ollama rm <model>                # Delete a model from disk
```

### 3.4 Exposing Ollama to Other Tools

By default Ollama listens on `127.0.0.1:11434`. Most tools connect there automatically. To allow Docker containers or network access:

```bash
# Temporary: listen on all interfaces
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Permanent: edit the systemd service
sudo systemctl edit ollama
# Add under [Service]:
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 3.5 CPU-Only Mode

If you have no GPU or want to force CPU operation:

```bash
OLLAMA_NUM_GPU=-1 ollama serve
```

Or permanently via systemd:

```bash
sudo systemctl edit ollama
# Add: Environment="OLLAMA_NUM_GPU=-1"
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Stick to small models on CPU: `phi`, `tinyllama`, `llama3.2:1b`.

### 3.6 FAQ - Ollama

**Q: Ollama fails to start with "address already in use".**
A: Another instance is already running. Check with `sudo lsof -i :11434` and stop it, or use a different port: `OLLAMA_HOST=127.0.0.1:8080 ollama serve`.

**Q: Model download fails or is corrupted.**
A: Remove the model with `ollama rm <model>` and re-pull it. Ensure you have enough disk space with `df -h`.

**Q: GPU is not detected.**
A: Run `nvidia-smi` to verify drivers are installed. Ollama requires NVIDIA driver 531+ with CUDA compute capability 5.0+. Check logs with `journalctl -u ollama -f` and enable debug mode: `OLLAMA_DEBUG=1 ollama serve`.

**Q: Model runs out of memory.**
A: Check `ollama ps` for memory usage. Use quantized or smaller models. A 7B Q4 model needs roughly 4GB RAM; a 13B model needs roughly 8GB.

**Q: Docker containers can't connect to Ollama.**
A: Either set `OLLAMA_HOST=0.0.0.0:11434` or run Docker with `--network=host`. From Docker containers, use `host.docker.internal:11434` as the Ollama URL.

---

## 4. Garak

NVIDIA Garak is an LLM vulnerability scanner that probes models for jailbreaks, prompt injection, encoding attacks, harmful content generation, and more.

### 4.1 Starting Garak

```bash
source /opt/ai-security-lab/venvs/garak/bin/activate

# List what's available
python -m garak --list_probes
python -m garak --list_generators
python -m garak --list_detectors
```

### 4.2 Running Scans

**Scan a local Ollama model:**

```bash
# Ensure Ollama is running first
python -m garak --target_type ollama --target_name mistral-nemo --probes dan
```

**Scan with specific probes:**

```bash
# DAN jailbreak probes
python -m garak -t ollama -n mistral-nemo -p dan

# Prompt injection
python -m garak -t ollama -n mistral-nemo -p promptinject

# Encoding-based attacks
python -m garak -t ollama -n mistral-nemo -p encoding

# Multiple probes at once
python -m garak -t ollama -n mistral-nemo -p "dan,encoding,promptinject"
```

**Scan an OpenAI model:**

```bash
export OPENAI_API_KEY="sk-..."
python -m garak --target_type openai --target_name gpt-4 --probes dan
```

**Scan a HuggingFace model:**

```bash
python -m garak --target_type huggingface --target_name gpt2 --probes encoding
```

### 4.3 Useful Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-g N` | Generations per prompt | `-g 10` |
| `--parallel_attempts N` | Parallel attempts | `--parallel_attempts 5` |
| `--config FILE` | YAML config file | `--config scan.yaml` |
| `-v` | Increase verbosity | `-v -v` |
| `--report_prefix NAME` | Custom report name | `--report_prefix my_scan` |
| `--probe_tags TAG` | Filter by OWASP tag | `--probe_tags "owasp:llm01"` |

### 4.4 Viewing Results

```bash
# Reports are saved as JSONL files
ls ~/.local/share/garak/garak_runs/

# View a report
cat ~/.local/share/garak/garak_runs/garak.*.report.jsonl | python -m json.tool | less
```

### 4.5 FAQ - Garak

**Q: `ConnectionError` when scanning Ollama model.**
A: Ollama must be running. Start it with `ollama serve` in another terminal. Verify with `curl http://127.0.0.1:11434/api/tags`.

**Q: Scan runs very slowly.**
A: Reduce generations per prompt with `-g 2`. Use a subset of probes instead of all. For remote models, add `--parallel_attempts 5`. For Ollama, use a smaller model.

**Q: `No such model` error.**
A: The model name must match exactly what Ollama has. Run `ollama list` to see available names.

**Q: Empty responses from the model.**
A: Garak retries up to 3 times. If persistent, verify the model works with `ollama run <model> "test"`. Try increasing timeout in a YAML config:
```yaml
plugins:
  generators:
    ollama.OllamaGeneratorChat:
      timeout: 120
```

**Q: Out of memory when scanning HuggingFace models.**
A: Use smaller models (`distilgpt2`) or use the HuggingFace Inference API instead: `--target_type huggingface.InferenceAPI --target_name gpt2`.

---

## 5. PyRIT

Microsoft's Python Risk Identification Toolkit for generative AI red-teaming. Supports single-turn and multi-turn attacks including Crescendo, TAP, and PAIR algorithms.

### 5.1 Starting PyRIT

```bash
source /opt/ai-security-lab/venvs/pyrit/bin/activate
```

PyRIT is primarily used as a Python library in scripts or Jupyter notebooks.

### 5.2 Configuration

Create a `.env` file at `~/.pyrit/.env`:

```bash
mkdir -p ~/.pyrit
cat > ~/.pyrit/.env << 'EOF'
# For OpenAI
OPENAI_CHAT_KEY=sk-your-key
OPENAI_CHAT_ENDPOINT=https://api.openai.com/v1/chat/completions
OPENAI_CHAT_MODEL=gpt-4o

# For Ollama (local)
# OPENAI_CHAT_KEY=local
# OPENAI_CHAT_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions
# OPENAI_CHAT_MODEL=mistral-nemo
EOF
```

### 5.3 Basic Usage (Python Script)

```python
from pyrit import initialize_pyrit_async
from pyrit.common import SimpleInitializer
from pyrit.chat import OpenAIChatTarget

# Initialize PyRIT
await initialize_pyrit_async(
    memory_db_type="InMemory",
    initializers=[SimpleInitializer()]
)

# Create a target pointing at your LLM
target = OpenAIChatTarget(
    api_key="local",
    endpoint="http://127.0.0.1:11434/v1/chat/completions",
    model="mistral-nemo"
)
```

### 5.4 Using with Ollama

```bash
# Ollama exposes an OpenAI-compatible API
# Set these environment variables:
export OPENAI_CHAT_KEY=local
export OPENAI_CHAT_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions
export OPENAI_CHAT_MODEL=mistral-nemo
```

### 5.5 FAQ - PyRIT

**Q: `ModuleNotFoundError: No module named 'pyrit'`**
A: Ensure you activated the venv: `source /opt/ai-security-lab/venvs/pyrit/bin/activate`.

**Q: `401 Unauthorized` when connecting to an API.**
A: Verify your API key. For Ollama, any non-empty string works as the key (e.g., `local`).

**Q: Connection refused to Ollama.**
A: Ensure Ollama is running and the endpoint URL is correct: `http://127.0.0.1:11434/v1/chat/completions`.

**Q: `429 Too Many Requests` rate-limiting.**
A: Set `max_requests_per_minute=10` in your orchestrator configuration. Ollama has no rate limits (local execution).

**Q: How do I run the example notebooks?**
A: Install Jupyter inside the venv, then launch:
```bash
source /opt/ai-security-lab/venvs/pyrit/bin/activate
pip install jupyter
jupyter notebook /opt/ai-security-lab/repos/  # if PyRIT notebooks are cloned
```

---

## 6. PromptFoo

PromptFoo is a Node.js-based LLM evaluation and red-teaming framework that runs entirely locally.

### 6.1 Getting Started

```bash
# Initialize a new project
mkdir ~/my-redteam && cd ~/my-redteam
promptfoo init

# Or initialize specifically for red-teaming
promptfoo redteam init
```

### 6.2 Running Evaluations

```bash
# Run evaluation against configured prompts and providers
promptfoo eval

# View results in browser
promptfoo view
```

### 6.3 Red-Teaming

```bash
# Interactive setup wizard
promptfoo redteam setup

# Run full red-team workflow (generate + evaluate)
promptfoo redteam run

# Generate adversarial test cases only
promptfoo redteam generate

# View findings report
promptfoo redteam report

# List available attack plugins
promptfoo redteam plugins
```

### 6.4 Configuring with Ollama

In your `promptfooconfig.yaml`:

```yaml
providers:
  - id: ollama:chat:mistral-nemo
    config:
      base_url: http://localhost:11434
      temperature: 0.7
```

### 6.5 Configuring with OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

```yaml
providers:
  - openai:gpt-4o
```

### 6.6 FAQ - PromptFoo

**Q: `OPENAI_API_KEY not found` error.**
A: Either set the environment variable or specify it in config. For Ollama you don't need an OpenAI key.

**Q: Evaluations time out.**
A: Increase timeouts:
```bash
export PROMPTFOO_EVAL_TIMEOUT_MS=60000       # 60s per request
export PROMPTFOO_MAX_EVAL_TIME_MS=600000     # 10 min total
```

**Q: `npx` or `node` command not found.**
A: Node.js may not be in your PATH. Try: `export PATH="/usr/bin:$PATH"` or run `node --version` to verify installation.

**Q: Python provider not working.**
A: Set the Python path: `export PROMPTFOO_PYTHON=/opt/ai-security-lab/venvs/garak/bin/python` (or whichever venv you need).

**Q: Native module compilation fails.**
A: Clear npm cache: `rm -rf ~/.npm/_npx` and reinstall.

**Q: How do I test a local API endpoint?**
A: Use an HTTP provider in your config:
```yaml
providers:
  - id: http
    config:
      url: http://localhost:8000/api/chat
      method: POST
      body:
        message: "{{prompt}}"
```

---

## 7. LM Eval Harness

EleutherAI's LM Evaluation Harness runs standardized benchmarks (HellaSwag, MMLU, ARC, etc.) against language models.

### 7.1 Starting LM Eval

```bash
source /opt/ai-security-lab/venvs/lm-eval-harness/bin/activate
```

### 7.2 Listing Available Tasks

```bash
lm-eval ls tasks
lm-eval ls tasks --tags mmlu
```

### 7.3 Running Evaluations

**With a HuggingFace model:**

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-neo-125m \
    --tasks hellaswag \
    --device cpu \
    --batch_size 8
```

**With Ollama (via local API):**

```bash
lm_eval --model local-completions \
    --model_args model=mistral-nemo,base_url=http://localhost:11434/v1/completions,num_concurrent=1 \
    --tasks hellaswag \
    --batch_size 1
```

**With OpenAI:**

```bash
export OPENAI_API_KEY=sk-...
lm_eval --model openai-completions \
    --model_args model=davinci-002 \
    --tasks hellaswag
```

**Quick test run (limit samples):**

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-neo-125m \
    --tasks hellaswag \
    --limit 10 \
    --device cpu
```

### 7.4 Saving Results

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-neo-125m \
    --tasks hellaswag \
    --output_path ./results/ \
    --log_samples
```

### 7.5 Key Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--model` | Backend type | `hf`, `vllm`, `local-completions` |
| `--model_args` | Model parameters | `pretrained=model-name` |
| `--tasks` | Benchmarks to run | `hellaswag,arc_easy,mmlu` |
| `--batch_size` | Batch size | `8`, `auto` |
| `--num_fewshot` | Few-shot examples | `5` |
| `--device` | Compute device | `cuda:0`, `cpu`, `mps` |
| `--limit` | Limit samples | `10`, `0.1` (10%) |
| `--output_path` | Save results | `./results/` |

### 7.6 FAQ - LM Eval Harness

**Q: CUDA out of memory.**
A: Reduce batch size (`--batch_size 1`), use quantization (`--model_args pretrained=model,load_in_4bit=True`), or use `--device cpu`.

**Q: Task not found.**
A: Task names are case-sensitive. Run `lm-eval ls tasks | grep -i hellaswag` to find the exact name.

**Q: Evaluation is extremely slow.**
A: Use `--limit 10` for testing. For production, switch to `vllm` backend if you have a GPU. Use `--batch_size auto` for automatic optimization.

**Q: GGUF model takes hours to load.**
A: Always provide a separate tokenizer: `--model_args pretrained=/path,gguf_file=model.gguf,tokenizer=/path/to/tokenizer`.

**Q: Ollama evaluation gives bad results.**
A: Set `num_concurrent=1` and `--batch_size 1`. Ollama's completion API may behave differently than HuggingFace for certain tasks.

---

## 8. Giskard

Giskard is an AI testing framework that automatically detects vulnerabilities in LLMs including hallucinations, prompt injection, harmful content, and bias.

### 8.1 Starting Giskard

```bash
source /opt/ai-security-lab/venvs/giskard/bin/activate
python
```

### 8.2 Quick Scan Example

```python
import giskard
import pandas as pd

# Configure LLM provider (needed for the scanning engine)
import os
os.environ["OPENAI_API_KEY"] = "your-key"
# OR for Ollama:
giskard.llm.set_llm_model("ollama/mistral-nemo", disable_structured_output=True,
                           api_base="http://localhost:11434")
giskard.llm.set_embedding_model("ollama/nomic-embed-text",
                                 api_base="http://localhost:11434")

# Wrap your model
def my_model(df: pd.DataFrame):
    return [your_llm_function(q) for q in df["question"].values]

model = giskard.Model(
    model=my_model,
    model_type="text_generation",
    name="My Assistant",
    description="Answers questions about climate science",
    feature_names=["question"],
)

# Run the scan
results = giskard.scan(model)
results.to_html("scan_report.html")
```

### 8.3 What the Scan Detects

- Hallucinations
- Harmful content generation
- Prompt injection vulnerabilities
- Robustness issues
- Sensitive information disclosure
- Stereotypes and discrimination

### 8.4 RAG Evaluation (RAGET)

```python
from giskard.rag import generate_testset, KnowledgeBase, evaluate

# Build knowledge base
kb = KnowledgeBase.from_pandas(df, columns=["content"])

# Generate test questions
testset = generate_testset(kb, num_questions=60, language="en")

# Evaluate your RAG
report = evaluate(your_rag_function, testset=testset, knowledge_base=kb)
report.to_html("rag_report.html")
```

### 8.5 FAQ - Giskard

**Q: `LLMImportError` - missing LLM packages.**
A: You need the LLM extras: `pip install "giskard[llm]" -U`.

**Q: `No embedding model set` error.**
A: Configure an embedding model before scanning:
```python
giskard.llm.set_embedding_model("text-embedding-3-small")
```

**Q: Scan is very slow.**
A: Limit which detectors run: `giskard.scan(model, only=["robustness", "prompt_injection"])`.

**Q: Ollama integration fails in Jupyter notebooks.**
A: Add `import nest_asyncio; nest_asyncio.apply()` at the top of your notebook.

**Q: Model wrapping fails with type errors.**
A: Your prediction function must accept a `pd.DataFrame` and return a **list** (one result per row), not a single string.

---

## 9. CleverHans

CleverHans is a library for generating adversarial examples to benchmark ML model robustness. It focuses on image classification attacks (FGSM, PGD) rather than LLM attacks.

### 9.1 Starting CleverHans

```bash
source /opt/ai-security-lab/venvs/cleverhans/bin/activate
python
```

### 9.2 Quick FGSM Attack Example

```python
import torch
import numpy as np
from cleverhans.torch.attacks import fast_gradient_method

model.eval()

# Generate adversarial images
adv_images = fast_gradient_method(
    model_fn=model,
    x=images,           # input tensor (e.g., MNIST batch)
    eps=0.3,            # perturbation magnitude
    norm=np.inf,        # L-infinity norm
    clip_min=0.0,
    clip_max=1.0,
)

# Test robustness
with torch.no_grad():
    predictions = model(adv_images).argmax(dim=1)
    accuracy = (predictions == labels).float().mean()
    print(f"Accuracy under FGSM attack: {accuracy:.2%}")
```

### 9.3 PGD Attack Example

```python
from cleverhans.torch.attacks import projected_gradient_descent

adv_images = projected_gradient_descent(
    model_fn=model,
    x=images,
    eps=0.03,           # 8/255 for CIFAR-10
    eps_iter=0.01,      # step size per iteration
    nb_iter=40,         # number of PGD steps
    norm=np.inf,
    clip_min=0.0,
    clip_max=1.0,
    rand_init=True,     # random start (Madry et al.)
)
```

### 9.4 Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `eps` | Max perturbation | 0.3 (MNIST), 0.03 (CIFAR) |
| `eps_iter` | Step size per iteration | eps/4 to eps/10 |
| `nb_iter` | PGD iterations | 40 |
| `norm` | Norm constraint | `np.inf` or `2` |
| `rand_init` | Random initialization | `True` |

### 9.5 FAQ - CleverHans

**Q: `ModuleNotFoundError: No module named 'cleverhans'`**
A: Ensure the venv is activated. If the PyPI version is outdated, install from GitHub: `pip install git+https://github.com/cleverhans-lab/cleverhans.git#egg=cleverhans`.

**Q: Import error for `fast_gradient_method`.**
A: Use framework-specific imports:
```python
# PyTorch
from cleverhans.torch.attacks import fast_gradient_method
# TensorFlow
from cleverhans.tf2.attacks import fast_gradient_method
# JAX
from cleverhans.jax.attacks import fast_gradient_method
```

**Q: Attack produces unchanged images.**
A: Make sure your model returns **raw logits**, not softmax probabilities. Check that `eps` is appropriate for your image normalization range.

**Q: What frameworks are supported?**
A: PyTorch (primary), JAX, and TensorFlow 2. Install at least one before using CleverHans.

---

## 10. Lab Environments

### 10.1 AI Red-Teaming Playground Labs

Microsoft's Docker-based AI red-teaming challenges.

**Location:** `/opt/ai-security-lab/repos/AI-Red-Teaming-Playground-Labs`

**Starting:**

```bash
cd /opt/ai-security-lab/repos/AI-Red-Teaming-Playground-Labs
cp .env.example .env
# Edit .env with your Azure OpenAI or OpenAI API keys

# For Azure OpenAI:
docker compose up

# For OpenAI:
docker compose -f docker-compose-openai.yaml up
```

**Access:** `http://localhost:5000/login?auth=YOUR-AUTH-KEY`

**FAQ:**

**Q: Containers fail to start.**
A: Check Docker is running (`sudo systemctl start docker`). Ensure your user is in the docker group (`newgrp docker`). Verify `.env` is correctly configured.

**Q: What API keys do I need?**
A: Either an Azure OpenAI endpoint + key (with `text-embedding-ada-002` deployment), or an OpenAI API key.

---

### 10.2 Damn Vulnerable LLM Agent

An intentionally vulnerable chatbot for learning prompt injection techniques.

**Location:** `/opt/ai-security-lab/repos/damn-vulnerable-llm-agent`

**Starting:**

```bash
source /opt/ai-security-lab/venvs/damn-vulnerable-llm-agent/bin/activate
cd /opt/ai-security-lab/repos/damn-vulnerable-llm-agent

# Configure your LLM provider
cp .env.ollama.template .env
# Edit .env with your Ollama settings

# Ensure Ollama is running and the model is pulled
ollama pull mistral-nemo

# Launch
python -m streamlit run main.py
```

**Access:** `http://localhost:8501`

**FAQ:**

**Q: The chatbot gives poor or incoherent answers.**
A: Smaller LLMs perform poorly as ReAct agents. Use `mistral-nemo` as recommended by the project authors. Ensure the model is fully downloaded (`ollama list`).

**Q: Streamlit fails to start.**
A: Install streamlit if missing: `pip install streamlit`. Check that the venv is activated.

**Docker alternative:**

```bash
cd /opt/ai-security-lab/repos/damn-vulnerable-llm-agent
docker build -t dvla .
docker run --env-file .env -p 8501:8501 dvla
```

---

### 10.3 Vulnerable LLMs

A Docker-based lab with a React frontend and FastAPI backend containing intentionally vulnerable LLM interactions.

**Location:** `/opt/ai-security-lab/repos/vulnerable-llms`

**Starting:**

```bash
cd /opt/ai-security-lab/repos/vulnerable-llms
docker-compose -f docker-compose.override.yml up
```

**Requirements:** At least 8GB of available RAM.

**FAQ:**

**Q: Startup takes a long time.**
A: On first run, Ollama models (llama3.2:1b, qwen3:0.6b) are downloaded (~1.5GB). Subsequent starts are faster. FastAPI initializes in ~7 seconds, ML models load in ~10 more seconds.

**Q: The UI loads but the chatbot doesn't respond.**
A: Wait for the backend health check to complete. Check Docker logs: `docker-compose logs -f backend`.

---

### 10.4 FinBot CTF

OWASP's financial chatbot CTF platform with built-in challenges.

**Location:** `/opt/ai-security-lab/repos/finbot-ctf`

**Starting:**

```bash
cd /opt/ai-security-lab/repos/finbot-ctf

# Check prerequisites
uv run python scripts/check_prerequisites.py

# Setup database (SQLite default)
uv run python scripts/setup_database.py

# Launch
uv run python run.py
```

**Access:** `http://localhost:8000`

**With PostgreSQL (optional):**

```bash
docker compose up -d postgres
uv run python scripts/setup_database.py --db-type postgresql
uv run python run.py
```

**FAQ:**

**Q: `uv` command not found.**
A: Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then add `~/.local/bin` to your PATH.

**Q: The main branch is unstable.**
A: This is noted by the maintainers. If you encounter issues, check for a stable release tag: `git tag -l`.

---

### 10.5 MAESTRO

Cloud Security Alliance's AI threat analysis framework using a Next.js web interface.

**Location:** `/opt/ai-security-lab/repos/MAESTRO`

**Starting (requires two terminals):**

```bash
# Terminal 1: Start the web interface
cd /opt/ai-security-lab/repos/MAESTRO
cp .env.example .env      # Edit with your LLM provider credentials
npm run dev

# Terminal 2: Start the AI flows backend
cd /opt/ai-security-lab/repos/MAESTRO
npm run genkit:watch
```

**Access:** `http://localhost:9002`

**Supported LLM providers:** Google Gemini, OpenAI, or Ollama. Configure in `.env`.

**FAQ:**

**Q: The UI loads but threat analysis fails.**
A: Check that your LLM provider credentials are set in `.env` and that `genkit:watch` is running in the second terminal.

**Q: npm install fails.**
A: Ensure Node.js 18+ is installed (`node --version`). Run `npm install` again. If it still fails, try `rm -rf node_modules package-lock.json && npm install`.

---

### 10.6 Vulnerable MCP Servers Lab

Nine intentionally vulnerable MCP (Model Context Protocol) servers for security research.

**Location:** `/opt/ai-security-lab/repos/vulnerable-mcp-servers-lab`

**Structure:** Each server is in its own subdirectory with its own README.

**Starting a Node.js server (example):**

```bash
cd /opt/ai-security-lab/repos/vulnerable-mcp-servers-lab/vulnerable-mcp-server-malicious-code-exec
npm install   # already done by setup script
node vulnerable-mcp-server-malicious-code-exec-mcp.js
```

**Starting a Python server (example):**

```bash
cd /opt/ai-security-lab/repos/vulnerable-mcp-servers-lab/vulnerable-mcp-server-filesystem-workspace-actions
python3 vulnerable-mcp-server-filesystem-workspace-actions-mcp.py /tmp/mcp-sandbox
```

**Configuring with Claude Desktop:**
Each server has a `claude_config.json` snippet. Merge it into Claude Desktop's MCP configuration (Settings -> Developer -> Edit config), updating all paths to absolute paths.

**FAQ:**

**Q: Which servers are Node.js vs Python?**
A: Check for `package.json` (Node.js) or `.py` files (Python) in each server directory.

**Q: How do I use these safely?**
A: Run in a disposable VM only. Use isolated networks. Never use real secrets or personal data. Treat all tool output as untrusted.

---

## 11. General FAQ

### Environment & Setup

**Q: How do I switch between tools?**
A: Deactivate the current venv and activate the new one:
```bash
deactivate
source /opt/ai-security-lab/venvs/<other-tool>/bin/activate
```

**Q: Docker permission denied.**
A: Your user needs to be in the `docker` group. Run `newgrp docker` or log out and back in after the setup script runs.

**Q: How do I update a tool?**
A: Activate its venv, then pip install the latest version:
```bash
source /opt/ai-security-lab/venvs/garak/bin/activate
pip install -U garak
```
For npm tools: `npm update -g promptfoo`.
For repos: `cd /opt/ai-security-lab/repos/<name> && git pull`.

**Q: Disk space is running low.**
A: Check Ollama models (`ollama list`) and remove unused ones (`ollama rm <model>`). Docker images can also be large; prune with `docker system prune`.

### API Keys

**Q: Which tools need API keys?**
A: It depends on your usage:
- **No API key needed:** Ollama (local), CleverHans (local models only)
- **Optional (can use Ollama instead):** Garak, PyRIT, PromptFoo, LM Eval Harness, Giskard
- **Required for their specific labs:** AI Red-Teaming Playground (OpenAI/Azure), MAESTRO (any LLM provider)

**Q: Can I run everything with just Ollama (no cloud APIs)?**
A: Most tools support Ollama as a backend. Garak, PyRIT, PromptFoo, and LM Eval Harness all work with local Ollama models. Giskard can use Ollama for both the LLM and embedding model. Some lab environments (AI Red-Teaming Playground) require cloud API keys.

### Networking & Security

**Q: Can I run this lab safely?**
A: This lab contains intentionally vulnerable applications. Always run in an isolated VM with no access to production systems. Never expose lab services to the internet.

**Q: What ports are used?**

| Service | Default Port |
|---------|-------------|
| Ollama | 11434 |
| AI Red-Teaming Playground | 5000 |
| Damn Vulnerable LLM Agent | 8501 |
| FinBot CTF | 8000 |
| MAESTRO | 9002 |

### Performance

**Q: Everything is slow without a GPU.**
A: Use the smallest Ollama models (`llama3.2:1b`, `tinyllama`). Reduce batch sizes in LM Eval. Limit scan scope in Garak (`-g 2`, specific probes only). Consider using cloud APIs for compute-intensive tasks.

**Q: How much RAM do I need?**
A: Minimum 8GB for basic usage. 16GB+ recommended if running Docker labs alongside Ollama. Each 7B model consumes roughly 4GB when loaded.

**Q: How much disk space does this lab need?**
A: The tools and repos themselves need roughly 5-10GB. Each Ollama model adds 1-5GB. Docker images for the labs add 2-5GB each. Budget at least 30-50GB total.
