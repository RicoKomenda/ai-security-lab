#!/usr/bin/env bash
# =============================================================================
# AI Security Lab - Debian 13 (Trixie) Setup Script
# =============================================================================
# Installs and configures:
#   - LM Eval Harness          (EleutherAI)
#   - PromptFoo                (LLM red-teaming / evaluation)
#   - CleverHans               (adversarial ML)
#   - Garak                    (NVIDIA LLM vulnerability scanner)
#   - Giskard                  (AI testing framework)
#   - PyRIT                    (Microsoft AI red-teaming toolkit)
#   - AI Red-Teaming Playground Labs (Microsoft)
#   - Ollama                   (local LLM runtime)
#   - Damn Vulnerable LLM Agent
#   - Vulnerable LLMs
#   - FinBot CTF               (OWASP)
#   - MAESTRO                  (Cloud Security Alliance)
#   - Vulnerable MCP Servers Lab
#
# Reference links (saved to file):
#   - https://www.llm-sec.dev/labs/vector-embedding-weakness
#
# Tested on: Debian 13 (Trixie)
# Run as:    sudo bash setup_ai_security_lab.sh
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# Prevent interactive prompts during package installation
export DEBIAN_FRONTEND=noninteractive

# ── Configuration ────────────────────────────────────────────────────────────
LAB_ROOT="/opt/ai-security-lab"
VENV_BASE="${LAB_ROOT}/venvs"
REPOS_DIR="${LAB_ROOT}/repos"
LOG_FILE="${LAB_ROOT}/install.log"
PYTHON_VERSION="3.11"                    # Garak needs 3.10-3.12, Giskard needs <=3.11
NODE_MAJOR=22                            # vulnerable-llms needs Node 22+

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()    { echo -e "${GREEN}[+]${NC} $*" | tee -a "$LOG_FILE"; }
warn()   { echo -e "${YELLOW}[!]${NC} $*" | tee -a "$LOG_FILE"; }
err()    { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"; }
header() { echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
           echo -e "${CYAN}  $*${NC}" | tee -a "$LOG_FILE"
           echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"; }

# ── Pre-flight checks ───────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    err "This script must be run as root (sudo)."
    exit 1
fi

REAL_USER="${SUDO_USER:-$(whoami)}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

mkdir -p "$LAB_ROOT" "$VENV_BASE" "$REPOS_DIR"
: > "$LOG_FILE"

header "AI Security Lab - Debian 13 Setup"
log "Lab root:  $LAB_ROOT"
log "User:      $REAL_USER"
log "Log file:  $LOG_FILE"
log "Started:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# =============================================================================
# 1. SYSTEM PACKAGES & PREREQUISITES
# =============================================================================
header "1/15 - System packages & prerequisites"

log "Updating apt package lists..."
if ! apt-get update -qq 2>&1 | tee -a "$LOG_FILE"; then
    err "Failed to update package lists. Check ${LOG_FILE} for details."
    warn "This might be due to network issues or repository configuration problems."
    exit 1
fi

log "Installing base build tools and libraries..."
if ! apt-get install -y -q \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libgdbm-dev \
    libnss3-dev \
    tk-dev \
    uuid-dev \
    xz-utils \
    jq \
    unzip \
    2>&1 | tee -a "$LOG_FILE"; then
    err "Failed to install base packages. Check ${LOG_FILE} for details."
    exit 1
fi

# Try to install software-properties-common separately (may not exist in all Debian versions)
if apt-cache show software-properties-common &>/dev/null; then
    log "Installing software-properties-common..."
    apt-get install -y -q software-properties-common 2>&1 | tee -a "$LOG_FILE" || warn "software-properties-common not available (optional package)"
else
    warn "software-properties-common not available in repositories (skipping - optional package)"
fi

log "Base build tools and libraries installed successfully."

# =============================================================================
# 2. PYTHON 3.11 (via deadsnakes or source build)
# =============================================================================
header "2/15 - Python ${PYTHON_VERSION}"

if command -v "python${PYTHON_VERSION}" &>/dev/null; then
    log "Python ${PYTHON_VERSION} already installed: $(python${PYTHON_VERSION} --version)"
else
    log "Installing Python ${PYTHON_VERSION} from source..."
    PYTHON_SRC_VERSION="3.11.11"
    cd /tmp
    wget -q "https://www.python.org/ftp/python/${PYTHON_SRC_VERSION}/Python-${PYTHON_SRC_VERSION}.tgz"
    tar xzf "Python-${PYTHON_SRC_VERSION}.tgz"
    cd "Python-${PYTHON_SRC_VERSION}"
    ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local 2>>"$LOG_FILE"
    make -j"$(nproc)" 2>>"$LOG_FILE"
    make altinstall 2>>"$LOG_FILE"
    cd /tmp && rm -rf "Python-${PYTHON_SRC_VERSION}" "Python-${PYTHON_SRC_VERSION}.tgz"
    log "Python ${PYTHON_VERSION} installed: $(python${PYTHON_VERSION} --version)"
fi

# Ensure pip is up-to-date
python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel 2>>"$LOG_FILE"

# =============================================================================
# 3. NODE.JS 22 (via NodeSource)
# =============================================================================
header "3/15 - Node.js ${NODE_MAJOR}"

if command -v node &>/dev/null && [[ "$(node --version | cut -d. -f1 | tr -d v)" -ge $NODE_MAJOR ]]; then
    log "Node.js already installed: $(node --version)"
else
    log "Installing Node.js ${NODE_MAJOR} via NodeSource..."
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg 2>>"$LOG_FILE"
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list
    apt-get update -qq 2>>"$LOG_FILE"
    apt-get install -y -qq nodejs 2>>"$LOG_FILE"
    log "Node.js installed: $(node --version)"
fi

# Install pnpm (needed by PromptFoo)
if ! command -v pnpm &>/dev/null; then
    log "Installing pnpm..."
    npm install -g pnpm 2>>"$LOG_FILE"
fi

# =============================================================================
# 4. DOCKER & DOCKER COMPOSE
# =============================================================================
header "4/15 - Docker & Docker Compose"

if command -v docker &>/dev/null; then
    log "Docker already installed: $(docker --version)"
else
    log "Installing Docker..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg \
        -o /etc/apt/keyrings/docker.asc 2>>"$LOG_FILE"
    chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq 2>>"$LOG_FILE"
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 2>>"$LOG_FILE"
    if command -v systemctl &>/dev/null && systemctl is-system-running &>/dev/null; then
        systemctl enable --now docker 2>>"$LOG_FILE"
    else
        warn "systemd not available - start Docker manually: dockerd &"
    fi
    usermod -aG docker "$REAL_USER" 2>>"$LOG_FILE" || true
    log "Docker installed: $(docker --version)"
fi

# =============================================================================
# 5. OLLAMA
# =============================================================================
header "5/15 - Ollama"

if command -v ollama &>/dev/null; then
    log "Ollama already installed: $(ollama --version 2>&1 || echo 'installed')"
else
    log "Installing Ollama..."
    if command -v timeout &>/dev/null; then
        INSTALL_CMD='timeout 300 bash -c "curl -fsSL https://ollama.com/install.sh | sh"'
    else
        INSTALL_CMD='bash -c "curl -fsSL https://ollama.com/install.sh | sh"'
    fi
    
    if eval "$INSTALL_CMD" 2>>"$LOG_FILE"; then
        log "Ollama installed successfully."
    else
        err "Ollama installation failed or timed out. Check ${LOG_FILE} for details."
        warn "You can install Ollama manually later with: curl -fsSL https://ollama.com/install.sh | sh"
    fi
fi

log "Checking Ollama service status..."
OLLAMA_PID=""

# Try to start Ollama service if it's not running
if command -v ollama &>/dev/null; then
    if ! pgrep -x ollama &>/dev/null; then
        log "Starting Ollama service..."
        # Try to start Ollama as the real user (it runs as a user service)
        if command -v systemctl &>/dev/null && systemctl is-system-running &>/dev/null; then
            # Check if there's a user service
            if systemctl --user -M "${REAL_USER}@" list-units ollama.service &>/dev/null 2>&1; then
                systemctl --user -M "${REAL_USER}@" start ollama.service 2>>"$LOG_FILE" || true
            fi
        fi
        # Give Ollama a moment to start
        sleep 2
    fi
    
    # Check if Ollama is now running
    if pgrep -x ollama &>/dev/null; then
        # Ollama process is running
        OLLAMA_RUNNING=true
    elif command -v timeout &>/dev/null && timeout 5 bash -c 'until curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do sleep 1; done' 2>/dev/null; then
        # Ollama API is responding
        OLLAMA_RUNNING=true
    else
        OLLAMA_RUNNING=false
    fi
    
    if [[ "${OLLAMA_RUNNING:-false}" == "true" ]]; then
        warn "Ollama service is running. Model pulls will be done manually after setup."
        warn "To pull models, run:"
        warn "  ollama pull llama3.2:1b"
        warn "  ollama pull mistral-nemo"
        warn "  ollama pull qwen3:0.6b"
    else
        warn "Ollama service not running - skipping model pull."
        warn "Start Ollama manually with: ollama serve"
        warn "Then pull models with: ollama pull llama3.2:1b"
    fi
else
    warn "Ollama command not found - skipping model pull."
fi

# =============================================================================
# HELPER: Create a Python venv and install into it
# =============================================================================
create_venv() {
    local name="$1"
    local venv_path="${VENV_BASE}/${name}"

    log "Creating venv: ${name}"
    if ! python${PYTHON_VERSION} -m venv "$venv_path" 2>>"$LOG_FILE"; then
        err "Failed to create venv: ${name}"
        return 1
    fi

    # Ensure pip exists
    if [[ ! -f "${venv_path}/bin/pip" ]]; then
        warn "pip missing in ${name} venv — bootstrapping with ensurepip"
        if ! "${venv_path}/bin/python" -m ensurepip --upgrade 2>>"$LOG_FILE"; then
            err "ensurepip failed for ${name}"
            return 1
        fi
    fi

    # Upgrade tooling safely
    if ! "${venv_path}/bin/python" -m pip install --upgrade pip setuptools wheel 2>>"$LOG_FILE"; then
        err "pip bootstrap failed for ${name}"
        return 1
    fi

    echo "$venv_path"
}


# =============================================================================
# 6. LM EVAL HARNESS
# =============================================================================
header "6/15 - LM Eval Harness (EleutherAI)"

VENV=$(create_venv "lm-eval-harness") || {
    err "Failed to create venv for lm-eval-harness. Check ${LOG_FILE} for details."
    exit 1
}
log "Cloning lm-evaluation-harness..."
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git \
    "${REPOS_DIR}/lm-evaluation-harness" 2>>"$LOG_FILE" || log "Already cloned."

log "Installing lm-eval with HuggingFace + API backends..."
"${VENV}/bin/pip" install -e "${REPOS_DIR}/lm-evaluation-harness[hf,api]" 2>>"$LOG_FILE"
log "LM Eval Harness installed. Activate: source ${VENV}/bin/activate"

# =============================================================================
# 7. PROMPTFOO
# =============================================================================
header "7/15 - PromptFoo"

log "Installing PromptFoo globally via npm..."
npm install -g promptfoo 2>>"$LOG_FILE"
log "PromptFoo installed: $(npx promptfoo --version 2>/dev/null || echo 'OK')"

# =============================================================================
# 8. CLEVERHANS
# =============================================================================
header "8/15 - CleverHans"

VENV=$(create_venv "cleverhans") || {
    err "Failed to create venv for cleverhans. Check ${LOG_FILE} for details."
    exit 1
}
log "Installing CleverHans with PyTorch backend..."
"${VENV}/bin/pip" install cleverhans torch torchvision 2>>"$LOG_FILE"
log "CleverHans installed. Activate: source ${VENV}/bin/activate"

# =============================================================================
# 9. GARAK
# =============================================================================
header "9/15 - Garak (NVIDIA)"

VENV=$(create_venv "garak") || {
    err "Failed to create venv for garak. Check ${LOG_FILE} for details."
    exit 1
}
log "Installing Garak..."
"${VENV}/bin/pip" install -U garak 2>>"$LOG_FILE"
log "Garak installed. Activate: source ${VENV}/bin/activate"

# =============================================================================
# 10. GISKARD
# =============================================================================
header "10/15 - Giskard"

VENV=$(create_venv "giskard") || {
    err "Failed to create venv for giskard. Check ${LOG_FILE} for details."
    exit 1
}
log "Installing Giskard with LLM extras..."
"${VENV}/bin/pip" install "giskard[llm]" -U 2>>"$LOG_FILE"
log "Giskard installed. Activate: source ${VENV}/bin/activate"

# =============================================================================
# 11. PYRIT
# =============================================================================
header "11/15 - PyRIT (Microsoft)"

VENV=$(create_venv "pyrit") || {
    err "Failed to create venv for pyrit. Check ${LOG_FILE} for details."
    exit 1
}
log "Installing PyRIT..."
"${VENV}/bin/pip" install pyrit 2>>"$LOG_FILE"
log "PyRIT installed. Activate: source ${VENV}/bin/activate"

# =============================================================================
# 12. GIT CLONE - LABS & VULNERABLE APPS
# =============================================================================
header "12/15 - Cloning lab repositories"

declare -A REPOS=(
    ["AI-Red-Teaming-Playground-Labs"]="https://github.com/microsoft/AI-Red-Teaming-Playground-Labs.git"
    ["damn-vulnerable-llm-agent"]="https://github.com/ReversecLabs/damn-vulnerable-llm-agent.git"
    ["vulnerable-llms"]="https://github.com/AImaginationLab/vulnerable-llms.git"
    ["finbot-ctf"]="https://github.com/OWASP-ASI/finbot-ctf.git"
    ["MAESTRO"]="https://github.com/CloudSecurityAlliance/MAESTRO.git"
    ["vulnerable-mcp-servers-lab"]="https://github.com/appsecco/vulnerable-mcp-servers-lab.git"
)

for name in "${!REPOS[@]}"; do
    url="${REPOS[$name]}"
    dest="${REPOS_DIR}/${name}"
    if [[ -d "$dest" ]]; then
        log "${name} already cloned."
    else
        log "Cloning ${name}..."
        git clone --depth 1 "$url" "$dest" 2>>"$LOG_FILE"
    fi
done

# =============================================================================
# 13. PER-REPO DEPENDENCY SETUP
# =============================================================================
header "13/15 - Installing per-repo dependencies"

# ── AI Red-Teaming Playground Labs (Docker-based) ───────────────────────────
log "[AI-Red-Teaming-Playground-Labs] Docker-based. Run with:"
log "  cd ${REPOS_DIR}/AI-Red-Teaming-Playground-Labs"
log "  cp .env.example .env   # then edit .env with your API keys"
log "  docker compose up"

# ── Damn Vulnerable LLM Agent ───────────────────────────────────────────────
log "[damn-vulnerable-llm-agent] Setting up venv..."
VENV=$(create_venv "damn-vulnerable-llm-agent") || {
    err "Failed to create venv for damn-vulnerable-llm-agent. Check ${LOG_FILE} for details."
    exit 1
}
if [[ -f "${REPOS_DIR}/damn-vulnerable-llm-agent/requirements.txt" ]]; then
    "${VENV}/bin/pip" install -r "${REPOS_DIR}/damn-vulnerable-llm-agent/requirements.txt" 2>>"$LOG_FILE"
    "${VENV}/bin/pip" install python-dotenv 2>>"$LOG_FILE"
fi
log "  Activate: source ${VENV}/bin/activate"
log "  Run:      python -m streamlit run ${REPOS_DIR}/damn-vulnerable-llm-agent/main.py"

# ── Vulnerable LLMs (Docker-based) ──────────────────────────────────────────
log "[vulnerable-llms] Docker-based. Run with:"
log "  cd ${REPOS_DIR}/vulnerable-llms"
log "  docker-compose -f docker-compose.override.yml up"

# ── FinBot CTF ──────────────────────────────────────────────────────────────
log "[finbot-ctf] Setting up with uv..."
if ! command -v uv &>/dev/null; then
    log "  Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>>"$LOG_FILE"
    export PATH="${REAL_HOME}/.local/bin:${PATH}"
    # Also install for root
    export PATH="/root/.local/bin:${PATH}"
fi
if command -v uv &>/dev/null; then
    cd "${REPOS_DIR}/finbot-ctf"
    uv sync 2>>"$LOG_FILE" || warn "  uv sync had issues - check manually."
    cd "$LAB_ROOT"
    log "  Run: cd ${REPOS_DIR}/finbot-ctf && uv run python scripts/setup_database.py && uv run python run.py"
else
    warn "  uv not found in PATH. Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# ── MAESTRO (Node.js / Next.js) ────────────────────────────────────────────
log "[MAESTRO] Installing Node.js dependencies..."
cd "${REPOS_DIR}/MAESTRO"
npm install 2>>"$LOG_FILE" || warn "  npm install had issues."
cd "$LAB_ROOT"
log "  Run:  cd ${REPOS_DIR}/MAESTRO && npm run dev"
log "  Also: cd ${REPOS_DIR}/MAESTRO && npm run genkit:watch  (in second terminal)"

# ── Vulnerable MCP Servers Lab ──────────────────────────────────────────────
log "[vulnerable-mcp-servers-lab] Installing Node dependencies for each server..."
for server_dir in "${REPOS_DIR}/vulnerable-mcp-servers-lab"/vulnerable-mcp-server-*/; do
    if [[ -f "${server_dir}/package.json" ]]; then
        server_name=$(basename "$server_dir")
        log "  Installing deps for ${server_name}..."
        cd "$server_dir"
        npm install 2>>"$LOG_FILE" || warn "  npm install failed for ${server_name}"
    fi
done
cd "$LAB_ROOT"

# =============================================================================
# 14. JUPYTER NOTEBOOK ENVIRONMENT
# =============================================================================
header "14/15 - Jupyter Notebook environment"

VENV_JUPYTER=$(create_venv "jupyter")
log "Installing JupyterLab and Jupyter Notebook..."
"${VENV_JUPYTER}/bin/pip" install jupyterlab notebook ipywidgets 2>>"$LOG_FILE"

log "Registering tool venvs as Jupyter kernels..."

declare -A KERNEL_VENVS=(
    ["lm-eval-harness"]="LM Eval Harness"
    ["cleverhans"]="CleverHans"
    ["garak"]="Garak"
    ["giskard"]="Giskard"
    ["pyrit"]="PyRIT"
    ["damn-vulnerable-llm-agent"]="Damn Vulnerable LLM Agent"
)

for venv_name in "${!KERNEL_VENVS[@]}"; do
    display_name="${KERNEL_VENVS[$venv_name]}"
    venv_path="${VENV_BASE}/${venv_name}"
    if [[ -d "$venv_path" ]]; then
        log "  Registering kernel: ${display_name}"
        "${venv_path}/bin/pip" install ipykernel 2>>"$LOG_FILE"
        "${venv_path}/bin/python" -m ipykernel install \
            --name "$venv_name" \
            --display-name "Python (${display_name})" \
            --prefix "${VENV_JUPYTER}" 2>>"$LOG_FILE"
    fi
done

# Create a default notebooks directory with a starter notebook
NOTEBOOKS_DIR="${LAB_ROOT}/notebooks"
mkdir -p "$NOTEBOOKS_DIR"

cat > "${NOTEBOOKS_DIR}/00_lab_overview.ipynb" << 'NBEOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AI Security Lab - Overview Notebook\n",
    "\n",
    "Welcome to the AI Security Lab. This notebook verifies your installation.\n",
    "\n",
    "**Switch the kernel** (top-right) to match the tool you want to use:\n",
    "- `Python (Garak)` - LLM vulnerability scanning\n",
    "- `Python (PyRIT)` - AI red-teaming\n",
    "- `Python (Giskard)` - AI testing & bias detection\n",
    "- `Python (LM Eval Harness)` - Benchmarking\n",
    "- `Python (CleverHans)` - Adversarial ML\n",
    "- `Python (Damn Vulnerable LLM Agent)` - DVLA lab\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Verify Ollama Connection\n",
    "Run this cell to check that Ollama is reachable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import urllib.request, json\n",
    "try:\n",
    "    resp = urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=5)\n",
    "    models = json.loads(resp.read())['models']\n",
    "    print(f'Ollama is running with {len(models)} model(s):')\n",
    "    for m in models:\n",
    "        print(f\"  - {m['name']}\")\n",
    "except Exception as e:\n",
    "    print(f'Ollama not reachable: {e}')\n",
    "    print('Start it with: ollama serve')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Quick Import Tests\n",
    "Switch to the appropriate kernel before running each cell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -- Switch to 'Python (Garak)' kernel first --\n",
    "import garak\n",
    "print(f'Garak version: {garak.__version__}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -- Switch to 'Python (Giskard)' kernel first --\n",
    "import giskard\n",
    "print(f'Giskard version: {giskard.__version__}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -- Switch to 'Python (CleverHans)' kernel first --\n",
    "import cleverhans\n",
    "import torch\n",
    "print(f'CleverHans imported successfully')\n",
    "print(f'PyTorch version: {torch.__version__}')\n",
    "print(f'CUDA available: {torch.cuda.is_available()}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Chat with a Local Model via Ollama\n",
    "This cell works with any kernel."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import urllib.request, json\n",
    "\n",
    "MODEL = 'llama3.2:1b'  # Change to your preferred model\n",
    "PROMPT = 'Explain prompt injection in one sentence.'\n",
    "\n",
    "data = json.dumps({'model': MODEL, 'prompt': PROMPT, 'stream': False}).encode()\n",
    "req = urllib.request.Request('http://127.0.0.1:11434/api/generate',\n",
    "                             data=data,\n",
    "                             headers={'Content-Type': 'application/json'})\n",
    "resp = urllib.request.urlopen(req, timeout=120)\n",
    "result = json.loads(resp.read())\n",
    "print(result['response'])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
NBEOF

log "Starter notebook created: ${NOTEBOOKS_DIR}/00_lab_overview.ipynb"
log "JupyterLab installed. Run with:"
log "  source ${VENV_JUPYTER}/bin/activate && jupyter lab --notebook-dir=${NOTEBOOKS_DIR}"

# =============================================================================
# 15. REFERENCE LINKS & DOCUMENTATION
# =============================================================================
header "15/15 - Reference links & documentation"

cat > "${LAB_ROOT}/REFERENCES.md" << 'REFEOF'
# AI Security Lab - Reference Links & Resources

## Installed Tools

| Tool | Location | Activation |
|------|----------|------------|
| LM Eval Harness | /opt/ai-security-lab/repos/lm-evaluation-harness | `source /opt/ai-security-lab/venvs/lm-eval-harness/bin/activate` |
| PromptFoo | global npm | `promptfoo` |
| CleverHans | /opt/ai-security-lab/venvs/cleverhans | `source /opt/ai-security-lab/venvs/cleverhans/bin/activate` |
| Garak | /opt/ai-security-lab/venvs/garak | `source /opt/ai-security-lab/venvs/garak/bin/activate` |
| Giskard | /opt/ai-security-lab/venvs/giskard | `source /opt/ai-security-lab/venvs/giskard/bin/activate` |
| PyRIT | /opt/ai-security-lab/venvs/pyrit | `source /opt/ai-security-lab/venvs/pyrit/bin/activate` |
| Ollama | system-wide | `ollama` |
| JupyterLab | /opt/ai-security-lab/venvs/jupyter | `source /opt/ai-security-lab/venvs/jupyter/bin/activate && jupyter lab` |

## Jupyter Notebooks

Default notebook directory: `/opt/ai-security-lab/notebooks/`

Pre-registered kernels (one per tool venv):
- Python (LM Eval Harness)
- Python (CleverHans)
- Python (Garak)
- Python (Giskard)
- Python (PyRIT)
- Python (Damn Vulnerable LLM Agent)

## Lab Repositories

| Lab | Location | Launch |
|-----|----------|--------|
| AI Red-Teaming Playground | /opt/ai-security-lab/repos/AI-Red-Teaming-Playground-Labs | `docker compose up` |
| Damn Vulnerable LLM Agent | /opt/ai-security-lab/repos/damn-vulnerable-llm-agent | `streamlit run main.py` (port 8501) |
| Vulnerable LLMs | /opt/ai-security-lab/repos/vulnerable-llms | `docker-compose -f docker-compose.override.yml up` |
| FinBot CTF | /opt/ai-security-lab/repos/finbot-ctf | `uv run python run.py` (port 8000) |
| MAESTRO | /opt/ai-security-lab/repos/MAESTRO | `npm run dev` (port 9002) |
| Vulnerable MCP Servers | /opt/ai-security-lab/repos/vulnerable-mcp-servers-lab | Per-server (see each README) |

## Additional Resources

- **LLM Security Labs - Vector Embedding Weakness**
  <https://www.llm-sec.dev/labs/vector-embedding-weakness>

## Quick Start

```bash
# Activate a Python tool (example: Garak)
source /opt/ai-security-lab/venvs/garak/bin/activate
garak --help

# Run PromptFoo
promptfoo eval

# Start Ollama and pull a model
ollama serve &
ollama pull llama3.2:1b

# Launch a Docker-based lab
cd /opt/ai-security-lab/repos/vulnerable-llms
docker-compose -f docker-compose.override.yml up
```

## Ollama Models

Pull additional models as needed:
```bash
ollama pull llama3.2:1b       # Small, fast model
ollama pull mistral-nemo      # Good for DVLA lab
ollama pull qwen3:0.6b        # Lightweight model
```

## Notes

- Each Python tool has its own virtual environment to avoid dependency conflicts.
- Docker-based labs need API keys configured in their .env files before launch.
- Run vulnerable labs only in isolated environments. Never expose to the internet.
- The vulnerable-mcp-servers-lab contains intentionally insecure code - use in sandboxes only.
REFEOF

log "Reference file created: ${LAB_ROOT}/REFERENCES.md"

# ── Additional resource links text file ──────────────────────────────────────
cat > "${LAB_ROOT}/RESOURCE_LINKS.txt" << 'LINKEOF'
AI Security Lab - Additional Resource Links
============================================

LLM Security Labs - Vector Embedding Weakness:
https://www.llm-sec.dev/labs/vector-embedding-weakness

Tool Documentation:
- LM Eval Harness:      https://github.com/EleutherAI/lm-evaluation-harness
- PromptFoo:            https://github.com/promptfoo/promptfoo
- CleverHans:           https://github.com/cleverhans-lab/cleverhans
- Garak:                https://github.com/NVIDIA/garak
- Giskard:              https://github.com/Giskard-AI/giskard
- PyRIT:                https://github.com/Azure/PyRIT
- Ollama:               https://ollama.com

Lab Repositories:
- AI Red-Teaming Playground: https://github.com/microsoft/AI-Red-Teaming-Playground-Labs
- Damn Vulnerable LLM Agent: https://github.com/ReversecLabs/damn-vulnerable-llm-agent
- Vulnerable LLMs:           https://github.com/AImaginationLab/vulnerable-llms
- FinBot CTF:                https://github.com/OWASP-ASI/finbot-ctf
- MAESTRO:                   https://github.com/CloudSecurityAlliance/MAESTRO
- Vulnerable MCP Servers:    https://github.com/appsecco/vulnerable-mcp-servers-lab
LINKEOF

log "Resource links file created: ${LAB_ROOT}/RESOURCE_LINKS.txt"

# =============================================================================
# FINAL SUMMARY
# =============================================================================
# Note: Ollama model pulls are done manually by the user after setup

# Fix ownership so the real user can work with everything
chown -R "$REAL_USER:$REAL_USER" "$LAB_ROOT"

header "Setup Complete!"

cat << EOF

$(echo -e "${GREEN}All tools installed successfully.${NC}")

  Lab root:       ${LAB_ROOT}
  Repos:          ${REPOS_DIR}/
  Python venvs:   ${VENV_BASE}/
  Notebooks:      ${LAB_ROOT}/notebooks/
  References:     ${LAB_ROOT}/REFERENCES.md
  Resource links: ${LAB_ROOT}/RESOURCE_LINKS.txt
  Install log:    ${LOG_FILE}

$(echo -e "${YELLOW}IMPORTANT: Post-install steps${NC}")

  1. Log out and back in (or run: newgrp docker) for Docker group membership.

  2. Configure API keys where needed:
     - AI Red-Teaming Playground: edit ${REPOS_DIR}/AI-Red-Teaming-Playground-Labs/.env
     - MAESTRO:                   edit ${REPOS_DIR}/MAESTRO/.env
     - FinBot CTF:                edit ${REPOS_DIR}/finbot-ctf/.env
     - Damn Vulnerable LLM Agent: edit ${REPOS_DIR}/damn-vulnerable-llm-agent/.env

  3. Pull Ollama models:
       ollama pull llama3.2:1b
       ollama pull mistral-nemo
       ollama pull qwen3:0.6b

  4. Launch JupyterLab:
       source ${VENV_BASE}/jupyter/bin/activate
       jupyter lab --notebook-dir=${LAB_ROOT}/notebooks

  5. Start exploring:
       cat ${LAB_ROOT}/REFERENCES.md

$(echo -e "${RED}WARNING: This lab contains intentionally vulnerable applications.${NC}")
$(echo -e "${RED}Run only in isolated environments. Never expose to the internet.${NC}")

EOF
