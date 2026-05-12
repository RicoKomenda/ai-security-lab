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

log()    { echo -e "${GREEN}[+]${NC} $*" | tee -a "$LOG_FILE" >&2; }
warn()   { echo -e "${YELLOW}[!]${NC} $*" | tee -a "$LOG_FILE" >&2; }
err()    { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE" >&2; }
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
# Sets the global variable VENV_PATH on success
# =============================================================================
create_venv() {
    local name="$1"
    VENV_PATH="${VENV_BASE}/${name}"

    log "Creating venv: ${name}"
    if ! python${PYTHON_VERSION} -m venv "$VENV_PATH" 2>>"$LOG_FILE"; then
        err "Failed to create venv: ${name}"
        return 1
    fi

    # Ensure pip exists
    if [[ ! -f "${VENV_PATH}/bin/pip" ]]; then
        warn "pip missing in ${name} venv — bootstrapping with ensurepip"
        if ! "${VENV_PATH}/bin/python" -m ensurepip --upgrade 2>>"$LOG_FILE"; then
            err "ensurepip failed for ${name}"
            return 1
        fi
    fi

    # Upgrade tooling safely
    if ! "${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel 2>>"$LOG_FILE"; then
        err "pip bootstrap failed for ${name}"
        return 1
    fi

    return 0
}


# =============================================================================
# 6. LM EVAL HARNESS
# =============================================================================
header "6/15 - LM Eval Harness (EleutherAI)"

create_venv "lm-eval-harness" || {
    err "Failed to create venv for lm-eval-harness. Check ${LOG_FILE} for details."
    exit 1
}

# DEBUG: Show what VENV_PATH contains
echo "DEBUG: VENV_PATH='${VENV_PATH}'" | tee -a "$LOG_FILE"
echo "DEBUG: pip exists? $(ls -la "${VENV_PATH}/bin/pip" 2>&1)" | tee -a "$LOG_FILE"

log "Cloning lm-evaluation-harness..."
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git \
    "${REPOS_DIR}/lm-evaluation-harness" 2>>"$LOG_FILE" || log "Already cloned."

log "Installing lm-eval with HuggingFace + API backends..."
"${VENV_PATH}/bin/pip" install -e "${REPOS_DIR}/lm-evaluation-harness[hf,api]" 2>>"$LOG_FILE"
log "LM Eval Harness installed. Activate: source ${VENV_PATH}/bin/activate"

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

create_venv "cleverhans" || {
    err "Failed to create venv for cleverhans. Check ${LOG_FILE} for details."
    exit 1
}
log "Installing CleverHans with PyTorch backend..."
"${VENV_PATH}/bin/pip" install cleverhans torch torchvision 2>>"$LOG_FILE"
log "CleverHans installed. Activate: source ${VENV_PATH}/bin/activate"

# =============================================================================
# 9. GARAK
# =============================================================================
header "9/15 - Garak (NVIDIA)"

create_venv "garak" || {
    err "Failed to create venv for garak. Check ${LOG_FILE} for details."
    exit 1
}
log "Installing Garak..."
"${VENV_PATH}/bin/pip" install -U garak 2>>"$LOG_FILE"
log "Garak installed. Activate: source ${VENV_PATH}/bin/activate"

# =============================================================================
# 10. GISKARD
# =============================================================================
header "10/15 - Giskard"

create_venv "giskard" || {
    err "Failed to create venv for giskard. Check ${LOG_FILE} for details."
    exit 1
}
log "Installing Giskard with LLM extras..."
"${VENV_PATH}/bin/pip" install "giskard[llm]" -U 2>>"$LOG_FILE"
log "Giskard installed. Activate: source ${VENV_PATH}/bin/activate"

# =============================================================================
# 11. PYRIT
# =============================================================================
header "11/15 - PyRIT (Microsoft)"

create_venv "pyrit" || {
    err "Failed to create venv for pyrit. Check ${LOG_FILE} for details."
    exit 1
}
log "Installing PyRIT..."
"${VENV_PATH}/bin/pip" install pyrit 2>>"$LOG_FILE"
log "PyRIT installed. Activate: source ${VENV_PATH}/bin/activate"

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
create_venv "damn-vulnerable-llm-agent" || {
    err "Failed to create venv for damn-vulnerable-llm-agent. Check ${LOG_FILE} for details."
    exit 1
}
if [[ -f "${REPOS_DIR}/damn-vulnerable-llm-agent/requirements.txt" ]]; then
    "${VENV_PATH}/bin/pip" install -r "${REPOS_DIR}/damn-vulnerable-llm-agent/requirements.txt" 2>>"$LOG_FILE"
    "${VENV_PATH}/bin/pip" install python-dotenv 2>>"$LOG_FILE"
fi
log "  Activate: source ${VENV_PATH}/bin/activate"
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

# Wire MAESTRO's Genkit OpenAI plugin to the unified router's base URL.
# Without this it ignores OPENAI_BASE_URL and always hits api.openai.com,
# breaking offline / non-OpenAI providers configured via `lab-llm configure`.
MAESTRO_GENKIT="${REPOS_DIR}/MAESTRO/src/ai/genkit.ts"
if [[ -f "$MAESTRO_GENKIT" ]] && ! grep -q "baseURL: process.env.OPENAI_BASE_URL" "$MAESTRO_GENKIT"; then
    log "[MAESTRO] Patching genkit.ts to honour OPENAI_BASE_URL..."
    sed -i.bak 's|openAI({apiKey: process.env.OPENAI_API_KEY})|openAI({apiKey: process.env.OPENAI_API_KEY, baseURL: process.env.OPENAI_BASE_URL})|' "$MAESTRO_GENKIT"
    rm -f "${MAESTRO_GENKIT}.bak"
fi

# Patch MAESTRO's package.json so `npm run genkit:watch` doesn't block on
# Genkit CLI's interactive analytics / update-notification prompts.
MAESTRO_PKG="${REPOS_DIR}/MAESTRO/package.json"
if [[ -f "$MAESTRO_PKG" ]] && grep -q '"genkit start -- ' "$MAESTRO_PKG"; then
    log "[MAESTRO] Patching package.json to skip Genkit interactive prompts..."
    sed -i.bak 's|"genkit start -- |"genkit start --non-interactive -- |' "$MAESTRO_PKG"
    rm -f "${MAESTRO_PKG}.bak"
fi

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

create_venv "jupyter" || {
    err "Failed to create venv for jupyter. Check ${LOG_FILE} for details."
    exit 1
}
VENV_JUPYTER="${VENV_PATH}"
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
    "## 1. Verify the unified LLM router\n",
    "All labs share `/opt/ai-security-lab/llm.env`. Configure once with:\n",
    "```\n",
    "sudo lab-llm configure --model gpt-4o-mini --api-key sk-...\n",
    "```\n",
    "Then run this cell to confirm the endpoint is reachable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '/opt/ai-security-lab/lib')\n",
    "from llm_client import MODEL, BASE_URL, API_KEY\n",
    "print(f'Model:    {MODEL}')\n",
    "print(f'Base URL: {BASE_URL}')\n",
    "print(f'API key:  {(API_KEY[:3] + \"…\" + API_KEY[-4:]) if len(API_KEY) >= 8 else \"(unset)\"}')"
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
    "## 3. Chat with the configured LLM\n",
    "Sends one chat completion through whatever endpoint `lab-llm configure` last set. Works with any kernel — uses only the standard library."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys, json, urllib.request\n",
    "sys.path.insert(0, '/opt/ai-security-lab/lib')\n",
    "from llm_client import MODEL, BASE_URL, API_KEY\n",
    "\n",
    "PROMPT = 'Explain prompt injection in one sentence.'\n",
    "req = urllib.request.Request(\n",
    "    BASE_URL.rstrip('/') + '/chat/completions',\n",
    "    data=json.dumps({\n",
    "        'model': MODEL,\n",
    "        'messages': [{'role': 'user', 'content': PROMPT}],\n",
    "    }).encode(),\n",
    "    headers={\n",
    "        'Content-Type': 'application/json',\n",
    "        'Authorization': f'Bearer {API_KEY}',\n",
    "    },\n",
    ")\n",
    "resp = urllib.request.urlopen(req, timeout=120)\n",
    "print(json.loads(resp.read())['choices'][0]['message']['content'])"
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
# 14b. UNIFIED LLM ROUTER (lab-llm CLI + shared config)
# =============================================================================
header "Unified LLM router"

LAB_BIN_DIR="${LAB_ROOT}/bin"
LAB_LIB_DIR="${LAB_ROOT}/lib"
LAB_LLM_ENV="${LAB_ROOT}/llm.env"
mkdir -p "$LAB_BIN_DIR" "$LAB_LIB_DIR"

log "Installing /usr/local/bin/lab-llm..."
cat > /usr/local/bin/lab-llm << 'LABLLMEOF'
#!/usr/bin/env bash
# lab-llm — unified LLM configuration for AI Security Lab.
# One CLI to point every tool at the same OpenAI-compatible endpoint.
set -euo pipefail

LAB_ROOT="/opt/ai-security-lab"
LLM_ENV="${LAB_ROOT}/llm.env"
DEFAULT_BASE_URL="https://api.openai.com/v1"

if [[ ${LAB_LLM_ALLOW_ROOT:-0} -eq 0 && $EUID -eq 0 && -z "${SUDO_USER:-}" ]]; then
    echo "Run via 'sudo' (so SUDO_USER is set) or as your user. Set LAB_LLM_ALLOW_ROOT=1 to override." >&2
    exit 1
fi
REAL_USER="${SUDO_USER:-$(whoami)}"
REAL_HOME=$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6 || true)
[[ -n "$REAL_HOME" ]] || REAL_HOME="$HOME"

usage() {
    cat <<USAGE
lab-llm — configure all AI Security Lab tools to use one OpenAI-compatible endpoint.

Usage:
  lab-llm configure --model MODEL --api-key KEY [--base-url URL]
  lab-llm show
  lab-llm doctor
  lab-llm -h | --help

Examples:
  sudo lab-llm configure --model gpt-4o-mini --api-key sk-...
  sudo lab-llm configure --model llama3.2 --api-key local \\
                         --base-url http://127.0.0.1:11434/v1
  lab-llm show
  lab-llm doctor

Default base URL: ${DEFAULT_BASE_URL}

Files written by 'configure':
  ${LLM_ENV}                              (canonical, mode 0600)
  \$HOME/.pyrit/.env                       (PyRIT-specific var names)
  ${LAB_ROOT}/repos/<repo>/.env           (where the repo was cloned)
  ${LAB_ROOT}/promptfooconfig.yaml        (PromptFoo starter config)
  ${LAB_ROOT}/garak-target.yaml           (Garak openai.OpenAICompatible config)
USAGE
}

# Idempotent upsert of KEY=VALUE in an env file. Preserves other lines.
upsert_env() {
    local file="$1" key="$2" value="$3"
    local dir
    dir=$(dirname "$file")
    [[ -d "$dir" ]] || install -d -m 0755 -o "$REAL_USER" -g "$REAL_USER" "$dir" 2>/dev/null \
                       || mkdir -p "$dir"
    [[ -f "$file" ]] || : > "$file"
    if grep -qE "^${key}=" "$file" 2>/dev/null; then
        local tmp="${file}.tmp.$$"
        awk -v k="$key" -v v="$value" '
            BEGIN { FS=OFS="=" }
            { if ($1 == k) { print k "=" v; next } print }
        ' "$file" > "$tmp"
        mv "$tmp" "$file"
    else
        printf '%s=%s\n' "$key" "$value" >> "$file"
    fi
    chown "$REAL_USER:$REAL_USER" "$file" 2>/dev/null || true
}

write_atomic() {
    local target="$1" mode="$2" content="$3"
    local dir
    dir=$(dirname "$target")
    [[ -d "$dir" ]] || install -d -m 0755 -o "$REAL_USER" -g "$REAL_USER" "$dir" 2>/dev/null \
                       || mkdir -p "$dir"
    local tmp="${target}.tmp.$$"
    printf '%s' "$content" > "$tmp"
    chmod "$mode" "$tmp"
    chown "$REAL_USER:$REAL_USER" "$tmp" 2>/dev/null || true
    mv "$tmp" "$target"
}

cmd_configure() {
    local model="" api_key="" base_url="$DEFAULT_BASE_URL"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)    model="${2:-}"; shift 2 || true;;
            --api-key)  api_key="${2:-}"; shift 2 || true;;
            --base-url) base_url="${2:-}"; shift 2 || true;;
            -h|--help)  usage; exit 0;;
            *) echo "Unknown argument: $1" >&2; usage; exit 2;;
        esac
    done
    [[ -n "$model" ]]   || { echo "--model is required" >&2; exit 2; }
    [[ -n "$api_key" ]] || { echo "--api-key is required" >&2; exit 2; }
    base_url="${base_url%/}"

    write_atomic "$LLM_ENV" 0600 \
"OPENAI_API_KEY=${api_key}
OPENAI_BASE_URL=${base_url}
OPENAI_MODEL=${model}
"

    local pyrit_env="${REAL_HOME}/.pyrit/.env"
    upsert_env "$pyrit_env" "OPENAI_CHAT_KEY"      "${api_key}"
    # PyRIT 0.11+ uses the OpenAI SDK under the hood and appends /chat/completions
    # itself. Passing the full path triggers a deprecation warning. Pass the base URL.
    upsert_env "$pyrit_env" "OPENAI_CHAT_ENDPOINT" "${base_url}"
    upsert_env "$pyrit_env" "OPENAI_CHAT_MODEL"    "${model}"
    chmod 0600 "$pyrit_env" 2>/dev/null || true

    local repo_envs=(
        "${LAB_ROOT}/repos/AI-Red-Teaming-Playground-Labs/.env"
        "${LAB_ROOT}/repos/finbot-ctf/.env"
        "${LAB_ROOT}/repos/damn-vulnerable-llm-agent/.env"
        "${LAB_ROOT}/repos/MAESTRO/.env"
    )
    for repo_env in "${repo_envs[@]}"; do
        [[ -d "$(dirname "$repo_env")" ]] || continue
        upsert_env "$repo_env" "OPENAI_API_KEY"  "${api_key}"
        upsert_env "$repo_env" "OPENAI_BASE_URL" "${base_url}"
        upsert_env "$repo_env" "OPENAI_MODEL"    "${model}"
        chmod 0600 "$repo_env" 2>/dev/null || true
    done

    # MAESTRO's Genkit config switches on LLM_PROVIDER and reads LLM_MODEL,
    # not OPENAI_MODEL. Set both alongside the OPENAI_* triplet so the
    # patched genkit.ts picks the OpenAI-compat plugin.
    local maestro_env="${LAB_ROOT}/repos/MAESTRO/.env"
    if [[ -d "$(dirname "$maestro_env")" ]]; then
        upsert_env "$maestro_env" "LLM_PROVIDER" "openai"
        upsert_env "$maestro_env" "LLM_MODEL"    "${model}"
        chmod 0600 "$maestro_env" 2>/dev/null || true
    fi

    write_atomic "${LAB_ROOT}/promptfooconfig.yaml" 0644 \
"# Generated by 'lab-llm configure'. Re-run to regenerate.
description: AI Security Lab — default OpenAI-compatible provider
providers:
  - id: openai:chat:${model}
    config:
      apiBaseUrl: ${base_url}
      apiKeyEnvar: OPENAI_API_KEY
prompts:
  - 'Answer concisely: {{prompt}}'
tests:
  - vars:
      prompt: 'What is prompt injection?'
"

    # Garak full config — openai.OpenAICompatible needs uri set via config,
    # not env. Use with: garak --config /opt/ai-security-lab/garak-target.yaml --probes ...
    write_atomic "${LAB_ROOT}/garak-target.yaml" 0644 \
"# Generated by 'lab-llm configure'. Re-run to regenerate.
plugins:
  model_type: openai.OpenAICompatible
  model_name: ${model}
  generators:
    openai:
      OpenAICompatible:
        uri: ${base_url}
"

    local redacted="${api_key:0:3}…${api_key: -4}"
    [[ ${#api_key} -lt 8 ]] && redacted="(short key)"
    echo "[+] Wrote ${LLM_ENV}"
    echo "[+] Wrote ${pyrit_env}"
    echo "[+] Updated per-repo .env files (where repos exist)"
    echo "[+] Wrote ${LAB_ROOT}/promptfooconfig.yaml"
    echo "[+] Wrote ${LAB_ROOT}/garak-target.yaml"
    echo
    echo "  Model:    ${model}"
    echo "  Base URL: ${base_url}"
    echo "  API key:  ${redacted}"
    echo
    echo "Next:"
    echo "  lab-llm doctor                                  # verify the endpoint"
    echo "  source ${LAB_ROOT}/bin/source-llm-env           # export vars in this shell"
}

cmd_show() {
    if [[ ! -f "$LLM_ENV" ]]; then
        echo "No config yet. Run: sudo lab-llm configure --model ... --api-key ..." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    set -a; . "$LLM_ENV"; set +a
    local key="${OPENAI_API_KEY:-}"
    local redacted="(unset)"
    if [[ -n "$key" ]]; then
        if [[ ${#key} -ge 8 ]]; then
            redacted="${key:0:3}…${key: -4}"
        else
            redacted="(short key)"
        fi
    fi
    echo "Model:    ${OPENAI_MODEL:-(unset)}"
    echo "Base URL: ${OPENAI_BASE_URL:-(unset)}"
    echo "API key:  ${redacted}"
}

cmd_doctor() {
    if [[ ! -f "$LLM_ENV" ]]; then
        echo "No config yet. Run: sudo lab-llm configure --model ... --api-key ..." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    set -a; . "$LLM_ENV"; set +a
    python3 - <<'PY'
import json, os, sys, urllib.request, urllib.error
url = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"
req = urllib.request.Request(
    url,
    data=json.dumps({
        "model": os.environ["OPENAI_MODEL"],
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    },
)
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        body = json.loads(r.read())
    print("[OK] endpoint responded.")
    if body.get("choices"):
        msg = body["choices"][0].get("message", {}).get("content", "")
        print(f"     reply: {msg!r}")
    sys.exit(0)
except urllib.error.HTTPError as e:
    snippet = e.read().decode(errors="replace")[:200]
    print(f"[FAIL] HTTP {e.code}: {snippet}")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)
PY
}

case "${1:-}" in
    configure) shift; cmd_configure "$@";;
    show)      shift; cmd_show      "$@";;
    doctor)    shift; cmd_doctor    "$@";;
    -h|--help|"") usage;;
    *) echo "Unknown command: $1" >&2; usage; exit 2;;
esac
LABLLMEOF
chmod 0755 /usr/local/bin/lab-llm
log "  /usr/local/bin/lab-llm installed."

log "Installing ${LAB_LIB_DIR}/llm_client.py..."
cat > "${LAB_LIB_DIR}/llm_client.py" << 'LLMCLIENTEOF'
"""AI Security Lab — shared LLM client.

Reads OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL from
/opt/ai-security-lab/llm.env (env vars in the current process win).

Usage:
    import sys
    sys.path.insert(0, "/opt/ai-security-lab/lib")
    from llm_client import get_client, MODEL
    client = get_client()
    r = client.chat.completions.create(model=MODEL, messages=[...])
"""
from __future__ import annotations
import os
from pathlib import Path

_ENV_FILE = Path("/opt/ai-security-lab/llm.env")


def _load_env_file() -> None:
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env_file()

MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY: str = os.environ.get("OPENAI_API_KEY", "")


def get_client():
    """Return a configured openai.OpenAI client. Requires `pip install openai`."""
    from openai import OpenAI
    return OpenAI(api_key=API_KEY or "missing", base_url=BASE_URL)
LLMCLIENTEOF

log "Installing ${LAB_BIN_DIR}/source-llm-env..."
cat > "${LAB_BIN_DIR}/source-llm-env" << 'SRCENVEOF'
# Source this file (don't execute it) to export OPENAI_* vars in the current shell:
#   source /opt/ai-security-lab/bin/source-llm-env
LAB_LLM_ENV="/opt/ai-security-lab/llm.env"
if [ -f "$LAB_LLM_ENV" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$LAB_LLM_ENV"
    # Tool-specific aliases for tools that don't read OPENAI_API_KEY directly.
    # Garak's openai.OpenAICompatible target reads from OPENAICOMPATIBLE_API_KEY.
    OPENAICOMPATIBLE_API_KEY="${OPENAI_API_KEY}"
    set +a
else
    echo "No LLM config at $LAB_LLM_ENV. Run: sudo lab-llm configure --model ... --api-key ..." >&2
    return 1 2>/dev/null || exit 1
fi
SRCENVEOF
chmod 0644 "${LAB_BIN_DIR}/source-llm-env"

# Canonical config stub (populated by `lab-llm configure`)
if [[ ! -f "$LAB_LLM_ENV" ]]; then
    cat > "$LAB_LLM_ENV" << 'LLMENVEOF'
# AI Security Lab — unified LLM configuration.
# Populated by `sudo lab-llm configure --model ... --api-key ... [--base-url ...]`
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
LLMENVEOF
    chmod 0600 "$LAB_LLM_ENV"
fi

log "Unified LLM router installed."
log "  Configure with:  sudo lab-llm configure --model gpt-4o-mini --api-key sk-..."
log "  Verify with:     sudo lab-llm doctor"

# =============================================================================
# 15. REFERENCE LINKS & DOCUMENTATION
# =============================================================================
header "15/15 - Reference links & documentation"

cat > "${LAB_ROOT}/REFERENCES.md" << 'REFEOF'
# AI Security Lab - Reference Links & Resources

## LLM Configuration (read this first)

All tools share a single OpenAI-compatible endpoint configured via `lab-llm`:

```bash
sudo lab-llm configure --model gpt-4o-mini --api-key sk-...
sudo lab-llm doctor                              # verify
source /opt/ai-security-lab/bin/source-llm-env   # export OPENAI_* in this shell
```

Canonical config lives at `/opt/ai-security-lab/llm.env`. Re-running `configure`
also updates the per-repo `.env` files, `~/.pyrit/.env`, and `promptfooconfig.yaml`.

Switch model or provider with one CLI call (no per-tool edits):

```bash
sudo lab-llm configure --model claude-3-5-sonnet --api-key sk-... \
                       --base-url https://openrouter.ai/api/v1
```

For offline use with a local backend, see the "Optional: offline backend with
Ollama" appendix in `AI_SECURITY_LAB_GUIDE.md`.

## Installed Tools

| Tool | Location | Activation |
|------|----------|------------|
| lab-llm | /usr/local/bin/lab-llm | `lab-llm --help` |
| LM Eval Harness | /opt/ai-security-lab/repos/lm-evaluation-harness | `source /opt/ai-security-lab/venvs/lm-eval-harness/bin/activate` |
| PromptFoo | global npm | `promptfoo` |
| CleverHans | /opt/ai-security-lab/venvs/cleverhans | `source /opt/ai-security-lab/venvs/cleverhans/bin/activate` |
| Garak | /opt/ai-security-lab/venvs/garak | `source /opt/ai-security-lab/venvs/garak/bin/activate` |
| Giskard | /opt/ai-security-lab/venvs/giskard | `source /opt/ai-security-lab/venvs/giskard/bin/activate` |
| PyRIT | /opt/ai-security-lab/venvs/pyrit | `source /opt/ai-security-lab/venvs/pyrit/bin/activate` |
| Ollama (optional, offline backend) | system-wide | `ollama` |
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
# 1. One-time LLM router configuration (do this first)
sudo lab-llm configure --model gpt-4o-mini --api-key sk-...
sudo lab-llm doctor

# 2. Activate a Python tool (example: Garak)
source /opt/ai-security-lab/venvs/garak/bin/activate
source /opt/ai-security-lab/bin/source-llm-env
garak --target_type openai.OpenAICompatible \
      --target_name "$OPENAI_MODEL" \
      --probes encoding.InjectBase64

# 3. Run PromptFoo using the generated starter config
promptfoo eval -c /opt/ai-security-lab/promptfooconfig.yaml

# 4. Launch a Docker-based lab (reads OPENAI_* from its repo .env)
cd /opt/ai-security-lab/repos/vulnerable-llms
docker-compose -f docker-compose.override.yml up
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
- Ollama (optional):    https://ollama.com

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

  2. Configure the LLM router (one CLI call propagates to every lab):
       sudo lab-llm configure --model gpt-4o-mini --api-key sk-...
       sudo lab-llm doctor                              # verify endpoint

     This writes ${LAB_LLM_ENV}, ~/.pyrit/.env, each repo's .env,
     and ${LAB_ROOT}/promptfooconfig.yaml. To switch model or provider
     later, just re-run 'lab-llm configure'.

  3. Launch JupyterLab:
       source ${VENV_BASE}/jupyter/bin/activate
       jupyter lab --notebook-dir=${LAB_ROOT}/notebooks

  4. Start exploring:
       cat ${LAB_ROOT}/REFERENCES.md

  Optional (offline backend):
     See "Optional: offline backend with Ollama" in AI_SECURITY_LAB_GUIDE.md.

$(echo -e "${RED}WARNING: This lab contains intentionally vulnerable applications.${NC}")
$(echo -e "${RED}Run only in isolated environments. Never expose to the internet.${NC}")

EOF
