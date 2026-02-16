#!/usr/bin/env bash
# =============================================================================
# AI Security Lab - Docker Test Runner
# =============================================================================
# Builds a Debian 13 container and runs the setup script inside it.
#
# Usage:
#   ./test_in_docker.sh              # Full install (takes 15-30 min)
#   ./test_in_docker.sh --shell      # Build only, drop into bash for manual testing
#   ./test_in_docker.sh --dry-run    # Syntax check only (shellcheck + bash -n)
#   ./test_in_docker.sh --fast       # Skip heavy installs (PyTorch, HuggingFace)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="ai-seclab-test"
CONTAINER_NAME="ai-seclab-test-run"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[TEST]${NC} $*"; }
warn() { echo -e "${YELLOW}[TEST]${NC} $*"; }
err()  { echo -e "${RED}[TEST]${NC} $*"; }

# ── Parse arguments ──────────────────────────────────────────────────────────
MODE="full"
case "${1:-}" in
    --shell)   MODE="shell"   ;;
    --dry-run) MODE="dry-run" ;;
    --fast)    MODE="fast"    ;;
    --help|-h)
        echo "Usage: $0 [--shell|--dry-run|--fast|--help]"
        echo ""
        echo "  (no args)   Full install inside Docker container"
        echo "  --shell     Build image, then drop into interactive bash"
        echo "  --dry-run   Syntax check only (shellcheck + bash -n)"
        echo "  --fast      Skip heavy packages (PyTorch, large pip installs)"
        echo "  --help      Show this help"
        exit 0
        ;;
esac

cd "$SCRIPT_DIR"

# ── Dry-run mode: syntax checks only ────────────────────────────────────────
if [[ "$MODE" == "dry-run" ]]; then
    log "Running syntax checks..."

    # bash -n: check for syntax errors
    if bash -n setup_ai_security_lab.sh; then
        log "bash -n: ${GREEN}PASS${NC} - no syntax errors"
    else
        err "bash -n: FAIL - syntax errors found"
        exit 1
    fi

    # shellcheck (if installed)
    if command -v shellcheck &>/dev/null; then
        # SC2034: unused variables (we use colors that appear unused)
        # SC2086: word splitting (intentional in some places)
        # SC2155: declare and assign separately (convenience pattern)
        if shellcheck -e SC2034,SC2086,SC2155 -s bash setup_ai_security_lab.sh; then
            log "shellcheck: ${GREEN}PASS${NC}"
        else
            warn "shellcheck: found warnings (see above)"
        fi
    else
        warn "shellcheck not installed - skipping. Install with: brew install shellcheck"
    fi

    log "Dry-run complete."
    exit 0
fi

# ── Check Docker / Colima is available ───────────────────────────────────────
if ! command -v docker &>/dev/null; then
    err "Docker CLI is not installed or not in PATH."
    err "If you use Colima: brew install docker colima && colima start"
    err "Or install Docker Desktop: https://docs.docker.com/desktop/install/mac-install/"
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    # Try auto-starting Colima if available
    if command -v colima &>/dev/null; then
        warn "Docker daemon not running. Attempting to start Colima..."
        colima start --cpu 4 --memory 6 2>&1 || true
        sleep 3
    fi
    if ! docker info &>/dev/null 2>&1; then
        err "Docker daemon is not running."
        err "  Colima users:  colima start --cpu 4 --memory 6"
        err "  Docker Desktop: start the Docker Desktop app"
        exit 1
    fi
fi
log "Docker runtime detected: $(docker info --format '{{.Name}}' 2>/dev/null || echo 'OK')"

# ── Build the test image ────────────────────────────────────────────────────
log "Building Debian 13 test image..."
docker build \
    -f Dockerfile.test \
    -t "$IMAGE_NAME" \
    "$SCRIPT_DIR"

log "Image built: ${IMAGE_NAME}"

# ── Shell mode: interactive bash ─────────────────────────────────────────────
if [[ "$MODE" == "shell" ]]; then
    log "Dropping into interactive shell..."
    warn "Run the setup script manually with:"
    warn "  sudo SUDO_USER=labuser bash /tmp/setup_ai_security_lab.sh"
    echo ""
    docker run -it --rm \
        --name "$CONTAINER_NAME" \
        --hostname ai-seclab \
        "$IMAGE_NAME" \
        /bin/bash
    exit 0
fi

# ── Fast mode: patch the script to skip heavy installs ───────────────────────
SETUP_CMD="sudo SUDO_USER=labuser bash /tmp/setup_ai_security_lab.sh"
if [[ "$MODE" == "fast" ]]; then
    warn "Fast mode: patching script to skip PyTorch and heavy downloads..."
    # We'll create a wrapper that modifies the script before running
    SETUP_CMD=$(cat << 'FASTEOF'
# Patch: replace heavy pip installs with lightweight stubs
cp /tmp/setup_ai_security_lab.sh /tmp/setup_fast.sh
# Skip Ollama model pulls (replace the background pull with a no-op)
sed -i 's|su - "\$REAL_USER" -c "ollama pull llama3.2:1b".*|log "FAST MODE: Skipping model pull"|' /tmp/setup_fast.sh
sed -i 's|wait \$OLLAMA_PID.*|true|' /tmp/setup_fast.sh
# Replace PyTorch install with CPU-only slim version
sed -i 's|install cleverhans torch torchvision|install cleverhans --index-url https://download.pytorch.org/whl/cpu torch torchvision|' /tmp/setup_fast.sh
sudo SUDO_USER=labuser bash /tmp/setup_fast.sh
FASTEOF
    )
fi

# ── Full / Fast mode: run the setup script ──────────────────────────────────
log "Starting setup inside container (mode: ${MODE})..."
log "This will take 15-30 minutes for full mode, 5-10 for fast mode."
echo ""

# Allocate 2GB+ memory for the container (needed for Python source build)
docker run --rm \
    --name "$CONTAINER_NAME" \
    --hostname ai-seclab \
    --memory=4g \
    "$IMAGE_NAME" \
    /bin/bash -c "$SETUP_CMD"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    log "${GREEN}SUCCESS${NC} - Setup script completed without errors."
    log ""
    log "Next steps:"
    log "  1. Run with --shell to explore the installed environment:"
    log "     $0 --shell"
    log "  2. Inside the container, check installed tools:"
    log "     source /opt/ai-security-lab/venvs/garak/bin/activate && python -m garak --help"
    log "     cat /opt/ai-security-lab/REFERENCES.md"
else
    echo ""
    err "FAILED - Setup script exited with code ${EXIT_CODE}."
    err "Run with --shell to debug interactively:"
    err "  $0 --shell"
    exit $EXIT_CODE
fi
