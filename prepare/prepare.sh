#!/bin/bash

set -e

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-./data}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Print usage
usage() {
    cat << EOF
Primus Data Preparation Script

Usage: $(basename $0) [OPTIONS]

Options:
  --all                     Prepare everything (datasets + tokenizers + tokenization)
  --dataset <name>          Download specific dataset (bookcorpus, c4)
  --tokenizer <name>        Download specific tokenizer (see --list-tokenizers)
  --tokenize                Tokenize downloaded data for Megatron
  --list-tokenizers         List available tokenizers
  --data-dir <path>         Data directory (default: ./data)
  --workers <n>             Number of workers (default: auto)
  --help, -h                Show this help message

Datasets:
  bookcorpus    BookCorpus dataset (https://huggingface.co/datasets/bookcorpus)
  c4            C4 dataset samples (https://huggingface.co/datasets/allenai/c4)

Environment Variables:
  HF_TOKEN      HuggingFace token for gated models (required for Llama, etc.)
  DATA_DIR      Override default data directory

Examples:
  # Download BookCorpus and prepare for Megatron with Llama tokenizer
  export HF_TOKEN=your_token
  $(basename $0) --dataset bookcorpus --tokenizer llama --tokenize

  # Download all supported datasets and tokenizers
  $(basename $0) --all

  # Just download the C4 dataset
  $(basename $0) --dataset c4

EOF
}

list_tokenizers() {
    python3 "${SCRIPT_DIR}/download_tokenizer.py" --list
}

download_bookcorpus() {
    local max_samples="${1:-10000}"
    log_info "Downloading BookCorpus dataset (${max_samples} samples)..."
    python3 "${SCRIPT_DIR}/download_bookcorpus.py" \
        --output-dir "${DATA_DIR}/bookcorpus" \
        --max-samples "${max_samples}"
    log_success "BookCorpus downloaded to ${DATA_DIR}/bookcorpus"
}

download_c4() {
    local max_samples="${1:-10000}"
    log_info "Downloading C4 dataset (${max_samples} samples)..."
    python3 "${SCRIPT_DIR}/download_c4.py" \
        --output-dir "${DATA_DIR}/c4" \
        --max-samples "${max_samples}"
    log_success "C4 samples downloaded to ${DATA_DIR}/c4"
}

download_tokenizer() {
    local model="$1"
    if [ -z "$model" ]; then
        log_error "Tokenizer name required. Use --list-tokenizers to see options."
    fi
    
    log_info "Downloading tokenizer: ${model}..."
    python3 "${SCRIPT_DIR}/download_tokenizer.py" \
        --output-dir "${DATA_DIR}/tokenizers" \
        "$model"
    log_success "Tokenizer downloaded to ${DATA_DIR}/tokenizers"
}

get_tokenizer_model() {
    local tokenizer_model="$1"
    
    if [ -z "$tokenizer_model" ]; then
        # Try to find a downloaded tokenizer
        if [ -d "${DATA_DIR}/tokenizers" ]; then
            local first_tokenizer=$(ls -1 "${DATA_DIR}/tokenizers" 2>/dev/null | head -1)
            if [ -n "$first_tokenizer" ]; then
                tokenizer_model="${DATA_DIR}/tokenizers/${first_tokenizer}"
            fi
        fi
    fi
    
    echo "$tokenizer_model"
}

tokenize_bookcorpus() {
    local tokenizer_model="$1"
    local input_file="${DATA_DIR}/bookcorpus/bookcorpus_megatron.json"
    local output_prefix="${DATA_DIR}/megatron/bookcorpus"
    
    if [ ! -f "$input_file" ]; then
        log_warn "BookCorpus input file not found: ${input_file}. Skipping."
        return 1
    fi
    
    tokenizer_model=$(get_tokenizer_model "$tokenizer_model")
    
    if [ -z "$tokenizer_model" ]; then
        log_error "No tokenizer specified. Run --tokenizer <name> first or specify model."
    fi
    
    mkdir -p "${DATA_DIR}/megatron"
    
    log_info "Tokenizing BookCorpus with ${tokenizer_model}..."
    python3 "${SCRIPT_DIR}/tokenize_megatron.py" \
        --input "$input_file" \
        --output-prefix "$output_prefix" \
        --tokenizer-model "$tokenizer_model" \
        --workers "$WORKERS" \
        --split-sentences \
        --append-eod
    
    log_success "BookCorpus tokenized to ${DATA_DIR}/megatron/"
}

tokenize_c4() {
    local tokenizer_model="$1"
    local input_file="${DATA_DIR}/c4/c4_megatron.json"
    local output_prefix="${DATA_DIR}/megatron/c4"
    
    if [ ! -f "$input_file" ]; then
        log_warn "C4 input file not found: ${input_file}. Skipping."
        return 1
    fi
    
    tokenizer_model=$(get_tokenizer_model "$tokenizer_model")
    
    if [ -z "$tokenizer_model" ]; then
        log_error "No tokenizer specified. Run --tokenizer <name> first or specify model."
    fi
    
    mkdir -p "${DATA_DIR}/megatron"
    
    log_info "Tokenizing C4 with ${tokenizer_model}..."
    python3 "${SCRIPT_DIR}/tokenize_megatron.py" \
        --input "$input_file" \
        --output-prefix "$output_prefix" \
        --tokenizer-model "$tokenizer_model" \
        --workers "$WORKERS" \
        --split-sentences \
        --append-eod
    
    log_success "C4 tokenized to ${DATA_DIR}/megatron/"
}

tokenize_data() {
    local tokenizer_model="$1"
    
    tokenizer_model=$(get_tokenizer_model "$tokenizer_model")
    
    if [ -z "$tokenizer_model" ]; then
        log_error "No tokenizer specified. Run --tokenizer <name> first or specify model."
    fi
    
    # Tokenize all available datasets
    tokenize_bookcorpus "$tokenizer_model" || true
    tokenize_c4 "$tokenizer_model" || true
    
    log_success "All available datasets tokenized to ${DATA_DIR}/megatron/"
}

prepare_all() {
    log_info "=== Preparing all datasets and tokenizers ==="
    
    if [ -z "$HF_TOKEN" ]; then
        log_warn "HF_TOKEN not set. Some downloads may fail."
    fi
    
    log_info "Step 1/4: Downloading BookCorpus..."
    download_bookcorpus
    
    log_info "Step 2/4: Downloading C4 samples..."
    download_c4 10000
    
    log_info "Step 3/4: Downloading tokenizers (llama, qwen)..."
    download_tokenizer "llama" || log_warn "Failed to download Llama tokenizer (may require HF_TOKEN)"
    download_tokenizer "qwen" || log_warn "Failed to download Qwen tokenizer"
    
    log_info "Step 4/4: Tokenizing datasets for Megatron..."
    if [ -d "${DATA_DIR}/tokenizers" ] && [ "$(ls -A ${DATA_DIR}/tokenizers 2>/dev/null)" ]; then
        log_info "  - Tokenizing BookCorpus..."
        tokenize_bookcorpus || log_warn "BookCorpus tokenization skipped"
        log_info "  - Tokenizing C4..."
        tokenize_c4 || log_warn "C4 tokenization skipped"
    else
        log_warn "No tokenizer available. Skipping tokenization."
    fi
    
    log_success "=== All preparation complete ==="
    echo ""
    echo "Data directory structure:"
    if command -v tree &> /dev/null; then
        tree -L 2 "${DATA_DIR}" 2>/dev/null || ls -la "${DATA_DIR}"
    else
        ls -la "${DATA_DIR}"
    fi
}

main() {
    local action=""
    local dataset=""
    local tokenizer=""
    local do_tokenize=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                action="all"
                shift
                ;;
            --dataset)
                dataset="$2"
                shift 2
                ;;
            --tokenizer)
                tokenizer="$2"
                shift 2
                ;;
            --tokenize)
                do_tokenize=true
                shift
                ;;
            --list-tokenizers)
                list_tokenizers
                exit 0
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --workers)
                WORKERS="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                ;;
        esac
    done
    
    mkdir -p "${DATA_DIR}"
    
    if [ "$action" = "all" ]; then
        prepare_all
        exit 0
    fi
    
    if [ -n "$dataset" ]; then
        case $dataset in
            bookcorpus)
                download_bookcorpus
                ;;
            c4)
                download_c4
                ;;
            *)
                log_error "Unknown dataset: ${dataset}. Available: bookcorpus, c4"
                ;;
        esac
    fi
    
    if [ -n "$tokenizer" ]; then
        download_tokenizer "$tokenizer"
    fi
    
    if [ "$do_tokenize" = true ]; then
        tokenize_data "$tokenizer"
    fi
    
    # Show help if no action specified
    if [ -z "$action" ] && [ -z "$dataset" ] && [ -z "$tokenizer" ] && [ "$do_tokenize" = false ]; then
        usage
    fi
}

main "$@"
