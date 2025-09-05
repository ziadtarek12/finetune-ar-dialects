#!/bin/bash

# Arabic Whisper Fine-tuning Experiment Runner
# This script runs both traditional fine-tuning and PEFT experiments

set -e

# Configuration
DIALECTS=("egyptian" "gulf" "iraqi" "levantine" "maghrebi" "all")
SEEDS=(42 84 168)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        return 0
    else
        print_warning "No GPU detected. Training will be slow on CPU."
        return 1
    fi
}

# Function to install requirements
install_requirements() {
    print_status "Installing requirements..."
    pip install -r requirements.txt
    print_success "Requirements installed."
}

# Function to run traditional fine-tuning
run_traditional_training() {
    local dialect=$1
    local seeds_str="${SEEDS[*]}"
    
    print_status "Running traditional fine-tuning for ${dialect} dialect..."
    python src/training/experiment_finetune.py --dialect ${dialect} --seeds ${seeds_str}
    print_success "Traditional fine-tuning completed for ${dialect}."
}

# Function to run PEFT training
run_peft_training() {
    local dialect=$1
    local seeds_str="${SEEDS[*]}"
    
    print_status "Running PEFT training for ${dialect} dialect..."
    python src/training/experiment_finetune_peft.py \
        --dialect ${dialect} \
        --use_peft \
        --load_in_8bit \
        --seeds ${seeds_str}
    print_success "PEFT training completed for ${dialect}."
}

# Function to compare results
compare_results() {
    local dialect=$1
    
    print_status "Comparing results for ${dialect} dialect..."
    
    # Check if results exist
    traditional_results="results/ex_finetune/results_whisper-small-finetune_${dialect}_seed42.json"
    peft_results="results/ex_peft/results_whisper-small-peft_${dialect}_seed42.json"
    
    if [[ -f "$traditional_results" && -f "$peft_results" ]]; then
        print_status "Results comparison for ${dialect}:"
        echo "Traditional fine-tuning results: $traditional_results"
        echo "PEFT training results: $peft_results"
        # You can add Python script here to parse and compare JSON results
    else
        print_warning "Results files not found for comparison."
    fi
}

# Main execution function
main() {
    echo "================================================"
    echo "Arabic Whisper Fine-tuning Experiment Runner"
    echo "================================================"
    
    # Parse command line arguments
    EXPERIMENT_TYPE="both"  # Default to both
    DIALECT_FILTER=""
    SKIP_INSTALL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                EXPERIMENT_TYPE="$2"
                shift 2
                ;;
            --dialect)
                DIALECT_FILTER="$2"
                shift 2
                ;;
            --skip-install)
                SKIP_INSTALL=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --type {both|traditional|peft}    Experiment type to run (default: both)"
                echo "  --dialect {dialect_name}           Run for specific dialect only"
                echo "  --skip-install                     Skip package installation"
                echo "  --help                              Show this help message"
                echo ""
                echo "Available dialects: ${DIALECTS[*]}"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check GPU
    check_gpu
    
    # Install requirements unless skipped
    if [[ "$SKIP_INSTALL" == false ]]; then
        install_requirements
    fi
    
    # Determine which dialects to process
    if [[ -n "$DIALECT_FILTER" ]]; then
        if [[ " ${DIALECTS[*]} " =~ " $DIALECT_FILTER " ]]; then
            DIALECTS_TO_PROCESS=("$DIALECT_FILTER")
        else
            print_error "Invalid dialect: $DIALECT_FILTER"
            print_status "Available dialects: ${DIALECTS[*]}"
            exit 1
        fi
    else
        DIALECTS_TO_PROCESS=("${DIALECTS[@]}")
    fi
    
    # Run experiments
    for dialect in "${DIALECTS_TO_PROCESS[@]}"; do
        echo ""
        print_status "Processing dialect: $dialect"
        echo "----------------------------------------"
        
        case $EXPERIMENT_TYPE in
            "traditional")
                run_traditional_training "$dialect"
                ;;
            "peft")
                run_peft_training "$dialect"
                ;;
            "both")
                run_traditional_training "$dialect"
                echo ""
                run_peft_training "$dialect"
                echo ""
                compare_results "$dialect"
                ;;
            *)
                print_error "Invalid experiment type: $EXPERIMENT_TYPE"
                exit 1
                ;;
        esac
        
        echo ""
    done
    
    print_success "All experiments completed!"
    echo ""
    echo "================================================"
    echo "Summary:"
    echo "- Processed dialects: ${DIALECTS_TO_PROCESS[*]}"
    echo "- Experiment type: $EXPERIMENT_TYPE"
    echo "- Random seeds: ${SEEDS[*]}"
    echo "================================================"
}

# Run main function with all arguments
main "$@"
