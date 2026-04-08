#!/bin/bash
# UV Environment Setup Script for lmms-eval
# This script sets up a Python virtual environment using uv and installs dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
PYTHON_VERSION="3.10"

# Print functions
print_info() {
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

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv first."
        echo "Install options:"
        echo "  - curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  - pip install uv"
        exit 1
    fi
    print_success "uv is installed ($(uv --version))"
}

# Check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python ${PYTHON_VERSION} or higher."
        exit 1
    fi
    
    local python_version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Found Python ${python_version}"
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment with uv..."
    
    # Remove existing venv if it exists
    if [ -d "$VENV_DIR" ]; then
        print_warning "Existing virtual environment found. Removing..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create new venv
    uv venv "$VENV_DIR" --python python3
    print_success "Virtual environment created at $VENV_DIR"
}

# Install dependencies
install_deps() {
    print_info "Installing dependencies..."
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Install project in editable mode with core dependencies
    uv pip install -e ".[core]"
    
    print_success "Core dependencies installed"
}

# Install all optional dependencies
install_all_deps() {
    print_info "Installing all dependencies (including optional ones)..."
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    uv pip install -e ".[all]"
    
    print_success "All dependencies installed"
}

# Install development dependencies
install_dev_deps() {
    print_info "Installing development dependencies..."
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    uv pip install -e ".[dev]"
    
    print_success "Development dependencies installed"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -a, --all       Install all optional dependencies"
    echo "  -d, --dev       Install development dependencies"
    echo "  -c, --clean     Clean existing virtual environment before setup"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Basic setup with core dependencies"
    echo "  $0 -a                 # Setup with all optional dependencies"
    echo "  $0 -d                 # Setup with development dependencies"
    echo "  $0 -a -d              # Setup with all + dev dependencies"
}

# Main function
main() {
    local install_all=ture
    local install_dev=false
    local clean=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                install_all=true
                shift
                ;;
            -d|--dev)
                install_dev=true
                shift
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_info "Starting uv environment setup for lmms-eval..."
    
    # Checks
    check_uv
    check_python
    
    # Clean if requested
    if [ "$clean" = true ] && [ -d "$VENV_DIR" ]; then
        print_info "Cleaning existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create virtual environment
    create_venv
    
    # Install dependencies based on options
    if [ "$install_all" = true ]; then
        install_all_deps
    else
        install_deps
    fi
    
    if [ "$install_dev" = true ]; then
        install_dev_deps
    fi
    
    # Activate virtual environment in current shell
    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Print summary
    echo ""
    print_success "Setup complete! 🎉"
    echo ""
    print_success "Virtual environment is now ACTIVE!"
    echo ""
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
    echo ""
    echo "Available commands:"
    echo "  lmms-eval          - Main evaluation CLI"
    echo "  lmms-eval-mcp      - MCP server CLI"
    echo "  lmms-eval-ui       - TUI interface"
    echo ""
    echo "To use uv run (recommended):"
    echo "  uv run python -m lmms_eval --help"
}

# Run main function
main "$@"
