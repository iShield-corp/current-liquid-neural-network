#!/bin/bash
# install_dependencies.sh - Robust dependency installer for Hybrid Liquid-Spiking Neural Network

set -e  # Exit on any error

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

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🧠 Installing Dependencies for Hybrid Liquid-Spiking Neural Network"
echo "===================================================================="

# Check Python version
log_info "Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_success "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    log_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Update pip
log_info "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install core dependencies first
log_info "Installing core dependencies..."
pip install torch>=2.0.0 torchvision>=0.15.0

# Install OpenCV with fallback options
log_info "Installing OpenCV (trying headless version first)..."
if pip install opencv-python-headless>=4.5.0; then
    log_success "OpenCV headless installed successfully"
else
    log_warning "OpenCV headless failed, trying standard version..."
    if pip install opencv-python>=4.5.0; then
        log_success "OpenCV standard installed successfully"
    else
        log_warning "OpenCV installation failed. Some robotics features may not work."
    fi
fi

# Install remaining dependencies
log_info "Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

# Test imports
log_info "Testing critical imports..."

python3 -c "
import sys
import torch
print(f'✅ PyTorch {torch.__version__}')

try:
    import cv2
    print(f'✅ OpenCV {cv2.__version__}')
except ImportError:
    print('⚠️  OpenCV not available (some features disabled)')

try:
    from src.core.main import LiquidSpikingNetwork
    print('✅ Core modules imported successfully')
except ImportError as e:
    print(f'❌ Core module import failed: {e}')
    sys.exit(1)

try:
    from src.datasets.advanced_programming_datasets import AdvancedProgrammingDataset
    print('✅ Dataset modules imported successfully')
except ImportError as e:
    print(f'❌ Dataset module import failed: {e}')
    sys.exit(1)

print('🎉 All critical imports successful!')
"

if [ $? -eq 0 ]; then
    log_success "All dependencies installed and tested successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Test training: python scripts/cli.py train --task llm --epochs 1"
    echo "2. Check system info: python scripts/cli.py info --system"
    echo "3. Run verification: python verify_setup.py"
else
    log_error "Some imports failed. Please check the error messages above."
    exit 1
fi
