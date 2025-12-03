# 🎨 GUI Quick Start

## Launch the GUI

The easiest way to use your Liquid-Spiking Neural Network!

### Method 1: Simple Launcher (Recommended)

```bash
python launch_gui.py
```

The launcher will:
- ✅ Check for required dependencies (PyQt6)
- ✅ Auto-install missing packages
- ✅ Launch the GUI application

### Method 2: Direct Launch

```bash
python scripts/gui.py
```

---

## First Time Setup

If PyQt6 is not installed:

```bash
pip install PyQt6
```

---

## What Can You Do?

### 🚀 Training Tab
- Select task type (LLM, Vision, Robotics)
- Configure architecture parameters
- Choose datasets (WikiText-103 recommended!)
- Start training with one click
- Monitor progress in real-time

### 💡 Inference Tab
- Load trained models
- Generate text with prompts
- Adjust temperature and length
- See results instantly

### ⚙️ Settings Tab
- Select GPU/CPU device
- Configure multi-GPU training
- Set output directories
- View system information

### 🤖 Model Types Tab
- Learn about architecture
- Compare LLM vs Vision vs Robotics
- Understand hybrid components
- See use cases and examples

---

## Quick Training Example

1. **Launch GUI**: `python launch_gui.py`
2. **Training Tab**: Select "LLM (Language Model)"
3. **Dataset**: Choose "WikiText-103 (100M tokens - RECOMMENDED)"
4. **Epochs**: Set to 30
5. **Click**: "🚀 Start Training"
6. **Wait**: ~2-3 hours on GPU
7. **Done**: Model saved to `./models/llm_final_model.pt`

---

## Quick Inference Example

1. **Inference Tab**: Click "📂 Browse"
2. **Select**: `./models/llm_final_model.pt`
3. **Load**: Click "📥 Load Model"
4. **Prompt**: Type "The future of artificial intelligence"
5. **Generate**: Click "✨ Generate"
6. **Read**: Your generated text!

---

## Full Documentation

See **GUI_GUIDE.md** for complete documentation including:
- Detailed tab explanations
- Configuration options
- Best practices
- Troubleshooting
- Advanced features

---

## Screenshots (Coming Soon)

The GUI features:
- Modern dark theme
- Real-time training progress
- Interactive parameter tuning
- Live log output
- Model information display

---

**Enjoy the GUI! 🎉**
