# 🖥️ Liquid-Spiking Neural Network - GUI Guide

## Overview

The GUI application provides an intuitive graphical interface for training, running inference, and managing your hybrid liquid-spiking neural networks. No command-line experience needed!

## Installation

### Prerequisites

The GUI requires PyQt6. It will auto-install if missing, but you can install manually:

```bash
pip install PyQt6
```

### Launch the GUI

```bash
cd /home/sovr610/ssn-cfc
python scripts/gui.py
```

Or use the launcher script:

```bash
python launch_gui.py
```

---

## 🎨 GUI Features

### Tab 1: 🚀 Training

**Train your neural network models with an easy-to-use interface**

#### Configuration Options

1. **Task Selection**
   - **LLM (Language Model)**: Text generation, completion, translation
   - **Vision**: Image classification, object recognition
   - **Robotics**: Control systems, sensor processing

2. **Config Presets** (Quick Start)
   - **Custom**: Build your own configuration
   - **GPT-4o Competitive**: ~100M parameters, competitive with GPT-4o
   - **Claude Sonnet 4**: ~150M parameters, matches Claude Sonnet 4
   - **Ultra-Advanced**: ~200M+ parameters, exceeds current LLMs

3. **Architecture Parameters**
   - **Number of Layers**: Stack depth (1-24)
   - **Hidden Dimension**: Internal representation size (64-2048)
   - **Liquid Units**: Continuous-time neurons (32-1024)
   - **Spiking Units**: Event-based neurons (32-512)
   - **Attention Heads**: Multi-head attention count (1-32)

4. **Training Parameters**
   - **Epochs**: Training iterations (1-1000)
   - **Batch Size**: Samples per step (1-256)
   - **Learning Rate**: Optimization speed (0.00001-0.1)
   - **Dropout**: Regularization rate (0.0-0.9)

5. **Dataset Selection** (LLM Only)
   - **WikiText-103**: 100M tokens - **RECOMMENDED** for large models
   - **WikiText-2**: 2M tokens - Only for tiny models (<10M params)
   - **BookCorpus**: 70M tokens - Diverse narrative text
   - **CC-News**: 76M tokens - News articles
   - **OpenWebText**: 40M tokens - General web content
   - **Combined**: 200M+ tokens - Maximum quality

6. **Tokenizer** (LLM Only)
   - gpt2, gpt3, gpt4, o200k, codellama, llama2

7. **Advanced Options**
   - ✅ **STDP Plasticity**: Bio-inspired synaptic learning
   - ✅ **Meta-Plasticity**: Learn-to-learn capabilities
   - ✅ **Mamba Integration**: State-space model enhancement
   - ✅ **Mixed Precision**: Faster training with FP16

#### Training Process

1. **Configure** your model using the left panel
2. **Click "🚀 Start Training"**
3. **Monitor Progress**:
   - Progress bar shows completion percentage
   - Current epoch, train loss, val loss displayed
   - Real-time log shows detailed training info
4. **Checkpoints** saved automatically every 5 epochs
5. **Final model** saved to `./models/` directory

#### Tips for Training

- **LLM Models**: Start with WikiText-103 dataset, 30+ epochs
- **Vision Models**: CIFAR-10 dataset, 20+ epochs  
- **Robotics Models**: Custom sensor data, 15+ epochs
- **GPU Recommended**: Training is much faster on CUDA devices
- **Batch Size**: Increase until GPU memory ~80% full
- **Learning Rate**: Start with 0.0001, adjust if loss plateaus

---

### Tab 2: 💡 Inference

**Generate text or run predictions with trained models**

#### Load Model

1. **Browse** for your trained model file (`.pt` extension)
2. **Click "📥 Load Model"**
3. **Model Information** displays:
   - Task type
   - Total parameters
   - Device (CPU/CUDA)
   - Architecture details

#### Generate Text (LLM)

1. **Set Generation Parameters**:
   - **Max Length**: How many tokens to generate (10-1000)
   - **Temperature**: Creativity level (0.1-2.0)
     - Low (0.5): Conservative, factual
     - Medium (0.8): Balanced
     - High (1.5): Creative, diverse

2. **Enter Prompt**: Type your input text

3. **Click "✨ Generate"**

4. **View Output**: Generated text appears in the output box

#### Examples

**Prompt**: "The future of artificial intelligence"  
**Output**: Complete sentences about AI (no more gibberish!)

**Prompt**: "Write a Python function to"  
**Output**: Code completion

---

### Tab 3: ⚙️ Settings

**Configure system and environment settings**

#### Device Settings

- **Compute Device**: 
  - Auto (CUDA if available) - Recommended
  - CPU - Slower but always works
  - CUDA - Specific GPU
  - CUDA:0, CUDA:1, etc. - Multi-GPU selection

- **Multi-GPU Strategy**:
  - None - Single device
  - DataParallel - Good for 2-4 GPUs
  - DistributedDataParallel - Best for 4+ GPUs

#### Output Directories

- **Model Save Directory**: Where trained models are saved (default: `./models`)
- **Dataset Cache Directory**: Where datasets are downloaded (default: `./data`)

#### System Information

Real-time display of:
- PyTorch version
- CUDA availability
- GPU count and names
- GPU memory capacity

---

### Tab 4: 🤖 Model Types

**Learn about the different neural network architectures**

#### Model Comparison Table

| Task Type | Input Domain | Use Cases | Architecture | Applications |
|-----------|-------------|-----------|--------------|--------------|
| 🔤 **LLM** | Text/Tokens | Text generation, translation | 8 layers, 512 hidden | Chatbots, code gen |
| 👁️ **Vision** | Images/Pixels | Classification, detection | 6 layers, 256 hidden | CIFAR-10, ImageNet |
| 🤖 **Robotics** | Sensors/State | Control, navigation | 4 layers, 256 hidden | Robot arms, drones |

#### Architecture Components

**Hybrid Architecture combines:**

1. **Liquid Neural Networks (LTC/CfC)**
   - Continuous-time dynamics
   - Adaptive time constants
   - Handles temporal patterns

2. **Spiking Neural Networks (SNN)**
   - Event-based processing
   - Energy efficient
   - Neuromorphic computation

3. **Multi-Head Attention**
   - Transformer-style attention
   - Global context awareness
   - Long-range dependencies

4. **STDP Plasticity** (Optional)
   - Bio-inspired learning
   - Unsupervised feature discovery
   - Synaptic weight adaptation

5. **Meta-Plasticity** (Optional)
   - Learning to learn
   - Faster adaptation
   - Continual learning

#### Advantages

✅ **Temporal Processing**: Native time-series handling  
✅ **Energy Efficiency**: Reduced computational cost  
✅ **Continual Learning**: No catastrophic forgetting  
✅ **Neuromorphic**: Compatible with specialized hardware  
✅ **Flexible**: Multi-domain adaptability  

---

## 📊 Quick Start Examples

### Example 1: Train LLM from Scratch

1. Go to **Training Tab**
2. Select: **LLM (Language Model)**
3. Choose: **WikiText-103** dataset
4. Set epochs: **30**
5. Enable: **Mixed Precision Training**
6. Click: **🚀 Start Training**
7. Wait ~2-3 hours (GPU) or 8-10 hours (CPU)
8. Model saved to `./models/llm_final_model.pt`

### Example 2: Generate Text

1. Go to **Inference Tab**
2. Browse: `./models/llm_final_model.pt`
3. Load model
4. Enter prompt: "The quick brown fox"
5. Set max length: 50
6. Temperature: 0.8
7. Click: **✨ Generate**
8. Read generated text!

### Example 3: Train Vision Model

1. Go to **Training Tab**
2. Select: **Vision (Image Recognition)**
3. Set epochs: **20**
4. Batch size: **64**
5. Click: **🚀 Start Training**
6. Model trains on CIFAR-10 images
7. Use for image classification tasks

---

## 🎯 Best Practices

### For LLM Training

- ✅ Use WikiText-103 for models >100M parameters
- ✅ Train for at least 30 epochs
- ✅ Enable mixed precision for speed
- ✅ Monitor validation loss - should decrease steadily
- ✅ If loss plateaus, decrease learning rate by 10x

### For Vision Training

- ✅ Use larger batch sizes (32-128)
- ✅ Enable data augmentation (built-in)
- ✅ Train for 20-50 epochs
- ✅ Monitor accuracy - should reach >80% on CIFAR-10

### For Robotics Training

- ✅ Prepare your sensor datasets first
- ✅ Use smaller batch sizes (8-16)
- ✅ Enable STDP for better adaptation
- ✅ Test in simulation before deployment

### Hardware Recommendations

- **CPU Only**: Small models (<10M params), be patient
- **1 GPU**: Most models, 8-16GB VRAM recommended
- **2-4 GPUs**: Use DataParallel strategy
- **4+ GPUs**: Use DistributedDataParallel for best scaling

---

## 🐛 Troubleshooting

### "CUDA out of memory"
- **Solution**: Reduce batch size by half
- Or reduce model size (hidden dim, layers)
- Or enable mixed precision

### "Training loss not decreasing"
- **Solution**: Check dataset size (WikiText-2 too small!)
- Increase learning rate
- Verify data loaded correctly (check logs)

### "Generated text is gibberish"
- **Solution**: Train longer (30+ epochs)
- Use larger dataset (WikiText-103)
- Check if model loaded correctly

### "GUI won't start"
- **Solution**: Install PyQt6: `pip install PyQt6`
- Update Python to 3.8+

### "Model file not found"
- **Solution**: Check path in `./models/` directory
- Ensure training completed successfully

---

## 🔧 Advanced Features

### Custom Datasets

You can add custom datasets by:
1. Modifying `src/core/main.py`
2. Adding to `DatasetFactory`
3. Selecting in GUI dropdown

### Export Models

After training, export to:
- ONNX format (use CLI tool)
- TorchScript (use CLI tool)
- Deploy to production

### Multi-Task Learning

Train on multiple tasks sequentially:
1. Enable continual learning in CLI
2. Train task 1
3. Train task 2 (without forgetting task 1!)

---

## 📚 Further Reading

- **DATASET_UPGRADE_WIKITEXT103.md**: Dataset implementation details
- **WIKITEXT103_QUICKSTART.md**: Quick start for WikiText-103
- **MAMBA_INTEGRATION.md**: Mamba architecture integration
- **QUICK_START_GUIDE.md**: CLI quick start guide

---

## 💬 Support

If you encounter issues:

1. Check training logs in GUI
2. Review error messages
3. Verify dataset loaded correctly
4. Check GPU memory usage
5. Consult documentation files

---

**Happy Training! 🚀**
