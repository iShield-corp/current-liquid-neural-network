# Configuration Repository Overview

## 📁 Directory Structure

The configuration repository has been reorganized into three task-specific folders with 20 different configurations:

```
configs/
├── llm/                    (8 configurations)
├── vision/                 (6 configurations)
└── robotics/              (6 configurations)
```

## 🧠 LLM Configurations (8 configs)

### 1. `tiny_llm.json` - Lightweight Language Model
- **Purpose**: Fast training and testing
- **Parameters**: 256 hidden_dim, 64 liquid_units, 32 spiking_units
- **Layers**: 3 layers, 4 attention heads
- **Training**: 5 epochs, batch_size 16
- **Use Case**: Quick prototyping and validation

### 2. `small_llm.json` - Small Scale Language Model
- **Purpose**: Balanced performance and efficiency
- **Parameters**: 512 hidden_dim, 128 liquid_units, 64 spiking_units
- **Layers**: 6 layers, 8 attention heads
- **Training**: 10 epochs, batch_size 8
- **Use Case**: Development and medium-scale experiments

### 3. `medium_llm.json` - Medium Scale Language Model
- **Purpose**: Good performance for most applications
- **Parameters**: 768 hidden_dim, 256 liquid_units, 128 spiking_units
- **Layers**: 12 layers, 12 attention heads
- **Training**: 15 epochs, batch_size 4
- **Use Case**: Production-ready medium models

### 4. `large_llm.json` - Large Scale Language Model
- **Purpose**: High-performance language modeling
- **Parameters**: 1024 hidden_dim, 512 liquid_units, 256 spiking_units
- **Layers**: 18 layers, 16 attention heads
- **Training**: 20 epochs, batch_size 2
- **Use Case**: Large-scale applications requiring high quality

### 5. `gpt4_tokenizer_llm.json` - GPT-4 Tokenizer Optimized
- **Purpose**: Optimized for GPT-4 level tokenization
- **Parameters**: 512 hidden_dim, 256 liquid_units, 128 spiking_units
- **Tokenizer**: GPT-4 vocabulary (100,277 tokens)
- **Backbone**: LTC liquid backbone
- **Use Case**: Advanced tokenization experiments

### 6. `code_llm.json` - Code Generation Specialist
- **Purpose**: Specialized for programming tasks
- **Parameters**: 512 hidden_dim, 192 liquid_units, 96 spiking_units
- **Tokenizer**: Code-focused vocabulary (32,000 tokens)
- **Backbone**: NCP liquid backbone
- **Use Case**: Code generation and programming assistance

### 7. `fast_training_llm.json` - Rapid Training Configuration
- **Purpose**: Quick iteration and fast training
- **Parameters**: 384 hidden_dim, 96 liquid_units, 48 spiking_units
- **Training**: 8 epochs, batch_size 12, high learning rate
- **Optimizations**: Low dropout, fast convergence
- **Use Case**: Rapid prototyping and quick experiments

### 8. `research_llm.json` - Research Configuration
- **Purpose**: Advanced research and experimentation
- **Parameters**: 640 hidden_dim, 320 liquid_units, 160 spiking_units
- **Layers**: 14 layers, 10 attention heads
- **Training**: 25 epochs, extended context
- **Use Case**: Research projects and advanced experiments

## 👁️ Vision Configurations (6 configs)

### 1. `cifar10_classification.json` - CIFAR-10 Classification
- **Dataset**: CIFAR-10 (10 classes)
- **Architecture**: 3072 input, conv layers [32,64,128]
- **Training**: 50 epochs, batch_size 32
- **Use Case**: Basic image classification

### 2. `cifar100_classification.json` - CIFAR-100 Classification
- **Dataset**: CIFAR-100 (100 classes)
- **Architecture**: Enhanced conv layers [64,128,256,512]
- **Training**: 80 epochs, batch_size 16
- **Use Case**: Complex multi-class classification

### 3. `mnist_classification.json` - MNIST Digit Recognition
- **Dataset**: MNIST (10 digit classes)
- **Architecture**: 784 input, lightweight conv layers
- **Training**: 30 epochs, batch_size 64
- **Use Case**: Digit recognition and basic CV

### 4. `fashion_mnist.json` - Fashion Item Classification
- **Dataset**: Fashion-MNIST (10 fashion classes)
- **Architecture**: NCP backbone, balanced parameters
- **Training**: 60 epochs, batch_size 24
- **Use Case**: Fashion and clothing classification

### 5. `stl10_classification.json` - STL-10 High Resolution
- **Dataset**: STL-10 (10 classes, higher resolution)
- **Architecture**: Large conv layers [64,128,256,512,1024]
- **Training**: 100 epochs, batch_size 8
- **Use Case**: High-resolution image classification

### 6. `object_detection.json` - Object Detection
- **Purpose**: Object detection and localization
- **Architecture**: Multi-scale conv processing
- **Training**: 40 epochs, specialized parameters
- **Use Case**: Object detection and computer vision research

## 🤖 Robotics Configurations (6 configs)

### 1. `arm_manipulation.json` - Robotic Arm Control
- **DOF**: 7 degrees of freedom
- **Purpose**: Robotic arm manipulation tasks
- **Training**: 100 epochs, sequence_length 50
- **Use Case**: Industrial robotics and manipulation

### 2. `mobile_navigation.json` - Mobile Robot Navigation
- **Purpose**: Autonomous mobile robot navigation
- **Architecture**: Vision + control integration
- **Training**: 200 epochs, extended sequences
- **Use Case**: Mobile robots and autonomous vehicles

### 3. `quadruped_control.json` - Quadruped Robot Control
- **DOF**: 12 degrees of freedom (quadruped)
- **Purpose**: Four-legged robot locomotion
- **Training**: 150 epochs, NCP backbone
- **Use Case**: Quadruped robots and legged locomotion

### 4. `autonomous_driving.json` - Self-Driving Systems
- **Purpose**: Autonomous vehicle control
- **Architecture**: Large input processing (1024 dim)
- **Training**: 180 epochs, complex vision processing
- **Use Case**: Autonomous driving and vehicle control

### 5. `drone_control.json` - Drone Flight Control
- **DOF**: 6 degrees of freedom (drone)
- **Purpose**: Unmanned aerial vehicle control
- **Training**: 120 epochs, specialized dynamics
- **Use Case**: Drone control and aerial robotics

### 6. `simple_control.json` - Basic Control Tasks
- **DOF**: 3 degrees of freedom
- **Purpose**: Simple control and learning tasks
- **Training**: 80 epochs, lightweight architecture
- **Use Case**: Basic robotics and control learning

## 🚀 Usage Examples

### LLM Training
```bash
# Train tiny LLM for quick testing
python scripts/cli.py train --task llm --config-path configs/llm/tiny_llm.json

# Train large LLM for production
python scripts/cli.py train --task llm --config-path configs/llm/large_llm.json

# Train with GPT-4 tokenizer
python scripts/cli.py train --task llm --config-path configs/llm/gpt4_tokenizer_llm.json
```

### Vision Training
```bash
# Train MNIST classifier
python scripts/cli.py train --task vision --config-path configs/vision/mnist_classification.json

# Train CIFAR-100 classifier
python scripts/cli.py train --task vision --config-path configs/vision/cifar100_classification.json

# Train object detection
python scripts/cli.py train --task vision --config-path configs/vision/object_detection.json
```

### Robotics Training
```bash
# Train arm manipulation
python scripts/cli.py train --task robotics --config-path configs/robotics/arm_manipulation.json

# Train mobile navigation
python scripts/cli.py train --task robotics --config-path configs/robotics/mobile_navigation.json

# Train drone control
python scripts/cli.py train --task robotics --config-path configs/robotics/drone_control.json
```

## ✅ Validation Results

All configurations have been tested and validated:
- ✅ **LLM configs**: All 8 configurations load and validate successfully
- ✅ **Vision configs**: All 6 configurations load and validate successfully  
- ✅ **Robotics configs**: All 6 configurations load and validate successfully

## 🎯 Key Features

### Diversity
- **3 liquid backbones**: CfC, LTC, NCP
- **Multiple scales**: From tiny (64 units) to large (512+ units)
- **Various applications**: Text, vision, robotics
- **Different training strategies**: Fast, research, production

### Optimization
- **Memory efficient**: Appropriate batch sizes for each scale
- **GPU optimized**: Mixed precision training enabled
- **Scalable**: Configurations from tiny to large scale
- **Task-specific**: Optimized parameters for each domain

### Flexibility
- **Easy modification**: JSON format for easy parameter changes
- **Hierarchical organization**: Clear folder structure
- **Backward compatible**: Works with existing CLI system
- **Research ready**: Advanced configurations for experimentation

Total configurations: **20 different configurations** across 3 domains, providing comprehensive coverage for liquid-spiking neural network training and research.