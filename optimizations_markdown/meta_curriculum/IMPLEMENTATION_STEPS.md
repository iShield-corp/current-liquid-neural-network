# Implementation Steps: Meta-Learning & Curriculum Learning

## 🚀 Quick Start (5 Minutes)

### Step 1: Add the Files
Place these three files in your project:
```
ssn-cfc/
├── src/
│   ├── core/
│   │   ├── main.py  (existing)
│   │   ├── meta_learning.py  (NEW)
│   │   ├── curriculum_learning.py  (NEW)
│   │   └── integration_guide.py  (NEW)
```

### Step 2: Modify cli.py
Add these arguments to your argument parser:

```python
# In scripts/cli.py, add to the train subparser:

train_parser.add_argument('--enable-curriculum', action='store_true',
                         help='Enable curriculum learning')
train_parser.add_argument('--curriculum-strategy', type=str, default='adaptive',
                         choices=['linear', 'exponential', 'step', 'adaptive', 'self_paced'],
                         help='Curriculum learning strategy')
train_parser.add_argument('--enable-meta-learning', action='store_true',
                         help='Enable MAML meta-learning')
train_parser.add_argument('--meta-frequency', type=int, default=5,
                         help='Perform meta-learning every N epochs')
```

### Step 3: Modify the Train Handler
Update the `_handle_train` method in cli.py:

```python
def _handle_train(self, args):
    # ... existing setup code ...
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # NEW: Check if enhanced features are requested
    if args.enable_curriculum or args.enable_meta_learning:
        from src.core.integration_guide import EnhancedTrainer
        
        enhanced_trainer = EnhancedTrainer(
            base_trainer=trainer,
            enable_meta_learning=args.enable_meta_learning,
            enable_curriculum=args.enable_curriculum,
            curriculum_strategy=args.curriculum_strategy
        )
        
        # Train with enhanced features
        enhanced_trainer.train(
            train_loader, 
            val_loader, 
            args.epochs,
            meta_learning_frequency=args.meta_frequency
        )
    else:
        # Regular training
        trainer.train(train_loader, val_loader, args.epochs)
```

## 📋 Usage Examples

### Example 1: Curriculum Learning Only (Recommended Starting Point)
```bash
# Adaptive curriculum - adjusts based on model performance
python scripts/cli.py train --task llm --epochs 50 \
  --enable-curriculum --curriculum-strategy adaptive

# Linear curriculum - steady progression
python scripts/cli.py train --task vision --epochs 30 \
  --enable-curriculum --curriculum-strategy linear

# Self-paced - model controls its own learning speed
python scripts/cli.py train --task robotics --epochs 40 \
  --enable-curriculum --curriculum-strategy self_paced
```

### Example 2: Meta-Learning Only
```bash
# Enable MAML with meta-learning every 5 epochs
python scripts/cli.py train --task llm --epochs 100 \
  --enable-meta-learning --meta-frequency 5

# More frequent meta-learning for faster adaptation
python scripts/cli.py train --task vision --epochs 80 \
  --enable-meta-learning --meta-frequency 3
```

### Example 3: Both Features Combined (Advanced)
```bash
# Full enhancement: curriculum + meta-learning
python scripts/cli.py train --task llm --epochs 100 \
  --liquid-units 384 --spiking-units 192 \
  --enable-curriculum --curriculum-strategy exponential \
  --enable-meta-learning --meta-frequency 5 \
  --batch-size 16 --learning-rate 2e-4

# For robotics with both features
python scripts/cli.py train --task robotics --epochs 80 \
  --enable-curriculum --curriculum-strategy adaptive \
  --enable-meta-learning --meta-frequency 10 \
  --sequence-length 100
```

## 🎯 What Each Strategy Does

### Curriculum Strategies

| Strategy | Best For | Behavior |
|----------|----------|----------|
| **linear** | Stable, predictable training | Steady, uniform progression |
| **exponential** | Fast early learning | Quick ramp-up, then stabilizes |
| **step** | Milestone-based training | Increases at specific epochs |
| **adaptive** | Optimal for most cases ⭐ | Advances when model performs well |
| **self_paced** | Advanced optimization | Model controls its own pace |

### Curriculum Progression

Your training will automatically:
1. **Start easy**: Shorter sequences (e.g., 64 tokens → 256 tokens)
2. **Increase complexity**: Fewer spike steps (8 → 32)
3. **Add harder samples**: Gradually include more difficult data
4. **Adapt dynamically**: Based on validation performance

### Meta-Learning Benefits

- ⚡ **Faster adaptation** to new tasks
- 🎯 **Better generalization** across domains
- 🔧 **Optimized liquid time constants** automatically
- 📉 **Reduced training time** for related tasks

## 🔧 Advanced Configuration

### Custom Curriculum Parameters

Create a custom curriculum configuration:

```python
from src.core.integration_guide import EnhancedTrainer

enhanced_trainer = EnhancedTrainer(
    base_trainer=trainer,
    enable_curriculum=True,
    curriculum_strategy="adaptive",
    curriculum_config={
        'initial_seq_length': 32,      # Start very short
        'final_seq_length': 512,       # End very long
        'initial_spike_steps': 4,      # Fewer spikes initially
        'final_spike_steps': 64,       # More spikes later
        'warmup_epochs': 10,           # Warm up for 10 epochs
        'performance_threshold': 0.80, # Advance at 80% accuracy
        'patience': 5                  # Wait 5 epochs before advancing
    }
)
```

### Custom Meta-Learning Parameters

```python
enhanced_trainer = EnhancedTrainer(
    base_trainer=trainer,
    enable_meta_learning=True,
    meta_learning_config={
        'meta_lr': 1e-3,              # Meta learning rate
        'inner_lr': 1e-2,             # Task adaptation rate
        'num_inner_steps': 5,         # Adaptation steps per task
        'first_order': False,         # Use second-order gradients
        'adapt_liquid_only': True,    # Only adapt liquid layers
        'n_way': 5,                   # 5 classes per task
        'k_shot': 5,                  # 5 examples per class
        'query_size': 15              # 15 test examples per class
    }
)
```

## 📊 Monitoring Progress

### What to Look For

**Curriculum Learning Indicators:**
```
📚 Curriculum: seq_len=64, spike_steps=8, difficulty=0.25
📚 Curriculum: seq_len=96, spike_steps=12, difficulty=0.40
📚 Curriculum: seq_len=128, spike_steps=16, difficulty=0.60
📈 Curriculum advanced at epoch 15
```

**Meta-Learning Indicators:**
```
🧠 Performing meta-learning step...
   Meta-loss: 1.234
   Avg task loss: 1.456
   Avg adaptation loss: 0.987
```

## 🐛 Troubleshooting

### Issue: Curriculum Not Advancing

**Problem**: Model stays at easy difficulty level

**Solutions**:
1. Lower `performance_threshold` (e.g., from 0.7 to 0.6)
2. Reduce `patience` (e.g., from 5 to 3 epochs)
3. Try different strategy: `--curriculum-strategy linear`

### Issue: Meta-Learning Too Slow

**Problem**: Meta-learning adds too much overhead

**Solutions**:
1. Increase `meta_frequency` (e.g., every 10 epochs instead of 5)
2. Enable `first_order` approximation for faster computation
3. Reduce `num_inner_steps` (e.g., from 5 to 3)

### Issue: Out of Memory

**Problem**: Enhanced features use more memory

**Solutions**:
1. Reduce `batch_size` by 50%
2. Enable `first_order=True` for meta-learning
3. Use gradient accumulation
4. Start with curriculum only, add meta-learning later

### Issue: No Performance Improvement

**Problem**: Features don't help your specific task

**Solutions**:
1. Try different curriculum strategies
2. Adjust curriculum parameters (warmup, thresholds)
3. Check if task benefits from curriculum (sequential tasks benefit most)
4. Meta-learning works best with diverse task distributions

## 🎓 Best Practices

### For LLM Tasks
```bash
# Recommended: Adaptive curriculum with periodic meta-learning
python scripts/cli.py train --task llm --epochs 100 \
  --enable-curriculum --curriculum-strategy adaptive \
  --enable-meta-learning --meta-frequency 10 \
  --sequence-length 256
```

### For Vision Tasks
```bash
# Recommended: Linear curriculum, more frequent meta-learning
python scripts/cli.py train --task vision --epochs 50 \
  --enable-curriculum --curriculum-strategy linear \
  --enable-meta-learning --meta-frequency 5 \
  --batch-size 32
```

### For Robotics Tasks
```bash
# Recommended: Self-paced curriculum for adaptive learning
python scripts/cli.py train --task robotics --epochs 80 \
  --enable-curriculum --curriculum-strategy self_paced \
  --sequence-length 100
```

## 📈 Expected Improvements

Based on typical curriculum learning and meta-learning results:

| Metric | Without Features | With Curriculum | With Both |
|--------|-----------------|-----------------|-----------|
| **Training Time** | 100% | 85-90% | 75-85% |
| **Final Accuracy** | Baseline | +3-7% | +5-12% |
| **Convergence Speed** | Baseline | 1.5-2x faster | 2-3x faster |
| **Few-Shot Performance** | Baseline | +10-15% | +20-30% |

## 🔬 Testing Your Implementation

### Verify Installation
```bash
# Test imports
python -c "from src.core.meta_learning import LiquidMAML; print('✓ Meta-learning OK')"
python -c "from src.core.curriculum_learning import CurriculumScheduler; print('✓ Curriculum OK')"
python -c "from src.core.integration_guide import EnhancedTrainer; print('✓ Integration OK')"
```

### Run Quick Test
```bash
# 10-epoch test with curriculum
python scripts/cli.py train --task llm --epochs 10 \
  --enable-curriculum --curriculum-strategy linear \
  --batch-size 4  # Small batch for quick test
```

## 📚 Further Reading

- **MAML Paper**: "Model-Agnostic Meta-Learning" (Finn et al., 2017)
- **Curriculum Learning**: "Curriculum Learning" (Bengio et al., 2009)
- **Self-Paced Learning**: "Self-Paced Learning for Latent Variable Models" (Kumar et al., 2010)

## 🤝 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all three new files are properly placed
3. Ensure imports work correctly
4. Start with curriculum only before adding meta-learning
5. Review the example outputs in the integration guide

---

**Pro Tip**: Start with `--enable-curriculum --curriculum-strategy adaptive` for immediate benefits, then add meta-learning once you're comfortable with the system!