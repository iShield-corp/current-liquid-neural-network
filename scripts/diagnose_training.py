#!/usr/bin/env python3
"""Comprehensive training diagnostic to identify why model isn't learning."""

import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from src.core.main import (
    create_llm_config, 
    LiquidSpikingNetwork, 
    WikiTextDataset, 
    TextDataset
)

print("="*80)
print("COMPREHENSIVE TRAINING DIAGNOSTIC")
print("="*80)

# Setup
config = create_llm_config('gpt2')
config.num_layers = 2  # Smaller for faster testing
config.batch_size = 8
config.sequence_length = 256

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create model
print("\n1. Model Setup")
print("-" * 80)
model = LiquidSpikingNetwork(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Load data
print("\n2. Dataset")
print("-" * 80)
texts = WikiTextDataset.load_wikitext2(split='train')
dataset = TextDataset(texts, tokenizer, seq_length=config.sequence_length)
print(f"Examples: {len(dataset):,}")
print(f"Tokens per example: {config.sequence_length}")
print(f"Total tokens: {len(dataset) * config.sequence_length:,}")
print(f"Parameters/Token ratio: {total_params / (len(dataset) * config.sequence_length):.2f}")

# Get one batch
print("\n3. Data Batch Inspection")
print("-" * 80)
batch_inputs = []
batch_targets = []
for i in range(8):
    inp, tgt = dataset[i]
    batch_inputs.append(inp)
    batch_targets.append(tgt)

inputs = torch.stack(batch_inputs).to(device)
targets = torch.stack(batch_targets).to(device)

print(f"Batch inputs shape: {inputs.shape}")
print(f"Batch targets shape: {targets.shape}")
print(f"Input range: [{inputs.min()}, {inputs.max()}]")
print(f"Target range: [{targets.min()}, {targets.max()}]")
print(f"Unique tokens in batch: {torch.unique(inputs).numel()}")

# Forward pass
print("\n4. Forward Pass (untrained model)")
print("-" * 80)
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")
    print(f"Output mean: {outputs.mean():.4f}")
    print(f"Output std: {outputs.std():.4f}")
    print(f"Output min: {outputs.min():.4f}")
    print(f"Output max: {outputs.max():.4f}")
    
    # Check if outputs are degenerate
    probs = F.softmax(outputs, dim=-1)
    max_probs = probs.max(dim=-1).values
    print(f"Max probability mean: {max_probs.mean():.6f}")
    print(f"Max probability std: {max_probs.std():.6f}")
    
    # Compute loss
    outputs_flat = outputs.reshape(-1, config.vocab_size)
    targets_flat = targets.reshape(-1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss = criterion(outputs_flat, targets_flat)
    
    # Add embedding regularization like training does
    embed_reg = 0.01 * torch.norm(model.token_embedding.weight, p=2)
    total_loss = loss + embed_reg
    
    print(f"Cross-entropy loss: {loss.item():.4f}")
    print(f"Embedding regularization: {embed_reg.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Expected random loss: {torch.tensor(config.vocab_size).log().item():.4f}")

# Test gradient flow
print("\n5. Gradient Flow Test")
print("-" * 80)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# One training step
optimizer.zero_grad()
outputs = model(inputs)
outputs_flat = outputs.reshape(-1, config.vocab_size)
targets_flat = targets.reshape(-1)
loss = criterion(outputs_flat, targets_flat)
embed_reg = 0.01 * torch.norm(model.token_embedding.weight, p=2)
total_loss = loss + embed_reg
total_loss.backward()

# Check gradients
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms[name] = grad_norm

print("Gradient norms by layer:")
for name, norm in sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name:50s}: {norm:.6f}")

print(f"\nTotal gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')):.6f}")

# Check for vanishing gradients
vanishing = sum(1 for v in grad_norms.values() if v < 1e-7)
print(f"Layers with vanishing gradients (<1e-7): {vanishing}/{len(grad_norms)}")

# Update and check if parameters change
param_before = {name: param.clone() for name, param in model.named_parameters()}
optimizer.step()

param_changes = {}
for name, param in model.named_parameters():
    change = (param - param_before[name]).abs().max().item()
    param_changes[name] = change

print("\nParameter changes after 1 step (top 10):")
for name, change in sorted(param_changes.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name:50s}: {change:.10f}")

# Test memorization
print("\n6. Memorization Test (10 steps on same batch)")
print("-" * 80)
model.train()
losses = []
for step in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    outputs_flat = outputs.reshape(-1, config.vocab_size)
    targets_flat = targets.reshape(-1)
    loss = criterion(outputs_flat, targets_flat)
    embed_reg = 0.01 * torch.norm(model.token_embedding.weight, p=2)
    total_loss = loss + embed_reg
    total_loss.backward()
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    losses.append(total_loss.item())
    
    print(f"Step {step:2d}: Loss={total_loss.item():.4f}, GradNorm={grad_norm:.4f}")

print(f"\nLoss change: {losses[0]:.4f} → {losses[-1]:.4f} (Δ={losses[-1]-losses[0]:.4f})")

if losses[-1] < losses[0] - 0.1:
    print("✅ Model CAN memorize (loss decreasing)")
else:
    print("❌ Model CANNOT memorize (loss not decreasing)")
    print("\nPOSSIBLE CAUSES:")
    print("  1. Gradient flow blocked somewhere in architecture")
    print("  2. Learning rate too low or optimizer issue")
    print("  3. Spiking network requires different training approach")
    print("  4. Label smoothing or regularization too strong")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
