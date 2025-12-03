#!/usr/bin/env python3
"""Simple training script that bypasses Trainer complexity."""

import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.core.main import (
    create_llm_config,
    LiquidSpikingNetwork,
    WikiTextDataset,
    TextDataset
)

def train_simple():
    """Simple training loop with minimal complexity."""
    
    # Setup
    print("="*80)
    print("SIMPLE TRAINING (bypasses Trainer class)")
    print("="*80)
    
    config = create_llm_config('gpt2')
    config.num_layers = 2  # Smaller model
    config.batch_size = 8
    config.sequence_length = 256
    config.learning_rate = 5e-5
    config.gradient_clip = 0.5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading WikiText-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    texts = WikiTextDataset.load_wikitext2(split='train')
    dataset = TextDataset(texts, tokenizer, seq_length=config.sequence_length)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset: {len(dataset):,} examples")
    print(f"Batches: {len(train_loader):,}")
    
    # Create model
    print("\nCreating model...")
    model = LiquidSpikingNetwork(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training
    print("\nStarting training...")
    print("Epochs: 3")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Gradient clip: {config.gradient_clip}")
    print("="*80)
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        total_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute loss
            outputs_flat = outputs.reshape(-1, config.vocab_size)
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            
            # Add embedding regularization (like Trainer does)
            if hasattr(model, 'token_embedding'):
                embed_reg = 0.01 * torch.norm(model.token_embedding.weight, p=2)
                loss = loss + embed_reg
            
            # Backward
            loss.backward()
            
            # Gradient clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip
            )
            
            # Optimizer step
            optimizer.step()
            
            # Track
            total_loss += loss.item()
            total_batches += 1
            
            # Update progress
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/total_batches:.4f}',
                'grad_norm': f'{grad_norm:.3f}'
            })
            
            # Early stopping for testing
            if batch_idx >= 100:
                print(f"\n[Stopping early at 100 batches for testing]")
                break
        
        avg_loss = total_loss / total_batches
        print(f"\nEpoch {epoch+1} complete - Average loss: {avg_loss:.4f}")
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

if __name__ == "__main__":
    train_simple()
