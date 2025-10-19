#!/usr/bin/env python3
"""
Evaluate the trained liquid-spiking LLM performance
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from src.core.main import LiquidSpikingNetwork, DatasetFactory, create_llm_config
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt

def evaluate_perplexity(model, dataloader, device='cuda'):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_tokens += targets.numel()
            
            if batch_idx % 10 == 0:
                current_ppl = torch.exp(torch.tensor(total_loss / total_tokens))
                print(f"Batch {batch_idx:3d}: Current perplexity = {current_ppl:.2f}")
    
    average_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(average_loss))
    
    return perplexity.item(), average_loss

def analyze_training_progress():
    """Analyze how loss decreased during training"""
    import os
    
    # Check what checkpoints we have
    checkpoints = []
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in os.listdir(parent_dir):
        if f.startswith('best_model_epoch_') and f.endswith('.pt'):
            epoch = int(f.split('_')[3].split('.')[0])
            checkpoints.append((epoch, os.path.join(parent_dir, f)))
    
    checkpoints.sort()
    print(f"Found {len(checkpoints)} training checkpoints")
    
    if not checkpoints:
        print("No checkpoints found for analysis")
        return
    
    # Load config
    config = create_llm_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset, tokenizer = DatasetFactory.create_llm_dataset(
        vocab_size=config.output_dim,
        seq_length=config.sequence_length,
        num_samples=500,  # Small test set
        tokenizer_name='gpt2'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    results = []
    
    # Evaluate each checkpoint
    for epoch, checkpoint_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating Epoch {epoch}: {checkpoint_path}")
        print(f"{'='*60}")
        
        try:
            # Create fresh model for each checkpoint
            model = LiquidSpikingNetwork(config)
            model = model.to(device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Try to load state dict, handling potential key mismatches
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded model state successfully")
            except RuntimeError as e:
                print(f"⚠️  Model structure mismatch: {e}")
                # Try partial loading
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"✅ Loaded {len(pretrained_dict)}/{len(checkpoint['model_state_dict'])} layers")
            
            # Evaluate
            perplexity, avg_loss = evaluate_perplexity(model, test_loader, device)
            
            results.append({
                'epoch': epoch,
                'perplexity': perplexity,
                'avg_loss': avg_loss,
                'checkpoint': checkpoint_path
            })
            
            print(f"\n📊 Results for Epoch {epoch}:")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"   Average Loss: {avg_loss:.4f}")
            
            # Also get training metrics if available
            if 'train_loss' in checkpoint:
                print(f"   Training Loss: {checkpoint['train_loss']:.4f}")
            if 'val_loss' in checkpoint:
                print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate epoch {epoch}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("📈 TRAINING PROGRESS SUMMARY")
    print(f"{'='*80}")
    
    if results:
        best_result = min(results, key=lambda x: x['perplexity'])
        print(f"Best performing model: Epoch {best_result['epoch']}")
        print(f"Best perplexity: {best_result['perplexity']:.2f}")
        print(f"Best average loss: {best_result['avg_loss']:.4f}")
        
        print(f"\nAll results:")
        for result in results:
            print(f"  Epoch {result['epoch']:2d}: Perplexity = {result['perplexity']:6.2f}, Loss = {result['avg_loss']:.4f}")
    else:
        print("No successful evaluations")

def sample_predictions(model, tokenizer, test_dataset, device='cuda', num_samples=5):
    """Show some sample predictions from the model"""
    print(f"\n{'='*60}")
    print("🎯 SAMPLE PREDICTIONS")
    print(f"{'='*60}")
    
    model.eval()
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            data, target = test_dataset[i]
            
            # Convert to batch format
            data_batch = data.unsqueeze(0).to(device)
            target_batch = target.unsqueeze(0).to(device)
            
            # Get model predictions
            outputs = model(data_batch)
            predictions = torch.argmax(outputs, dim=-1).squeeze(0)
            
            # Decode texts
            input_text = tokenizer.decode(data, skip_special_tokens=True)
            target_text = tokenizer.decode(target, skip_special_tokens=True) 
            predicted_text = tokenizer.decode(predictions, skip_special_tokens=True)
            
            print(f"\nSample {i+1}:")
            print(f"Input:     {input_text[:100]}...")
            print(f"Target:    {target_text[:100]}...")
            print(f"Predicted: {predicted_text[:100]}...")
            
            # Calculate token-level accuracy
            correct = (predictions == target_batch.squeeze(0)).float().mean()
            print(f"Token accuracy: {correct:.2%}")

def main():
    print("=" * 80)
    print("🧠 LIQUID-SPIKING LLM EVALUATION")
    print("=" * 80)
    print(f"🖥️  PyTorch: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    # Analyze training progress across epochs
    analyze_training_progress()
    
    print("\n" + "=" * 80)
    print("🎉 Evaluation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
