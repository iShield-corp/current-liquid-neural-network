"""
Multi-Modal Fusion for Hybrid Liquid-Spiking Neural Networks
Implements early, late, and hybrid fusion strategies with cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Different fusion strategies for multi-modal learning."""
    EARLY = "early"        # Fuse at feature level before encoding
    LATE = "late"          # Fuse at decision level after separate processing
    HYBRID = "hybrid"      # Combination of early and late fusion
    HIERARCHICAL = "hierarchical"  # Progressive fusion at multiple levels


class ModalityType(Enum):
    """Supported modality types."""
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"
    SENSOR = "sensor"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"


class ModalityEncoder(nn.Module):
    """
    Modality-specific encoder with separate liquid-spike processing.
    Each modality gets its own hybrid encoder for optimal feature extraction.
    """
    
    def __init__(
        self,
        modality_type: ModalityType,
        input_dim: int,
        output_dim: int,
        liquid_units: int,
        spiking_units: int,
        num_spike_steps: int = 32,
        beta: float = 0.95,
        backbone: str = 'cfc',
        preprocessing: Optional[nn.Module] = None
    ):
        """
        Args:
            modality_type: Type of input modality
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            liquid_units: Number of liquid neurons
            spiking_units: Number of spiking neurons
            num_spike_steps: Spiking time steps
            beta: Membrane potential decay
            backbone: Liquid network backbone ('cfc', 'ltc', 'ncp')
            preprocessing: Optional preprocessing module (e.g., CNN for images)
        """
        super().__init__()
        
        self.modality_type = modality_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Modality-specific preprocessing
        self.preprocessing = preprocessing
        
        # Import from main (assuming available)
        try:
            from .main import SpikingEncoder, LiquidCell
        except ImportError:
            logger.warning("Could not import from main.py, using simplified versions")
            SpikingEncoder = nn.Linear  # Fallback
            LiquidCell = nn.LSTM
        
        # Spike encoder for this modality
        self.spike_encoder = SpikingEncoder(
            input_dim=input_dim,
            output_dim=spiking_units,
            num_steps=num_spike_steps,
            beta=beta
        )
        
        # Liquid cell for this modality
        self.liquid_cell = LiquidCell(
            input_dim=spiking_units,
            units=liquid_units,
            backbone=backbone
        )
        
        # Fusion of liquid and spike features
        self.modality_fusion = nn.Sequential(
            nn.Linear(liquid_units + spiking_units, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Modality-specific attention (self-attention within modality)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        logger.info(f"📊 {modality_type.value} encoder initialized: "
                   f"{input_dim}→{liquid_units}L+{spiking_units}S→{output_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Encode modality-specific input.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, channels, H, W]
            hidden_state: Hidden state for liquid cell
        
        Returns:
            encoded_features: Encoded features [batch, seq_len, output_dim]
            hidden_state: Updated hidden state
            modality_info: Additional modality information
        """
        # Apply preprocessing if provided
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        
        # Ensure correct shape [batch, seq_len, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, features = x.shape
        
        # Spike encoding
        spike_out = self.spike_encoder(x)
        
        # Handle spike output dimensions
        if len(spike_out.shape) == 3:
            spike_features = spike_out.mean(dim=1)  # Average over time
        else:
            spike_features = spike_out
        
        # Liquid processing
        liquid_out, hidden_state = self.liquid_cell(spike_features, hidden_state)
        
        # Ensure correct dimensions
        if liquid_out.dim() == 3:
            liquid_out = liquid_out.squeeze(1)
        
        # Repeat for sequence length if needed
        if spike_features.dim() == 2:
            spike_features = spike_features.unsqueeze(1).expand(-1, seq_len, -1)
        if liquid_out.dim() == 2:
            liquid_out = liquid_out.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Fuse liquid and spike features
        combined = torch.cat([liquid_out, spike_features], dim=-1)
        encoded = self.modality_fusion(combined)
        
        # Apply self-attention within modality
        attended, attention_weights = self.self_attention(encoded, encoded, encoded)
        encoded = encoded + attended  # Residual connection
        
        # Collect modality information
        modality_info = {
            'spike_features': spike_features,
            'liquid_features': liquid_out,
            'attention_weights': attention_weights,
            'modality_type': self.modality_type.value
        }
        
        return encoded, hidden_state, modality_info


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention for fusing features from different modalities.
    Allows modalities to attend to each other for rich interaction.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality names to their dimensions
            fusion_dim: Common fusion dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Project each modality to common fusion dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim)
            for name, dim in modality_dims.items()
        })
        
        # Cross-modal attention layers
        self.cross_attentions = nn.ModuleDict({
            f"{source}_to_{target}": nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for source in modality_dims.keys()
            for target in modality_dims.keys()
            if source != target
        })
        
        # Fusion gate to control information flow
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * len(modality_dims), fusion_dim),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * len(modality_dims), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        logger.info(f"🔗 Cross-modal attention fusion initialized")
        logger.info(f"   Modalities: {list(modality_dims.keys())}")
        logger.info(f"   Fusion dim: {fusion_dim}, Heads: {num_heads}")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        return_attention_maps: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Fuse features from multiple modalities using cross-attention.
        
        Args:
            modality_features: Dictionary of modality features
                {modality_name: tensor of shape [batch, seq_len, dim]}
            return_attention_maps: Whether to return attention maps
        
        Returns:
            fused_features: Fused representation [batch, seq_len, fusion_dim]
            attention_maps: Optional attention maps between modalities
        """
        # Project all modalities to common dimension
        projected = {}
        for name, features in modality_features.items():
            projected[name] = self.modality_projections[name](features)
        
        # Apply cross-modal attention
        attended_features = {}
        attention_maps = {} if return_attention_maps else None
        
        for source_name, source_features in projected.items():
            # Collect attended features from all other modalities
            attended = []
            
            for target_name, target_features in projected.items():
                if source_name == target_name:
                    # Self-attention (identity)
                    attended.append(source_features)
                else:
                    # Cross-attention
                    attn_key = f"{source_name}_to_{target_name}"
                    attended_features_ij, attn_weights = self.cross_attentions[attn_key](
                        source_features,  # query
                        target_features,  # key
                        target_features   # value
                    )
                    attended.append(attended_features_ij)
                    
                    if return_attention_maps:
                        attention_maps[attn_key] = attn_weights
            
            # Average attended features
            attended_features[source_name] = torch.stack(attended, dim=0).mean(dim=0)
        
        # Concatenate all attended features
        all_attended = torch.cat(list(attended_features.values()), dim=-1)
        
        # Apply fusion gate
        gate = self.fusion_gate(all_attended)
        
        # Final fusion
        fused = self.fusion_layer(all_attended)
        fused = fused * gate  # Gated fusion
        
        return fused, attention_maps


class MultiModalFusionNetwork(nn.Module):
    """
    Complete multi-modal fusion network with configurable fusion strategy.
    
    Supports:
    - Early fusion: Concatenate raw features, then encode
    - Late fusion: Encode separately, then fuse decisions
    - Hybrid fusion: Mix of early and late fusion
    - Hierarchical fusion: Progressive fusion at multiple levels
    """
    
    def __init__(
        self,
        modality_configs: Dict[str, Dict],
        fusion_strategy: FusionStrategy = FusionStrategy.HYBRID,
        fusion_dim: int = 512,
        num_attention_heads: int = 8,
        output_dim: int = 10,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_configs: Configuration for each modality
                {modality_name: {
                    'type': ModalityType,
                    'input_dim': int,
                    'encoder_dim': int,
                    'liquid_units': int,
                    'spiking_units': int,
                    'preprocessing': Optional[nn.Module]
                }}
            fusion_strategy: Strategy for fusing modalities
            fusion_dim: Dimension for fusion layer
            num_attention_heads: Attention heads for cross-modal attention
            output_dim: Final output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modality_configs = modality_configs
        self.fusion_strategy = fusion_strategy
        self.fusion_dim = fusion_dim
        self.modality_names = list(modality_configs.keys())
        
        # Create modality-specific encoders
        self.encoders = nn.ModuleDict()
        modality_dims = {}
        
        for name, config in modality_configs.items():
            encoder = ModalityEncoder(
                modality_type=config['type'],
                input_dim=config['input_dim'],
                output_dim=config['encoder_dim'],
                liquid_units=config.get('liquid_units', 256),
                spiking_units=config.get('spiking_units', 128),
                num_spike_steps=config.get('num_spike_steps', 32),
                beta=config.get('beta', 0.95),
                backbone=config.get('backbone', 'cfc'),
                preprocessing=config.get('preprocessing')
            )
            self.encoders[name] = encoder
            modality_dims[name] = config['encoder_dim']
        
        # Fusion components based on strategy
        if fusion_strategy in [FusionStrategy.EARLY, FusionStrategy.HYBRID]:
            # Early fusion: simple concatenation + projection
            total_dim = sum(modality_dims.values())
            self.early_fusion = nn.Sequential(
                nn.Linear(total_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        if fusion_strategy in [FusionStrategy.LATE, FusionStrategy.HYBRID, FusionStrategy.HIERARCHICAL]:
            # Late fusion: cross-modal attention
            self.cross_modal_fusion = CrossModalAttentionFusion(
                modality_dims=modality_dims,
                fusion_dim=fusion_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
        
        if fusion_strategy == FusionStrategy.HIERARCHICAL:
            # Hierarchical fusion: multiple fusion levels
            self.hierarchical_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                for _ in range(3)  # 3 fusion levels
            ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
        logger.info(f"🌐 Multi-modal fusion network initialized")
        logger.info(f"   Strategy: {fusion_strategy.value}")
        logger.info(f"   Modalities: {self.modality_names}")
        logger.info(f"   Fusion dim: {fusion_dim}")
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_modality_features: bool = False,
        return_attention_maps: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through multi-modal network.
        
        Args:
            inputs: Dictionary of modality inputs
                {modality_name: tensor}
            return_modality_features: Return intermediate modality features
            return_attention_maps: Return attention maps
        
        Returns:
            output: Final predictions [batch, output_dim]
            Optional: Dictionary with intermediate features and attention maps
        """
        # Encode each modality separately
        encoded_features = {}
        hidden_states = {}
        modality_info = {}
        
        for name in self.modality_names:
            if name not in inputs:
                raise ValueError(f"Missing input for modality: {name}")
            
            encoded, hidden, info = self.encoders[name](
                inputs[name],
                hidden_states.get(name)
            )
            encoded_features[name] = encoded
            hidden_states[name] = hidden
            modality_info[name] = info
        
        # Fuse based on strategy
        if self.fusion_strategy == FusionStrategy.EARLY:
            # Early fusion: concatenate and project
            concatenated = torch.cat(list(encoded_features.values()), dim=-1)
            fused = self.early_fusion(concatenated)
            attention_maps = None
            
        elif self.fusion_strategy == FusionStrategy.LATE:
            # Late fusion: cross-modal attention
            fused, attention_maps = self.cross_modal_fusion(
                encoded_features,
                return_attention_maps=return_attention_maps
            )
            
        elif self.fusion_strategy == FusionStrategy.HYBRID:
            # Hybrid fusion: both early and late
            # Early component
            concatenated = torch.cat(list(encoded_features.values()), dim=-1)
            early_fused = self.early_fusion(concatenated)
            
            # Late component
            late_fused, attention_maps = self.cross_modal_fusion(
                encoded_features,
                return_attention_maps=return_attention_maps
            )
            
            # Combine early and late
            fused = (early_fused + late_fused) / 2
            
        elif self.fusion_strategy == FusionStrategy.HIERARCHICAL:
            # Hierarchical fusion: progressive fusion
            fused, attention_maps = self.cross_modal_fusion(
                encoded_features,
                return_attention_maps=return_attention_maps
            )
            
            # Apply hierarchical fusion layers
            for fusion_layer in self.hierarchical_fusion:
                fused = fusion_layer(fused) + fused  # Residual
        
        # Average over sequence dimension if present
        if fused.dim() == 3:
            fused = fused.mean(dim=1)
        
        # Generate output
        output = self.output_head(fused)
        
        # Return additional information if requested
        if return_modality_features or return_attention_maps:
            additional_info = {
                'modality_features': encoded_features if return_modality_features else None,
                'attention_maps': attention_maps if return_attention_maps else None,
                'modality_info': modality_info if return_modality_features else None
            }
            return output, additional_info
        
        return output


def create_vision_text_fusion(
    image_size: Tuple[int, int, int] = (3, 224, 224),
    text_vocab_size: int = 50257,
    text_seq_length: int = 128,
    fusion_dim: int = 512,
    output_dim: int = 1000
) -> MultiModalFusionNetwork:
    """
    Create a vision + text multi-modal fusion network.
    Common for image captioning, VQA, visual reasoning.
    """
    # Vision preprocessing (CNN)
    vision_preprocessing = nn.Sequential(
        nn.Conv2d(image_size[0], 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((7, 7)),
        
        nn.Flatten(),
        nn.Linear(256 * 7 * 7, 2048)
    )
    
    # Text preprocessing (embedding)
    text_preprocessing = nn.Sequential(
        nn.Embedding(text_vocab_size, 512),
        nn.Dropout(0.1)
    )
    
    modality_configs = {
        'vision': {
            'type': ModalityType.VISION,
            'input_dim': 2048,
            'encoder_dim': fusion_dim,
            'liquid_units': 256,
            'spiking_units': 128,
            'preprocessing': vision_preprocessing
        },
        'text': {
            'type': ModalityType.TEXT,
            'input_dim': 512,
            'encoder_dim': fusion_dim,
            'liquid_units': 256,
            'spiking_units': 128,
            'preprocessing': text_preprocessing
        }
    }
    
    return MultiModalFusionNetwork(
        modality_configs=modality_configs,
        fusion_strategy=FusionStrategy.HYBRID,
        fusion_dim=fusion_dim,
        output_dim=output_dim
    )


# Example usage
if __name__ == "__main__":
    print("Testing Multi-Modal Fusion...")
    
    # Create vision + text fusion network
    model = create_vision_text_fusion(
        fusion_dim=512,
        output_dim=1000
    )
    
    # Create sample inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text_tokens = torch.randint(0, 50257, (batch_size, 128))
    
    inputs = {
        'vision': images,
        'text': text_tokens
    }
    
    # Forward pass
    output, info = model(
        inputs,
        return_modality_features=True,
        return_attention_maps=True
    )
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Vision features shape: {info['modality_features']['vision'].shape}")
    print(f"✓ Text features shape: {info['modality_features']['text'].shape}")
    print(f"✓ Attention maps: {list(info['attention_maps'].keys())}")
    print("\n✅ Multi-modal fusion working!")