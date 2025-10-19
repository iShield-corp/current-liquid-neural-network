"""
Modality-Specific Preprocessing Modules
Specialized preprocessing for different input modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VisionPreprocessor(nn.Module):
    """
    Advanced vision preprocessing with residual connections.
    Suitable for images and video frames.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 2048,
        architecture: str = 'resnet'  # 'resnet', 'efficient', 'simple'
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        if architecture == 'resnet':
            self.backbone = self._build_resnet_backbone()
        elif architecture == 'efficient':
            self.backbone = self._build_efficient_backbone()
        else:
            self.backbone = self._build_simple_backbone()
        
        logger.info(f"👁️  Vision preprocessor initialized: {architecture}")
    
    def _build_resnet_backbone(self):
        """ResNet-inspired architecture with skip connections."""
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out
        
        return nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.output_dim)
        )
    
    def _build_efficient_backbone(self):
        """Efficient architecture with depthwise separable convolutions."""
        return nn.Sequential(
            # Stem
            nn.Conv2d(self.input_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            # Depthwise separable blocks
            self._make_ds_block(32, 64, stride=2),
            self._make_ds_block(64, 128, stride=2),
            self._make_ds_block(128, 256, stride=2),
            self._make_ds_block(256, 512, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.output_dim)
        )
    
    def _make_ds_block(self, in_channels, out_channels, stride=1):
        """Depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
    
    def _build_simple_backbone(self):
        """Simple CNN backbone."""
        return nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(512 * 16, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images [batch, channels, height, width]
        Returns:
            features: [batch, output_dim]
        """
        return self.backbone(x)


class AudioPreprocessor(nn.Module):
    """
    Audio preprocessing with spectrogram conversion and temporal modeling.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        n_mels: int = 128,
        output_dim: int = 512,
        use_raw_audio: bool = False
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.use_raw_audio = use_raw_audio
        
        if not use_raw_audio:
            # Mel spectrogram transform
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels
            )
            
            # Process spectrogram with CNN
            self.spec_processor = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                
                nn.Flatten(),
                nn.Linear(128 * 16, output_dim)
            )
        else:
            # Process raw audio with 1D convolutions
            self.raw_processor = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=80, stride=16),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(4),
                
                nn.Conv1d(32, 64, kernel_size=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(4),
                
                nn.Conv1d(64, 128, kernel_size=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(16),
                
                nn.Flatten(),
                nn.Linear(128 * 16, output_dim)
            )
        
        logger.info(f"🔊 Audio preprocessor initialized: "
                   f"{'raw' if use_raw_audio else 'spectrogram'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Audio [batch, time_samples] or [batch, channels, time_samples]
        Returns:
            features: [batch, output_dim]
        """
        if not self.use_raw_audio:
            # Convert to mel spectrogram
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add channel dimension
            
            spec = self.mel_transform(x)
            # Log scale
            spec = torch.log(spec + 1e-9)
            
            return self.spec_processor(spec)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return self.raw_processor(x)


class TextPreprocessor(nn.Module):
    """
    Text preprocessing with embeddings and positional encoding.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        embedding_dim: int = 512,
        max_seq_length: int = 512,
        use_positional: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        if use_positional:
            self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        else:
            self.pos_embedding = None
        
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"📝 Text preprocessor initialized: vocab={vocab_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices [batch, seq_len]
        Returns:
            embeddings: [batch, seq_len, embedding_dim]
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.token_embedding(x)
        
        # Add positional embeddings
        if self.pos_embedding is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            positions = positions.expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            embeddings = token_emb + pos_emb
        else:
            embeddings = token_emb
        
        return self.dropout(embeddings)


class SensorPreprocessor(nn.Module):
    """
    Sensor data preprocessing for robotics and IoT applications.
    Handles multi-channel time-series sensor data.
    """
    
    def __init__(
        self,
        num_sensors: int,
        output_dim: int = 256,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        
        # Per-sensor processing with 1D convolutions
        self.sensor_conv = nn.Sequential(
            nn.Conv1d(num_sensors, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        
        # Temporal attention for important time steps
        if use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=4,
                batch_first=True
            )
        else:
            self.temporal_attention = None
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        logger.info(f"📡 Sensor preprocessor initialized: {num_sensors} sensors")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sensor data [batch, num_sensors, time_steps]
        Returns:
            features: [batch, output_dim]
        """
        # Convolutional processing
        features = self.sensor_conv(x)  # [batch, 256, 8]
        
        # Temporal attention
        if self.temporal_attention is not None:
            features = features.transpose(1, 2)  # [batch, 8, 256]
            attended, _ = self.temporal_attention(features, features, features)
            features = attended.transpose(1, 2)  # [batch, 256, 8]
        
        # Final projection
        return self.projection(features)


class VideoPreprocessor(nn.Module):
    """
    Video preprocessing that handles temporal dimension.
    Processes videos as sequences of frames.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        num_frames: int = 16,
        use_3d_conv: bool = False
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.use_3d_conv = use_3d_conv
        
        if use_3d_conv:
            # 3D convolutions for spatiotemporal features
            self.backbone = nn.Sequential(
                nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                         stride=(1, 2, 2), padding=(1, 3, 3)),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                
                nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d((2, 2, 2)),
                
                nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                
                nn.Flatten(),
                nn.Linear(256, output_dim)
            )
        else:
            # 2D convolutions per frame + temporal pooling
            self.frame_encoder = VisionPreprocessor(
                input_channels=input_channels,
                output_dim=output_dim,
                architecture='simple'
            )
            
            self.temporal_aggregation = nn.LSTM(
                input_size=output_dim,
                hidden_size=output_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=False
            )
        
        logger.info(f"🎥 Video preprocessor initialized: "
                   f"{'3D' if use_3d_conv else '2D+LSTM'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video [batch, channels, frames, height, width] or
               [batch, frames, channels, height, width]
        Returns:
            features: [batch, output_dim]
        """
        if self.use_3d_conv:
            # Ensure correct format [batch, channels, frames, H, W]
            if x.dim() == 5 and x.shape[1] > x.shape[2]:
                # Likely [batch, frames, channels, H, W]
                x = x.permute(0, 2, 1, 3, 4)
            return self.backbone(x)
        else:
            # Process frame by frame
            batch_size = x.shape[0]
            
            # Reshape for frame processing
            if x.shape[1] == self.num_frames:
                # [batch, frames, channels, H, W]
                x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            else:
                # [batch, channels, frames, H, W]
                x = x.permute(0, 2, 1, 3, 4)
                x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            
            # Encode each frame
            frame_features = self.frame_encoder(x)  # [batch*frames, output_dim]
            
            # Reshape to sequence
            frame_features = frame_features.view(batch_size, self.num_frames, -1)
            
            # Temporal aggregation with LSTM
            _, (hidden, _) = self.temporal_aggregation(frame_features)
            
            return hidden[-1]  # Last hidden state


# Factory function to create preprocessors
def create_preprocessor(
    modality_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create appropriate preprocessor.
    
    Args:
        modality_type: Type of modality
        **kwargs: Modality-specific arguments
    
    Returns:
        Preprocessor module
    """
    if modality_type == 'vision':
        return VisionPreprocessor(**kwargs)
    elif modality_type == 'audio':
        return AudioPreprocessor(**kwargs)
    elif modality_type == 'text':
        return TextPreprocessor(**kwargs)
    elif modality_type == 'sensor':
        return SensorPreprocessor(**kwargs)
    elif modality_type == 'video':
        return VideoPreprocessor(**kwargs)
    else:
        raise ValueError(f"Unknown modality type: {modality_type}")


# Example usage
if __name__ == "__main__":
    print("Testing Modality Preprocessors...")
    
    # Test vision
    vision_prep = VisionPreprocessor(output_dim=512)
    images = torch.randn(4, 3, 224, 224)
    vision_features = vision_prep(images)
    print(f"✓ Vision: {images.shape} → {vision_features.shape}")
    
    # Test audio
    audio_prep = AudioPreprocessor(output_dim=512)
    audio = torch.randn(4, 16000)
    audio_features = audio_prep(audio)
    print(f"✓ Audio: {audio.shape} → {audio_features.shape}")
    
    # Test text
    text_prep = TextPreprocessor(embedding_dim=512)
    text_tokens = torch.randint(0, 50257, (4, 128))
    text_features = text_prep(text_tokens)
    print(f"✓ Text: {text_tokens.shape} → {text_features.shape}")
    
    # Test sensor
    sensor_prep = SensorPreprocessor(num_sensors=10, output_dim=512)
    sensor_data = torch.randn(4, 10, 100)
    sensor_features = sensor_prep(sensor_data)
    print(f"✓ Sensor: {sensor_data.shape} → {sensor_features.shape}")
    
    print("\n✅ All preprocessors working!")