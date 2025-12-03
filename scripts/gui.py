#!/usr/bin/env python3
"""
Liquid-Spiking Neural Network GUI Application

A comprehensive graphical interface for training, inference, and managing
hybrid liquid-spiking neural networks.

Usage:
    python gui.py
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import threading
import queue
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PyQt6 imports
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
        QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar, QFileDialog,
        QGroupBox, QGridLayout, QScrollArea, QMessageBox, QSlider,
        QTableWidget, QTableWidgetItem, QHeaderView, QSplitter
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QTextCursor, QPalette, QColor, QIcon
except ImportError:
    print("PyQt6 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6"])
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
        QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar, QFileDialog,
        QGroupBox, QGridLayout, QScrollArea, QMessageBox, QSlider,
        QTableWidget, QTableWidgetItem, QHeaderView, QSplitter
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QTextCursor, QPalette, QColor, QIcon

import torch
from torch.utils.data import DataLoader

# Import from main.py
from src.core.main import (
    TaskType, ModelConfig, LiquidSpikingNetwork, LiquidSpikingTrainer,
    DatasetFactory, create_llm_config, create_vision_config, create_robotics_config,
    load_model, generate_text, get_model_parameter_count
)


class TrainingThread(QThread):
    """Background thread for model training."""
    
    progress_update = pyqtSignal(int, float, float)  # epoch, train_loss, val_loss
    log_message = pyqtSignal(str)
    training_complete = pyqtSignal(str)  # final model path
    training_error = pyqtSignal(str)
    
    def __init__(self, config, train_loader, val_loader, num_epochs, output_dir):
        super().__init__()
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.should_stop = False
        
    def run(self):
        try:
            self.log_message.emit("🚀 Initializing model...")
            
            # Create model
            model = LiquidSpikingNetwork(self.config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            self.log_message.emit(f"📊 Model created with {total_params:,} parameters")
            self.log_message.emit(f"🖥️  Using device: {device}")
            
            # Create trainer
            trainer = LiquidSpikingTrainer(model, self.config)
            
            # Training loop
            for epoch in range(self.num_epochs):
                if self.should_stop:
                    self.log_message.emit("⚠️ Training stopped by user")
                    break
                
                self.log_message.emit(f"\n📈 Epoch {epoch + 1}/{self.num_epochs}")
                
                # Train epoch
                train_loss = trainer.train_epoch(self.train_loader)
                
                # Validate
                val_loss, val_acc = trainer.validate(self.val_loader)
                
                # Update progress
                progress = int((epoch + 1) / self.num_epochs * 100)
                self.progress_update.emit(progress, train_loss, val_loss)
                
                self.log_message.emit(
                    f"✓ Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}"
                )
                
                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(
                        self.output_dir,
                        f"{self.config.task_type.value}_epoch_{epoch + 1}.pt"
                    )
                    trainer.save_checkpoint(checkpoint_path)
                    self.log_message.emit(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Save final model
            task_name = self.config.task_type.value
            final_path = os.path.join(self.output_dir, f"{task_name}_final_model.pt")
            trainer.save_checkpoint(final_path)
            
            self.log_message.emit(f"\n✅ Training completed!")
            self.log_message.emit(f"📁 Final model saved: {final_path}")
            self.training_complete.emit(final_path)
            
        except Exception as e:
            self.training_error.emit(f"❌ Training error: {str(e)}")
    
    def stop(self):
        self.should_stop = True


class InferenceThread(QThread):
    """Background thread for model inference."""
    
    result_ready = pyqtSignal(str)
    log_message = pyqtSignal(str)
    inference_error = pyqtSignal(str)
    
    def __init__(self, model, config, tokenizer, prompt, max_length, temperature):
        super().__init__()
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
        self.temperature = temperature
        
    def run(self):
        try:
            self.log_message.emit(f"🤔 Generating text...")
            self.log_message.emit(f"📝 Prompt: {self.prompt}")
            
            # Generate text
            generated = generate_text(
                self.model, self.config, self.tokenizer,
                self.prompt, self.max_length, self.temperature
            )
            
            self.log_message.emit(f"\n✅ Generation complete!")
            self.result_ready.emit(generated)
            
        except Exception as e:
            self.inference_error.emit(f"❌ Inference error: {str(e)}")


class LiquidSpikingGUI(QMainWindow):
    """Main GUI application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Liquid-Spiking Neural Network - Training & Inference GUI")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize state
        self.current_model = None
        self.current_config = None
        self.current_tokenizer = None
        self.training_thread = None
        
        # Setup UI
        self.init_ui()
        
        # Apply dark theme
        self.apply_dark_theme()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create tabs
        self.training_tab = self.create_training_tab()
        self.inference_tab = self.create_inference_tab()
        self.settings_tab = self.create_settings_tab()
        self.models_tab = self.create_models_tab()
        
        self.tabs.addTab(self.training_tab, "🚀 Training")
        self.tabs.addTab(self.inference_tab, "💡 Inference")
        self.tabs.addTab(self.settings_tab, "⚙️ Settings")
        self.tabs.addTab(self.models_tab, "🤖 Model Types")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_header(self):
        """Create header section."""
        header = QWidget()
        layout = QVBoxLayout(header)
        
        title = QLabel("🧠 Hybrid Liquid-Spiking Neural Network")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle = QLabel("Advanced Neural Architecture with Liquid Time-Constants & Spiking Dynamics")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888;")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        return header
    
    def create_training_tab(self):
        """Create training configuration tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left panel - Configuration
        left_panel = QScrollArea()
        left_panel.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Task selection
        task_group = QGroupBox("Task Configuration")
        task_layout = QGridLayout()
        
        task_layout.addWidget(QLabel("Task Type:"), 0, 0)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["LLM (Language Model)", "Vision (Image Recognition)", "Robotics (Control)"])
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        task_layout.addWidget(self.task_combo, 0, 1)
        
        task_layout.addWidget(QLabel("Config Preset:"), 1, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "GPT-4o Competitive", "Claude Sonnet 4", "Ultra-Advanced"])
        task_layout.addWidget(self.preset_combo, 1, 1)
        
        task_group.setLayout(task_layout)
        left_layout.addWidget(task_group)
        
        # Architecture parameters
        arch_group = QGroupBox("Architecture Parameters")
        arch_layout = QGridLayout()
        
        arch_layout.addWidget(QLabel("Number of Layers:"), 0, 0)
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 24)
        self.num_layers_spin.setValue(8)
        arch_layout.addWidget(self.num_layers_spin, 0, 1)
        
        arch_layout.addWidget(QLabel("Hidden Dimension:"), 1, 0)
        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(64, 2048)
        self.hidden_dim_spin.setSingleStep(64)
        self.hidden_dim_spin.setValue(512)
        arch_layout.addWidget(self.hidden_dim_spin, 1, 1)
        
        arch_layout.addWidget(QLabel("Liquid Units:"), 2, 0)
        self.liquid_units_spin = QSpinBox()
        self.liquid_units_spin.setRange(32, 1024)
        self.liquid_units_spin.setSingleStep(32)
        self.liquid_units_spin.setValue(256)
        arch_layout.addWidget(self.liquid_units_spin, 2, 1)
        
        arch_layout.addWidget(QLabel("Spiking Units:"), 3, 0)
        self.spiking_units_spin = QSpinBox()
        self.spiking_units_spin.setRange(32, 512)
        self.spiking_units_spin.setSingleStep(32)
        self.spiking_units_spin.setValue(128)
        arch_layout.addWidget(self.spiking_units_spin, 3, 1)
        
        arch_layout.addWidget(QLabel("Attention Heads:"), 4, 0)
        self.num_heads_spin = QSpinBox()
        self.num_heads_spin.setRange(1, 32)
        self.num_heads_spin.setValue(8)
        arch_layout.addWidget(self.num_heads_spin, 4, 1)
        
        arch_group.setLayout(arch_layout)
        left_layout.addWidget(arch_group)
        
        # Training parameters
        train_group = QGroupBox("Training Parameters")
        train_layout = QGridLayout()
        
        train_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(30)
        train_layout.addWidget(self.epochs_spin, 0, 1)
        
        train_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(8)
        train_layout.addWidget(self.batch_size_spin, 1, 1)
        
        train_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0001)
        train_layout.addWidget(self.lr_spin, 2, 1)
        
        train_layout.addWidget(QLabel("Dropout:"), 3, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setSingleStep(0.1)
        self.dropout_spin.setValue(0.1)
        train_layout.addWidget(self.dropout_spin, 3, 1)
        
        train_group.setLayout(train_layout)
        left_layout.addWidget(train_group)
        
        # Dataset selection (LLM only)
        self.dataset_group = QGroupBox("Dataset Selection (LLM)")
        dataset_layout = QGridLayout()
        
        dataset_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "WikiText-103 (100M tokens - RECOMMENDED)",
            "WikiText-2 (2M tokens - Too Small)",
            "BookCorpus (70M tokens)",
            "CC-News (76M tokens)",
            "OpenWebText (40M tokens)",
            "Combined (200M+ tokens)"
        ])
        dataset_layout.addWidget(self.dataset_combo, 0, 1)
        
        dataset_layout.addWidget(QLabel("Tokenizer:"), 1, 0)
        self.tokenizer_combo = QComboBox()
        self.tokenizer_combo.addItems(["gpt2", "gpt3", "gpt4", "o200k", "codellama", "llama2"])
        dataset_layout.addWidget(self.tokenizer_combo, 1, 1)
        
        self.dataset_group.setLayout(dataset_layout)
        left_layout.addWidget(self.dataset_group)
        
        # Advanced options
        adv_group = QGroupBox("Advanced Options")
        adv_layout = QVBoxLayout()
        
        self.use_stdp_check = QCheckBox("Enable STDP Plasticity")
        self.use_meta_check = QCheckBox("Enable Meta-Plasticity")
        self.use_mamba_check = QCheckBox("Enable Mamba Integration")
        self.mixed_precision_check = QCheckBox("Mixed Precision Training")
        self.mixed_precision_check.setChecked(True)
        
        adv_layout.addWidget(self.use_stdp_check)
        adv_layout.addWidget(self.use_meta_check)
        adv_layout.addWidget(self.use_mamba_check)
        adv_layout.addWidget(self.mixed_precision_check)
        
        adv_group.setLayout(adv_layout)
        left_layout.addWidget(adv_group)
        
        # Start training button
        self.start_train_btn = QPushButton("🚀 Start Training")
        self.start_train_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.start_train_btn.setMinimumHeight(50)
        self.start_train_btn.clicked.connect(self.start_training)
        left_layout.addWidget(self.start_train_btn)
        
        self.stop_train_btn = QPushButton("⏹️ Stop Training")
        self.stop_train_btn.setFont(QFont("Arial", 12))
        self.stop_train_btn.setMinimumHeight(40)
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        left_layout.addWidget(self.stop_train_btn)
        
        left_layout.addStretch()
        left_panel.setWidget(left_widget)
        
        # Right panel - Progress and logs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        
        # Metrics display
        metrics_layout = QGridLayout()
        metrics_layout.addWidget(QLabel("Current Epoch:"), 0, 0)
        self.epoch_label = QLabel("0 / 0")
        metrics_layout.addWidget(self.epoch_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Train Loss:"), 1, 0)
        self.train_loss_label = QLabel("N/A")
        metrics_layout.addWidget(self.train_loss_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Val Loss:"), 2, 0)
        self.val_loss_label = QLabel("N/A")
        metrics_layout.addWidget(self.val_loss_label, 2, 1)
        
        progress_layout.addLayout(metrics_layout)
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setFont(QFont("Courier", 10))
        log_layout.addWidget(self.train_log)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        return tab
    
    def create_inference_tab(self):
        """Create inference tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model loading
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to trained model...")
        model_layout.addWidget(self.model_path_edit)
        
        browse_btn = QPushButton("📂 Browse")
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_btn)
        
        load_btn = QPushButton("📥 Load Model")
        load_btn.clicked.connect(self.load_model_for_inference)
        model_layout.addWidget(load_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Model info
        self.model_info_group = QGroupBox("Model Information")
        self.model_info_layout = QVBoxLayout()
        self.model_info_label = QLabel("No model loaded")
        self.model_info_layout.addWidget(self.model_info_label)
        self.model_info_group.setLayout(self.model_info_layout)
        layout.addWidget(self.model_info_group)
        
        # Inference parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Max Length:"), 0, 0)
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(10, 1000)
        self.max_length_spin.setValue(100)
        params_layout.addWidget(self.max_length_spin, 0, 1)
        
        params_layout.addWidget(QLabel("Temperature:"), 1, 0)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.8)
        params_layout.addWidget(self.temperature_spin, 1, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Prompt input
        prompt_group = QGroupBox("Input Prompt")
        prompt_layout = QVBoxLayout()
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt here...")
        self.prompt_edit.setMaximumHeight(100)
        prompt_layout.addWidget(self.prompt_edit)
        
        self.generate_btn = QPushButton("✨ Generate")
        self.generate_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.clicked.connect(self.generate_text_inference)
        self.generate_btn.setEnabled(False)
        prompt_layout.addWidget(self.generate_btn)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Output
        output_group = QGroupBox("Generated Output")
        output_layout = QVBoxLayout()
        
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setFont(QFont("Courier", 11))
        output_layout.addWidget(self.output_edit)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        return tab
    
    def create_settings_tab(self):
        """Create settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Device settings
        device_group = QGroupBox("Device Settings")
        device_layout = QGridLayout()
        
        device_layout.addWidget(QLabel("Compute Device:"), 0, 0)
        self.device_combo = QComboBox()
        devices = ["Auto (CUDA if available)", "CPU", "CUDA"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"CUDA:{i}")
        self.device_combo.addItems(devices)
        device_layout.addWidget(self.device_combo, 0, 1)
        
        device_layout.addWidget(QLabel("Multi-GPU Strategy:"), 1, 0)
        self.multi_gpu_combo = QComboBox()
        self.multi_gpu_combo.addItems(["None (Single GPU/CPU)", "DataParallel", "DistributedDataParallel"])
        device_layout.addWidget(self.multi_gpu_combo, 1, 1)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # Output directories
        output_group = QGroupBox("Output Directories")
        output_layout = QGridLayout()
        
        output_layout.addWidget(QLabel("Model Save Directory:"), 0, 0)
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setText("./models")
        output_layout.addWidget(self.model_dir_edit, 0, 1)
        
        browse_model_dir_btn = QPushButton("Browse")
        browse_model_dir_btn.clicked.connect(lambda: self.browse_directory(self.model_dir_edit))
        output_layout.addWidget(browse_model_dir_btn, 0, 2)
        
        output_layout.addWidget(QLabel("Dataset Cache Directory:"), 1, 0)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setText("./data")
        output_layout.addWidget(self.data_dir_edit, 1, 1)
        
        browse_data_dir_btn = QPushButton("Browse")
        browse_data_dir_btn.clicked.connect(lambda: self.browse_directory(self.data_dir_edit))
        output_layout.addWidget(browse_data_dir_btn, 1, 2)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # System info
        sys_info_group = QGroupBox("System Information")
        sys_info_layout = QVBoxLayout()
        
        info_text = f"""
<b>PyTorch Version:</b> {torch.__version__}<br>
<b>CUDA Available:</b> {torch.cuda.is_available()}<br>
"""
        if torch.cuda.is_available():
            info_text += f"""
<b>CUDA Version:</b> {torch.version.cuda}<br>
<b>GPU Count:</b> {torch.cuda.device_count()}<br>
<b>GPU Name:</b> {torch.cuda.get_device_name(0)}<br>
<b>GPU Memory:</b> {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB
"""
        
        sys_info_label = QLabel(info_text)
        sys_info_layout.addWidget(sys_info_label)
        
        sys_info_group.setLayout(sys_info_layout)
        layout.addWidget(sys_info_group)
        
        layout.addStretch()
        
        return tab
    
    def create_models_tab(self):
        """Create models comparison tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Overview
        overview = QLabel("""
        <h2>Liquid-Spiking Neural Network Model Types</h2>
        <p>This hybrid architecture supports three main task types, each optimized for different domains:</p>
        """)
        layout.addWidget(overview)
        
        # Model comparison table
        table_group = QGroupBox("Model Type Comparison")
        table_layout = QVBoxLayout()
        
        table = QTableWidget(3, 5)
        table.setHorizontalHeaderLabels([
            "Task Type", "Input Domain", "Use Cases", "Default Architecture", "Example Applications"
        ])
        
        # LLM row
        table.setItem(0, 0, QTableWidgetItem("🔤 LLM (Language)"))
        table.setItem(0, 1, QTableWidgetItem("Text / Tokens"))
        table.setItem(0, 2, QTableWidgetItem("Text generation, translation, summarization"))
        table.setItem(0, 3, QTableWidgetItem("8 layers, 512 hidden, 256 liquid, 128 spiking"))
        table.setItem(0, 4, QTableWidgetItem("Chatbots, code generation, Q&A systems"))
        
        # Vision row
        table.setItem(1, 0, QTableWidgetItem("👁️ Vision"))
        table.setItem(1, 1, QTableWidgetItem("Images / Pixels"))
        table.setItem(1, 2, QTableWidgetItem("Image classification, object detection"))
        table.setItem(1, 3, QTableWidgetItem("6 layers, 256 hidden, 128 liquid, 64 spiking"))
        table.setItem(1, 4, QTableWidgetItem("CIFAR-10, ImageNet, medical imaging"))
        
        # Robotics row
        table.setItem(2, 0, QTableWidgetItem("🤖 Robotics"))
        table.setItem(2, 1, QTableWidgetItem("Sensor Data / State"))
        table.setItem(2, 2, QTableWidgetItem("Control, navigation, manipulation"))
        table.setItem(2, 3, QTableWidgetItem("4 layers, 256 hidden, 128 liquid, 64 spiking"))
        table.setItem(2, 4, QTableWidgetItem("Robot arms, autonomous vehicles, drones"))
        
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        table_layout.addWidget(table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Architecture details
        arch_group = QGroupBox("Hybrid Architecture Components")
        arch_layout = QVBoxLayout()
        
        arch_text = QLabel("""
        <h3>Key Components:</h3>
        <ul>
            <li><b>Liquid Neural Networks (LTC/CfC):</b> Continuous-time dynamics with adaptive time constants
                <br>→ Handles temporal patterns and long-term dependencies</li>
            <li><b>Spiking Neural Networks (SNN):</b> Event-based, energy-efficient spike processing
                <br>→ Enables neuromorphic computation and temporal credit assignment</li>
            <li><b>Multi-Head Attention:</b> Transformer-style attention for global context
                <br>→ Captures long-range dependencies in sequences</li>
            <li><b>STDP Plasticity (Optional):</b> Spike-timing dependent synaptic plasticity
                <br>→ Biologically-inspired learning for unsupervised feature discovery</li>
            <li><b>Meta-Plasticity (Optional):</b> Learning to learn - adaptive learning rates
                <br>→ Faster adaptation to new tasks and domains</li>
        </ul>
        
        <h3>Advantages:</h3>
        <ul>
            <li>✓ <b>Temporal Processing:</b> Native handling of time-series and sequential data</li>
            <li>✓ <b>Energy Efficiency:</b> Spiking dynamics reduce computational cost</li>
            <li>✓ <b>Continual Learning:</b> Learn new tasks without catastrophic forgetting</li>
            <li>✓ <b>Neuromorphic:</b> Compatible with neuromorphic hardware (Intel Loihi, IBM TrueNorth)</li>
            <li>✓ <b>Flexible:</b> Adaptable architecture for multiple domains</li>
        </ul>
        """)
        arch_text.setWordWrap(True)
        
        arch_layout.addWidget(arch_text)
        arch_group.setLayout(arch_layout)
        layout.addWidget(arch_group)
        
        layout.addStretch()
        
        return tab
    
    def apply_dark_theme(self):
        """Apply a modern dark theme."""
        palette = QPalette()
        
        # Colors
        dark_bg = QColor(45, 45, 48)
        darker_bg = QColor(30, 30, 30)
        text_color = QColor(255, 255, 255)
        highlight = QColor(0, 120, 215)
        
        palette.setColor(QPalette.ColorRole.Window, dark_bg)
        palette.setColor(QPalette.ColorRole.WindowText, text_color)
        palette.setColor(QPalette.ColorRole.Base, darker_bg)
        palette.setColor(QPalette.ColorRole.AlternateBase, dark_bg)
        palette.setColor(QPalette.ColorRole.ToolTipBase, text_color)
        palette.setColor(QPalette.ColorRole.ToolTipText, text_color)
        palette.setColor(QPalette.ColorRole.Text, text_color)
        palette.setColor(QPalette.ColorRole.Button, dark_bg)
        palette.setColor(QPalette.ColorRole.ButtonText, text_color)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        self.setPalette(palette)
        
        # Stylesheet for additional styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #555;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: white;
                padding: 10px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QTabBar::tab:hover {
                background-color: #505050;
            }
        """)
    
    def on_task_changed(self, index):
        """Handle task type change."""
        # Show/hide dataset group for LLM
        is_llm = index == 0
        self.dataset_group.setVisible(is_llm)
    
    def start_training(self):
        """Start model training."""
        try:
            self.log_to_train("🚀 Preparing for training...")
            
            # Get task type
            task_index = self.task_combo.currentIndex()
            if task_index == 0:
                task_type = TaskType.LLM
                config = create_llm_config(self.tokenizer_combo.currentText())
            elif task_index == 1:
                task_type = TaskType.VISION
                config = create_vision_config()
            else:
                task_type = TaskType.ROBOTICS
                config = create_robotics_config()
            
            # Update config with GUI parameters
            config.num_layers = self.num_layers_spin.value()
            config.hidden_dim = self.hidden_dim_spin.value()
            config.liquid_units = self.liquid_units_spin.value()
            config.spiking_units = self.spiking_units_spin.value()
            config.num_attention_heads = self.num_heads_spin.value()
            config.batch_size = self.batch_size_spin.value()
            config.learning_rate = self.lr_spin.value()
            config.dropout = self.dropout_spin.value()
            config.mixed_precision = self.mixed_precision_check.isChecked()
            
            # STDP settings
            if self.use_stdp_check.isChecked():
                config.use_stdp = True
                config.stdp_type = 'homeostatic'
            
            # Meta-plasticity settings
            if self.use_meta_check.isChecked():
                config.use_meta_plasticity = True
            
            self.log_to_train(f"✓ Configuration created for {task_type.value}")
            
            # Create datasets
            self.log_to_train("📚 Loading datasets...")
            
            if task_type == TaskType.LLM:
                # Get dataset type
                dataset_map = {
                    0: 'wikitext103',
                    1: 'wikitext2',
                    2: 'bookcorpus',
                    3: 'ccnews',
                    4: 'openwebtext',
                    5: 'combined'
                }
                dataset_type = dataset_map[self.dataset_combo.currentIndex()]
                
                dataset, tokenizer = DatasetFactory.create_llm_dataset(
                    vocab_size=config.vocab_size,
                    seq_length=config.sequence_length,
                    tokenizer_type=self.tokenizer_combo.currentText(),
                    dataset_type=dataset_type,
                    cache_dir=self.data_dir_edit.text()
                )
            elif task_type == TaskType.VISION:
                dataset = DatasetFactory.create_vision_dataset()
            else:
                dataset = DatasetFactory.create_robotics_dataset()
            
            # Split dataset
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config.batch_size, 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config.batch_size, 
                shuffle=False
            )
            
            self.log_to_train(f"✓ Datasets loaded: {train_size:,} train, {val_size:,} val")
            
            # Create output directory
            output_dir = self.model_dir_edit.text()
            os.makedirs(output_dir, exist_ok=True)
            
            # Start training thread
            num_epochs = self.epochs_spin.value()
            self.training_thread = TrainingThread(
                config, train_loader, val_loader, num_epochs, output_dir
            )
            
            # Connect signals
            self.training_thread.progress_update.connect(self.on_progress_update)
            self.training_thread.log_message.connect(self.log_to_train)
            self.training_thread.training_complete.connect(self.on_training_complete)
            self.training_thread.training_error.connect(self.on_training_error)
            
            # Update UI
            self.start_train_btn.setEnabled(False)
            self.stop_train_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            
            # Start training
            self.training_thread.start()
            self.log_to_train("🎯 Training started!")
            
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Failed to start training:\n{str(e)}")
            self.log_to_train(f"❌ Error: {str(e)}")
    
    def stop_training(self):
        """Stop training."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.log_to_train("⏹️ Stopping training...")
    
    def on_progress_update(self, progress, train_loss, val_loss):
        """Update progress display."""
        self.progress_bar.setValue(progress)
        current_epoch = int(progress / 100 * self.epochs_spin.value())
        self.epoch_label.setText(f"{current_epoch} / {self.epochs_spin.value()}")
        self.train_loss_label.setText(f"{train_loss:.4f}")
        self.val_loss_label.setText(f"{val_loss:.4f}")
    
    def on_training_complete(self, model_path):
        """Handle training completion."""
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        QMessageBox.information(
            self, 
            "Training Complete", 
            f"Training completed successfully!\n\nModel saved to:\n{model_path}"
        )
    
    def on_training_error(self, error_msg):
        """Handle training error."""
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        QMessageBox.critical(self, "Training Error", error_msg)
    
    def log_to_train(self, message):
        """Add message to training log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.train_log.append(f"[{timestamp}] {message}")
        self.train_log.moveCursor(QTextCursor.MoveOperation.End)
    
    def browse_model(self):
        """Browse for model file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "./models",
            "PyTorch Models (*.pt *.pth)"
        )
        if filename:
            self.model_path_edit.setText(filename)
    
    def load_model_for_inference(self):
        """Load model for inference."""
        try:
            model_path = self.model_path_edit.text()
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "File Not Found", "Model file does not exist!")
                return
            
            self.output_edit.append("📂 Loading model...")
            
            # Determine task type from filename
            if 'llm' in model_path.lower():
                task_type = TaskType.LLM
            elif 'vision' in model_path.lower():
                task_type = TaskType.VISION
            else:
                task_type = TaskType.ROBOTICS
            
            # Load model
            self.current_model, self.current_config = load_model(model_path, task_type)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.current_model = self.current_model.to(device)
            
            # Load tokenizer for LLM
            if task_type == TaskType.LLM:
                from transformers import AutoTokenizer
                self.current_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # Display model info
            total_params = sum(p.numel() for p in self.current_model.parameters())
            info_text = f"""
<b>Model loaded successfully!</b><br>
<b>Task Type:</b> {task_type.value}<br>
<b>Total Parameters:</b> {total_params:,}<br>
<b>Device:</b> {device}<br>
<b>Layers:</b> {self.current_config.num_layers}<br>
<b>Hidden Dim:</b> {self.current_config.hidden_dim}
            """
            self.model_info_label.setText(info_text)
            
            self.output_edit.append("✅ Model loaded successfully!")
            self.generate_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}")
            self.output_edit.append(f"❌ Error: {str(e)}")
    
    def generate_text_inference(self):
        """Generate text using loaded model."""
        if self.current_model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first!")
            return
        
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter a prompt!")
            return
        
        # Start inference thread
        self.inference_thread = InferenceThread(
            self.current_model,
            self.current_config,
            self.current_tokenizer,
            prompt,
            self.max_length_spin.value(),
            self.temperature_spin.value()
        )
        
        self.inference_thread.result_ready.connect(self.on_inference_complete)
        self.inference_thread.log_message.connect(lambda msg: self.output_edit.append(msg))
        self.inference_thread.inference_error.connect(
            lambda err: QMessageBox.critical(self, "Inference Error", err)
        )
        
        self.generate_btn.setEnabled(False)
        self.output_edit.clear()
        self.inference_thread.start()
    
    def on_inference_complete(self, generated_text):
        """Handle inference completion."""
        self.output_edit.append("\n" + "="*60)
        self.output_edit.append(generated_text)
        self.output_edit.append("="*60)
        self.generate_btn.setEnabled(True)
    
    def browse_directory(self, line_edit):
        """Browse for directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            line_edit.setText(directory)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Liquid-Spiking Neural Network")
    
    # Set application-wide font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    # Create and show main window
    window = LiquidSpikingGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
