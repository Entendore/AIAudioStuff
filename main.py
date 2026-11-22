import sys
import os
import json
import time
import shutil
import logging
import math
import csv
import gc
import copy
import subprocess
import threading
import inspect
import platform
import base64
import zipfile
import re
import itertools
import random
import wave
import queue
import traceback
import faulthandler
import tempfile
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque
from functools import partial, lru_cache
from logging.handlers import RotatingFileHandler
from enum import Enum

# Enable faulthandler immediately to catch silent C-level crashes
CWD = os.getcwd()
FAULT_LOG_FILE = os.path.join(CWD, "crash_traceback.log")
try:
    fault_log_file = open(FAULT_LOG_FILE, "w", buffering=1)
    faulthandler.enable(file=fault_log_file)
except Exception as e:
    print(f"Could not enable faulthandler: {e}")

import numpy as np
from scipy.io import wavfile
import scipy.signal

# Qt Imports
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QUrl, QTimer, QPoint, QSize, QLineF,
    QStandardPaths, QEvent, QByteArray, QRect, QPointF, QProcess
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSlider, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter,
    QDialog, QFormLayout, QMenu, QStyle, QInputDialog, QPlainTextEdit,
    QSizePolicy, QFrame, QToolButton, QStatusBar, QListWidget, QListWidgetItem,
    QDockWidget, QScrollArea, QToolTip, QCompleter, QSystemTrayIcon
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import (
    QAction, QIcon, QPixmap, QColor, QPalette, QFont, QPainter, QPen,
    QMouseEvent, QKeySequence, QShortcut, QDesktopServices, QLinearGradient,
    QPainterPath, QPolygon, QPolygonF, QBrush, QImage, QTextCursor
)

# ML Imports - Core
try:
    import torch
    from diffusers import AudioLDM2Pipeline
    import transformers
    import diffusers
    ML_AVAILABLE = True
    
    # FIX: Monkey-patch GPT2Model to restore _update_model_kwargs_for_generation
    # This fixes compatibility with transformers >= 4.50.0
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
        if not hasattr(GPT2Model, "_update_model_kwargs_for_generation"):
            def _update_model_kwargs_for_generation_patch(self, outputs, model_kwargs, is_encoder_decoder=False, standardize_cache_format=False):
                model_kwargs["past_key_values"] = self._extract_past_from_model_output(outputs, standardize_cache_format=standardize_cache_format)
                if "token_type_ids" in model_kwargs:
                    token_type_ids = model_kwargs["token_type_ids"]
                    model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)
                if not is_encoder_decoder:
                    if "attention_mask" in model_kwargs:
                        attention_mask = model_kwargs["attention_mask"]
                        model_kwargs["attention_mask"] = torch.cat(
                            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                        )
                return model_kwargs
            GPT2Model._update_model_kwargs_for_generation = _update_model_kwargs_for_generation_patch
    except Exception as e:
        print(f"Failed to monkey-patch GPT2Model: {e}")

except ImportError:
    ML_AVAILABLE = False

# ML Imports - Optional Features
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import torchaudio
    from demucs.pretrained import get_model as get_demucs_model
    from demucs.apply import apply_model as apply_demucs_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    from transformers import ClapModel, ClapProcessor
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False

try:
    from pedalboard import Pedalboard, Reverb, Delay, Chorus, Distortion, Compressor, LowpassFilter, HighpassFilter, Gain
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False


# --- Configuration & Constants ---
APP_NAME = "AudioLDM2 Studio - SFX & Long-Form Edition"
APP_VERSION = "4.2.9-expert"
SETTINGS_ORG = "AudioLDM2"
SETTINGS_APP = "Studio"

DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
BATCH_STATE_FILE = os.path.join(CWD, "batch_state.json")
LOG_FILE = os.path.join(CWD, f"{APP_NAME}.log")
TEMP_DIR = tempfile.gettempdir()
FX_PREVIEW_FILE = os.path.join(TEMP_DIR, "audioldm2_fx_preview.wav")
TEMP_LOAD_WAV = os.path.join(TEMP_DIR, "audioldm2_temp_load.wav")

DEFAULT_MODELS = [
    "cvssp/audioldm2",
    "cvssp/audioldm2-music",
    "cvssp/audioldm2-gigaspeech",
    "cvssp/audioldm2-ljspeech",
]

DEFAULT_PROMPT_PRESETS = [
    "Cinematic riser and massive impact, sub bass drop",
    "Sci-fi UI confirmation beep, clean digital tone",
    "Distant thunder rolling across a valley, rain ambience",
    "Heavy sword draw and metallic scrape",
    "Continuous ambient forest soundscape, birds, wind",
    "Evolving dark drone, low frequency rumble",
    "Foley: footsteps on gravel, slow pace",
    "Whoosh transition, fast air movement"
]

NEGATIVE_PROMPT_PRESETS = [
    "noise, distortion, artifacts, low quality",
    "static, hum, buzzing, interference",
    "echo, reverb, muffled sound",
    "clipping, peaking, distortion",
]

AUDIO_TAGS = [
    "rain", "thunder", "ocean", "waves", "wind", "fire", "crackling", "storm",
    "bass", "bass drop", "drums", "kick", "snare", "hi-hat", "cymbals",
    "guitar", "acoustic", "electric", "distorted", "piano", "synth",
    "synthesizer", "pad", "strings", "violin", "cello", "orchestra",
    "brass", "trumpet", "saxophone", "flute", "vocals", "choir", "humming",
    "engine", "car", "traffic", "train", "airplane", "motorcycle",
    "birds", "chirping", "dog", "barking", "cat", "meowing", "insects",
    "footsteps", "door", "knock", "bell", "ring", "alarm", "siren",
    "explosion", "gunshot", "impact", "crash", "glass", "breaking",
    "babbling brook", "stream", "river", "waterfall", "drip", "splash",
    "ambient", "cinematic", "epic", "lofi", "jazz", "classical", "rock",
    "metal", "electronic", "techno", "house", "drone", "8-bit", "retro",
    "riser", "whoosh", "swoosh", "UI", "beep", "boop", "mechanical"
]

OVERLAP_S = 1.0
MIN_CHUNK_S = 2.0
MAX_HISTORY = 500
MAX_PROMPT_HISTORY = 50
FADE_DURATION_S = 0.05
TRIM_THRESHOLD = 0.01
ETA_SMOOTHING_ALPHA = 0.3
AUTO_SAVE_INTERVAL_MS = 60_000
GPU_MEM_POLL_MS = 2_000
NOTIFY_INTERVAL_MS = 50
PROMPT_MATRIX_MAX = 200
THUMB_WIDTH = 120
THUMB_HEIGHT = 28
THUMB_BARS = 60

STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QGroupBox {
    border: 1px solid #3c3c3c; border-radius: 4px; margin-top: 10px;
    padding-top: 10px; font-weight: bold; color: #cccccc;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 5px; background-color: #2b2b2b;
}
QTabWidget::pane {
    border: 1px solid #3c3c3c; border-radius: 2px;
    background-color: #2b2b2b; top: -1px;
}
QTabBar::tab {
    background: #3c3c3c; color: #aaaaaa; padding: 8px 18px;
    margin-right: 2px; border-top-left-radius: 3px;
    border-top-right-radius: 3px; min-width: 70px; font-weight: bold;
}
QTabBar::tab:selected { background: #2A82DA; color: white; }
QTabBar::tab:hover:!selected { background: #4a4a4a; color: white; }
QPushButton {
    background-color: #3c3c3c; border: 1px solid #4a4a4a; color: #e0e0e0;
    padding: 6px 12px; border-radius: 3px; min-height: 20px;
}
QPushButton:hover { background-color: #4a4a4a; border-color: #2A82DA; }
QPushButton:pressed { background-color: #2b2b2b; }
QPushButton:disabled { background-color: #333333; color: #666666; border-color: #333333; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background-color: #333333; border: 1px solid #3c3c3c; padding: 4px 6px;
    border-radius: 2px; color: #e0e0e0; selection-background-color: #2A82DA;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1px solid #2A82DA;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView, QAbstractItemView {
    background-color: #333333; border: 1px solid #3c3c3c;
    selection-background-color: #2A82DA; color: #e0e0e0;
}
QTableWidget {
    gridline-color: #3c3c3c; background-color: #2b2b2b;
    alternate-background-color: #333333; border: 1px solid #3c3c3c; border-radius: 2px;
}
QHeaderView::section {
    background-color: #3c3c3c; padding: 6px; border: none;
    font-weight: bold; color: #cccccc;
}
QProgressBar {
    border: 1px solid #3c3c3c; border-radius: 2px; text-align: center;
    background-color: #333333; color: white; min-height: 22px;
}
QProgressBar::chunk { background-color: #2A82DA; border-radius: 1px; }
QScrollBar:vertical { background: #2b2b2b; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #4a4a4a; min-height: 20px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #2A82DA; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #2b2b2b; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #4a4a4a; min-width: 20px; border-radius: 4px; }
QScrollBar::handle:horizontal:hover { background: #2A82DA; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollArea { border: none; background-color: transparent; }
QCheckBox { color: #e0e0e0; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px; border-radius: 2px;
    border: 1px solid #4a4a4a; background: #333333;
}
QCheckBox::indicator:checked { background: #2A82DA; border-color: #2A82DA; }
QLabel { color: #d0d0d0; }
QToolButton {
    background-color: #3c3c3c; border: 1px solid #4a4a4a;
    border-radius: 3px; padding: 4px; color: #e0e0e0;
}
QToolButton:hover { background-color: #4a4a4a; border-color: #2A82DA; }
QStatusBar { background-color: #2b2b2b; color: #aaaaaa; }
QStatusBar::item { border: none; }
QMenuBar { background-color: #2b2b2b; color: #d0d0d0; border-bottom: 1px solid #3c3c3c; }
QMenuBar::item:selected { background-color: #3c3c3c; }
QMenu { background-color: #333333; border: 1px solid #3c3c3c; color: #d0d0d0; }
QMenu::item:selected { background-color: #2A82DA; }
QSlider::groove:horizontal {
    border: 1px solid #3c3c3c; height: 4px; background: #333333; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #2A82DA; border: 1px solid #2A82DA; width: 12px;
    margin: -5px 0; border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #3a92ea; }
QSplitter::handle { background-color: #3c3c3c; }
QSplitter::handle:horizontal { width: 2px; }
QListWidget { background-color: #2b2b2b; border: 1px solid #3c3c3c; border-radius: 2px; }
QListWidget::item:selected { background-color: #2A82DA; color: white; }
"""

# --- Robust Logging & Exception Hooks ---
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
        ]
    )
except Exception as e:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    print(f"Failed to initialize file logging: {e}")

logger = logging.getLogger(APP_NAME)

def install_exception_hooks():
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception in main thread", exc_info=(exc_type, exc_value, exc_traceback))
            
    sys.excepthook = global_exception_handler

    def threading_exception_handler(args):
        logger.critical(f"Uncaught exception in thread {args.thread.name}", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        
    threading.excepthook = threading_exception_handler

install_exception_hooks()

class JsonSettings:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self._dirty = False
        self._lock = threading.RLock()
        self.load()

    def load(self):
        with self._lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, 'r', encoding='utf-8') as f:
                        self.data = json.load(f)
                except Exception as e:
                    logging.error(f"Failed to load JSON settings: {e}", exc_info=True)
                    self.data = {}

    def save(self):
        with self._lock:
            if not self._dirty: return
            data_copy = copy.deepcopy(self.data)
            self._dirty = False
            
        try:
            tmp = self.path + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data_copy, f, indent=4)
            os.replace(tmp, self.path)
        except Exception as e:
            logging.error(f"Failed to save JSON settings: {e}", exc_info=True)
            with self._lock:
                self._dirty = True

    def value(self, key, default=None, type=None):
        with self._lock:
            val = self.data.get(key, default)
            if isinstance(val, (list, dict)):
                val = copy.deepcopy(val)

        if val is None:
            if isinstance(default, (list, dict)):
                return copy.deepcopy(default)
            return default

        if isinstance(val, str) and val.startswith("base64:"):
            return QByteArray.fromBase64(val[7:].encode('utf-8'))

        if type == bool:
            if isinstance(val, str): return val.lower() in ('true', '1', 'yes')
            return bool(val)
        if type == int:
            try: return int(val)
            except (ValueError, TypeError): return default
        if type == float:
            try: return float(val)
            except (ValueError, TypeError): return default
        if isinstance(val, (list, dict)):
            return copy.deepcopy(val)
        return val

    def setValue(self, key, value):
        with self._lock:
            if isinstance(value, QByteArray):
                value = "base64:" + base64.b64encode(value.data()).decode('utf-8')
            elif isinstance(value, (list, dict)):
                value = copy.deepcopy(value)
            if self.data.get(key) != value:
                self.data[key] = value
                self._dirty = True

    def remove(self, key):
        with self._lock:
            if key in self.data:
                del self.data[key]
                self._dirty = True
        self.save()

    def sync(self):
        self.save()

    def clear(self):
        with self._lock:
            self.data = {}
            self._dirty = True
        self.save()


def log_system_info():
    logger.info("=" * 60)
    logger.info(f"{APP_NAME} v{APP_VERSION} starting")
    logger.info(f"Python: {sys.version}")
    logger.info(f"OS: {platform.platform()}")
    if ML_AVAILABLE:
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"Diffusers: {diffusers.__version__}")
        logger.info(f"Transformers: {transformers.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("=" * 60)

def sanitize_filename(text: str, max_length: int = 30) -> str:
    safe = "".join(c for c in text if c.isalnum() or c in (' ', '_', '-')).rstrip()
    safe = safe[:max_length].replace(' ', '_') if safe else "audio"
    return safe

def format_time(seconds) -> str:
    try:
        total_secs = int(float(seconds))
        if total_secs < 0: total_secs = 0
        minutes, secs = divmod(total_secs, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    except Exception:
        return "00:00"

def normalize_audio_data(data: np.ndarray) -> np.ndarray:
    if data is None or data.size == 0: return np.zeros(0, dtype=np.float32)
    if data.ndim > 1: data = data.mean(axis=1)
    if data.dtype == np.int16: return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32: return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8: return (data.astype(np.float32) - 128.0) / 128.0
    if data.dtype in (np.float32, np.float64): return data.astype(np.float32)
    return data.astype(np.float32)

def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    if audio.size == 0: return audio.copy()
    audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak == 0: return audio
    target_peak = 10 ** (target_db / 20.0)
    return audio * (target_peak / peak)

def apply_fade(audio: np.ndarray, sample_rate: int, fade_in_s: float = 0.05, fade_out_s: float = 0.05) -> np.ndarray:
    audio = audio.astype(np.float32, copy=True)
    if audio.size == 0: return audio
    fade_in_samples = int(fade_in_s * sample_rate)
    fade_out_samples = int(fade_out_s * sample_rate)
    if fade_in_samples + fade_out_samples > len(audio):
        total_fade = len(audio)
        fade_in_samples = total_fade // 2
        fade_out_samples = total_fade - fade_in_samples
    if fade_in_samples > 0:
        audio[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
    if fade_out_samples > 0:
        audio[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
    return audio

def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if audio.size == 0: return audio.copy()
    audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    above_threshold = np.where(np.abs(audio) > threshold)[0]
    if above_threshold.size == 0: return audio
    return audio[above_threshold[0]:above_threshold[-1] + 1]

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    if not isinstance(v0, torch.Tensor): v0 = torch.tensor(v0)
    if not isinstance(v1, torch.Tensor): v1 = torch.tensor(v1)
    v0 = v0.flatten().float()
    v1 = v1.flatten().float()
    v0_norm = v0 / (v0.norm() + 1e-9)
    v1_norm = v1 / (v1.norm() + 1e-9)
    dot = torch.dot(v0_norm, v1_norm)
    if torch.abs(dot) > DOT_THRESHOLD:
        return torch.lerp(v0, v1, t).view(v0.shape)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    slerp_v = (torch.sin((1.0 - t) * omega) / so) * v0 + (torch.sin(t * omega) / so) * v1
    return slerp_v.view(v0.shape)

class BatchState(Enum):
    IDLE = 0
    RUNNING = 1
    CANCELLED = 2
    COMPLETED = 3

@dataclass(slots=True)
class GenerationParams:
    prompt: str
    negative_prompt: str
    model_name: str
    duration: float
    steps: int
    guidance: float
    seed: int
    device: str
    cache_dir: str
    output_dir: str
    num_variations: int = 1
    normalize: bool = True
    fade_in: bool = True
    fade_out: bool = True
    trim: bool = True
    chunk_size: float = 10.0
    use_cpu_offload: bool = False
    section: str = ""
    high_pass_freq: float = 0.0
    low_pass_freq: float = 0.0
    conditioning_audio: str = ""
    inpaint_source: str = ""
    inpaint_start_s: float = 0.0
    inpaint_end_s: float = 0.0
    use_seed_travel: bool = False
    travel_start_seed: int = -1
    travel_end_seed: int = 100
    travel_steps: int = 5

@dataclass(slots=True)
class GenerationStatus:
    variation: int
    total_variations: int
    chunk: int
    total_chunks: int
    step: int
    total_steps: int
    
    def format(self) -> str:
        s = f"Variation {self.variation}/{self.total_variations}"
        if self.total_chunks > 1:
            s += f" - Chunk {self.chunk}/{self.total_chunks}"
        s += f" - Step {self.step}/{self.total_steps}"
        return s

class BatchOrchestrator(QObject):
    batch_started = Signal()
    batch_progress = Signal(int, int)
    batch_finished = Signal()
    batch_log = Signal(str)
    next_prompt_ready = Signal(str, float)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.batch_state = BatchState.IDLE
        self.batch_prompts = []
        self.batch_index = 0
        self.is_timeline_batch = False
        self._master_wav_writer = None
        self.master_timeline_path = ""
        self._overlap_buffer = None
        self._master_fr = 0

    def start_batch(self, prompts, is_timeline=False):
        self.batch_prompts = prompts
        self.batch_index = 0
        self.is_timeline_batch = is_timeline
        self.batch_state = BatchState.RUNNING
        self.batch_started.emit()
        
        if is_timeline:
            self.master_timeline_path = os.path.join(self.main_window.output_dir_edit.text().strip() or os.getcwd(), "timeline_master.wav")
            self._master_wav_writer = wave.open(self.master_timeline_path, 'wb')
            self._master_wav_writer.setnchannels(1)
            self._master_wav_writer.setsampwidth(2)
            self._overlap_buffer = None
            
        self._process_next()

    def _process_next(self):
        if self.batch_state != BatchState.RUNNING: return
        if self.batch_index < len(self.batch_prompts):
            self.batch_progress.emit(self.batch_index + 1, len(self.batch_prompts))
            item = self.batch_prompts[self.batch_index]
            prompt = item['prompt'] if isinstance(item, dict) else item
            duration = item.get('duration', 10.0) if isinstance(item, dict) else 10.0
            self.next_prompt_ready.emit(prompt, duration)
        else:
            self.batch_state = BatchState.COMPLETED
            if self._master_wav_writer:
                if self._overlap_buffer is not None:
                    self._master_wav_writer.writeframes(self._overlap_buffer.tobytes())
                self._master_wav_writer.close()
                self._master_wav_writer = None
                self._overlap_buffer = None
                QMessageBox.information(self.main_window, "Timeline Complete", f"Master timeline saved to:\n{self.master_timeline_path}")
            self.batch_finished.emit()

    def on_generation_finished(self, metadata_list):
        if self.batch_state == BatchState.RUNNING:
            if self._master_wav_writer:
                for meta in metadata_list:
                    path = meta['path']
                    try:
                        with wave.open(path, 'rb') as wf_in:
                            if self._master_fr == 0:
                                self._master_fr = wf_in.getframerate()
                                self._master_wav_writer.setframerate(self._master_fr)
                            
                            data = np.frombuffer(wf_in.readframes(wf_in.getnframes()), dtype=np.int16)
                            
                            if self._overlap_buffer is not None and len(self._overlap_buffer) > 0:
                                overlap_len = min(len(self._overlap_buffer), len(data) // 2)
                                if overlap_len > 0:
                                    fade_out = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
                                    fade_in = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)
                                    crossfaded = (self._overlap_buffer[:overlap_len].astype(np.float32) * fade_out) + (data[:overlap_len].astype(np.float32) * fade_in)
                                    data[:overlap_len] = np.clip(crossfaded, -32768, 32767).astype(np.int16)
                            
                            overlap_size = int(self._master_fr * OVERLAP_S)
                            if len(data) > overlap_size:
                                self._master_wav_writer.writeframes(data[:-overlap_size].tobytes())
                                self._overlap_buffer = data[-overlap_size:]
                            else:
                                self._master_wav_writer.writeframes(data.tobytes())
                                self._overlap_buffer = data
                                
                    except Exception as e:
                        logger.error(f"Failed to append to master timeline: {e}", exc_info=True)
            
            self.batch_index += 1
            QTimer.singleShot(int(self.main_window.batch_delay_spin.value() * 1000), self._process_next)

    def cancel(self):
        self.batch_state = BatchState.CANCELLED
        if self._master_wav_writer:
            self._master_wav_writer.close()
            self._master_wav_writer = None
            self._overlap_buffer = None


class AudioGenerationWorker(QObject):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    structured_status_updated = Signal(object)
    eta_updated = Signal(str)
    generation_completed = Signal(str, dict)
    error_occurred = Signal(str)
    cancelled = Signal()

    _shared_pipe = None
    _shared_pipe_model = None
    _shared_pipe_device = None
    _shared_pipe_offload = None
    _pipe_lock = threading.Lock()

    def __init__(self, params: GenerationParams) -> None:
        super().__init__()
        self.params = params
        self._cancel_event = threading.Event()
        self.pipe = None
        self.ema_step_time = None
        self.last_step_time = None

    @property
    def is_cancelled(self) -> bool: return self._cancel_event.is_set()

    @is_cancelled.setter
    def is_cancelled(self, val: bool):
        if val: self._cancel_event.set()
        else: self._cancel_event.clear()

    def _load_pipeline(self, actual_device: str):
        with self._pipe_lock:
            if (AudioGenerationWorker._shared_pipe is not None
                and AudioGenerationWorker._shared_pipe_model == self.params.model_name
                and AudioGenerationWorker._shared_pipe_device == actual_device
                and AudioGenerationWorker._shared_pipe_offload == self.params.use_cpu_offload
                and not self.is_cancelled):
                self.pipe = AudioGenerationWorker._shared_pipe
                logger.info("Reusing cached pipeline in memory.")
                return

            self._unload_pipeline_unsafe()
            self.status_updated.emit("Loading model weights...")
            self.progress_updated.emit(5)

            dtype = torch.float16 if actual_device != "cpu" else torch.float32
            load_kwargs = {"torch_dtype": dtype, "cache_dir": self.params.cache_dir}
            if dtype == torch.float16: load_kwargs["variant"] = "fp16"

            load_attempts = [{"use_safetensors": True}, {"use_safetensors": False}]
            last_err, pipe = None, None

            for attempt in load_attempts:
                if self.is_cancelled: raise InterruptedError("User cancelled generation")
                try:
                    pipe = AudioLDM2Pipeline.from_pretrained(self.params.model_name, **load_kwargs, **attempt)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "variant" in msg or "fp16" in msg:
                        logger.warning("fp16 variant not found, falling back to default weights.")
                        load_kwargs.pop("variant", None)
                        try:
                            pipe = AudioLDM2Pipeline.from_pretrained(self.params.model_name, **load_kwargs, **attempt)
                            break
                        except Exception as e2:
                            last_err = e2
                            logger.warning(f"Load attempt failed ({attempt} no variant): {e2}")
                    else:
                        last_err = e
                        logger.warning(f"Load attempt failed ({attempt}): {e}")
                    pipe = None
                    gc.collect()
                    if ML_AVAILABLE and torch.cuda.is_available(): torch.cuda.empty_cache()

            if pipe is None:
                raise RuntimeError(f"Failed to load model after multiple attempts. Last error: {last_err}")

            if self.is_cancelled:
                del pipe
                gc.collect()
                if ML_AVAILABLE and torch.cuda.is_available(): torch.cuda.empty_cache()
                raise InterruptedError("User cancelled generation")

            self.status_updated.emit("Configuring model for inference...")
            self.progress_updated.emit(10)

            try:
                if self.params.use_cpu_offload and actual_device != "cpu":
                    if not hasattr(pipe, 'enable_model_cpu_offload'):
                        raise RuntimeError("CPU Offload is requested but the 'accelerate' library is not installed.")
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to(actual_device)

                if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_tiling"): pipe.enable_vae_tiling()
            except RuntimeError as e:
                del pipe
                gc.collect()
                if ML_AVAILABLE and torch.cuda.is_available(): torch.cuda.empty_cache()
                if "out of memory" in str(e).lower(): raise RuntimeError("GPU Out of Memory during model configuration.")
                raise

            AudioGenerationWorker._shared_pipe = pipe
            AudioGenerationWorker._shared_pipe_model = self.params.model_name
            AudioGenerationWorker._shared_pipe_device = actual_device
            AudioGenerationWorker._shared_pipe_offload = self.params.use_cpu_offload
            self.pipe = pipe

    def _on_step(self, step, timestep, latents, chunk_idx=0, **kwargs):
        if self.is_cancelled: raise InterruptedError("User cancelled generation")
        p = self.params
        
        step, chunk_idx, num_chunks_int = int(step), int(chunk_idx), int(self._num_chunks)
        p_steps_int, variation_int, p_num_variations_int = int(p.steps), int(self._variation), int(self._total_variations)

        steps_done = self._steps_done_before_variation + (chunk_idx * p_steps_int) + (step + 1)
        total_progress = 15 + int(80 * steps_done / max(self._total_steps_all, 1))
        self.progress_updated.emit(min(total_progress, 95))

        current_time = time.time()
        if self.last_step_time is not None:
            dt = current_time - self.last_step_time
            if dt > 0:
                if self.ema_step_time is None: self.ema_step_time = dt
                else: self.ema_step_time = ETA_SMOOTHING_ALPHA * dt + (1 - ETA_SMOOTHING_ALPHA) * self.ema_step_time
                remaining_steps = max(0, p_steps_int - step - 1)
                remaining_chunks = max(0, num_chunks_int - chunk_idx - 1)
                remaining_variations = max(0, p_num_variations_int - variation_int - 1)
                eta_seconds = (remaining_steps * self.ema_step_time + remaining_chunks * p_steps_int * self.ema_step_time + remaining_variations * num_chunks_int * p_steps_int * self.ema_step_time)
                self.eta_updated.emit(format_time(eta_seconds))

        self.last_step_time = current_time
        
        status = GenerationStatus(variation_int + 1, p_num_variations_int, chunk_idx + 1, num_chunks_int, step + 1, p_steps_int)
        self.structured_status_updated.emit(status)
        self.status_updated.emit(status.format())

    def run(self):
        gen_start_time = time.time()
        try:
            p = self.params
            logger.info(f"Starting generation. Prompt: '{p.prompt}' | Model: {p.model_name} | Duration: {p.duration}s")

            cache_dir = os.path.normpath(p.cache_dir or DEFAULT_CACHE_DIR)
            try: os.makedirs(cache_dir, exist_ok=True)
            except OSError as e: raise RuntimeError(f"Cannot create or access cache directory '{cache_dir}':\n{e}")

            final_out_dir = p.output_dir
            if p.section:
                safe_section = sanitize_filename(p.section, max_length=50)
                final_out_dir = os.path.join(p.output_dir, safe_section)
            os.makedirs(final_out_dir, exist_ok=True)

            os.environ["HF_HOME"] = cache_dir
            os.environ["HF_HUB_CACHE"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["DIFFUSERS_CACHE"] = cache_dir

            actual_device = p.device
            if actual_device == "auto":
                if torch.cuda.is_available(): actual_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): actual_device = "mps"
                else: actual_device = "cpu"

            self._load_pipeline(actual_device)
            self.status_updated.emit("Generating audio...")
            self.progress_updated.emit(15)

            saved_paths, metadata_list = [], []
            MAX_CHUNK_S = max(MIN_CHUNK_S, p.chunk_size)
            use_chunking = p.duration > MAX_CHUNK_S and not p.inpaint_source

            total_variations = max(2, p.travel_steps) if p.use_seed_travel else p.num_variations

            for variation in range(total_variations):
                if self.is_cancelled: raise InterruptedError("User cancelled generation")

                if p.use_seed_travel:
                    base_seed = p.travel_start_seed if p.travel_start_seed >= 0 else int(time.time() * 1000) % (2**31)
                    end_seed = p.travel_end_seed if p.travel_end_seed >= 0 else base_seed + 100
                elif p.seed >= 0:
                    base_seed = p.seed + variation
                else:
                    base_seed = int(time.time() * 1000) % (2**31) + variation

                variation_start = time.time()

                if use_chunking:
                    num_chunks = math.ceil((p.duration - OVERLAP_S) / (MAX_CHUNK_S - OVERLAP_S))
                    last_chunk_len = p.duration - (num_chunks - 1) * (MAX_CHUNK_S - OVERLAP_S)
                    if last_chunk_len < MIN_CHUNK_S and num_chunks > 1:
                        num_chunks -= 1
                    self.status_updated.emit(f"Variation {variation + 1}/{total_variations} - Streaming {num_chunks} chunks to disk...")
                else:
                    num_chunks = 1

                sample_rate = getattr(getattr(getattr(self.pipe, "vae", None), "config", None), "sample_rate", 16000)
                self.ema_step_time, self.last_step_time = None, None

                self._steps_done_before_variation = variation * num_chunks * p.steps
                self._total_steps_all = total_variations * num_chunks * p.steps
                self._num_chunks = num_chunks
                self._variation = variation
                self._total_variations = total_variations

                travel_latents = None
                if p.use_seed_travel and variation > 0:
                    t = variation / (total_variations - 1)
                    logger.info(f"Seed Travel: Interpolating latents at t={t:.2f}")
                
                safe_prompt = sanitize_filename(p.prompt)
                suffix = f"_v{variation + 1}" if total_variations > 1 else ""
                if p.use_seed_travel: suffix = f"_travel{variation + 1}"
                prefix = "inpaint_" if p.inpaint_source else ""
                filename = f"{prefix}{int(time.time())}_{safe_prompt}{suffix}.wav"
                out_path = os.path.join(final_out_dir, filename)
                
                wav_writer = wave.open(out_path, 'wb')
                wav_writer.setnchannels(1)
                wav_writer.setsampwidth(2)
                wav_writer.setframerate(sample_rate)
                
                overlap_buffer = None
                last_audio_chunk = None

                for chunk_idx in range(num_chunks):
                    if self.is_cancelled:
                        wav_writer.close()
                        raise InterruptedError("User cancelled generation")

                    if use_chunking:
                        if chunk_idx < num_chunks - 1:
                            chunk_len = MAX_CHUNK_S
                        else:
                            chunk_len = max(MIN_CHUNK_S, p.duration - (num_chunks - 1) * (MAX_CHUNK_S - OVERLAP_S))
                    else:
                        chunk_len = p.duration

                    chunk_seed = (base_seed + chunk_idx * 7919) % (2**31)
                    gen_device = "cpu" if p.use_cpu_offload else actual_device
                    
                    if p.use_seed_travel:
                        if not hasattr(self, '_latent_shape'):
                            try:
                                vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
                                num_waveform_samples = int(chunk_len * sample_rate)
                                num_frames = num_waveform_samples // vae_scale_factor
                                self._latent_shape = (1, self.pipe.unet.config.in_channels, num_frames, self.pipe.vae.config.sample_size)
                            except Exception as e:
                                logger.error(f"Failed to calculate latent shape: {e}. Disabling seed travel.", exc_info=True)
                                p.use_seed_travel = False

                        if p.use_seed_travel:
                            gen_start = torch.Generator(device=gen_device).manual_seed(p.travel_start_seed if p.travel_start_seed >=0 else chunk_seed)
                            gen_end = torch.Generator(device=gen_device).manual_seed(p.travel_end_seed if p.travel_end_seed >=0 else chunk_seed + 100)
                            noise_start = torch.randn(self._latent_shape, generator=gen_start, device=gen_device, dtype=torch.float32)
                            noise_end = torch.randn(self._latent_shape, generator=gen_end, device=gen_device, dtype=torch.float32)
                            target_dtype = torch.float16 if actual_device != "cpu" else torch.float32
                            travel_latents = slerp(t, noise_start, noise_end).to(target_dtype)

                    generator = torch.Generator(device=gen_device).manual_seed(chunk_seed)

                    cb_new = partial(self._on_step, chunk_idx=chunk_idx)
                    cb_old = partial(self._on_step, chunk_idx=chunk_idx)

                    kwargs = dict(
                        prompt=p.prompt,
                        negative_prompt=p.negative_prompt if p.negative_prompt else None,
                        num_inference_steps=p.steps,
                        audio_length_in_s=chunk_len,
                        guidance_scale=p.guidance,
                        generator=generator,
                    )

                    if p.use_seed_travel and travel_latents is not None:
                        kwargs["latents"] = travel_latents

                    sig = inspect.signature(self.pipe.__call__)
                    if "callback_on_step_end" in sig.parameters:
                        kwargs["callback_on_step_end"] = cb_new
                        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                    else:
                        kwargs["callback"] = cb_old
                        kwargs["callback_steps"] = 1

                    with torch.inference_mode():
                        audio_chunk = self.pipe(**kwargs).audios[0]
                        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                    if num_chunks > 1:
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

                    if p.high_pass_freq > 0 or p.low_pass_freq > 0:
                        from scipy.signal import butter, sosfilt
                        nyq = 0.5 * sample_rate
                        if p.high_pass_freq > 0:
                            sos = butter(4, min(p.high_pass_freq / nyq, 0.99), 'hp', output='sos')
                            audio_chunk = sosfilt(sos, audio_chunk).astype(np.float32)
                        if p.low_pass_freq > 0:
                            sos = butter(4, max(p.low_pass_freq / nyq, 0.01), 'lp', output='sos')
                            audio_chunk = sosfilt(sos, audio_chunk).astype(np.float32)

                    if p.trim and not p.inpaint_source: audio_chunk = trim_silence(audio_chunk)
                    if p.normalize and not p.inpaint_source: audio_chunk = normalize_audio(audio_chunk)
                    if p.fade_in or p.fade_out:
                        if not p.inpaint_source:
                            audio_chunk = apply_fade(audio_chunk, sample_rate, fade_in_s=FADE_DURATION_S if p.fade_in else 0, fade_out_s=FADE_DURATION_S if p.fade_out else 0)

                    if p.conditioning_audio and os.path.exists(p.conditioning_audio):
                        try:
                            c_sr, c_data = wavfile.read(p.conditioning_audio)
                            c_data = normalize_audio_data(c_data)
                            if len(c_data) < len(audio_chunk): c_data = np.pad(c_data, (0, len(audio_chunk) - len(c_data)))
                            else: c_data = c_data[:len(audio_chunk)]
                            window_size = max(1, int(0.05 * c_sr))
                            c_env = np.convolve(np.abs(c_data), np.ones(window_size)/window_size, mode='same')
                            max_env = np.max(c_env)
                            if max_env > 0:
                                audio_chunk = audio_chunk * (c_env / max_env)
                        except Exception as e:
                            logger.error(f"Conditioning failed: {e}", exc_info=True)

                    chunk_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    
                    if overlap_buffer is not None and use_chunking:
                        overlap_samples = min(len(overlap_buffer), len(chunk_int16) // 2)
                        if overlap_samples > 0:
                            fade_out_arr = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
                            fade_in_arr  = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
                            crossfaded = (overlap_buffer[-overlap_samples:] * fade_out_arr + chunk_int16[:overlap_samples] * fade_in_arr).astype(np.int16)
                            chunk_int16[:overlap_samples] = crossfaded
                    
                    if use_chunking and len(chunk_int16) > OVERLAP_S * sample_rate:
                        overlap_buffer = chunk_int16[-int(OVERLAP_S * sample_rate):]
                        wav_writer.writeframes(chunk_int16[:-int(OVERLAP_S * sample_rate)].tobytes())
                    else:
                        wav_writer.writeframes(chunk_int16.tobytes())
                        
                    last_audio_chunk = audio_chunk

                wav_writer.close()

                if p.inpaint_source and os.path.exists(p.inpaint_source):
                    try:
                        orig_sr, orig_data = wavfile.read(p.inpaint_source)
                        orig_data = normalize_audio_data(orig_data)
                        if orig_sr != sample_rate:
                            logger.warning("Inpaint source SR mismatch. Skipping splice.")
                        else:
                            gen_data = last_audio_chunk
                            start_samp = int(p.inpaint_start_s * sample_rate)
                            end_samp = start_samp + len(gen_data)
                            fade_len = min(int(0.1 * sample_rate), len(gen_data)//2, len(orig_data)//2)
                            if fade_len > 0:
                                gen_data[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                                gen_data[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                                orig_data[start_samp : start_samp + fade_len] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                                orig_data[end_samp - fade_len : end_samp] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                            orig_data[start_samp:end_samp] = gen_data[:len(orig_data[start_samp:end_samp])]
                            final_int16 = (np.clip(orig_data, -1.0, 1.0) * 32767).astype(np.int16)
                            wavfile.write(out_path, orig_sr, final_int16)
                            logger.info("Inpainting splice successful.")
                    except Exception as e:
                        logger.error(f"Inpainting splice failed: {e}", exc_info=True)

                metadata = {
                    'prompt': p.prompt, 'negative_prompt': p.negative_prompt, 'model': p.model_name,
                    'duration': p.duration, 'steps': p.steps, 'guidance': p.guidance, 'seed': base_seed,
                    'sample_rate': sample_rate, 'device': actual_device, 'variation': variation + 1,
                    'total_variations': total_variations, 'generation_time': time.time() - variation_start,
                    'timestamp': datetime.now().isoformat(), 'chunked_generation': num_chunks > 1,
                    'num_chunks': num_chunks, 'path': out_path, 'filename': filename,
                    'use_seed_travel': p.use_seed_travel, 'tags': [], 'favorite': False
                }

                saved_paths.append(out_path)
                metadata_list.append(metadata)
                logger.info(f"Variation {variation + 1} generated: {out_path}")

            manifest_path = os.path.join(final_out_dir, "manifest.json")
            manifest_data = []
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f: manifest_data = json.load(f)
                except Exception: manifest_data = []

            manifest_data.extend(metadata_list)
            try:
                with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(manifest_data, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save manifest.json: {e}", exc_info=True)

            self.progress_updated.emit(100)
            self.status_updated.emit("Generation complete")
            self.eta_updated.emit("")

            self.generation_completed.emit(saved_paths[0], {'paths': saved_paths, 'metadata': metadata_list, 'gen_time': time.time() - gen_start_time})

        except InterruptedError:
            logger.warning("Generation interrupted by user.")
            self.status_updated.emit("Cancelled")
            self.progress_updated.emit(0)
            self.eta_updated.emit("")
            self.cancelled.emit()

        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"Generation failed: {err_trace}")
            if "out of memory" in str(e).lower():
                friendly_msg = ("GPU Out of Memory!\n\nYour graphics card ran out of VRAM. Try:\n"
                                "- Lowering the Chunk Size (e.g., 5s or 10s)\n- Lowering the number of Steps\n"
                                "- Enabling CPU Offload (Low VRAM Mode)\n- Closing other GPU-intensive applications")
                self.error_occurred.emit(friendly_msg)
            else:
                self.error_occurred.emit(f"Generation Failed:\n{err_trace}")

        finally:
            self.clear_cache()

    def clear_cache(self):
        self.pipe = None
        self.ema_step_time = None
        self.last_step_time = None
        if hasattr(self, '_latent_shape'): del self._latent_shape
        gc.collect()
        if ML_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}", exc_info=True)

    @classmethod
    def unload_pipeline(cls):
        with cls._pipe_lock: cls._unload_pipeline_unsafe()

    @classmethod
    def _unload_pipeline_unsafe(cls):
        if cls._shared_pipe is not None:
            try:
                logger.info(f"Unloading model '{cls._shared_pipe_model}' from memory...")
                pipe = cls._shared_pipe
                if hasattr(pipe, "remove_hooks"):
                    try: pipe.remove_hooks()
                    except Exception: pass
                del pipe
                cls._shared_pipe = None
                cls._shared_pipe_model = None
                cls._shared_pipe_device = None
                cls._shared_pipe_offload = None
                gc.collect()
                if ML_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                logger.info("Model unloaded successfully.")
            except Exception as e:
                logger.error(f"Error unloading pipeline: {e}", exc_info=True)


class FFmpegWorker(QThread):
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def run(self):
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(self.cmd, capture_output=True, text=True, creationflags=creationflags)
            if result.returncode == 0:
                self.finished.emit(True, "")
            else:
                logger.error(f"FFmpeg failed to encode: {result.stderr}")
                self.finished.emit(False, result.stderr[-1000:])
        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"FFmpegWorker crashed: {err_trace}")
            self.error.emit(f"FFmpeg Error:\n{err_trace}")


class DemucsWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, audio_path, device="cpu"):
        super().__init__()
        self.audio_path = audio_path
        self.device = device

    def run(self):
        try:
            model = get_demucs_model('htdemucs')
            model.to(self.device)
            wav, sr = torchaudio.load(self.audio_path)
            if sr != model.samplerate: wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            if wav.shape[0] == 1: wav = wav.repeat(2, 1)
            wav = wav.unsqueeze(0)
            with torch.inference_mode():
                stems = apply_demucs_model(model, wav, split=True, overlap=0.25)
            stem_names = ['drums', 'bass', 'other', 'vocals']
            base, _ = os.path.splitext(self.audio_path)
            out_dir = base + "_stems"
            os.makedirs(out_dir, exist_ok=True)
            saved = []
            for i, name in enumerate(stem_names):
                path = os.path.join(out_dir, f"{name}.wav")
                torchaudio.save(path, stems[0, i].cpu(), model.samplerate)
                saved.append(path)
            self.finished.emit(saved)
        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"DemucsWorker crashed: {err_trace}")
            self.error.emit(f"Demucs Error:\n{err_trace}")

class WhisperWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            model = whisper.load_model("base")
            result = model.transcribe(self.audio_path)
            self.finished.emit(result["text"])
        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"WhisperWorker crashed: {err_trace}")
            self.error.emit(f"Whisper Error:\n{err_trace}")

class CLAPWorker(QThread):
    finished = Signal(float)
    error = Signal(str)

    def __init__(self, audio_path, prompt):
        super().__init__()
        self.audio_path = audio_path
        self.prompt = prompt

    def run(self):
        try:
            processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            sr, data = wavfile.read(self.audio_path)
            data = normalize_audio_data(data)
            inputs = processor(text=[self.prompt], audios=data, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.inference_mode():
                audio_embed = model.get_audio_features(**inputs)
                text_embed = model.get_text_features(**inputs)
                audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                sim = (audio_embed @ text_embed.T).item()
                score = (sim + 1) / 2 * 100
            self.finished.emit(score)
        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"CLAPWorker crashed: {err_trace}")
            self.error.emit(f"CLAP Error:\n{err_trace}")


class AudioStatsWorker(QThread):
    statsReady = Signal(float, float)
    error = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            with wave.open(self.path, 'rb') as wf:
                sr = wf.getframerate()
                frames_to_read = min(wf.getnframes(), sr * 60)
                raw = wf.readframes(frames_to_read)
            
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if data.size == 0:
                self.statsReady.emit(-120.0, -120.0)
                return
            peak = float(np.max(np.abs(data)))
            rms = float(np.sqrt(np.mean(np.square(data))))
            self.statsReady.emit(peak, rms)
        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"AudioStatsWorker crashed: {err_trace}")
            self.error.emit(f"Stats Error:\n{err_trace}")


class SpectrogramWorker(QThread):
    arrayReady = Signal(object)
    
    def __init__(self, audio_data, sample_rate):
        super().__init__()
        self.audio = audio_data
        self.sr = sample_rate

    def run(self):
        try:
            if self.audio is None or self.audio.size == 0:
                self.arrayReady.emit(None)
                return

            nperseg = 2048 if self.sr >= 44100 else 1024
            nperseg = min(nperseg, len(self.audio))

            freqs, times, Sxx = scipy.signal.spectrogram(
                self.audio, fs=self.sr, nperseg=nperseg, noverlap=nperseg // 2, window='hann'
            )

            Sxx_log = 10 * np.log10(Sxx + 1e-10)
            vmin = float(np.percentile(Sxx_log, 10))
            vmax = float(np.percentile(Sxx_log, 99))
            if vmax <= vmin: vmax = vmin + 1.0
            Sxx_norm = np.clip((Sxx_log - vmin) / (vmax - vmin), 0.0, 1.0)

            del Sxx_log, Sxx

            xp = [0.0, 0.25, 0.5, 0.75, 1.0]
            fp_r = [68, 59, 33, 53, 253]
            fp_g = [1, 82, 145, 148, 231]
            fp_b = [84, 139, 140, 27, 36]
            r = np.interp(Sxx_norm, xp, fp_r)
            g = np.interp(Sxx_norm, xp, fp_g)
            b = np.interp(Sxx_norm, xp, fp_b)
            del Sxx_norm

            rgb_array = np.stack((r, g, b), axis=-1).astype(np.uint8)
            rgb_array = rgb_array[::-1]
            rgb_array = np.ascontiguousarray(rgb_array)
            
            self.arrayReady.emit(rgb_array)
        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"SpectrogramWorker crashed: {err_trace}")
            self.arrayReady.emit(None)


class SpectrogramWidget(QWidget):
    seekRequested = Signal(float)
    selectionChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_audio = None
        self._sample_rate = 16000
        self._qimg = None
        self._qpix = None
        self.playback_position = 0.0
        self.duration_s = 0.0
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._pending_path = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(60)
        self._render_timer.timeout.connect(self._flush_pending_audio)

        self.inpaint_mode = False
        self._sel_start = -1.0
        self._sel_end = -1.0
        self._is_dragging = False
        
        self._spec_worker = None

    def set_inpaint_mode(self, enabled):
        self.inpaint_mode = enabled
        if not enabled:
            self._sel_start = -1.0
            self._sel_end = -1.0
            self.selectionChanged.emit(-1.0, -1.0)
        self.update()

    def get_selection(self):
        if self._sel_start < 0 or self._sel_end < 0: return -1.0, -1.0
        return min(self._sel_start, self._sel_end), max(self._sel_start, self._sel_end)

    def set_audio(self, path: str):
        if path == self._pending_path and self._render_timer.isActive(): return
        self._pending_path = path
        self._render_timer.start()

    def _flush_pending_audio(self):
        path = self._pending_path
        if not path: return
        try:
            with wave.open(path, 'rb') as wf:
                self._sample_rate = wf.getframerate()
                frames_to_read = min(wf.getnframes(), self._sample_rate * 60)
                raw = wf.readframes(frames_to_read)
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                self.duration_s = wf.getnframes() / self._sample_rate
            
            self._raw_audio = data
            
            if self._spec_worker and self._spec_worker.isRunning():
                self._spec_worker.quit()
                self._spec_worker.wait(500)
            
            self._qimg = None
            self._qpix = None
            self.update()
            
            self._spec_worker = SpectrogramWorker(self._raw_audio, self._sample_rate)
            self._spec_worker.arrayReady.connect(self._on_spec_ready)
            self._spec_worker.start()
            
        except Exception as e:
            logger.error(f"Failed to load spectrogram: {e}", exc_info=True)
            self._raw_audio = None
            self._qimg = None
            self._qpix = None
            self.update()

    def _on_spec_ready(self, rgb_array):
        if rgb_array is None:
            self._qimg = None
            self._qpix = None
        else:
            h, w, _ = rgb_array.shape
            self._qimg = QImage(rgb_array.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self._qpix = QPixmap.fromImage(self._qimg)
            self._raw_audio = None
        self.update()

    def set_playback_position(self, position: float):
        self.playback_position = max(0.0, min(1.0, position))
        self.update()

    def clear(self):
        self._raw_audio = None
        self._qimg = None
        self._qpix = None
        self.duration_s = 0.0
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.inpaint_mode and event.button() == Qt.LeftButton:
            pos = event.position().x() / max(self.width(), 1)
            self._sel_start = pos * self.duration_s
            self._sel_end = self._sel_start
            self._is_dragging = True
            self.update()
            return

        if self._qpix is not None and event.button() == Qt.LeftButton:
            pos = event.position().x() / max(self.width(), 1)
            self.seekRequested.emit(pos)
            self.set_playback_position(pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.inpaint_mode and self._is_dragging:
            pos = event.position().x() / max(self.width(), 1)
            self._sel_end = pos * self.duration_s
            self.update()
            return

        if self._qpix is not None and self.duration_s > 0 and self._sample_rate > 0:
            x, y = event.position().x(), event.position().y()
            t_s = (x / max(self.width(), 1)) * self.duration_s
            nyquist = self._sample_rate / 2
            freq = nyquist * (1 - (y / max(self.height(), 1)))

            if freq >= 1000: text = f"Time: {format_time(t_s)} | Freq: {freq/1000:.2f} kHz"
            else: text = f"Time: {format_time(t_s)} | Freq: {freq:.0f} Hz"
            QToolTip.showText(event.globalPosition().toPoint(), text, self)

            if event.buttons() & Qt.LeftButton:
                pos = x / max(self.width(), 1)
                self.seekRequested.emit(pos)
                self.set_playback_position(pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.inpaint_mode and self._is_dragging:
            self._is_dragging = False
            start, end = self.get_selection()
            if end - start > 0.1:
                self.selectionChanged.emit(start, end)
            else:
                self._sel_start = -1.0
                self._sel_end = -1.0
            self.selectionChanged.emit(-1.0, -1.0)
            self.update()

    def leaveEvent(self, event):
        QToolTip.hideText()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 12))

        if self._qpix is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "Computing spectrogram..." if self._raw_audio is not None else "No spectrogram loaded")
            return

        painter.drawPixmap(self.rect(), self._qpix)

        painter.setPen(QPen(QColor(255, 255, 255, 100), 1, Qt.DashLine))
        painter.setFont(QFont("Segoe UI", 8))

        nyquist = self._sample_rate / 2
        num_y_steps = 4
        for i in range(1, num_y_steps):
            y = int((i / num_y_steps) * self.height())
            freq = int(nyquist * (1 - (i / num_y_steps)))
            painter.drawLine(0, y, self.width(), y)
            label = f"{freq/1000:.1f}kHz" if freq >= 1000 else f"{freq}Hz"
            painter.drawText(5, y - 2, label)

        if self.duration_s > 0:
            step_s = 1.0
            if self.duration_s > 3600: step_s = 600.0
            elif self.duration_s > 600: step_s = 60.0
            elif self.duration_s > 60: step_s = 15.0
            elif self.duration_s > 30: step_s = 5.0
            elif self.duration_s > 10: step_s = 2.0

            t = step_s
            while t < self.duration_s:
                x = int((t / self.duration_s) * self.width())
                painter.drawLine(x, 0, x, self.height())
                painter.drawText(x + 3, self.height() - 5, format_time(t))
                t += step_s

        if self.inpaint_mode and self._sel_start >= 0 and self._sel_end >= 0:
            start, end = self.get_selection()
            x1 = int((start / max(self.duration_s, 1e-6)) * self.width())
            x2 = int((end / max(self.duration_s, 1e-6)) * self.width())
            painter.fillRect(QRect(x1, 0, x2 - x1, self.height()), QColor(42, 130, 218, 100))
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.drawLine(x1, 0, x1, self.height())
            painter.drawLine(x2, 0, x2, self.height())

        if self.playback_position >= 0 and self._qpix is not None:
            play_x = int(self.playback_position * (self.width() - 1))
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.drawLine(play_x, 0, play_x, self.height())


class WaveformWidget(QWidget):
    seekRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_audio = None
        self._peaks = None
        self._rms = None
        self._peaks_width = 0
        self._poly_peak = None
        self._poly_rms = None
        self.sample_rate = 16000
        self.duration_s = 0.0
        self.setMinimumHeight(120)
        self.setMaximumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.playback_position = 0.0
        self.hover_position = -1.0
        self.setMouseTracking(True)

    def set_audio(self, path: str):
        if not path or not os.path.exists(path):
            self._reset_state()
            return
        try:
            with wave.open(path, 'rb') as wf:
                self.sample_rate = wf.getframerate()
                frames_to_read = min(wf.getnframes(), self.sample_rate * 60)
                raw = wf.readframes(frames_to_read)
                self._raw_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                self.duration_s = wf.getnframes() / self.sample_rate
                
            self._peaks = None
            self._poly_peak = None
            self._poly_rms = None
            self._ensure_peaks(self.width())
            self.update()
        except Exception as e:
            logger.error(f"Failed to load waveform: {e}", exc_info=True)
            self._reset_state()

    def _reset_state(self):
        self._raw_audio = None
        self._peaks = None
        self._poly_peak = None
        self._poly_rms = None
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._peaks = None
        self._poly_peak = None
        self._poly_rms = None
        self.update()

    def _ensure_peaks(self, width: int):
        if self._peaks is not None and self._peaks_width == width: return
        if self._raw_audio is None or width <= 0:
            self._peaks, self._poly_peak, self._poly_rms = None, None, None
            return

        n = len(self._raw_audio)
        if n == 0:
            self._peaks = np.zeros(width, dtype=np.float32)
            self._rms = np.zeros(width, dtype=np.float32)
        else:
            abs_audio = np.abs(self._raw_audio)
            if n <= width:
                self._peaks = np.pad(abs_audio, (0, width - n), 'constant').astype(np.float32)
                self._rms = np.pad(np.sqrt(np.mean(abs_audio**2)), (0, width - 1), 'constant').astype(np.float32) if n > 0 else np.zeros(width, dtype=np.float32)
            else:
                chunk_size = max(1, n // width)
                n_full = (n // chunk_size) * chunk_size
                trimmed = abs_audio[:n_full].reshape(-1, chunk_size)
                self._peaks = trimmed.max(axis=1).astype(np.float32)
                self._rms = np.sqrt(np.mean(trimmed**2, axis=1)).astype(np.float32)
                if len(self._peaks) < width:
                    pad = width - len(self._peaks)
                    self._peaks = np.pad(self._peaks, (0, pad), 'constant')
                    self._rms = np.pad(self._rms, (0, pad), 'constant')
            del abs_audio
            self._peaks_width = width

        self._poly_peak = self._build_poly(self._peaks, width)
        self._poly_rms = self._build_poly(self._rms, width)
        self.update()

    def _build_poly(self, heights, width):
        mid = self.height() / 2.0
        max_h = self.height() * 0.42
        poly = QPolygonF()
        if width == 0: return poly

        poly << QPointF(0, mid)
        for x, h in enumerate(heights):
            poly << QPointF(x, mid - (h * max_h))

        poly << QPointF(width, mid)
        for x in range(width - 1, -1, -1):
            h = heights[x]
            poly << QPointF(x, mid + (h * max_h))
        return poly

    def set_playback_position(self, position: float):
        self.playback_position = max(0.0, min(1.0, position))
        self.update()

    def clear(self):
        self._raw_audio = None
        self._peaks = None
        self._poly_peak = None
        self._poly_rms = None
        self.duration_s = 0.0
        self.update()

    def rms_at(self, norm_pos: float) -> float:
        if self._rms is None or len(self._rms) == 0: return 0.0
        idx = int(max(0.0, min(1.0, norm_pos)) * (len(self._rms) - 1))
        return float(self._rms[idx])

    def mousePressEvent(self, event: QMouseEvent):
        if self._raw_audio is not None and event.button() == Qt.LeftButton:
            pos = event.position().x() / max(self.width(), 1)
            self.seekRequested.emit(pos)
            self.set_playback_position(pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._raw_audio is not None:
            self.hover_position = event.position().x() / max(self.width(), 1)
            if event.buttons() & Qt.LeftButton:
                pos = self.hover_position
                self.seekRequested.emit(pos)
                self.set_playback_position(pos)
            self.update()

    def leaveEvent(self, event):
        self.hover_position = -1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()

        painter.fillRect(self.rect(), QColor(22, 22, 26))

        if self._poly_peak is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "No audio loaded")
            return

        mid = height / 2.0
        play_x = int(self.playback_position * (width - 1))

        if self.duration_s > 0:
            step_s = 1.0
            if self.duration_s > 3600: step_s = 600.0
            elif self.duration_s > 600: step_s = 60.0
            elif self.duration_s > 60: step_s = 15.0
            elif self.duration_s > 30: step_s = 5.0
            elif self.duration_s > 10: step_s = 2.0

            painter.setPen(QPen(QColor(50, 50, 55), 1, Qt.DashLine))
            painter.setFont(QFont("Segoe UI", 8))

            t = 0.0
            while t < self.duration_s:
                gx = int((t / self.duration_s) * width)
                painter.drawLine(gx, 10, gx, height - 10)
                painter.drawText(gx + 3, 12, 40, 15, Qt.AlignLeft, format_time(t))
                t += step_s

        painter.setClipRect(play_x, 0, width - play_x, height)
        painter.setBrush(QColor(60, 65, 75))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(self._poly_peak)

        painter.setBrush(QColor(80, 90, 100))
        painter.drawPolygon(self._poly_rms)

        painter.setClipRect(0, 0, play_x + 1, height)
        grad = QLinearGradient(0, 0, 0, height)
        grad.setColorAt(0.0, QColor(40, 160, 240))
        grad.setColorAt(0.5, QColor(80, 220, 255))
        grad.setColorAt(1.0, QColor(40, 160, 240))
        painter.setBrush(grad)
        painter.drawPolygon(self._poly_peak)

        painter.setBrush(QColor(150, 240, 255, 220))
        painter.drawPolygon(self._poly_rms)

        painter.setClipping(False)

        painter.setPen(QPen(QColor(35, 35, 40), 1))
        painter.drawLine(0, mid, width, mid)

        if 0 <= self.hover_position <= 1.0:
            hx = int(self.hover_position * (width - 1))
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.DashLine))
            painter.drawLine(hx, 0, hx, height)

        if self.playback_position >= 0 and self._raw_audio is not None:
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.drawLine(play_x, 0, play_x, height)

            painter.setBrush(QColor(255, 255, 255, 220))
            triangle = QPolygon([QPoint(play_x - 5, 0), QPoint(play_x + 5, 0), QPoint(play_x, 6)])
            painter.drawPolygon(triangle)


class MiniWaveformItem(QTableWidgetItem):
    @staticmethod
    @lru_cache(maxsize=500)
    def _cached_pixmap(path: str, mtime: float) -> QPixmap:
        p = None
        try:
            with wave.open(path, 'rb') as wf:
                sr = wf.getframerate()
                frames_to_read = min(wf.getnframes(), sr * 30)
                raw = wf.readframes(frames_to_read)
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                
            if data.size == 0: return None

            chunk = max(1, len(data) // THUMB_BARS)
            n_full = (len(data) // chunk) * chunk
            peaks = np.abs(data[:n_full]).reshape(-1, chunk).max(axis=1)
            if len(peaks) < THUMB_BARS:
                peaks = np.pad(peaks, (0, THUMB_BARS - len(peaks)), 'constant')
            peaks = peaks[:THUMB_BARS]

            peaks_list = [float(v) for v in peaks]

            pix = QPixmap(THUMB_WIDTH, THUMB_HEIGHT)
            pix.fill(Qt.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(42, 130, 218), 2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)

            mid = THUMB_HEIGHT // 2
            max_h = (THUMB_HEIGHT // 2) - 1
            for i, pk in enumerate(peaks_list):
                h = min(pk * 12.0, float(max_h))
                h_int = int(round(h))
                x = i * 2
                p.drawLine(int(x), mid - h_int, int(x), mid + h_int)
            return pix
        except Exception as e:
            logger.error(f"MiniWaveformItem thumbnail failed for {path}: {e}", exc_info=True)
            return None
        finally:
            if p is not None: p.end()

    @staticmethod
    def generate_pixmap(path: str) -> QPixmap:
        if not path or not os.path.exists(path): return None
        mtime = os.path.getmtime(path)
        return MiniWaveformItem._cached_pixmap(path, mtime)

    def __init__(self, path: str, duration: float):
        super().__init__()
        self.setText(format_time(duration))
        self.setTextAlignment(Qt.AlignCenter)
        self.setToolTip(path)
        self._path = path
        self._pixmap = None

    def set_pixmap(self, pix: QPixmap):
        self._pixmap = pix
        self.setIcon(QIcon(pix))
        self.setSizeHint(QSize(THUMB_WIDTH + 10, THUMB_HEIGHT + 4))


class ThumbnailWorker(QThread):
    thumbReady = Signal(int, QPixmap)

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self._table = table
        self._queue = queue.Queue()
        self._running = True

    def enqueue(self, row: int, path: str):
        self._queue.put((row, path))

    def stop(self):
        self._running = False
        self._queue.put(None)

    def run(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None: break
            row, path = item
            try:
                pix = MiniWaveformItem.generate_pixmap(path)
                if pix is not None and self._running:
                    self.thumbReady.emit(row, pix)
            except Exception as e:
                logger.error(f"ThumbnailWorker crashed on item {path}: {e}", exc_info=True)


class ClickableLabel(QLabel):
    clicked = Signal()
    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.clicked.emit()


class VariationCard(QFrame):
    playRequested = Signal(str)
    selected = Signal(str)

    def __init__(self, path: str, index: int, duration: float, seed: int, parent=None):
        super().__init__(parent)
        self.path = path
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(110, 130)
        self.setStyleSheet("""
            QFrame { background-color: #333333; border: 1px solid #3c3c3c; border-radius: 4px; }
            QFrame:hover { border-color: #2A82DA; background-color: #3c3c3c; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header_row = QHBoxLayout()
        title = QLabel(f"Var {index + 1}")
        title.setStyleSheet("font-weight: bold; color: #2A82DA;")
        header_row.addWidget(title)
        header_row.addStretch()
        dur_lbl = QLabel(format_time(duration))
        dur_lbl.setStyleSheet("font-size: 10px; color: #888;")
        header_row.addWidget(dur_lbl)
        layout.addLayout(header_row)

        self.thumb_label = ClickableLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setFixedSize(THUMB_WIDTH + 10, THUMB_HEIGHT + 10)
        self.thumb_label.setStyleSheet("background-color: #222; border-radius: 2px;")
        self.thumb_label.setCursor(Qt.PointingHandCursor)
        self.thumb_label.clicked.connect(lambda: self.selected.emit(self.path))
        layout.addWidget(self.thumb_label)

        seed_lbl = QLabel(f"Seed: {seed}")
        seed_lbl.setStyleSheet("font-size: 10px; color: #aaa;")
        layout.addWidget(seed_lbl)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(26, 26)
        self.play_btn.setStyleSheet("QPushButton { background-color: #2A82DA; color: white; border-radius: 3px; font-weight: bold; } QPushButton:hover { background-color: #3A92EA; }")
        self.play_btn.clicked.connect(lambda: self.playRequested.emit(self.path))
        btn_row.addWidget(self.play_btn)

        load_btn = QPushButton("Load")
        load_btn.setFixedHeight(26)
        load_btn.setStyleSheet("QPushButton { background-color: #3c3c3c; color: white; border-radius: 3px; font-size: 11px; } QPushButton:hover { background-color: #4a4a4a; }")
        load_btn.clicked.connect(lambda: self.selected.emit(self.path))  # FIX: Syntax error corrected
        btn_row.addWidget(load_btn)

        layout.addLayout(btn_row)

    def set_thumbnail(self, pixmap: QPixmap):
        self.thumb_label.setPixmap(pixmap)


class PromptHistoryLineEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.history_index = -1
        self._saved_text = ""
        self.setPlaceholderText("Describe the audio... (Ctrl+Enter to generate, Ctrl+Up/Down for history)")
        self.setMaximumHeight(70)
        self.setTabChangesFocus(True)

        self.c = QCompleter(AUDIO_TAGS, self)
        self.c.setCaseSensitivity(Qt.CaseInsensitive)
        self.c.setWidget(self)
        self.c.setCompletionMode(QCompleter.PopupCompletion)
        self.c.activated.connect(self.insert_completion)

    def insert_completion(self, completion):
        if self.c.widget() != self: return
        cursor = self.textCursor()
        extra = len(completion) - len(self.c.completionPrefix())
        cursor.movePosition(QTextCursor.Left)
        cursor.movePosition(QTextCursor.EndOfWord)
        cursor.insertText(completion[-extra:])
        self.setTextCursor(cursor)

    def text_under_cursor(self):
        tc = self.textCursor()
        tc.select(QTextCursor.WordUnderCursor)
        word = tc.selectedText()
        start = tc.selectionStart()
        end = tc.selectionEnd()
        text = self.toPlainText()
        while start > 0 and text[start-1] not in (' ', '\n', ','):
            start -= 1
        return text[start:end]

    def keyPressEvent(self, event):
        if self.c.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape, Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return

        if event.key() == Qt.Key_Up and (event.modifiers() & Qt.ControlModifier):
            self.navigate_history(-1)
            return
        elif event.key() == Qt.Key_Down and (event.modifiers() & Qt.ControlModifier):
            self.navigate_history(1)
            return
        elif event.key() == Qt.Key_Return and (event.modifiers() & Qt.ControlModifier):
            self.window().start_generation()
            return

        super().keyPressEvent(event)

        ctrl_or_shift = event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)
        if ctrl_or_shift and len(event.text()) == 0: return

        prefix = self.text_under_cursor()
        if prefix != self.c.completionPrefix():
            self.c.setCompletionPrefix(prefix)
            self.c.popup().setCurrentIndex(self.c.completionModel().index(0, 0))

        cr = self.cursorRect()
        if len(prefix) >= 2 and self.c.completionCount() > 0:
            cr.setWidth(self.c.popup().sizeHintForColumn(0) + self.c.popup().verticalScrollBar().sizeHint().width())
            self.c.complete(cr)
        else:
            self.c.popup().hide()

    def navigate_history(self, direction):
        if not self.history: return
        if self.history_index == -1: self._saved_text = self.toPlainText()
        new_index = max(-1, min(len(self.history) - 1, self.history_index + direction))
        if new_index == self.history_index: return
        self.history_index = new_index
        self.setPlainText(self._saved_text if self.history_index == -1 else self.history[self.history_index])

    def add_to_history(self, text):
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            if len(self.history) > MAX_PROMPT_HISTORY: self.history.pop(0)
            self.history_index = -1


class FxChainDialog(QDialog):
    PLUGIN_PARAMS = {
        "Reverb": ["room_size", "damping", "wet_level", "dry_level", "width", "freeze_mode"],
        "Delay": ["delay_seconds", "feedback", "mix"],
        "Chorus": ["rate_hz", "depth", "centre_delay_ms", "feedback", "mix"],
        "Distortion": ["drive_db"],
        "Compressor": ["threshold_db", "ratio", "attack_ms", "release_ms"],
        "LowpassFilter": ["cutoff_frequency_hz"],
        "HighpassFilter": ["cutoff_frequency_hz"],
        "Gain": ["gain_db"]
    }

    def __init__(self, parent, current_chain=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time FX Chain")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.chain = current_chain if current_chain else []
        
        layout = QVBoxLayout(self)
        
        avail_group = QGroupBox("Available Plugins")
        avail_layout = QHBoxLayout(avail_group)
        self.avail_list = QListWidget()
        self.plugin_classes = {
            "Reverb": Reverb, "Delay": Delay, "Chorus": Chorus, 
            "Distortion": Distortion, "Compressor": Compressor,
            "LowpassFilter": LowpassFilter, "HighpassFilter": HighpassFilter, "Gain": Gain
        }
        for name in self.plugin_classes.keys():
            self.avail_list.addItem(name)
        self.avail_list.setFixedWidth(150)
        avail_layout.addWidget(self.avail_list)
        
        add_btn = QPushButton("→ Add →")
        add_btn.clicked.connect(self.add_plugin)
        avail_layout.addWidget(add_btn, alignment=Qt.AlignCenter)
        layout.addWidget(avail_group)
        
        active_group = QGroupBox("Active Chain (Top to Bottom)")
        active_layout = QHBoxLayout(active_group)
        
        self.active_list = QListWidget()
        self.active_list.setFixedWidth(150)
        self.update_active_list()
        active_layout.addWidget(self.active_list)
        
        controls_layout = QVBoxLayout()
        self.params_layout = QFormLayout()
        controls_layout.addLayout(self.params_layout)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_plugin)
        controls_layout.addWidget(remove_btn)
        active_layout.addLayout(controls_layout, 1)
        
        layout.addWidget(active_group)
        
        btn_box = QHBoxLayout()
        apply_btn = QPushButton("Apply & Close")
        apply_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addStretch()
        btn_box.addWidget(apply_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)
        
        self.active_list.currentRowChanged.connect(self.load_params_ui)

    def update_active_list(self):
        self.active_list.clear()
        for plugin in self.chain:
            self.active_list.addItem(plugin.__class__.__name__)

    def add_plugin(self):
        item = self.avail_list.currentItem()
        if not item: return
        plugin_name = item.text()
        plugin_class = self.plugin_classes[plugin_name]
        new_plugin = plugin_class()
        self.chain.append(new_plugin)
        self.update_active_list()
        self.active_list.setCurrentRow(len(self.chain) - 1)

    def remove_plugin(self):
        row = self.active_list.currentRow()
        if row >= 0 and row < len(self.chain):
            del self.chain[row]
            self.update_active_list()
            self.clear_params_ui()

    def clear_params_ui(self):
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)

    def load_params_ui(self, row):
        self.clear_params_ui()
        if row < 0 or row >= len(self.chain): return
            
        plugin = self.chain[row]
        plugin_name = plugin.__class__.__name__
        attrs = self.PLUGIN_PARAMS.get(plugin_name, [])
        
        for attr_name in attrs:
            val = getattr(plugin, attr_name, 0.0)
            if isinstance(val, (float, int)):
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 1.0 if val <= 1.0 else 20000.0)
                spin.setSingleStep(0.01 if val <= 1.0 else 1.0)
                spin.setValue(float(val))
                spin.valueChanged.connect(lambda v, p=plugin, a=attr_name: setattr(p, a, v))
                self.params_layout.addRow(attr_name + ":", spin)

    def get_chain(self):
        return self.chain


class AudioLDM2Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1320, 880)
        self.setMinimumSize(1000, 650)
        log_system_info()

        try:
            if not ML_AVAILABLE:
                logger.error("ML libraries (torch, diffusers, transformers) are not installed.")
                sys.exit(1)

            self.settings = JsonSettings(os.path.join(CWD, "config.json"))

            self.worker_thread = None
            self.worker = None
            
            self.base_audio_path = None
            self.current_audio_path = None

            self.batch_orchestrator = BatchOrchestrator(self)
            self.batch_orchestrator.next_prompt_ready.connect(self._on_batch_prompt_ready)
            self.batch_orchestrator.batch_finished.connect(self._on_batch_finished)

            self.prompt_history = []
            self.setAcceptDrops(True)

            self.demucs_worker = None
            self.whisper_worker = None
            self.clap_worker = None
            self.stats_worker = None
            self.ffmpeg_worker = None

            self._inpaint_active = False
            self._inpaint_source = ""
            self._inpaint_start = 0.0
            self._inpaint_end = 0.0
            
            self.fx_chain = []
            if PEDALBOARD_AVAILABLE:
                saved_chain_data = self.settings.value("fx_chain", [])
                self.fx_chain = self.deserialize_fx_chain(saved_chain_data)

            self.check_library_versions()
            self.setup_ui()
            self.setup_audio()
            self.setup_statusbar()
            self.setup_tray_icon()
            self.load_settings()
            self.setup_shortcuts()
            
            QTimer.singleShot(1000, self.check_interrupted_batch)

            self._thumb_worker = ThumbnailWorker(self.playlist_table, self)
            self._thumb_worker.thumbReady.connect(self._on_thumb_ready)
            self._thumb_worker.start()

            self.autosave_timer = QTimer(self)
            self.autosave_timer.timeout.connect(self.auto_save_state)
            self.autosave_timer.start(AUTO_SAVE_INTERVAL_MS)
            
        except Exception:
            logger.critical("Failed to initialize AudioLDM2Studio UI", exc_info=True)
            sys.exit(1)

    def deserialize_fx_chain(self, data):
        chain = []
        if not PEDALBOARD_AVAILABLE: return chain
        plugin_map = {
            "Reverb": Reverb, "Delay": Delay, "Chorus": Chorus, 
            "Distortion": Distortion, "Compressor": Compressor,
            "LowpassFilter": LowpassFilter, "HighpassFilter": HighpassFilter, "Gain": Gain
        }
        for item in data:
            p_class = plugin_map.get(item.get("class"))
            if p_class:
                params = item.get("params", {})
                try:
                    chain.append(p_class(**params))
                except TypeError:
                    chain.append(p_class())
        return chain

    def serialize_fx_chain(self):
        data = []
        for plugin in self.fx_chain:
            attrs = {a: getattr(plugin, a) for a in dir(plugin) if not a.startswith('_') and not callable(getattr(plugin, a))}
            data.append({"class": plugin.__class__.__name__, "params": attrs})
        return data

    def _on_thumb_ready(self, row: int, pix: QPixmap):
        item = self.playlist_table.item(row, 3)
        if isinstance(item, MiniWaveformItem):
            item.set_pixmap(pix)

    def check_library_versions(self):
        try:
            t_ver = transformers.__version__.split('.')
            d_ver = diffusers.__version__.split('.')
            if int(t_ver[0]) >= 4 and int(t_ver[1]) >= 40:
                if int(d_ver[0]) == 0 and int(d_ver[1]) < 29:
                    logger.warning("Potential library version conflict detected.")
        except Exception:
            pass

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        self.create_left_panel(splitter)
        self.create_right_panel(splitter)
        splitter.setSizes([450, 870])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.create_menus()

        try:
            geo = self.settings.value("geometry")
            if geo: self.restoreGeometry(geo)
            state = self.settings.value("windowState")
            if state: self.restoreState(state)
        except Exception as e:
            logger.warning(f"Failed to restore window geometry: {e}", exc_info=True)

    def create_left_panel(self, parent):
        left_panel = QWidget()
        left_panel.setMinimumWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.mode_tabs = QTabWidget()
        left_layout.addWidget(self.mode_tabs)

        generate_widget = QWidget()
        gen_outer_layout = QVBoxLayout(generate_widget)
        gen_outer_layout.setContentsMargins(0, 0, 0, 0)

        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_scroll.setFrameShape(QFrame.NoFrame)
        gen_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        gen_content = QWidget()
        gen_layout = QVBoxLayout(gen_content)
        gen_layout.setContentsMargins(8, 8, 8, 8)
        gen_layout.setSpacing(8)

        prompt_group = QGroupBox("Prompt")
        p_layout = QVBoxLayout(prompt_group)
        p_layout.setSpacing(6)

        chips_layout = QHBoxLayout()
        chips_layout.setSpacing(2)
        chips_lbl = QLabel("Quick:")
        chips_lbl.setFixedWidth(40)
        chips_layout.addWidget(chips_lbl)

        quick_tags = ["💥 Impact", "🌬️ Whoosh", "🌧️ Ambient", "🎛️ UI Beep", "🚗 Engine", "🔥 Fire"]
        for tag in quick_tags:
            btn = QPushButton(tag)
            btn.setStyleSheet("padding: 2px 4px; font-size: 10px; min-height: 18px;")
            btn.clicked.connect(lambda checked, t=tag: self.prompt_input.setPlainText(
                self.prompt_input.toPlainText() + " " + t.split(" ", 1)[1]
            ))
            chips_layout.addWidget(btn)
        chips_layout.addStretch()
        p_layout.addLayout(chips_layout)

        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Presets:")
        preset_lbl.setFixedWidth(50)
        preset_row.addWidget(preset_lbl)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([""] + DEFAULT_PROMPT_PRESETS)
        self.preset_combo.currentTextChanged.connect(self.on_preset_selected)
        preset_row.addWidget(self.preset_combo, 1)
        
        enhance_btn = QPushButton("✨ Enhance")
        enhance_btn.setFixedWidth(80)
        enhance_btn.setToolTip("Expand simple prompts into rich, detailed descriptions using a local NLP dictionary.")
        enhance_btn.clicked.connect(self.enhance_prompt)
        preset_row.addWidget(enhance_btn)
        
        random_btn = QToolButton()
        random_btn.setText("🎲")
        random_btn.setToolTip("Generate random prompt from tags")
        random_btn.clicked.connect(self.generate_random_prompt)
        preset_row.addWidget(random_btn)
        
        p_layout.addLayout(preset_row)

        self.prompt_input = PromptHistoryLineEdit()
        p_layout.addWidget(self.prompt_input)

        neg_row = QHBoxLayout()
        neg_lbl = QLabel("Negative:")
        neg_lbl.setFixedWidth(50)
        neg_row.addWidget(neg_lbl)
        self.neg_preset_combo = QComboBox()
        self.neg_preset_combo.addItems([""] + NEGATIVE_PROMPT_PRESETS)
        self.neg_preset_combo.currentTextChanged.connect(self.on_neg_preset_selected)
        neg_row.addWidget(self.neg_preset_combo, 1)
        p_layout.addLayout(neg_row)

        self.neg_prompt_input = QLineEdit()
        self.neg_prompt_input.setPlaceholderText("What to avoid...")
        p_layout.addWidget(self.neg_prompt_input)
        gen_layout.addWidget(prompt_group)

        param_group = QGroupBox("Parameters")
        f_layout = QFormLayout(param_group)
        f_layout.setSpacing(6)
        f_layout.setLabelAlignment(Qt.AlignRight)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(DEFAULT_MODELS)
        model_row.addWidget(self.model_combo, 1)

        save_model_btn = QToolButton()
        save_model_btn.setText("+")
        save_model_btn.setToolTip("Save current model to list")
        save_model_btn.clicked.connect(self.save_current_model)
        model_row.addWidget(save_model_btn)
        f_layout.addRow("Model:", model_row)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 14400.0)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSuffix(" s")
        f_layout.addRow("Duration:", self.duration_spin)

        self.chunk_size_spin = QDoubleSpinBox()
        self.chunk_size_spin.setRange(MIN_CHUNK_S, 60.0)
        self.chunk_size_spin.setValue(10.0)
        self.chunk_size_spin.setSingleStep(0.5)
        self.chunk_size_spin.setSuffix(" s")
        f_layout.addRow("Chunk Size:", self.chunk_size_spin)

        self.chunk_preview_label = QLabel("")
        self.chunk_preview_label.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        f_layout.addRow("", self.chunk_preview_label)

        self.duration_spin.valueChanged.connect(self.update_chunk_preview)
        self.chunk_size_spin.valueChanged.connect(self.update_chunk_preview)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 500)
        self.steps_spin.setValue(200)
        f_layout.addRow("Steps:", self.steps_spin)

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(3.5)
        self.guidance_spin.setSingleStep(0.5)
        f_layout.addRow("Guidance:", self.guidance_spin)

        seed_row = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 999999999)
        self.seed_spin.setValue(-1)
        seed_row.addWidget(self.seed_spin, 1)

        random_seed_btn = QToolButton()
        random_seed_btn.setText("🎲")
        random_seed_btn.setToolTip("Random seed")
        random_seed_btn.clicked.connect(lambda: self.seed_spin.setValue(-1))
        seed_row.addWidget(random_seed_btn)
        f_layout.addRow("Seed:", seed_row)

        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(1, 10)
        self.variations_spin.setValue(1)
        f_layout.addRow("Variations:", self.variations_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        if torch.cuda.is_available():
            self.device_combo.setCurrentText("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device_combo.setCurrentText("mps")
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        f_layout.addRow("Device:", self.device_combo)

        cache_row = QHBoxLayout()
        self.cache_dir_edit = QLineEdit()
        self.cache_dir_edit.setPlaceholderText(DEFAULT_CACHE_DIR)
        cache_browse_btn = QPushButton("…")
        cache_browse_btn.setFixedWidth(30)
        cache_browse_btn.clicked.connect(self.browse_cache_dir)
        cache_row.addWidget(self.cache_dir_edit)
        cache_row.addWidget(cache_browse_btn)
        f_layout.addRow("Cache:", cache_row)

        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("./generated_audio")
        out_browse_btn = QPushButton("…")
        out_browse_btn.setFixedWidth(30)
        out_browse_btn.clicked.connect(self.browse_output_dir)
        out_row.addWidget(self.output_dir_edit)
        out_row.addWidget(out_browse_btn)

        open_out_btn = QPushButton("📂")
        open_out_btn.setFixedWidth(30)
        open_out_btn.clicked.connect(self.open_output_folder)
        out_row.addWidget(open_out_btn)
        f_layout.addRow("Output:", out_row)

        self.section_edit = QLineEdit()
        self.section_edit.setPlaceholderText("e.g., SFX, Music (leave empty for root)")
        f_layout.addRow("Section:", self.section_edit)

        self.cpu_offload_check = QCheckBox("Enable CPU Offload (Low VRAM)")
        f_layout.addRow("", self.cpu_offload_check)
        gen_layout.addWidget(param_group)
        
        adv_group = QGroupBox("Advanced Workflows")
        adv_layout = QVBoxLayout(adv_group)
        
        cond_row = QHBoxLayout()
        cond_lbl = QLabel("Conditioning:")
        cond_lbl.setFixedWidth(80)
        cond_row.addWidget(cond_lbl)
        self.cond_audio_edit = QLineEdit()
        self.cond_audio_edit.setPlaceholderText("Hum/beat audio path (rhythm match)")
        cond_row.addWidget(self.cond_audio_edit)
        cond_browse_btn = QPushButton("…")
        cond_browse_btn.setFixedWidth(30)
        cond_browse_btn.clicked.connect(self.browse_cond_audio)
        cond_row.addWidget(cond_browse_btn)
        cond_clear_btn = QPushButton("✖")
        cond_clear_btn.setFixedWidth(30)
        cond_clear_btn.clicked.connect(self.cond_audio_edit.clear)
        cond_row.addWidget(cond_clear_btn)
        adv_layout.addLayout(cond_row)
        
        travel_row1 = QHBoxLayout()
        self.travel_check = QCheckBox("🧬 Seed Travel (Morph Latents)")
        self.travel_check.toggled.connect(self.toggle_seed_travel)
        travel_row1.addWidget(self.travel_check)
        adv_layout.addLayout(travel_row1)
        
        self.travel_widget = QWidget()
        travel_layout = QFormLayout(self.travel_widget)
        travel_layout.setContentsMargins(20, 0, 0, 0)
        self.travel_start_seed = QSpinBox()
        self.travel_start_seed.setRange(-1, 999999999)
        self.travel_start_seed.setValue(100)
        travel_layout.addRow("Start Seed:", self.travel_start_seed)
        
        self.travel_end_seed = QSpinBox()
        self.travel_end_seed.setRange(-1, 999999999)
        self.travel_end_seed.setValue(200)
        travel_layout.addRow("End Seed:", self.travel_end_seed)
        
        self.travel_steps_spin = QSpinBox()
        self.travel_steps_spin.setRange(2, 20)
        self.travel_steps_spin.setValue(5)
        travel_layout.addRow("Steps:", self.travel_steps_spin)
        adv_layout.addWidget(self.travel_widget)
        self.travel_widget.setEnabled(False)

        gen_layout.addWidget(adv_group)

        post_group = QGroupBox("Post-Processing")
        post_layout = QVBoxLayout(post_group)
        post_layout.setSpacing(4)
        self.normalize_check = QCheckBox("Normalize audio volume")
        self.normalize_check.setChecked(True)
        post_layout.addWidget(self.normalize_check)
        self.trim_check = QCheckBox("Trim silence")
        self.trim_check.setChecked(True)
        post_layout.addWidget(self.trim_check)

        fade_row = QHBoxLayout()
        self.fade_in_check = QCheckBox("Fade in")
        self.fade_in_check.setChecked(True)
        fade_row.addWidget(self.fade_in_check)
        self.fade_out_check = QCheckBox("Fade out")
        self.fade_out_check.setChecked(True)
        fade_row.addWidget(self.fade_out_check)
        post_layout.addLayout(fade_row)

        eq_row = QHBoxLayout()
        eq_row.addWidget(QLabel("HP:"))
        self.hp_spin = QDoubleSpinBox()
        self.hp_spin.setRange(0, 20000)
        self.hp_spin.setSuffix(" Hz")
        self.hp_spin.setSpecialValueText("Off")
        eq_row.addWidget(self.hp_spin)

        eq_row.addWidget(QLabel("LP:"))
        self.lp_spin = QDoubleSpinBox()
        self.lp_spin.setRange(0, 20000)
        self.lp_spin.setSuffix(" Hz")
        self.lp_spin.setSpecialValueText("Off")
        eq_row.addWidget(self.lp_spin)
        post_layout.addLayout(eq_row)

        gen_layout.addWidget(post_group)

        gen_btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Audio")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2A82DA; color: white; font-size: 13px; font-weight: bold; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3A92EA; }"
            "QPushButton:pressed { background-color: #1A72CA; }"
            "QPushButton:disabled { background-color: #444444; color: #888888; }"
        )
        self.generate_btn.clicked.connect(self.start_generation)
        gen_btn_row.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-size: 13px; font-weight: bold; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d0493b; }"
            "QPushButton:pressed { background-color: #b0291b; }"
            "QPushButton:disabled { background-color: #444444; color: #888888; }"
        )
        self.cancel_btn.clicked.connect(self.cancel_generation)
        gen_btn_row.addWidget(self.cancel_btn)
        gen_layout.addLayout(gen_btn_row)
        gen_layout.addStretch()

        gen_scroll.setWidget(gen_content)
        gen_outer_layout.addWidget(gen_scroll)
        self.mode_tabs.addTab(generate_widget, "Generate")

        batch_widget = QWidget()
        batch_outer_layout = QVBoxLayout(batch_widget)
        batch_outer_layout.setContentsMargins(0, 0, 0, 0)

        batch_scroll = QScrollArea()
        batch_scroll.setWidgetResizable(True)
        batch_scroll.setFrameShape(QFrame.NoFrame)
        batch_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        batch_content = QWidget()
        batch_layout = QVBoxLayout(batch_content)
        batch_layout.setContentsMargins(8, 8, 8, 8)
        batch_layout.setSpacing(8)

        batch_layout.addWidget(QLabel("Load a .txt file with one prompt per line:"))
        self.batch_file_btn = QPushButton("Select Text File")
        self.batch_file_btn.clicked.connect(self.select_batch_file)
        batch_layout.addWidget(self.batch_file_btn)

        self.batch_file_label = QLabel("No file selected")
        self.batch_file_label.setWordWrap(True)
        self.batch_file_label.setStyleSheet("color: #888; font-style: italic;")
        batch_layout.addWidget(self.batch_file_label)

        batch_settings_group = QGroupBox("Batch Settings")
        bsl = QFormLayout(batch_settings_group)
        bsl.setSpacing(6)
        bsl.setLabelAlignment(Qt.AlignRight)
        self.batch_variations_spin = QSpinBox()
        self.batch_variations_spin.setRange(1, 10)
        self.batch_variations_spin.setValue(1)
        bsl.addRow("Variations:", self.batch_variations_spin)
        self.batch_delay_spin = QDoubleSpinBox()
        self.batch_delay_spin.setRange(0, 30)
        self.batch_delay_spin.setValue(1.0)
        self.batch_delay_spin.setSuffix(" s")
        bsl.addRow("Delay:", self.batch_delay_spin)
        self.batch_continue_on_error_check = QCheckBox("Continue on Error")
        self.batch_continue_on_error_check.setChecked(True)  # FIX: Syntax error corrected
        bsl.addRow("", self.batch_continue_on_error_check)
        batch_layout.addWidget(batch_settings_group)

        self.batch_start_btn = QPushButton("Start Batch Generation")
        self.batch_start_btn.setMinimumHeight(40)
        self.batch_start_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-size: 13px; font-weight: bold; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #37be70; }"
            "QPushButton:pressed { background-color: #179e50; }"
            "QPushButton:disabled { background-color: #444444; color: #888888; }"
        )
        self.batch_start_btn.clicked.connect(self.start_batch)
        batch_layout.addWidget(self.batch_start_btn)

        self.batch_progress_label = QLabel("")
        self.batch_progress_label.setWordWrap(True)
        batch_layout.addWidget(self.batch_progress_label)

        batch_layout.addWidget(QLabel("Batch Log:"))
        self.batch_log = QPlainTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_log.setMaximumBlockCount(1000)
        batch_layout.addWidget(self.batch_log)
        batch_layout.addStretch()

        batch_scroll.setWidget(batch_content)
        batch_outer_layout.addWidget(batch_scroll)
        self.mode_tabs.addTab(batch_widget, "Batch")
        
        timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(timeline_widget)
        timeline_layout.addWidget(QLabel("Timeline Scripting (Long-Form Soundscapes):"))
        timeline_layout.addWidget(QLabel("Format: [Start-End] Prompt\nExample: [0:00-5:00] Ambient rain"))
        
        self.timeline_edit = QPlainTextEdit()
        self.timeline_edit.setPlaceholderText("[0:00-5:00] Ambient drone, low rumble\n[5:00-5:30] Thunder strike\n[5:30-10:00] Rain fading out")
        timeline_layout.addWidget(self.timeline_edit)
        
        tl_btn_row = QHBoxLayout()
        self.tl_load_btn = QPushButton("Load Script")
        self.tl_load_btn.clicked.connect(self.load_timeline_script)
        tl_btn_row.addWidget(self.tl_load_btn)
        
        self.tl_start_btn = QPushButton("Generate Long-Form Audio")
        self.tl_start_btn.setMinimumHeight(40)
        self.tl_start_btn.setStyleSheet(
            "QPushButton { background-color: #8e44ad; color: white; font-size: 13px; font-weight: bold; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #9e54bd; }"
            "QPushButton:pressed { background-color: #7e349d; }"
        )
        self.tl_start_btn.clicked.connect(self.start_timeline_generation)
        tl_btn_row.addWidget(self.tl_start_btn)
        timeline_layout.addLayout(tl_btn_row)
        
        self.mode_tabs.addTab(timeline_widget, "Timeline")

        prog_group = QGroupBox("Status")
        prog_layout = QVBoxLayout(prog_group)
        prog_layout.setSpacing(4)
        prog_layout.setContentsMargins(8, 16, 8, 8)
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prog_row.addWidget(self.progress_bar, 1)
        self.eta_label = QLabel("")
        self.eta_label.setMinimumWidth(70)
        self.eta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prog_row.addWidget(self.eta_label)
        prog_layout.addLayout(prog_row)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        prog_layout.addWidget(self.status_label)
        left_layout.addWidget(prog_group)
        parent.addWidget(left_panel)

    def toggle_seed_travel(self, checked):
        self.travel_widget.setEnabled(checked)
        if checked:
            self.variations_spin.setEnabled(False)
        else:
            self.variations_spin.setEnabled(True)

    def create_right_panel(self, parent):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.right_tabs = QTabWidget()
        right_layout.addWidget(self.right_tabs)

        player_widget = QWidget()
        player_layout = QVBoxLayout(player_widget)
        player_layout.setContentsMargins(8, 8, 8, 8)
        player_layout.setSpacing(8)

        visualizer_group = QGroupBox("Audio Visualizer")
        visualizer_layout = QVBoxLayout(visualizer_group)

        self.visualizer_tabs = QTabWidget()

        self.waveform_widget = WaveformWidget()
        self.waveform_widget.seekRequested.connect(self._seek_to)
        self.visualizer_tabs.addTab(self.waveform_widget, "Oscillogram (Time)")

        self.spectrogram_widget = SpectrogramWidget()
        self.spectrogram_widget.seekRequested.connect(self._seek_to)
        self.spectrogram_widget.selectionChanged.connect(self.on_spectrogram_selection_changed)
        self.visualizer_tabs.addTab(self.spectrogram_widget, "Spectrogram (Freq)")

        vis_ctrl_layout = QHBoxLayout()
        self.inpaint_check = QCheckBox("Inpaint Mode (Drag on Spectrogram)")
        self.inpaint_check.toggled.connect(self.spectrogram_widget.set_inpaint_mode)
        self.inpaint_check.toggled.connect(self.toggle_inpaint_button)
        vis_ctrl_layout.addWidget(self.inpaint_check)
        vis_ctrl_layout.addStretch()
        
        self.inpaint_btn = QPushButton("Inpaint Selection")
        self.inpaint_btn.setEnabled(False)
        self.inpaint_btn.clicked.connect(self.start_inpainting)
        vis_ctrl_layout.addWidget(self.inpaint_btn)
        visualizer_layout.addLayout(vis_ctrl_layout)

        visualizer_layout.addWidget(self.visualizer_tabs)

        self.audio_stats_label = QLabel("Peak: 0.0 dB | RMS: 0.0 dB | Headroom: 0.0 dB")
        self.audio_stats_label.setAlignment(Qt.AlignCenter)
        self.audio_stats_label.setStyleSheet("color: #888; font-size: 11px; font-family: monospace; padding: 2px;")
        visualizer_layout.addWidget(self.audio_stats_label)

        player_layout.addWidget(visualizer_group)

        play_group = QGroupBox("Playback & FX")
        play_layout = QVBoxLayout(play_group)
        play_layout.setSpacing(6)

        playback_main_row = QHBoxLayout()
        playback_main_row.setSpacing(8)

        play_btn_col = QVBoxLayout()
        play_btn_col.setSpacing(6)
        play_btn_row = QHBoxLayout()
        play_btn_row.setSpacing(6)

        self.load_btn = QPushButton("Load")
        self.load_btn.setFixedHeight(36)
        self.load_btn.clicked.connect(self.open_audio)
        play_btn_row.addWidget(self.load_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedHeight(36)
        self.play_btn.setMinimumWidth(80)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        play_btn_row.addWidget(self.play_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_audio)
        play_btn_row.addWidget(self.save_btn)
        
        self.fx_btn = QPushButton("FX Chain")
        self.fx_btn.setFixedHeight(36)
        self.fx_btn.setEnabled(PEDALBOARD_AVAILABLE)
        self.fx_btn.setToolTip("Real-time effects chain" if PEDALBOARD_AVAILABLE else "Install 'pedalboard' to enable FX")
        self.fx_btn.clicked.connect(self.open_fx_chain_dialog)
        play_btn_row.addWidget(self.fx_btn)
        
        self.clear_fx_btn = QPushButton("Clear FX")
        self.clear_fx_btn.setFixedHeight(36)
        self.clear_fx_btn.setEnabled(False)
        self.clear_fx_btn.setToolTip("Bypass FX and restore original audio")
        self.clear_fx_btn.clicked.connect(self.clear_fx)
        play_btn_row.addWidget(self.clear_fx_btn)

        play_btn_col.addLayout(play_btn_row)
        playback_main_row.addLayout(play_btn_col, 1)

        self.vu_meter = QProgressBar()
        self.vu_meter.setOrientation(Qt.Vertical)
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setFixedWidth(20)
        self.vu_meter.setFixedHeight(70)
        self.vu_meter.setTextVisible(False)
        self.vu_meter.setStyleSheet("""
            QProgressBar { background-color: #1a1a1a; border: 1px solid #3c3c3c; border-radius: 2px; }
            QProgressBar::chunk { background-color: #2ecc71; border-radius: 1px; }
        """)
        playback_main_row.addWidget(self.vu_meter)

        play_layout.addLayout(playback_main_row)

        vol_layout = QHBoxLayout()
        vol_layout.setSpacing(6)
        vol_lbl = QLabel("Volume:")
        vol_lbl.setFixedWidth(50)
        vol_layout.addWidget(vol_lbl)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(50)
        self.vol_slider.valueChanged.connect(self.set_volume)
        vol_layout.addWidget(self.vol_slider)
        self.vol_label = QLabel("50%")
        self.vol_label.setFixedWidth(35)
        vol_layout.addWidget(self.vol_label)
        
        self.autoplay_check = QCheckBox("Auto-play on finish")
        self.autoplay_check.setChecked(True)
        vol_layout.addWidget(self.autoplay_check)
        
        play_layout.addLayout(vol_layout)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2A82DA;")
        play_layout.addWidget(self.time_label)
        player_layout.addWidget(play_group)

        ai_group = QGroupBox("AI Analysis & Tools")
        ai_layout = QHBoxLayout(ai_group)
        ai_layout.setSpacing(6)

        self.demucs_btn = QPushButton("Split Stems\n(Demucs)")
        self.demucs_btn.setToolTip("Split audio into Drums, Bass, Vocals, and Other using AI.\nRequires: pip install demucs")
        self.demucs_btn.setEnabled(DEMUCS_AVAILABLE)
        self.demucs_btn.clicked.connect(self.run_demucs)
        ai_layout.addWidget(self.demucs_btn)

        self.whisper_btn = QPushButton("Transcribe\n(Whisper)")
        self.whisper_btn.setToolTip("Transcribe speech in the audio using AI.\nRequires: pip install openai-whisper")
        self.whisper_btn.setEnabled(WHISPER_AVAILABLE)
        self.whisper_btn.clicked.connect(self.run_whisper)
        ai_layout.addWidget(self.whisper_btn)

        self.clap_btn = QPushButton("Score Match\n(Clap)")
        self.clap_btn.setToolTip("Calculate how well the audio matches your prompt using AI.\nRequires: pip install transformers")
        self.clap_btn.setEnabled(CLAP_AVAILABLE)
        self.clap_btn.clicked.connect(self.run_clap)
        ai_layout.addWidget(self.clap_btn)

        player_layout.addWidget(ai_group)

        var_group = QGroupBox("Generated Variations")
        var_layout = QVBoxLayout(var_group)
        var_scroll = QScrollArea()
        var_scroll.setWidgetResizable(True)
        var_scroll.setFixedHeight(140)
        var_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        var_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        var_scroll.setStyleSheet("QScrollArea { border: none; }")

        self.variations_container = QWidget()
        self.variations_layout = QHBoxLayout(self.variations_container)
        self.variations_layout.setContentsMargins(4, 4, 4, 4)
        self.variations_layout.setSpacing(8)
        self.variations_layout.addStretch()

        var_scroll.setWidget(self.variations_container)
        var_layout.addWidget(var_scroll)
        player_layout.addWidget(var_group)

        player_layout.addStretch()
        self.right_tabs.addTab(player_widget, "Player")

        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(8, 8, 8, 8)
        history_layout.setSpacing(6)

        header_row = QHBoxLayout()
        hist_lbl = QLabel("History")
        hist_lbl.setStyleSheet("font-size: 13px; font-weight: bold;")
        header_row.addWidget(hist_lbl)
        header_row.addStretch()
        
        self.fav_filter_check = QCheckBox("★ Favorites")
        self.fav_filter_check.toggled.connect(self.filter_history)
        header_row.addWidget(self.fav_filter_check)
        
        header_row.addWidget(QLabel("Tag:"))
        self.tag_filter_combo = QComboBox()
        self.tag_filter_combo.setFixedWidth(120)
        self.tag_filter_combo.currentTextChanged.connect(self.filter_history)
        header_row.addWidget(self.tag_filter_combo)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search...")
        self.search_edit.setFixedWidth(150)
        header_row.addWidget(self.search_edit)

        export_hist_btn = QPushButton("Export")
        export_hist_btn.setFixedHeight(28)
        export_hist_btn.clicked.connect(self.export_history)
        header_row.addWidget(export_hist_btn)

        clear_hist_btn = QPushButton("Clear")
        clear_hist_btn.setFixedHeight(28)
        clear_hist_btn.clicked.connect(self.clear_history)
        header_row.addWidget(clear_hist_btn)
        history_layout.addLayout(header_row)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(200)
        self._search_timer.timeout.connect(lambda: self._do_filter(self.search_edit.text()))
        self.search_edit.textChanged.connect(self._search_timer.start)

        self.playlist_table = QTableWidget()
        self.playlist_table.setColumnCount(7)
        self.playlist_table.setHorizontalHeaderLabels(["★", "Tags", "Prompt", "Waveform", "Date", "Seed", "Model"])
        self.playlist_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.playlist_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.playlist_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.playlist_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        for i in range(4, 7):
            self.playlist_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.playlist_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.playlist_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.playlist_table.setAlternatingRowColors(True)
        self.playlist_table.itemDoubleClicked.connect(self.play_from_playlist)
        self.playlist_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.playlist_table.customContextMenuRequested.connect(self.show_history_context_menu)
        history_layout.addWidget(self.playlist_table)
        self.right_tabs.addTab(history_widget, "History")
        parent.addWidget(right_panel)

    def parse_time_str(self, t_str):
        try:
            parts = t_str.split(':')
            if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
            if len(parts) == 2: return int(parts[0])*60 + int(parts[1])
            return int(parts[0])
        except ValueError:
            raise ValueError(f"Invalid time format: '{t_str}'. Use MM:SS or HH:MM:SS.")

    def load_timeline_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Timeline Script", "", "Text Files (*.txt)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.timeline_edit.setPlainText(f.read())
            except Exception as e:
                logger.error(f"Failed to load timeline script: {e}", exc_info=True)

    def start_timeline_generation(self):
        script = self.timeline_edit.toPlainText().strip()
        if not script:
            logger.warning("Timeline script is empty.")
            return
            
        lines = script.split('\n')
        batch_prompts = []
        
        try:
            for line in lines:
                if not line.strip(): continue
                match = re.match(r'\[(\d+:\d+(?::\d+)?)\s*-\s*(\d+:\d+(?::\d+)?)\]\s*(.*)', line)
                if match:
                    start_s = self.parse_time_str(match.group(1))
                    end_s = self.parse_time_str(match.group(2))
                    prompt = match.group(3).strip()
                    duration = end_s - start_s
                    if duration > 0 and prompt:
                        batch_prompts.append({'prompt': prompt, 'duration': duration, 'start_time': start_s})
            
            if not batch_prompts:
                logger.warning("No valid timeline entries found in script.")
                return
                
            self.batch_orchestrator.start_batch(batch_prompts, is_timeline=True)
            self.batch_start_btn.setEnabled(False)
            self.tl_start_btn.setEnabled(False)
            self.batch_log.clear()
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Timeline Generation...")
            self.mode_tabs.setCurrentIndex(0)
            
        except Exception as e:
            logger.error(f"Timeline script error: {e}", exc_info=True)

    def _on_batch_prompt_ready(self, prompt, duration):
        self.prompt_input.setPlainText(prompt)
        self.duration_spin.setValue(duration)
        if not self.start_generation():
            self.batch_orchestrator.cancel()
            self._on_batch_finished()

    def _on_batch_finished(self):
        self.batch_start_btn.setEnabled(True)
        self.tl_start_btn.setEnabled(True)
        self.batch_progress_label.setText("Batch finished or stopped.")
        if os.path.exists(BATCH_STATE_FILE):
            try: os.remove(BATCH_STATE_FILE)
            except OSError: pass

    def open_fx_chain_dialog(self):
        dialog = FxChainDialog(self, self.fx_chain)
        if dialog.exec() == QDialog.Accepted:
            self.fx_chain = dialog.get_chain()
            self.settings.setValue("fx_chain", self.serialize_fx_chain())
            self.settings.sync()
            self.status_bar.showMessage("FX Chain updated.", 3000)
            
            if self.base_audio_path:
                if self.fx_chain:
                    self.apply_fx_to_current()
                else:
                    self.clear_fx()

    def clear_fx(self):
        if not self.base_audio_path: return
        self.current_audio_path = self.base_audio_path
        self.waveform_widget.set_audio(self.base_audio_path)
        self.spectrogram_widget.set_audio(self.base_audio_path)
        self.update_audio_stats(self.base_audio_path)
        self.media_player.setSource(QUrl.fromLocalFile(self.base_audio_path))
        self.clear_fx_btn.setEnabled(False)
        self.status_bar.showMessage("FX bypassed. Original audio restored.", 3000)

    def apply_fx_to_current(self):
        if not PEDALBOARD_AVAILABLE or not self.fx_chain or not self.base_audio_path:
            return
            
        try:
            with wave.open(self.base_audio_path, 'rb') as wf_in:
                sr = wf_in.getframerate()
                n_frames = wf_in.getnframes()
                with wave.open(FX_PREVIEW_FILE, 'wb') as wf_out:
                    wf_out.setnchannels(1)
                    wf_out.setsampwidth(2)
                    wf_out.setframerate(sr)
                    
                    board = Pedalboard(self.fx_chain)
                    chunk_size = sr * 10 
                    
                    for i in range(0, n_frames, chunk_size):
                        raw = wf_in.readframes(min(chunk_size, n_frames - i))
                        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        if data.size == 0: break
                        
                        data = np.expand_dims(data, axis=0)
                        processed = board(data, sr)
                        processed_int16 = (np.clip(processed[0], -1.0, 1.0) * 32767).astype(np.int16)
                        wf_out.writeframes(processed_int16.tobytes())
            
            self.current_audio_path = FX_PREVIEW_FILE
            self.waveform_widget.set_audio(FX_PREVIEW_FILE)
            self.spectrogram_widget.set_audio(FX_PREVIEW_FILE)
            self.update_audio_stats(FX_PREVIEW_FILE)
            self.media_player.setSource(QUrl.fromLocalFile(FX_PREVIEW_FILE))
            
            self.clear_fx_btn.setEnabled(True)
            self.status_bar.showMessage("FX applied to preview successfully!", 3000)
        except Exception as e:
            logger.error(f"FX Chain application failed: {e}", exc_info=True)

    def setup_audio(self):
        self.audio_output = QAudioOutput(self)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.mediaStatusChanged.connect(self.on_media_status)
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.set_volume(50)

    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_led = QLabel()
        self.status_led.setFixedSize(12, 12)
        self.status_led.setStyleSheet("background-color: #555; border-radius: 6px; margin: 2px;")
        self.status_bar.addPermanentWidget(self.status_led)

        device_info = "CPU"
        if torch.cuda.is_available():
            device_info = f"GPU: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info = "MPS"
        self.device_status_label = QLabel(f"Device: {device_info}")
        self.status_bar.addPermanentWidget(self.device_status_label)

        self.gpu_mem_label = ClickableLabel()
        self.gpu_mem_label.clicked.connect(self.show_gpu_memory)
        if torch.cuda.is_available():
            self.update_gpu_memory_label()
            self.gpu_mem_timer = QTimer(self)
            self.gpu_mem_timer.timeout.connect(self.update_gpu_memory_label)
            self.gpu_mem_timer.start(GPU_MEM_POLL_MS)
        else:
            self.gpu_mem_label.setText("VRAM: N/A")
        self.status_bar.addPermanentWidget(self.gpu_mem_label)
        self.status_bar.showMessage("Ready")

    def update_gpu_memory_label(self):
        if not torch.cuda.is_available(): return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        self.gpu_mem_label.setText(f"VRAM: {allocated:.1f}/{total:.1f} GB")

    def setup_tray_icon(self):
        try:
            self.tray_icon = QSystemTrayIcon(self)
            self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
            self.tray_icon.setToolTip(APP_NAME)
            self.tray_icon.show()
        except Exception as e:
            logger.warning(f"Failed to setup system tray icon: {e}")

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.start_generation)
        sc_space = QShortcut(QKeySequence("Space"), self)
        sc_space.setContext(Qt.WidgetWithChildrenShortcut)
        sc_space.activated.connect(self.toggle_playback)
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.open_audio)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_audio)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.cancel_generation)
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(lambda: self.prompt_input.setFocus())
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.regenerate_new_seed)
        QShortcut(QKeySequence("Ctrl+Shift+F"), self).activated.connect(self.open_fx_chain_dialog)

    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("Export Session", self.export_session)
        file_menu.addAction("Import Session", self.import_session)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close).setShortcut("Ctrl+Q")

        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction("Show GPU Memory", self.show_gpu_memory)
        tools_menu.addAction("Clear GPU Cache", self.clear_gpu_cache)
        if PEDALBOARD_AVAILABLE:
            tools_menu.addAction("Open FX Chain", self.open_fx_chain_dialog)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("Clear Prompt", lambda: self.prompt_input.clear()).setShortcut("Ctrl+Shift+X")
        edit_menu.addAction("Paste from Clipboard", self.paste_from_clipboard).setShortcut("Ctrl+V")
        edit_menu.addAction("Copy Settings as JSON", self.copy_settings_to_clipboard)
        edit_menu.addSeparator()
        edit_menu.addAction("Save Settings Preset", self.save_preset)
        edit_menu.addAction("Load Settings Preset", self.load_preset)
        edit_menu.addSeparator()
        edit_menu.addAction("Reset All Settings", self.reset_settings)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("Documentation", self.open_docs)
        help_menu.addAction("Prompt Tips", self.show_prompt_tips)
        help_menu.addSeparator()
        help_menu.addAction("About", self.show_about)

    def show_gpu_memory(self):
        if not ML_AVAILABLE or not torch.cuda.is_available():
            logger.info("CUDA is not available.")
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}\nTotal VRAM: {total:.2f} GB\nAllocated: {allocated:.2f} GB\nReserved: {reserved:.2f} GB\nFree (approx): {total - reserved:.2f} GB")

    def clear_gpu_cache(self):
        if not ML_AVAILABLE or not torch.cuda.is_available(): return
        gc.collect()
        torch.cuda.empty_cache()
        self.update_gpu_memory_label()
        self.status_bar.showMessage("GPU cache cleared", 3000)

    def reset_settings(self):
        reply = QMessageBox.question(self, "Reset Settings",
            "Reset all UI settings to defaults? (Model cache and generated files are not affected.)",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.settings.clear()
            self.load_settings()
            self.status_bar.showMessage("Settings reset to defaults", 3000)

    def on_preset_selected(self, text):
        if text: self.prompt_input.setPlainText(text)

    def on_neg_preset_selected(self, text):
        if text: self.neg_prompt_input.setText(text)

    def enhance_prompt(self):
        text = self.prompt_input.toPlainText().strip()
        if not text: text = "ambient sound effect"
        
        templates = [
            "{base}, {quality}, {detail}",
            "{base} in the {env}, {quality}",
            "{adj} {base} with {detail}, {quality}"
        ]
        qualities = ["high quality", "professional recording", "cinematic", "ASMR", "immersive"]
        envs = ["background", "distance", "studio", "forest", "city"]
        details = ["distant rumbling", "subtle echoes", "crisp transients", "deep bass response", "natural decay"]
        adjs = ["Heavy", "Gentle", "Epic", "Subtle", "Vibrant"]
        
        base = text
        if len(text.split()) < 4:
            template = random.choice(templates)
            base = template.format(
                base=text, quality=random.choice(qualities), detail=random.choice(details),
                env=random.choice(envs), adj=random.choice(adjs)
            )
        else:
            base = f"{text}, {random.choice(qualities)}, {random.choice(details)}"
            
        self.prompt_input.setPlainText(base)
        self.status_bar.showMessage("Prompt enhanced!", 2000)

    def generate_random_prompt(self):
        instruments = ["guitar", "piano", "synth", "drums", "violin", "trumpet", "flute", "pad"]
        genres = ["lofi", "cinematic", "ambient", "electronic", "classical", "rock"]
        qualities = ["high quality", "professional recording", "immersive"]
        
        prompt = f"{random.choice(genres)} {random.choice(instruments)}, {random.choice(qualities)}"
        self.prompt_input.setPlainText(prompt)

    def save_current_model(self):
        model = self.model_combo.currentText().strip()
        if model and self.model_combo.findText(model) == -1:
            self.model_combo.addItem(model)
            self.model_combo.setCurrentText(model)
            custom_models = self.settings.value("custom_models", [])
            if model not in custom_models:
                custom_models.append(model)
                self.settings.setValue("custom_models", custom_models)
                self.settings.sync()

    def browse_cache_dir(self):
        cur = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        path = QFileDialog.getExistingDirectory(self, "Select Model Cache Folder", cur if os.path.isdir(cur) else "")
        if path: self.cache_dir_edit.setText(path)

    def browse_output_dir(self):
        cur = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        path = QFileDialog.getExistingDirectory(self, "Select Output Audio Folder", cur if os.path.isdir(cur) else "")
        if path: self.output_dir_edit.setText(path)

    def browse_cond_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Conditioning Audio", "", "Audio Files (*.wav *.mp3 *.flac)")
        if path: self.cond_audio_edit.setText(path)

    def open_output_folder(self):
        out = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        if self.section_edit.text().strip():
            out = os.path.join(out, sanitize_filename(self.section_edit.text().strip()))
        os.makedirs(out, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(out))

    def update_chunk_preview(self):
        duration = self.duration_spin.value()
        chunk = self.chunk_size_spin.value()
        if chunk <= OVERLAP_S:
            self.chunk_preview_label.setText("⚠️ Chunk size must be > 1.0s")
            self.chunk_preview_label.setStyleSheet("color: #e74c3c; font-style: italic; font-size: 10px;")
            return
        if duration > chunk:
            n = math.ceil((duration - OVERLAP_S) / (chunk - OVERLAP_S))
            warn = " ⚠️ massive generation" if n > 500 else ""
            self.chunk_preview_label.setText(f"→ Will stream {n} chunks of ~{chunk:.1f}s with 1.0s crossfade{warn}")
            self.chunk_preview_label.setStyleSheet(
                "color: #e74c3c; font-style: italic; font-size: 10px;" if n > 500
                else "color: #888; font-style: italic; font-size: 10px;"
            )
        else:
            self.chunk_preview_label.setText("→ Will generate as a single chunk")
            self.chunk_preview_label.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")

    def on_device_changed(self, text: str):
        self.cpu_offload_check.setEnabled(text != "cpu")
        if text == "cpu": self.cpu_offload_check.setChecked(False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(STYLESHEET + "QMainWindow { border: 3px dashed #2A82DA; }")

    def dragLeaveEvent(self, event):
        self.setStyleSheet(STYLESHEET)

    def dropEvent(self, event):
        self.setStyleSheet(STYLESHEET)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.wav', '.mp3', '.flac')):
                self._load_audio_file(path)
                break
            elif path.lower().endswith('.txt'):
                self._load_batch_file(path)
                break

    def update_audio_stats(self, path: str):
        if self.stats_worker and self.stats_worker.isRunning():
            self.stats_worker.quit()
            self.stats_worker.wait(500)
            
        self.stats_worker = AudioStatsWorker(path)
        self.stats_worker.statsReady.connect(self._on_stats_ready)
        self.stats_worker.error.connect(self.on_ml_error)
        self.stats_worker.start()

    def _on_stats_ready(self, peak: float, rms: float):
        if peak > 0:
            peak_db = 20 * math.log10(peak)
            rms_db = 20 * math.log10(rms) if rms > 0 else -120.0
            headroom = 0.0 - peak_db
            self.audio_stats_label.setText(f"Peak: {peak_db:.1f} dB | RMS: {rms_db:.1f} dB | Headroom: {headroom:.1f} dB")
            self.audio_stats_label.setStyleSheet("color: #2A82DA; font-size: 11px; font-family: monospace; padding: 2px;")
        else:
            self.audio_stats_label.setText("Peak: N/A | RMS: N/A | Headroom: N/A")
            self.audio_stats_label.setStyleSheet("color: #888; font-size: 11px; font-family: monospace; padding: 2px;")

    def _load_audio_file(self, path):
        if not path or not os.path.exists(path): return
        try:
            with open(path, 'rb') as f: f.read(4)
        except OSError as e:
            logger.warning(f"Cannot read file: {e}")
            return
        
        if not path.lower().endswith('.wav'):
            if not shutil.which("ffmpeg") and not os.path.exists("ffmpeg.exe"):
                logger.warning("FFmpeg is required to load non-WAV files for visualization.")
            else:
                cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "44100", TEMP_LOAD_WAV]
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                subprocess.run(cmd, capture_output=True, text=True, creationflags=creationflags)
                path = TEMP_LOAD_WAV
        
        self._clear_variations()
        try:
            with wave.open(path, 'rb') as wf:
                dur = wf.getnframes() / wf.getframerate()
        except Exception as e:
            logger.error(f"Failed to read WAV header: {e}", exc_info=True)
            dur = 0.0
        self._add_variation_card(path, 0, dur, -1)
        
        self.base_audio_path = path
        self.current_audio_path = path
        
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.demucs_btn.setEnabled(DEMUCS_AVAILABLE)
        self.whisper_btn.setEnabled(WHISPER_AVAILABLE)
        self.clap_btn.setEnabled(CLAP_AVAILABLE)
        
        if PEDALBOARD_AVAILABLE and self.fx_chain:
            self.apply_fx_to_current()
        else:
            self.clear_fx_btn.setEnabled(False)
            self.waveform_widget.set_audio(path)
            self.spectrogram_widget.set_audio(path)
            self.update_audio_stats(path)
        
        self.right_tabs.setCurrentIndex(0)
        self.settings.setValue("last_audio_path", path)
        self.settings.sync()
        
        self._update_variation_card_style(path)

    def expand_prompt_matrix(self, text: str) -> list:
        matches = list(re.finditer(r'\[([^]]+)\]', text))
        if not matches: return [text]
        options = [m.group(1).split(',') for m in matches]
        options = [[opt.strip() for opt in group] for group in options]
        
        total = 1
        for g in options:
            total *= len(g)
            if total > PROMPT_MATRIX_MAX:
                raise ValueError(
                    f"Prompt matrix expands to {total}+ combinations (cap={PROMPT_MATRIX_MAX}). "
                    "Reduce the number of bracketed groups."
                )
                
        combinations = list(itertools.product(*options))

        results = []
        for combo in combinations:
            t = text
            for i, m in enumerate(matches):
                t = t.replace(m.group(0), combo[i], 1)
            results.append(t)
        return results

    def _load_batch_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

            expanded_prompts = []
            for line in lines:
                expanded_prompts.extend(self.expand_prompt_matrix(line))

            self.batch_orchestrator.batch_prompts = expanded_prompts
            self.batch_file_label.setText(path)
            self.batch_log.appendPlainText(f"Loaded {len(lines)} prompts. Expanded to {len(expanded_prompts)} matrix variations.")
            self.mode_tabs.setCurrentIndex(1)
        except Exception as e:
            logger.error(f"Batch file error: {e}", exc_info=True)

    def toggle_inpaint_button(self, checked):
        self.inpaint_btn.setEnabled(checked and self.current_audio_path is not None)

    def on_spectrogram_selection_changed(self, start, end):
        valid = start >= 0 and end > start
        self.inpaint_btn.setEnabled(valid and self.inpaint_check.isChecked())

    def start_inpainting(self):
        if not self.base_audio_path or not self.base_audio_path.endswith('.wav'):
            logger.warning("Inpainting error: Please load a WAV file to inpaint.")
            return
            
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            logger.warning("Input error: Please enter a prompt for the inpainted region.")
            return
            
        start, end = self.spectrogram_widget.get_selection()
        if end - start < 0.5:
            logger.warning("Selection error: Please select a region longer than 0.5s on the spectrogram.")
            return
            
        self._inpaint_active = True
        self._inpaint_source = self.base_audio_path
        self._inpaint_start = start
        self._inpaint_end = end
        
        self.status_bar.showMessage(f"Inpainting region {format_time(start)} to {format_time(end)}...")
        self.start_generation()

    def start_generation(self) -> bool:
        if self.worker_thread and self.worker_thread.isRunning():
            logger.info("The engine is still cleaning up from the last generation. Please wait a moment.")
            return False

        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            logger.warning("Input error: Please enter a prompt.")
            return False

        model = self.model_combo.currentText().strip()
        if not model or '/' not in model:
            logger.warning("Invalid model: Model name must be in 'org/model' format.")
            return False

        out_dir = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Output error: Cannot create output folder:\n{e}")
            return False

        cache_dir = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Cache directory error: Cannot create or access cache directory '{cache_dir}'.")
            return False

        self.prompt_input.add_to_history(prompt)

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.play_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.demucs_btn.setEnabled(False)
        self.whisper_btn.setEnabled(False)
        self.clap_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.eta_label.setText("")
        self.status_label.setText("Starting...")
        self.status_bar.showMessage("Generation in progress...")
        self.status_led.setStyleSheet("background-color: #f1c40f; border-radius: 6px; margin: 2px;")

        duration = self.duration_spin.value()
        inpaint_source = ""
        inpaint_start = 0.0
        inpaint_end = 0.0
        
        if self._inpaint_active:
            duration = self._inpaint_end - self._inpaint_start
            inpaint_source = self._inpaint_source
            inpaint_start = self._inpaint_start
            inpaint_end = self._inpaint_end

        params = GenerationParams(
            prompt=prompt,
            negative_prompt=self.neg_prompt_input.text().strip(),
            model_name=model,
            duration=duration,
            steps=self.steps_spin.value(),
            guidance=self.guidance_spin.value(),
            seed=self.seed_spin.value(),
            device=self.device_combo.currentText(),
            cache_dir=cache_dir,
            output_dir=out_dir,
            num_variations=self.variations_spin.value(),
            normalize=self.normalize_check.isChecked(),
            fade_in=self.fade_in_check.isChecked(),
            fade_out=self.fade_out_check.isChecked(),
            trim=self.trim_check.isChecked(),
            chunk_size=self.chunk_size_spin.value(),
            use_cpu_offload=self.cpu_offload_check.isChecked(),
            section=self.section_edit.text().strip(),
            high_pass_freq=self.hp_spin.value(),
            low_pass_freq=self.lp_spin.value(),
            conditioning_audio=self.cond_audio_edit.text().strip(),
            inpaint_source=inpaint_source,
            inpaint_start_s=inpaint_start,
            inpaint_end_s=inpaint_end,
            use_seed_travel=self.travel_check.isChecked(),
            travel_start_seed=self.travel_start_seed.value(),
            travel_end_seed=self.travel_end_seed.value(),
            travel_steps=self.travel_steps_spin.value()
        )

        self.worker_thread = QThread()
        self.worker = AudioGenerationWorker(params)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        self.worker.status_updated.connect(self.status_label.setText, Qt.QueuedConnection)
        self.worker.eta_updated.connect(self.eta_label.setText, Qt.QueuedConnection)
        self.worker.generation_completed.connect(self.generation_finished, Qt.QueuedConnection)
        self.worker.error_occurred.connect(self.generation_error, Qt.QueuedConnection)
        self.worker.cancelled.connect(self.generation_cancelled, Qt.QueuedConnection)

        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()
        return True

    def regenerate_new_seed(self):
        self.seed_spin.setValue(-1)
        self.start_generation()

    def cancel_generation(self):
        if self.batch_orchestrator.batch_state == BatchState.RUNNING:
            reply = QMessageBox.question(self, "Cancel Batch",
                "Cancelling will stop the entire batch. Continue?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes: return
            self.batch_orchestrator.cancel()

        if self.worker:
            self.worker.is_cancelled = True
            self.status_label.setText("Cancelling...")
            self.cancel_btn.setEnabled(False)
            self.status_bar.showMessage("Cancelling generation...")

        if self.batch_orchestrator.batch_state == BatchState.CANCELLED:
            self.batch_progress_label.setText("Batch cancelled.")
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Batch cancelled by user.")

    def generation_cancelled(self):
        self.progress_bar.setVisible(False)
        self.eta_label.SetText("")
        self.status_label.setText("Ready")
        self.status_led.setStyleSheet("background-color: #555; border-radius: 6px; margin: 2px;")
        if self.batch_orchestrator.batch_state == BatchState.RUNNING:
            self.batch_orchestrator.cancel()
        self.status_bar.showMessage("Generation cancelled", 3000)

    def generation_finished(self, path, metadata):
        self.base_audio_path = path
        self.current_audio_path = path
        
        self.progress_bar.setVisible(False)
        self.eta_label.SetText("")
        self.status_label.SetText("Ready")
        self.status_led.setStyleSheet("background-color: #555; border-radius: 6px; margin: 2px;")
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.demucs_btn.setEnabled(DEMUCS_AVAILABLE)
        self.whisper_btn.setEnabled(WHISPER_AVAILABLE)
        self.clap_btn.setEnabled(CLAP_AVAILABLE)

        self._clear_variations()
        meta_list = metadata['metadata']
        
        for i, meta in enumerate(meta_list):
            self._add_variation_card(
                path=meta['path'], index=i, duration=meta['duration'], seed=meta['seed']
            )

        if metadata.get('paths'):
            self._load_variation(metadata['paths'][0])

        history_data = self.settings.value("history", [])
        row = self.playlist_table.rowCount()
        
        all_tags = set()
        for entry in history_data:
            if entry.get("tags"): all_tags.update(entry["tags"])
        for meta in meta_list:
            if meta.get("tags"): all_tags.update(meta["tags"])
        self.tag_filter_combo.clear()
        self.tag_filter_combo.addItem("All")
        self.tag_filter_combo.addItems(sorted(list(all_tags)))

        for i, meta in enumerate(meta_list):
            history_data.append(meta)

            self.playlist_table.insertRow(row + i)
            
            fav_item = QTableWidgetItem()
            if meta.get('favorite'):
                fav_item.setText("★")
                fav_item.setForeground(QColor('#f1c40f'))
            else:
                fav_item.setText("")
            fav_item.setTextAlignment(Qt.AlignCenter)
            self.playlist_table.setItem(row + i, 0, fav_item)
            
            tags_item = QTableWidgetItem(", ".join(meta.get('tags', [])))
            self.playlist_table.setItem(row + i, 1, tags_item)
            
            prompt_item = QTableWidgetItem(meta['prompt'])
            prompt_item.setToolTip(meta['path'])
            if meta['path'] and not os.path.exists(meta['path']):
                prompt_item.setForeground(QColor('red'))
            self.playlist_table.setItem(row + i, 2, prompt_item)
            
            mini_item = MiniWaveformItem(meta['path'], meta['duration'])
            self.playlist_table.setItem(row + i, 3, mini_item)
            if self._thumb_worker.isRunning():
                self._thumb_worker.enqueue(row + i, meta['path'])
            else:
                pix = MiniWaveformItem.generate_pixmap(meta['path'])
                if pix: mini_item.set_pixmap(pix)
                
            self.playlist_table.setItem(row + i, 4, QTableWidgetItem(meta['timestamp'][:10]))
            self.playlist_table.setItem(row + i, 5, QTableWidgetItem(str(meta['seed'])))
            self.playlist_table.setItem(row + i, 6, QTableWidgetItem(meta['model'].split('/')[-1]))

        if len(history_data) > MAX_HISTORY:
            history_data = history_data[-MAX_HISTORY:]
            while self.playlist_table.rowCount() > MAX_HISTORY:
                self.playlist_table.removeRow(0)

        self.settings.setValue("history", history_data)
        self.settings.sync()

        self.settings.setValue("last_audio_path", path)
        self.settings.sync()

        gen_time = metadata.get('gen_time', 0.0)
        self.right_tabs.setCurrentIndex(0)

        self.status_bar.showMessage(f"Generated {len(metadata['paths'])} audio file(s) in {format_time(gen_time)}", 5000)
        
        if not self.isActiveWindow() and hasattr(self, 'tray_icon'):
            self.tray_icon.showMessage("Generation Complete", f"Generated: {os.path.basename(path)}", QSystemTrayIcon.Information, 5000)
        
        self._inpaint_active = False
        self._inpaint_source = ""
        self.inpaint_check.setChecked(False)
        
        if self.batch_orchestrator.batch_state == BatchState.RUNNING:
            self.batch_orchestrator.on_generation_finished(meta_list)
        
        if self.autoplay_check.isChecked():
            QTimer.singleShot(100, self.toggle_playback)

    def _clear_variations(self):
        while self.variations_layout.count() > 1:
            item = self.variations_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def _add_variation_card(self, path, index, duration, seed):
        card = VariationCard(path, index, duration, seed)
        card.playRequested.connect(self._play_variation)
        card.selected.connect(self._load_variation)
        self.variations_layout.insertWidget(self.variations_layout.count() - 1, card)
        pix = MiniWaveformItem.generate_pixmap(path)
        if pix:
            card.set_thumbnail(pix)

    def _play_variation(self, path):
        self._load_variation(path)
        self.toggle_playback()

    def _load_variation(self, path):
        if not path or not os.path.exists(path): return
        self._load_audio_file(path)

    def _update_variation_card_style(self, active_path):
        for i in range(self.variations_layout.count()):
            item = self.variations_layout.itemAt(i)
            if item and item.widget():
                card = item.widget()
                if isinstance(card, VariationCard):
                    if card.path == active_path:
                        card.setStyleSheet("QFrame { background-color: #1a3d5c; border: 1px solid #2A82DA; border-radius: 4px; }")
                    else:
                        card.setStyleSheet("QFrame { background-color: #333333; border: 1px solid #3c3c3c; border-radius: 4px; } QFrame:hover { border-color: #2A82DA; background-color: #3c3c3c; }")

    def generation_error(self, msg):
        self.progress_bar.setVisible(False)
        self.eta_label.SetText("")
        self.status_label.SetText("Error")
        self.status_led.setStyleSheet("background-color: #c0392b; border-radius: 6px; margin: 2px;")
        if self.batch_orchestrator.batch_state == BatchState.RUNNING:
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {msg[:100]}...")
            if not self.batch_continue_on_error_check.isChecked():
                self.batch_orchestrator.cancel()
        else:
            logger.error(f"Generation error: {msg}")
        
        self._inpaint_active = False
        self._inpaint_source = ""
        self.inpaint_check.setChecked(False)

    def _trigger_next_batch_item(self):
        if self.batch_orchestrator.batch_state != BatchState.RUNNING: return
        if not self.start_generation():
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Failed to start generation. Skipping prompt.")
            self.on_worker_thread_finished()

    def on_worker_thread_finished(self):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if self.batch_orchestrator.batch_state != BatchState.RUNNING:
            self.batch_orchestrator.batch_state = BatchState.IDLE
            self.status_label.SetText("Ready")

    def toggle_playback(self):
        fw = QApplication.focusWidget()
        if fw and fw.inherits("QAbstractSpinBox") or isinstance(fw, (QLineEdit, QPlainTextEdit, QComboBox)):
            return
        if not self.current_audio_path or not os.path.exists(self.current_audio_path): return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.stop()
            self.status_led.setStyleSheet("background-color: #555; border-radius: 6px; margin: 2px;")
        else:
            self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_path))
            self.media_player.play()
            self.status_led.setStyleSheet("background-color: #2ecc71; border-radius: 6px; margin: 2px;")

    def on_playback_state_changed(self, state):
        self.play_btn.setText("Stop" if state == QMediaPlayer.PlayingState else "Play")
        if state != QMediaPlayer.PlayingState:
            self.status_led.setStyleSheet("background-color: #555; border-radius: 6px; margin: 2px;")
            self.vu_meter.setValue(0)

    def play_from_playlist(self, item):
        if isinstance(item, QTableWidgetItem):
            path = self.playlist_table.item(item.row(), 2).toolTip()
        else:
            path = item
        self._play_path(path)

    def _play_path(self, path):
        if path and os.path.exists(path):
            self._load_audio_file(path)
            self.toggle_playback()
        else:
            logger.warning("File missing: This audio file has been moved or deleted.")

    def show_history_context_menu(self, pos):
        item = self.playlist_table.itemAt(pos)
        if not item: return
        row = item.row()
        path = self.playlist_table.item(row, 2).toolTip()
        history_data = self.settings.value("history", [])
        
        menu = QMenu(self)
        menu.addAction("Play", lambda: self._play_path(path))
        menu.addAction("Open in Folder", lambda: self.open_file_in_folder(path))
        menu.addAction("Copy File Path", lambda: QApplication.clipboard().setText(path))
        menu.addAction("Copy Prompt", lambda: QApplication.clipboard().setText(self.playlist_table.item(row, 2).text()))
        menu.addAction("Reuse Settings", lambda: self.reuse_settings(row))
        menu.addSeparator()
        
        is_fav = history_data[row].get('favorite', False)
        fav_action = menu.addAction("Remove Favorite" if is_fav else "Add Favorite")
        fav_action.triggered.connect(lambda: self.toggle_favorite(row))
        
        menu.addAction("Add Tag...", lambda: self.add_tag_to_row(row))
        menu.addAction("Clear Tags", lambda: self.clear_tags(row))
        
        menu.addSeparator()

        rev_action = menu.addAction("Reverse Audio & Load")
        rev_action.triggered.connect(lambda: self.reverse_audio_file(path))

        menu.addAction("Export Selected to ZIP", self.export_selected_to_zip)

        menu.addSeparator()
        menu.addAction("Delete Entry", lambda: self.delete_history_row(row))
        menu.exec(self.playlist_table.viewport().mapToGlobal(pos))

    def toggle_favorite(self, row):
        history_data = self.settings.value("history", [])
        if 0 <= row < len(history_data):
            history_data[row]['favorite'] = not history_data[row].get('favorite', False)
            self.settings.setValue("history", history_data)
            self.settings.sync()
            self.load_history()

    def add_tag_to_row(self, row):
        history_data = self.settings.value("history", [])
        if 0 <= row < len(history_data):
            tag, ok = QInputDialog.getText(self, "Add Tag", "Enter tag (e.g., sfx, music, ambient):")
            if ok and tag:
                if 'tags' not in history_data[row]:
                    history_data[row]['tags'] = []
                if tag not in history_data[row]['tags']:
                    history_data[row]['tags'].append(tag)
                    self.settings.setValue("history", history_data)
                    self.settings.sync()
                    self.load_history()

    def clear_tags(self, row):
        history_data = self.settings.value("history", [])
        if 0 <= row < len(history_data):
            history_data[row]['tags'] = []
            self.settings.setValue("history", history_data)
            self.settings.sync()
            self.load_history()

    def filter_history(self):
        search_text = self.search_edit.text().lower()
        only_favs = self.fav_filter_check.isChecked()
        tag_filter = self.tag_filter_combo.currentText()
        
        for row in range(self.playlist_table.rowCount()):
            should_hide = False
            item_fav = self.playlist_table.item(row, 0)
            if only_favs and (not item_fav or item_fav.text() != "★"):
                should_hide = True
            item_tags = self.playlist_table.item(row, 1)
            if tag_filter != "All" and (not item_tags or tag_filter not in item_tags.text()):
                should_hide = True
            item_prompt = self.playlist_table.item(row, 2)
            if search_text and (not item_prompt or search_text not in item_prompt.text().lower()):
                should_hide = True
                
            self.playlist_table.setRowHidden(row, should_hide)

    def reverse_audio_file(self, path: str):
        try:
            with wave.open(path, 'rb') as wf_in:
                sr = wf_in.getframerate()
                n_frames = wf_in.getnframes()
                
                base, _ = os.path.splitext(path)
                out_path = base + "_reversed.wav"
                
                with wave.open(out_path, 'wb') as wf_out:
                    wf_out.setnchannels(1)
                    wf_out.setsampwidth(2)
                    wf_out.setframerate(sr)
                    
                    chunk_size = sr * 10 
                    chunks = []
                    
                    for i in range(0, n_frames, chunk_size):
                        raw = wf_in.readframes(min(chunk_size, n_frames - i))
                        data = np.frombuffer(raw, dtype=np.int16)
                        chunks.append(data)
                        
                    for chunk in reversed(chunks):
                        wf_out.writeframes(chunk[::-1].tobytes())

            self._load_audio_file(out_path)
            self.status_bar.showMessage("Audio reversed and loaded successfully!", 3000)
        except Exception as e:
            logger.error(f"Failed to reverse audio: {e}", exc_info=True)

    def export_selected_to_zip(self):
        paths = []
        for item in self.playlist_table.selectedItems():
            row = item.row()
            path = self.playlist_table.item(row, 2).toolTip()
            if path and os.path.exists(path) and path not in paths:
                paths.append(path)

        if not paths:
            logger.warning("No files selected for ZIP export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export to ZIP", "audio_export.zip", "ZIP Files (*.zip)")
        if not path: return

        try:
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p in paths:
                    zf.write(p, arcname=os.path.basename(p))
            self.status_bar.showMessage(f"Exported {len(paths)} files to ZIP successfully!", 5000)
        except Exception as e:
            logger.error(f"ZIP export failed: {e}", exc_info=True)

    def open_file_in_folder(self, path):
        if not path or not os.path.exists(path): return
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', '/select,', path])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', '-R', path])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))

    def reuse_settings(self, row):
        history_data = self.settings.value("history", [])
        if 0 <= row < len(history_data):
            entry = history_data[row]
            self.prompt_input.setPlainText(entry.get("prompt", ""))
            self.neg_prompt_input.setText(entry.get("negative_prompt", ""))
            self.duration_spin.setValue(entry.get("duration", 10.0))
            self.steps_spin.setValue(entry.get("steps", 200))
            self.guidance_spin.setValue(entry.get("guidance", 3.5))
            self.seed_spin.setValue(entry.get("seed", -1))
            self.model_combo.setCurrentText(entry.get("model", "cvssp/audioldm2"))
        self.mode_tabs.setCurrentIndex(0)

    def delete_history_row(self, row):
        self.playlist_table.removeRow(row)
        history_data = self.settings.value("history", [])
        if 0 <= row < len(history_data):
            history_data.pop(row)
            self.settings.setValue("history", history_data)
            self.settings.sync()

    def on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.play_btn.setText("Play")
            self.waveform_widget.set_playback_position(0.0)
            self.spectrogram_widget.set_playback_position(0.0)
            self.vu_meter.setValue(0)

    def on_position_changed(self, position):
        duration = self.media_player.duration()
        if duration > 0:
            self.time_label.setText(f"{format_time(position / 1000)} / {format_time(duration / 1000)}")
            self.waveform_widget.set_playback_position(position / duration)
            self.spectrogram_widget.set_playback_position(position / duration)

            vu_val = int(min(100, self.waveform_widget.rms_at(position / duration) * 150))
            self.vu_meter.setValue(vu_val)

    def on_duration_changed(self, duration):
        if duration > 0:
            self.time_label.setText(f"00:00 / {format_time(duration / 1000)}")

    def _seek_to(self, pos: float):
        if self.media_player.duration() > 0:
            self.media_player.setPosition(int(pos * self.media_player.duration()))

    def set_volume(self, val):
        self.audio_output.setVolume(val / 100.0)
        self.vol_label.setText(f"{val}%")

    def save_audio(self):
        if not self.base_audio_path: return
        default_name = os.path.basename(self.base_audio_path).replace('.wav', '')
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio/Video", default_name,
            "WAV Audio (*.wav);;MP3 Audio (*.mp3);;FLAC Audio (*.flac);;MP4 Video (*.mp4)"
        )
        if not path: return
        
        is_fx_active = PEDALBOARD_AVAILABLE and self.fx_chain and self.current_audio_path == FX_PREVIEW_FILE
        
        if is_fx_active:
            try:
                with wave.open(self.base_audio_path, 'rb') as wf_in:
                    sr = wf_in.getframerate()
                    n_frames = wf_in.getnframes()
                    
                    temp_wav = path + ".tmp.wav"
                    with wave.open(temp_wav, 'wb') as wf_out:
                        wf_out.setnchannels(1)
                        wf_out.setsampwidth(2)
                        wf_out.setframerate(sr)
                        
                        board = Pedalboard(self.fx_chain)
                        chunk_size = sr * 10 
                        
                        for i in range(0, n_frames, chunk_size):
                            raw = wf_in.readframes(min(chunk_size, n_frames - i))
                            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                            if data.size == 0: break
                            data = np.expand_dims(data, axis=0)
                            processed = board(data, sr)
                            processed_int16 = (np.clip(processed[0], -1.0, 1.0) * 32767).astype(np.int16)
                            wf_out.writeframes(processed_int16.tobytes())
                            
                if path.endswith('.wav'):
                    shutil.move(temp_wav, path)
                    self.status_bar.showMessage("Audio saved with FX successfully!", 3000)
                    return
                else:
                    self.export_to_media(temp_wav, path, path.split('.')[-1])
                    try: os.remove(temp_wav)
                    except: pass
                    return
            except Exception as e:
                logger.error(f"Error saving with FX: {e}", exc_info=True)

        if path.endswith('.mp4'):
            self.export_to_media(self.base_audio_path, path, 'mp4')
        elif path.endswith('.mp3'):
            self.export_to_media(self.base_audio_path, path, 'mp3')
        elif path.endswith('.flac'):
            self.export_to_media(self.base_audio_path, path, 'flac')
        else:
            try:
                shutil.copy2(self.base_audio_path, path)
                self.status_bar.showMessage("Audio saved successfully!", 3000)
            except Exception as e:
                logger.error(f"Failed to save audio: {e}", exc_info=True)

    def export_to_media(self, wav_path: str, out_path: str, fmt: str):
        if not shutil.which("ffmpeg") and not os.path.exists("ffmpeg.exe"):
            logger.error("FFmpeg is missing. Cannot export media files.")
            return

        cmd = ["ffmpeg", "-y", "-i", wav_path]

        if fmt == 'mp4':
            cmd += [
                "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=30",
                "-c:v", "libx264", "-c:a", "aac",
                "-shortest", "-pix_fmt", "yuv420p"
            ]
        elif fmt == 'mp3':
            cmd += ["-c:a", "libmp3lame", "-b:a", "320k"]
        elif fmt == 'flac':
            cmd += ["-c:a", "flac"]

        cmd.append(out_path)

        self.status_bar.showMessage(f"Encoding {fmt.upper()}... Please wait.")
        QApplication.processEvents()

        self.ffmpeg_worker = FFmpegWorker(cmd)
        self.ffmpeg_worker.finished.connect(lambda success, err: self._on_ffmpeg_finished(success, err, fmt, out_path))
        self.ffmpeg_worker.error.connect(self._on_ffmpeg_error)
        self.ffmpeg_worker.start()

    def _on_ffmpeg_finished(self, success, err, fmt, out_path):
        if success:
            self.status_bar.showMessage(f"{fmt.upper()} exported successfully!", 5000)
        else:
            logger.error(f"FFmpeg failed to encode {fmt}: {err}")

    def _on_ffmpeg_error(self, err):
        logger.error(f"Error exporting media: {err}")

    def open_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.wav *.mp3 *.flac)")
        if path: self._load_audio_file(path)

    def paste_from_clipboard(self):
        text = QApplication.clipboard().text()
        if text: self.prompt_input.setPlainText(text)

    def select_batch_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Text File", "", "Text Files (*.txt)")
        if path: self._load_batch_file(path)

    def start_batch(self):
        if not self.batch_orchestrator.batch_prompts or not os.path.exists(self.batch_file_label.text()):
            logger.warning("Batch error: Please select a valid text file first.")
            return
        reply = QMessageBox.question(self, "Start Batch",
            f"About to generate {len(self.batch_orchestrator.batch_prompts)} prompts with {self.batch_variations_spin.value()} variation(s) each.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return

        self.batch_orchestrator.is_timeline_batch = False
        self.variations_spin.setValue(self.batch_variations_spin.value())
        
        self.batch_orchestrator.start_batch(self.batch_orchestrator.batch_prompts, is_timeline=False)
        self.batch_start_btn.setEnabled(False)
        self.batch_file_btn.setEnabled(False)
        
        self.batch_progress_label.setText(f"Batch Progress: 1 / {len(self.batch_orchestrator.batch_prompts)}")
        self.batch_log.clear()
        self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Starting batch...")
        self.save_batch_state()
        self.mode_tabs.setCurrentIndex(0)

    def save_batch_state(self):
        if self.batch_orchestrator.batch_state == BatchState.RUNNING and self.batch_orchestrator.batch_prompts:
            try:
                with open(BATCH_STATE_FILE, "w") as f:
                    json.dump({
                        "prompts": self.batch_orchestrator.batch_prompts, 
                        "index": self.batch_orchestrator.batch_index, 
                        "file": self.batch_file_label.text(), 
                        "is_timeline": self.batch_orchestrator.is_timeline_batch
                    }, f)
            except Exception as e:
                logger.error(f"Failed to save batch state: {e}", exc_info=True)

    def check_interrupted_batch(self):
        if os.path.exists(BATCH_STATE_FILE):
            reply = QMessageBox.question(self, "Resume Batch",
                "An interrupted batch generation was detected. Would you like to resume?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    with open(BATCH_STATE_FILE, "r") as f:
                        state = json.load(f)
                    self.batch_orchestrator.batch_prompts = state["prompts"]
                    self.batch_orchestrator.batch_index = state["index"]
                    self.batch_file_label.setText(state.get("file", "Resumed Batch"))
                    self.batch_orchestrator.is_timeline_batch = state.get("is_timeline", False)
                    self.batch_orchestrator.batch_state = BatchState.RUNNING
                    self.batch_start_btn.setEnabled(False)
                    self.batch_file_btn.setEnabled(False)
                    
                    if not (0 <= self.batch_orchestrator.batch_index < len(self.batch_orchestrator.batch_prompts)):
                        os.remove(BATCH_STATE_FILE)
                        return
                    
                    next_item = self.batch_orchestrator.batch_prompts[self.batch_orchestrator.batch_index]
                    next_prompt = next_item['prompt'] if isinstance(next_item, dict) else next_item
                    self.prompt_input.setPlainText(next_prompt)
                    if isinstance(next_item, dict) and 'duration' in next_item:
                        self.duration_spin.setValue(next_item['duration'])
                        
                    self.batch_progress_label.setText(f"Batch Progress: {self.batch_orchestrator.batch_index + 1} / {len(self.batch_orchestrator.batch_prompts)}")
                    self.batch_log.clear()
                    self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Resuming batch...")
                    self.mode_tabs.setCurrentIndex(0)
                    if not self.start_generation():
                        self.batch_orchestrator.cancel()
                        self._on_batch_finished()
                except Exception as e:
                    logger.error(f"Failed to resume batch: {e}", exc_info=True)
                    try: os.remove(BATCH_STATE_FILE)
                    except OSError: pass
            else:
                try: os.remove(BATCH_STATE_FILE)
                except OSError: pass

    def copy_settings_to_clipboard(self):
        settings_dict = {
            "prompt": self.prompt_input.toPlainText(),
            "negative_prompt": self.neg_prompt_input.text(),
            "model": self.model_combo.currentText(),
            "duration": self.duration_spin.value(),
            "chunk_size": self.chunk_size_spin.value(),
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
            "seed": self.seed_spin.value(),
            "variations": self.variations_spin.value(),
            "device": self.device_combo.currentText(),
            "cpu_offload": self.cpu_offload_check.isChecked(),
            "conditioning_audio": self.cond_audio_edit.text().strip(),
            "use_seed_travel": self.travel_check.isChecked(),
            "travel_start_seed": self.travel_start_seed.value(),
            "travel_end_seed": self.travel_end_seed.value(),
            "travel_steps": self.travel_steps_spin.value()
        }
        QApplication.clipboard().setText(json.dumps(settings_dict, indent=2))
        self.status_bar.showMessage("Settings copied as JSON", 3000)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip(): return
        preset = {
            "prompt": self.prompt_input.toPlainText(), "negative_prompt": self.neg_prompt_input.text(),
            "model": self.model_combo.currentText(), "duration": self.duration_spin.value(),
            "chunk_size": self.chunk_size_spin.value(), "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(), "variations": self.variations_spin.value(),
            "normalize": self.normalize_check.isChecked(), "trim": self.trim_check.isChecked(),
            "fade_in": self.fade_in_check.isChecked(), "fade_out": self.fade_out_check.isChecked(),
        }
        presets = self.settings.value("presets", {})
        presets[name] = preset
        self.settings.setValue("presets", presets)
        self.settings.sync()

    def load_preset(self):
        presets = self.settings.value("presets", {})
        if not presets: return
        name, ok = QInputDialog.getItem(self, "Load Preset", "Select preset:", list(presets.keys()), 0, False)
        if not ok or not name: return
        p = presets[name]
        self.prompt_input.setPlainText(p.get("prompt", ""))
        self.neg_prompt_input.setText(p.get("negative_prompt", ""))
        self.model_combo.setCurrentText(p.get("model", "cvssp/audioldm2"))
        self.duration_spin.setValue(p.get("duration", 10.0))
        self.chunk_size_spin.setValue(p.get("chunk_size", 10.0))
        self.steps_spin.setValue(p.get("steps", 200))
        self.guidance_spin.setValue(p.get("guidance", 3.5))
        self.variations_spin.setValue(p.get("variations", 1))
        self.normalize_check.setChecked(p.get("normalize", True))
        self.trim_check.setChecked(p.get("trim", True))
        self.fade_in_check.setChecked(p.get("fade_in", True))
        self.fade_out_check.setChecked(p.get("fade_out", True))

    def export_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Session", "session.json", "JSON Files (*.json)")
        if not path: return
        session = {
            "prompt": self.prompt_input.toPlainText(), "negative_prompt": self.neg_prompt_input.text(),
            "settings": {
                "model": self.model_combo.currentText(), "duration": self.duration_spin.value(),
                "chunk_size": self.chunk_size_spin.value(), "steps": self.steps_spin.value(),
                "guidance": self.guidance_spin.value(), "seed": self.seed_spin.value(),
                "variations": self.variations_spin.value(),
                "use_seed_travel": self.travel_check.isChecked(),
                "travel_start_seed": self.travel_start_seed.value(),
                "travel_end_seed": self.travel_end_seed.value(),
                "travel_steps": self.travel_steps_spin.value()
            },
            "current_audio": self.base_audio_path
        }
        try:
            with open(path, "w") as f: json.dump(session, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to export session: {e}", exc_info=True)

    def import_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Session", "", "JSON Files (*.json)")
        if not path: return
        try:
            with open(path, "r") as f: session = json.load(f)
            self.prompt_input.setPlainText(session.get("prompt", ""))
            self.neg_prompt_input.setText(session.get("negative_prompt", ""))
            s = session.get("settings", {})
            self.model_combo.setCurrentText(s.get("model", "cvssp/audioldm2"))
            self.duration_spin.setValue(s.get("duration", 10.0))
            self.chunk_size_spin.setValue(s.get("chunk_size", 10.0))
            self.steps_spin.setValue(s.get("steps", 200))
            self.guidance_spin.setValue(s.get("guidance", 3.5))
            self.seed_spin.setValue(s.get("seed", -1))
            self.variations_spin.setValue(s.get("variations", 1))
            if "use_seed_travel" in s:
                self.travel_check.setChecked(s.get("use_seed_travel", False))
                self.travel_start_seed.setValue(s.get("travel_start_seed", 100))
                self.travel_end_seed.setValue(s.get("travel_end_seed", 200))
                self.travel_steps_spin.setValue(s.get("travel_steps", 5))
            if session.get("current_audio") and os.path.exists(session["current_audio"]):
                self._load_audio_file(session["current_audio"])
        except Exception as e:
            logger.error(f"Failed to import session: {e}", exc_info=True)

    def open_docs(self):
        logger.info("Opening documentation...")

    def show_prompt_tips(self):
        logger.info("Showing prompt tips...")

    def show_about(self):
        logger.info(f"About {APP_NAME} v{APP_VERSION}")

    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export History", "history.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Favorite', 'Tags', 'Prompt', 'Duration', 'Date', 'Seed', 'Model', 'File Path'])
                for row in range(self.playlist_table.rowCount()):
                    writer.writerow([
                        self.playlist_table.item(row, 0).text() if self.playlist_table.item(row, 0) else "",
                        self.playlist_table.item(row, 1).text() if self.playlist_table.item(row, 1) else "",
                        self.playlist_table.item(row, 2).text() if self.playlist_table.item(row, 2) else "",
                        self.playlist_table.item(row, 3).text() if self.playlist_table.item(row, 3) else "",
                        self.playlist_table.item(row, 4).text() if self.playlist_table.item(row, 4) else "",
                        self.playlist_table.item(row, 5).text() if self.playlist_table.item(row, 5) else "",
                        self.playlist_table.item(row, 6).text() if self.playlist_table.item(row, 6) else "",
                        self.playlist_table.item(row, 2).toolTip()
                    ])
        except Exception as e:
            logger.error(f"Failed to export history: {e}", exc_info=True)

    def _do_filter(self, text):
        self.filter_history()

    def load_history(self):
        history_data = self.settings.value("history", [])
        if not history_data: return

        if len(history_data) > MAX_HISTORY:
            history_data = history_data[-MAX_HISTORY:]
            self.settings.setValue("history", history_data)
            self.settings.sync()

        all_tags = set()
        for entry in history_data:
            if entry.get("tags"):
                all_tags.update(entry["tags"])
        self.tag_filter_combo.clear()
        self.tag_filter_combo.addItem("All")
        self.tag_filter_combo.addItems(sorted(list(all_tags)))

        self.playlist_table.setUpdatesEnabled(False)
        self.playlist_table.setRowCount(0)
        self.playlist_table.setRowCount(len(history_data))
        try:
            for row, entry in enumerate(history_data):
                path = entry.get("path", "")
                
                fav_item = QTableWidgetItem()
                if entry.get('favorite'):
                    fav_item.setText("★")
                    fav_item.setForeground(QColor('#f1c40f'))
                else:
                    fav_item.setText("")
                fav_item.setTextAlignment(Qt.AlignCenter)
                self.playlist_table.setItem(row, 0, fav_item)
                
                tags_item = QTableWidgetItem(", ".join(entry.get('tags', [])))
                self.playlist_table.setItem(row, 1, tags_item)
                
                prompt_item = QTableWidgetItem(entry.get("prompt", ""))
                prompt_item.setToolTip(path)
                if path and not os.path.exists(path):
                    prompt_item.setForeground(QColor('red'))
                self.playlist_table.setItem(row, 2, prompt_item)
                
                mini_item = MiniWaveformItem(path, entry.get('duration', 0.0))
                self.playlist_table.setItem(row, 3, mini_item)
                if hasattr(self, '_thumb_worker') and self._thumb_worker.isRunning():
                    self._thumb_worker.enqueue(row, path)
                else:
                    QTimer.singleShot(0, lambda r=row, p=path: self._defer_thumb(r, p))
                    
                self.playlist_table.setItem(row, 4, QTableWidgetItem(entry.get("timestamp", "")[:10]))
                self.playlist_table.setItem(row, 5, QTableWidgetItem(str(entry.get("seed", ""))))
                self.playlist_table.setItem(row, 6, QTableWidgetItem(entry.get("model", "").split('/')[-1]))
        finally:
            self.playlist_table.setUpdatesEnabled(True)

    def _defer_thumb(self, row, path):
        if hasattr(self, '_thumb_worker') and self._thumb_worker.isRunning():
            self._thumb_worker.enqueue(row, path)
        else:
            pix = MiniWaveformItem.generate_pixmap(path)
            if pix:
                self._on_thumb_ready(row, pix)

    def clear_history(self):
        if QMessageBox.question(self, "Clear History", "Remove all history entries?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.playlist_table.setRowCount(0)
            self.settings.remove("history")
            self.settings.sync()
            self.tag_filter_combo.clear()
            self.tag_filter_combo.addItem("All")

    def load_settings(self):
        self.prompt_input.setPlainText(self.settings.value("prompt", "Cinematic riser and massive impact, sub bass drop"))
        self.neg_prompt_input.setText(self.settings.value("negative_prompt", ""))
        self.model_combo.setCurrentText(self.settings.value("model", "cvssp/audioldm2"))
        
        self.duration_spin.setValue(float(self.settings.value("duration", 10.0, type=float)))
        self.chunk_size_spin.setValue(float(self.settings.value("chunk_size", 10.0, type=float)))
        self.steps_spin.setValue(int(self.settings.value("steps", 200, type=int)))
        self.guidance_spin.setValue(float(self.settings.value("guidance", 3.5, type=float)))
        self.seed_spin.setValue(int(self.settings.value("seed", -1, type=int)))
        self.variations_spin.setValue(int(self.settings.value("variations", 1, type=int)))

        default_device = "cpu"
        if torch.cuda.is_available(): default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): default_device = "mps"
        self.device_combo.setCurrentText(self.settings.value("device", default_device, type=str))

        self.cache_dir_edit.setText(self.settings.value("cache_dir", DEFAULT_CACHE_DIR, type=str))
        self.output_dir_edit.setText(self.settings.value("output_dir", os.path.join(os.getcwd(), "generated_audio"), type=str))
        self.section_edit.setText(self.settings.value("section", "", type=str))

        for model in self.settings.value("custom_models", []):
            if model not in DEFAULT_MODELS and self.model_combo.findText(model) == -1:
                self.model_combo.addItem(model)

        self.normalize_check.setChecked(self.settings.value("normalize", True, type=bool))
        self.trim_check.setChecked(self.settings.value("trim", True, type=bool))
        self.fade_in_check.setChecked(self.settings.value("fade_in", True, type=bool))
        self.fade_out_check.setChecked(self.settings.value("fade_out", True, type=bool))
        self.cpu_offload_check.setChecked(self.settings.value("cpu_offload", False, type=bool))
        self.hp_spin.setValue(float(self.settings.value("high_pass_freq", 0.0, type=float)))
        self.lp_spin.setValue(float(self.settings.value("low_pass_freq", 0.0, type=float)))

        self.batch_variations_spin.setValue(int(self.settings.value("batch_variations", 1, type=int)))
        self.batch_delay_spin.setValue(float(self.settings.value("batch_delay", 1.0, type=float)))
        self.batch_continue_on_error_check.setChecked(self.settings.value("batch_continue_on_error", True, type=bool))
        self.vol_slider.setValue(int(self.settings.value("volume", 50, type=int)))

        self.travel_check.setChecked(self.settings.value("use_seed_travel", False, type=bool))
        self.travel_start_seed.setValue(int(self.settings.value("travel_start_seed", 100, type=int)))
        self.travel_end_seed.setValue(int(self.settings.value("travel_end_seed", 200, type=int)))
        self.travel_steps_spin.setValue(int(self.settings.value("travel_steps", 5, type=int)))
        
        self.autoplay_check.setChecked(self.settings.value("autoplay", True, type=bool))

        self.mode_tabs.setCurrentIndex(int(self.settings.value("left_tab", 0, type=int)))
        self.right_tabs.setCurrentIndex(int(self.settings.value("right_tab", 0, type=int)))
        self.prompt_input.history = self.settings.value("prompt_history", [])

        self.on_device_changed(self.device_combo.currentText())
        self.update_chunk_preview()
        self.load_history()

        last_audio = self.settings.value("last_audio_path", "", type=str)
        if last_audio and os.path.exists(last_audio):
            self._load_audio_file(last_audio)

    def auto_save_state(self):
        self.settings.setValue("prompt", self.prompt_input.toPlainText())
        self.settings.setValue("negative_prompt", self.neg_prompt_input.text())
        self.settings.setValue("steps", self.steps_spin.value())
        self.settings.setValue("guidance", self.guidance_spin.value())
        self.settings.setValue("seed", self.seed_spin.value())
        self.settings.setValue("duration", self.duration_spin.value())
        self.settings.setValue("chunk_size", self.chunk_size_spin.value())
        self.settings.setValue("variations", self.variations_spin.value())
        self.settings.setValue("use_seed_travel", self.travel_check.isChecked())
        self.settings.setValue("travel_start_seed", self.travel_start_seed.value())
        self.settings.setValue("travel_end_seed", self.travel_end_seed.value())
        self.settings.setValue("travel_steps", self.travel_steps_spin.value())
        self.settings.setValue("autoplay", self.autoplay_check.isChecked())
        self.settings.sync()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.isMinimized() and hasattr(self, 'gpu_mem_timer'):
                self.gpu_mem_timer.stop()
            elif hasattr(self, 'gpu_mem_timer'):
                self.gpu_mem_timer.start(GPU_MEM_POLL_MS)
        super().changeEvent(event)

    def run_demucs(self):
        if not self.base_audio_path: return
        self.demucs_btn.setEnabled(False)
        self.status_bar.showMessage("Running Demucs stem separation... This might take a while.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.demucs_worker = DemucsWorker(self.base_audio_path, device)
        self.demucs_worker.finished.connect(self.on_demucs_finished)
        self.demucs_worker.error.connect(self.on_ml_error)
        self.demucs_worker.start()

    def on_demucs_finished(self, paths):
        self.demucs_btn.setEnabled(DEMUCS_AVAILABLE and bool(self.base_audio_path))
        self.status_bar.showMessage("Demucs stems generated!", 5000)
        logger.info(f"Stems saved to: {os.path.dirname(paths[0])}")

    def run_whisper(self):
        if not self.base_audio_path: return
        self.whisper_btn.setEnabled(False)
        self.status_bar.showMessage("Running Whisper transcription...")
        self.whisper_worker = WhisperWorker(self.base_audio_path)
        self.whisper_worker.finished.connect(self.on_whisper_finished)
        self.whisper_worker.error.connect(self.on_ml_error)
        self.whisper_worker.start()

    def on_whisper_finished(self, text):
        self.whisper_btn.setEnabled(WHISPER_AVAILABLE and bool(self.base_audio_path))
        self.status_bar.showMessage("Whisper transcription complete!", 5000)
        logger.info(f"Whisper transcription: {text}")

    def run_clap(self):
        if not self.base_audio_path: return
        self.clap_btn.setEnabled(False)
        self.status_bar.showMessage("Running CLAP prompt matching...")
        prompt = self.prompt_input.toPlainText().strip()
        self.clap_worker = CLAPWorker(self.base_audio_path, prompt)
        self.clap_worker.finished.connect(self.on_clap_finished)
        self.clap_worker.error.connect(self.on_ml_error)
        self.clap_worker.start()

    def on_clap_finished(self, score):
        self.clap_btn.setEnabled(CLAP_AVAILABLE and bool(self.base_audio_path))
        self.status_bar.showMessage(f"CLAP Match Score: {score:.1f}%", 5000)
        logger.info(f"CLAP Match Score: {score:.1f}%")

    def on_ml_error(self, err):
        self.demucs_btn.setEnabled(DEMUCS_AVAILABLE and bool(self.base_audio_path))
        self.whisper_btn.setEnabled(WHISPER_AVAILABLE and bool(self.base_audio_path))
        self.clap_btn.setEnabled(CLAP_AVAILABLE and bool(self.base_audio_path))
        self.status_bar.showMessage("AI Tool Error", 3000)
        logger.error(f"AI Tool Error: {err}")

    def closeEvent(self, event):
        if hasattr(self, '_thumb_worker'):
            self._thumb_worker.stop()

        if self.worker:
            self.worker.is_cancelled = True

        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            if not self.worker_thread.wait(3000):
                logger.warning("Worker thread did not finish in time. Terminating.")
                self.worker_thread.terminate()
                self.worker_thread.wait(1000)

        for worker in [self.demucs_worker, self.whisper_worker, self.clap_worker, self.stats_worker, self.ffmpeg_worker]:
            if worker and worker.isRunning():
                worker.quit()
                worker.wait(2000)

        if hasattr(self, '_spec_worker') and self._spec_worker and self._spec_worker.isRunning():
            self._spec_worker.quit()
            self._spec_worker.wait(1000)

        AudioGenerationWorker.unload_pipeline()

        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("left_tab", self.mode_tabs.currentIndex())
        self.settings.setValue("right_tab", self.right_tabs.currentIndex())
        self.settings.setValue("prompt", self.prompt_input.toPlainText())
        self.settings.setValue("negative_prompt", self.neg_prompt_input.text())
        self.settings.setValue("model", self.model_combo.currentText())
        self.settings.setValue("duration", self.duration_spin.value())
        self.settings.setValue("chunk_size", self.chunk_size_spin.value())
        self.settings.setValue("steps", self.steps_spin.value())
        self.settings.setValue("guidance", self.guidance_spin.value())
        self.settings.setValue("seed", self.seed_spin.value())
        self.settings.setValue("variations", self.variations_spin.value())
        self.settings.setValue("device", self.device_combo.currentText())
        self.settings.setValue("cache_dir", self.cache_dir_edit.text().strip())
        self.settings.setValue("output_dir", self.output_dir_edit.text().strip())
        self.settings.setValue("section", self.section_edit.text().strip())
        self.settings.setValue("normalize", self.normalize_check.isChecked())
        self.settings.setValue("trim", self.trim_check.isChecked())
        self.settings.setValue("fade_in", self.fade_in_check.isChecked())
        self.settings.setValue("fade_out", self.fade_out_check.isChecked())
        self.settings.setValue("cpu_offload", self.cpu_offload_check.isChecked())
        self.settings.setValue("high_pass_freq", self.hp_spin.value())
        self.settings.setValue("low_pass_freq", self.lp_spin.value())
        self.settings.setValue("batch_variations", self.batch_variations_spin.value())
        self.settings.setValue("batch_delay", self.batch_delay_spin.value())
        self.settings.setValue("batch_continue_on_error", self.batch_continue_on_error_check.isChecked())
        self.settings.setValue("volume", self.vol_slider.value())
        self.settings.setValue("prompt_history", self.prompt_input.history[-MAX_PROMPT_HISTORY:])
        
        self.settings.setValue("use_seed_travel", self.travel_check.isChecked())
        self.settings.setValue("travel_start_seed", self.travel_start_seed.value())
        self.settings.setValue("travel_end_seed", self.travel_end_seed.value())
        self.settings.setValue("travel_steps", self.travel_steps_spin.value())
        
        self.settings.setValue("fx_chain", self.serialize_fx_chain())

        if self.base_audio_path and os.path.exists(self.base_audio_path):
            self.settings.setValue("last_audio_path", self.base_audio_path)
        else:
            self.settings.remove("last_audio_path")

        try: self.media_player.stop()
        except Exception: pass

        if os.path.exists(FX_PREVIEW_FILE):
            try: os.remove(FX_PREVIEW_FILE)
            except: pass
            
        if os.path.exists(TEMP_LOAD_WAV):
            try: os.remove(TEMP_LOAD_WAV)
            except: pass

        self.settings.sync()
        event.accept()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setApplicationName(APP_NAME)
        app.setApplicationVersion(APP_VERSION)
        app.setOrganizationName(SETTINGS_ORG)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        app.setPalette(palette)
        app.setStyleSheet(STYLESHEET)
        app.setFont(QFont("Segoe UI", 9))

        window = AudioLDM2Studio()
        window.show()
        
        def signal_handler(sig, frame):
            if window:
                window.close()
            app.quit()
        signal.signal(signal.SIGINT, signal_handler)
        
        timer = QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)
        
        sys.exit(app.exec())
    except Exception:
        logger.critical("Fatal Startup Error", exc_info=True)
        sys.exit(1)