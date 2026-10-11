#!/usr/bin/env python3
"""
AudioLDM2 Studio - Complete Edition
Smart Performance | Model Management | Download Manager | Batch Settings | 1000 Variations
"""

import sys
import os
import json
import time
import shutil
import logging
import math
import gc
import threading
import inspect
import signal
import wave
import random
import traceback
import faulthandler
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from collections import deque
from typing import Optional

CWD = os.getcwd()

# Create a dedicated config directory to keep the working folder clean
CONFIG_DIR = os.path.join(CWD, "audioldm2_studio_config")
os.makedirs(CONFIG_DIR, exist_ok=True)

APP_LOG_PREFIX = "audioldm2_studio"
FAULT_LOG_FILE = os.path.join(CONFIG_DIR, f"{APP_LOG_PREFIX}_faulthandler.log")

try:
    fault_log_file = open(FAULT_LOG_FILE, "w", buffering=1)
    faulthandler.enable(file=fault_log_file)
except Exception as e:
    print(f"Could not enable faulthandler: {e}")

import numpy as np

from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QUrl, QTimer, QPointF
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSlider, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter,
    QDialog, QFormLayout, QMenu, QPlainTextEdit, QSizePolicy, QFrame,
    QStatusBar, QScrollArea, QListWidget, QListWidgetItem, QTabWidget,
    QInputDialog, QButtonGroup, QRadioButton
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import (
    QColor, QPalette, QFont, QPainter, QPen,
    QMouseEvent, QKeySequence, QShortcut, QLinearGradient, QPolygonF, QAction
)

try:
    import torch
    from diffusers import AudioLDM2Pipeline
    import transformers
    import diffusers
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    torch = None
    AudioLDM2Pipeline = None

# =========================================================================
# CONSTANTS
# =========================================================================

APP_NAME = "AudioLDM2 Studio - Complete Edition"
APP_VERSION = "6.0.0-complete"
SETTINGS_ORG = "AudioLDM2"
SETTINGS_APP = "Studio"
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
LOG_FILE = os.path.join(CONFIG_DIR, f"{APP_LOG_PREFIX}.log")

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
    "Continuous ambient forest soundscape, birds, wind",
    "Evolving dark drone, low frequency rumble",
]

MAX_DURATION_S = 86400.0
LONG_FORM_THRESHOLD_S = 60.0
AUTO_CHUNK_SIZE_S = 30.0
OVERLAP_S = 2.0
MIN_CHUNK_S = 5.0

MAX_HISTORY = 5000
MAX_PROMPT_HISTORY = 50
FADE_DURATION_S = 0.05
TRIM_THRESHOLD = 0.01
ETA_SMOOTHING_ALPHA = 0.3
PROGRESS_THROTTLE_S = 0.033

# --- Performance Presets ---
PERFORMANCE_PRESETS = {
    "Custom":   {"steps": None,  "guidance": None,  "note": "Manual control"},
    "Draft":    {"steps": 20,   "guidance": 2.0,   "note": "Ultra-fast preview (~5x faster)"},
    "Speed":    {"steps": 50,   "guidance": 2.5,   "note": "Fast iteration (~3x faster)"},
    "Balanced": {"steps": 100,  "guidance": 3.5,   "note": "Good quality, moderate speed"},
    "Quality":  {"steps": 200,  "guidance": 3.5,   "note": "Best quality, slower"},
    "Extreme":  {"steps": 500,  "guidance": 4.0,   "note": "Maximum quality, very slow"},
}

# --- Smart Performance ---
ETA_HISTORY_SIZE = 30
ETA_LEARNING_FILE = os.path.join(CONFIG_DIR, "eta_learning.json")
BATCH_STATE_FILE = os.path.join(CONFIG_DIR, "batch_state.json")
VRAM_WARNING_MB = 4000
VRAM_CRITICAL_MB = 2000
DISK_SPACE_WARNING_MB = 500
LARGE_BATCH_THRESHOLD = 10
AGGRESSIVE_GC_THRESHOLD = 50

# --- Model Management ---
MODEL_CACHE_MAX = 2
MODEL_PROFILES_FILE = os.path.join(CONFIG_DIR, "model_profiles.json")
MODEL_FAVORITES_FILE = os.path.join(CONFIG_DIR, "model_favorites.json")

MODEL_VRAM_ESTIMATES = {
    "cvssp/audioldm2": 3500,
    "cvssp/audioldm2-music": 3500,
    "cvssp/audioldm2-gigaspeech": 3500,
    "cvssp/audioldm2-ljspeech": 3500,
}
MODEL_DESCRIPTIONS = {
    "cvssp/audioldm2": "General-purpose audio generation (SFX, ambience, music)",
    "cvssp/audioldm2-music": "Music-focused generation (instruments, compositions)",
    "cvssp/audioldm2-gigaspeech": "Speech-focused generation (voice, dialogue)",
    "cvssp/audioldm2-ljspeech": "Single-speaker TTS (LJSpeech dataset)",
}
MODEL_DEFAULT_PROFILES = {
    "cvssp/audioldm2": {"steps": 200, "guidance": 3.5, "duration": 10.0},
    "cvssp/audioldm2-music": {"steps": 200, "guidance": 3.0, "duration": 10.0},
    "cvssp/audioldm2-gigaspeech": {"steps": 150, "guidance": 4.0, "duration": 10.0},
    "cvssp/audioldm2-ljspeech": {"steps": 150, "guidance": 3.5, "duration": 10.0},
}

# --- Download System ---
DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_RETRY_BASE_DELAY = 2
DOWNLOAD_SPEED_WINDOW = 15
DOWNLOAD_MONITOR_INTERVAL = 0.5
DOWNLOAD_SETTINGS_FILE = os.path.join(CONFIG_DIR, "download_settings.json")
DOWNLOAD_LOG_FILE = os.path.join(CONFIG_DIR, "download_log.json")

try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    HF_TRANSFER_AVAILABLE = True
except ImportError:
    HF_TRANSFER_AVAILABLE = False

DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

DOWNLOAD_FILE_EXTENSIONS = (
    '.safetensors', '.bin', '.json', '.txt', '.model',
    '.tiktoken', '.vocab', '.merges', '.jsonl', '.zip',
    '.npz', '.npy', '.wav', '.flac',
)

# --- Batch System ---
BATCH_CONFIG_FILE = os.path.join(CONFIG_DIR, "batch_config.json")
BATCH_TEMPLATES_FILE = os.path.join(CONFIG_DIR, "batch_templates.json")
BATCH_STATS_FILE = os.path.join(CONFIG_DIR, "batch_stats.json")
SETTINGS_FILE_NAME = "settings.json"

SEED_ALGORITHMS_ADVANCED = {
    "Sequential":     "Each variation: base_seed + i * step",
    "Random":         "Completely random seed per variation",
    "Fixed":          "Same seed for all variations",
    "Golden Ratio":   "Golden ratio spacing for natural variation",
    "Interpolated":   "Linear interpolation between two seed endpoints",
    "Geometric":      "Geometric progression (multiply by ratio)",
    "Harmonic":       "Harmonic series spacing (1, 1/2, 1/3, ...)",
    "Noise Walk":     "Random walk with bounded step (smooth changes)",
    "Linspace":       "Evenly distributed seeds from 0 to max",
    "Birthday":       "Random but guaranteed unique (no collisions)",
    "Perlin":         "Perlin noise-based (organic transitions)",
    "Fibonacci":      "Fibonacci sequence-based spacing",
}

OUTPUT_ORG_MODES = {
    "flat":       "All files in output directory",
    "by_date":    "Subfolder per date (YYYY-MM-DD)",
    "by_prompt":  "Subfolder per prompt",
    "by_model":   "Subfolder per model",
    "by_batch":   "Subfolder per batch ID",
    "by_tags":    "Subfolder per tag",
    "custom":     "Custom pattern",
}

BATCH_PRESETS = {
    "Quick Preview": {"num_variations": 10, "performance_mode": "Draft", "duration": 5.0, "seed_algo": "Random", "skip_on_error": True, "early_preview": True, "aggressive_gc": False},
    "Sample Pack": {"num_variations": 20, "performance_mode": "Speed", "duration": 10.0, "seed_algo": "Random", "skip_on_error": True, "early_preview": True, "aggressive_gc": False},
    "Production Run": {"num_variations": 50, "performance_mode": "Quality", "duration": 10.0, "seed_algo": "Golden Ratio", "skip_on_error": True, "early_preview": True, "aggressive_gc": False},
    "Mass Generation": {"num_variations": 100, "performance_mode": "Balanced", "duration": 10.0, "seed_algo": "Golden Ratio", "skip_on_error": True, "early_preview": True, "aggressive_gc": True},
    "Mega Batch": {"num_variations": 500, "performance_mode": "Speed", "duration": 5.0, "seed_algo": "Noise Walk", "skip_on_error": True, "early_preview": True, "aggressive_gc": True},
    "Ultra Batch": {"num_variations": 1000, "performance_mode": "Draft", "duration": 5.0, "seed_algo": "Noise Walk", "skip_on_error": True, "early_preview": True, "aggressive_gc": True},
}

AUTO_ADJUST_THRESHOLDS = {
    "reduce_steps": 50, "speed_mode": 100, "draft_mode": 500,
    "aggressive_gc": 50, "skip_errors": 20, "early_preview": 3,
}

NOTIFICATION_OPTIONS = {
    "none": "No notification", "sound": "Play a sound",
    "visual": "Flash window and show dialog", "system": "System tray", "all": "All notifications",
}

# =========================================================================
# STYLESHEET
# =========================================================================

STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QGroupBox { border: 1px solid #3c3c3c; border-radius: 4px; margin-top: 10px; padding-top: 10px; font-weight: bold; color: #cccccc; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px; background-color: #2b2b2b; }
QPushButton { background-color: #3c3c3c; border: 1px solid #4a4a4a; color: #e0e0e0; padding: 6px 12px; border-radius: 3px; min-height: 20px; }
QPushButton:hover { background-color: #4a4a4a; border-color: #2A82DA; }
QPushButton:pressed { background-color: #2b2b2b; }
QPushButton:disabled { background-color: #333333; color: #666666; border-color: #333333; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit { background-color: #333333; border: 1px solid #3c3c3c; padding: 4px 6px; border-radius: 2px; color: #e0e0e0; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus { border: 1px solid #2A82DA; }
QComboBox QAbstractItemView { background-color: #333333; border: 1px solid #3c3c3c; selection-background-color: #2A82DA; color: #e0e0e0; }
QTableWidget { background-color: #2b2b2b; alternate-background-color: #333333; border: 1px solid #3c3c3c; gridline-color: #3c3c3c; }
QHeaderView::section { background-color: #3c3c3c; padding: 6px; border: none; font-weight: bold; color: #cccccc; }
QProgressBar { border: 1px solid #3c3c3c; border-radius: 2px; text-align: center; background-color: #333333; color: white; min-height: 22px; }
QProgressBar::chunk { background-color: #2A82DA; border-radius: 1px; }
QScrollBar:vertical { background: #2b2b2b; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #4a4a4a; min-height: 20px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #2A82DA; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QCheckBox { color: #e0e0e0; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border-radius: 2px; border: 1px solid #4a4a4a; background: #333333; }
QCheckBox::indicator:checked { background: #2A82DA; border-color: #2A82DA; }
QLabel { color: #d0d0d0; }
QStatusBar { background-color: #2b2b2b; color: #aaaaaa; }
QSlider::groove:horizontal { border: 1px solid #3c3c3c; height: 4px; background: #333333; border-radius: 2px; }
QSlider::handle:horizontal { background: #2A82DA; border: 1px solid #2A82DA; width: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: #3a92ea; }
QSplitter::handle { background-color: #3c3c3c; }
QSplitter::handle:horizontal { width: 2px; }
QListWidget { background-color: #2b2b2b; border: 1px solid #3c3c3c; color: #e0e0e0; }
QListWidget::item { padding: 4px 6px; }
QListWidget::item:selected { background-color: #2A82DA; color: white; }
QTabWidget::pane { border: 1px solid #3c3c3c; background-color: #2b2b2b; }
QTabBar::tab { background-color: #333333; color: #aaaaaa; padding: 8px 16px; border: 1px solid #3c3c3c; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background-color: #2b2b2b; color: #2A82DA; border-bottom: 2px solid #2A82DA; }
QTabBar::tab:hover { background-color: #3c3c3c; }
"""

# =========================================================================
# LOGGING
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s] %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(APP_NAME)

# =========================================================================
# GPT2LMHeadModel PATCH
# =========================================================================

try:
    if ML_AVAILABLE:
        from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
        import torch.nn as nn

        _orig_fp = AudioLDM2Pipeline.from_pretrained

        @classmethod
        def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            cache_dir = kwargs.get("cache_dir")
            torch_dtype = kwargs.get("torch_dtype")
            pipe = _orig_fp.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
            lm = getattr(pipe, "language_model", None)
            if lm is not None and isinstance(lm, GPT2Model) and not isinstance(lm, GPT2LMHeadModel):
                logger.info("AudioLDM2: Reloading language_model as GPT2LMHeadModel.")
                try:
                    reload_kwargs = {"subfolder": "language_model", "cache_dir": cache_dir}
                    if torch_dtype is not None:
                        reload_kwargs["torch_dtype"] = torch_dtype
                    new_lm = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, **reload_kwargs)
                    if hasattr(lm, "device"):
                        new_lm = new_lm.to(lm.device)
                    pipe.language_model = new_lm
                    logger.info("AudioLDM2: GPT2LMHeadModel loaded successfully.")
                except Exception as e_reload:
                    logger.warning(f"AudioLDM2: Reload failed ({e_reload}), using wrapper.")
                    try:
                        new_lm = GPT2LMHeadModel(lm.config)
                        new_lm.transformer = lm
                        new_lm = new_lm.to(lm.device)
                        if torch_dtype is not None:
                            new_lm = new_lm.to(torch_dtype)
                        pipe.language_model = new_lm
                    except Exception as e_wrap:
                        logger.error(f"AudioLDM2: Wrapper fallback failed: {e_wrap}")
            return pipe

        AudioLDM2Pipeline.from_pretrained = _patched_from_pretrained
        print("✅ Expert patch applied: GPT2Model → GPT2LMHeadModel conversion.")
except Exception as e:
    print(f"⚠️ Could not apply GPT2LMHeadModel patch: {e}")

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def sanitize_filename(text: str, max_length: int = 40) -> str:
    """Convert arbitrary text into a safe, readable filename fragment."""
    if not text:
        return "audio"
    # Normalize unicode (e.g., é → e), keep alphanumerics + separators
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    safe = "".join(c for c in text if c.isalnum() or c in (' ', '_', '-')).strip()
    # Collapse multiple spaces/underscores into single hyphen
    safe = re.sub(r"[\s_]+", "-", safe)
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    # Truncate on word boundary
    if len(safe) > max_length:
        cut = safe[:max_length].rsplit("-", 1)[0] or safe[:max_length]
        safe = cut
    return safe.lower() or "audio"


def format_time(seconds) -> str:
    try:
        total_secs = int(float(seconds))
        if total_secs < 0:
            total_secs = 0
        minutes, secs = divmod(total_secs, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"
    except Exception:
        return "00:00"


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    if audio.size == 0:
        return audio.copy()
    audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak == 0:
        return audio
    target_peak = 10 ** (target_db / 20.0)
    return audio * (target_peak / peak)


def apply_fade(audio: np.ndarray, sample_rate: int, fade_in_s: float = 0.05, fade_out_s: float = 0.05) -> np.ndarray:
    audio = audio.astype(np.float32, copy=True)
    if audio.size == 0:
        return audio
    fi = int(fade_in_s * sample_rate)
    fo = int(fade_out_s * sample_rate)
    if fi > 0:
        fi = min(fi, audio.size)
        audio[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        fo = min(fo, audio.size)
        audio[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return audio


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if audio.size == 0:
        return audio.copy()
    audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    above = np.where(np.abs(audio) > threshold)[0]
    return audio[above[0]:above[-1] + 1] if above.size > 0 else audio


SEED_ALGORITHMS = set(SEED_ALGORITHMS_ADVANCED.keys())


def parse_multi_prompt_line(line: str, default_seed_algo: str, default_duration: float) -> dict:
    parts = [p.strip() for p in line.split('|')]
    prompt = parts[0].strip()
    seed_algo = default_seed_algo
    duration = default_duration
    for idx in range(1, min(len(parts), 3)):
        val = parts[idx]
        if not val:
            continue
        if val in SEED_ALGORITHMS:
            seed_algo = val
        else:
            try:
                d = float(val)
                if d > 0:
                    duration = min(d, MAX_DURATION_S)
            except ValueError:
                pass
    return {"prompt": prompt, "seed_algo": seed_algo, "duration": duration}


def compute_chunks_for_duration(duration: float) -> list:
    if duration <= LONG_FORM_THRESHOLD_S:
        return [max(MIN_CHUNK_S, duration)]
    effective = AUTO_CHUNK_SIZE_S - OVERLAP_S
    n = math.ceil((duration - OVERLAP_S) / effective)
    last_len = duration - (n - 1) * effective
    if last_len < MIN_CHUNK_S and n > 1:
        n -= 1
    chunks = []
    for i in range(n):
        if i == n - 1:
            cl = max(MIN_CHUNK_S, duration - (n - 1) * effective)
        else:
            cl = AUTO_CHUNK_SIZE_S
        chunks.append(cl)
    return chunks


def format_bytes(bytes_val: float) -> str:
    if bytes_val == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_val) < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


# =========================================================================
# SMART PERFORMANCE HELPERS
# =========================================================================

def get_gpu_info():
    if not ML_AVAILABLE or not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        total_mb = props.total_memory / (1024 * 1024)
        allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        return {"name": props.name, "total_mb": total_mb, "free_mb": total_mb - allocated_mb, "allocated_mb": allocated_mb}
    except Exception:
        return None


def get_free_disk_mb(path):
    try:
        return shutil.disk_usage(path).free / (1024 * 1024)
    except Exception:
        return None


def load_eta_history():
    try:
        if os.path.exists(ETA_LEARNING_FILE):
            with open(ETA_LEARNING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {"records": [], "last_updated": 0}


def save_eta_history(history):
    try:
        history["last_updated"] = time.time()
        with open(ETA_LEARNING_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save ETA history: {e}")


def record_eta_point(model_name, device, steps, duration, chunks, gen_time):
    hist = load_eta_history()
    records = hist.get("records", [])
    sps = (steps * chunks) / gen_time if gen_time > 0 else 0
    records.append({"model": model_name, "device": device, "steps": steps, "duration": duration, "chunks": chunks, "gen_time": gen_time, "steps_per_sec": sps, "timestamp": time.time()})
    if len(records) > ETA_HISTORY_SIZE * 3:
        records = records[-ETA_HISTORY_SIZE * 3:]
    hist["records"] = records
    save_eta_history(hist)


def get_learned_steps_per_sec(model_name, device):
    hist = load_eta_history()
    records = [r for r in hist.get("records", []) if r["model"] == model_name and r["device"] == device]
    if not records:
        return None
    recent = records[-ETA_HISTORY_SIZE:]
    sps = sorted([r["steps_per_sec"] for r in recent if r["steps_per_sec"] > 0])
    return sps[len(sps) // 2] if sps else None


def load_batch_state():
    try:
        if os.path.exists(BATCH_STATE_FILE):
            with open(BATCH_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def save_batch_state(state):
    try:
        with open(BATCH_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save batch state: {e}")


def clear_batch_state():
    try:
        if os.path.exists(BATCH_STATE_FILE):
            os.remove(BATCH_STATE_FILE)
    except Exception:
        pass


def estimate_batch_time(num_variations, steps, chunks, learned_sps=None):
    if learned_sps and learned_sps > 0:
        return (steps * chunks / learned_sps) * num_variations
    return steps * chunks * 0.15 * num_variations


def estimate_disk_usage_mb(duration, num_variations, sample_rate=16000):
    return (duration * sample_rate * 2 * num_variations) / (1024 * 1024)


# =========================================================================
# DOWNLOAD HELPERS
# =========================================================================

def load_download_settings():
    try:
        if os.path.exists(DOWNLOAD_SETTINGS_FILE):
            with open(DOWNLOAD_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {"max_retries": DOWNLOAD_MAX_RETRIES, "use_hf_transfer": HF_TRANSFER_AVAILABLE, "hf_endpoint": DEFAULT_HF_ENDPOINT, "auto_download": False, "max_parallel": 4, "skip_errors": True}


def save_download_settings(settings):
    try:
        with open(DOWNLOAD_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save download settings: {e}")


def load_download_log():
    try:
        if os.path.exists(DOWNLOAD_LOG_FILE):
            with open(DOWNLOAD_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def add_to_download_log(model_name, status, file_count, total_size, duration, error=""):
    log = load_download_log()
    log.append({"model": model_name, "status": status, "files": file_count, "size_bytes": total_size, "duration_s": duration, "error": error, "timestamp": datetime.now().isoformat()})
    log = log[-100:]
    try:
        with open(DOWNLOAD_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass


def get_model_cache_size(cache_dir, model_name):
    try:
        cache_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
        if not os.path.exists(cache_path):
            return 0
        total = 0
        for dirpath, _, filenames in os.walk(cache_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return total
    except Exception:
        return 0


def delete_model_cache(cache_dir, model_name):
    try:
        cache_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            return True
    except Exception as e:
        logger.error(f"Failed to delete model cache for {model_name}: {e}")
    return False


# =========================================================================
# MODEL CACHE
# =========================================================================

class ModelCache:
    def __init__(self, max_models=MODEL_CACHE_MAX):
        self._cache = {}
        self._max_models = max_models
        self._lock = threading.Lock()

    def get(self, model_name, device, offload):
        key = (model_name, device, offload)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry["last_used"] = time.time()
                logger.info(f"ModelCache HIT: {model_name}")
                return entry["pipe"]
            return None

    def put(self, model_name, device, offload, pipe, vram_mb=0):
        key = (model_name, device, offload)
        with self._lock:
            while len(self._cache) >= self._max_models:
                lru_key = min(self._cache, key=lambda k: self._cache[k]["last_used"])
                old = self._cache.pop(lru_key)
                self._unload(old["pipe"])
            self._cache[key] = {"pipe": pipe, "last_used": time.time(), "vram_mb": vram_mb}
            logger.info(f"ModelCache PUT: {model_name} (cache: {len(self._cache)})")

    def evict(self, model_name, device=None, offload=None):
        with self._lock:
            for key in list(self._cache):
                if key[0] == model_name and (device is None or key[1] == device) and (offload is None or key[2] == offload):
                    self._unload(self._cache.pop(key)["pipe"])

    def clear(self):
        with self._lock:
            for key, entry in list(self._cache.items()):
                self._unload(entry["pipe"])
            self._cache.clear()

    def list_loaded(self):
        with self._lock:
            return [{"model": k[0], "device": k[1], "offload": k[2], "last_used": v["last_used"], "vram_mb": v["vram_mb"]} for k, v in self._cache.items()]

    def is_cached(self, model_name, device=None, offload=None):
        with self._lock:
            for key in self._cache:
                if key[0] == model_name and (device is None or key[1] == device) and (offload is None or key[2] == offload):
                    return True
            return False

    def set_max_models(self, n):
        with self._lock:
            self._max_models = n
            while len(self._cache) > n:
                lru = min(self._cache, key=lambda k: self._cache[k]["last_used"])
                self._unload(self._cache.pop(lru)["pipe"])

    @staticmethod
    def _unload(pipe):
        try:
            if hasattr(pipe, "remove_hooks"):
                pipe.remove_hooks()
        except Exception:
            pass
        try:
            if ML_AVAILABLE and torch.cuda.is_available():
                pipe.to("cpu")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        del pipe
        gc.collect()


model_cache = ModelCache()


# =========================================================================
# MODEL PROFILE MANAGER
# =========================================================================

class ModelProfileManager:
    def __init__(self):
        self.profiles = self._load()

    def _load(self):
        try:
            if os.path.exists(MODEL_PROFILES_FILE):
                with open(MODEL_PROFILES_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save(self):
        try:
            with open(MODEL_PROFILES_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save model profiles: {e}")

    def get_profile(self, model_name):
        return self.profiles.get(model_name, MODEL_DEFAULT_PROFILES.get(model_name, {"steps": 200, "guidance": 3.5, "duration": 10.0}))

    def save_profile(self, model_name, steps, guidance, duration, seed_algo="Sequential", performance_mode="Custom"):
        self.profiles[model_name] = {"steps": steps, "guidance": guidance, "duration": duration, "seed_algo": seed_algo, "performance_mode": performance_mode, "last_used": time.time()}
        self._save()

    def apply_profile(self, model_name, window):
        p = self.get_profile(model_name)
        if "steps" in p:
            window.steps_spin.setValue(p["steps"])
        if "guidance" in p:
            window.guidance_spin.setValue(p["guidance"])
        if "duration" in p:
            window.duration_spin.setValue(p["duration"])
        if "seed_algo" in p:
            window.seed_algo_combo.setCurrentText(p.get("seed_algo", "Sequential"))
        if "performance_mode" in p:
            window.perf_mode_combo.setCurrentText(p.get("performance_mode", "Custom"))


profile_manager = ModelProfileManager()


# =========================================================================
# GENERATION PARAMS
# =========================================================================

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
    use_cpu_offload: bool = False
    section: str = ""
    use_multi_prompt: bool = False
    multi_prompts: str = ""
    seed_algo: str = "Sequential"
    performance_mode: str = "Custom"
    skip_on_error: bool = True
    resume_from_variation: int = 0
    batch_id: str = ""
    early_preview: bool = True
    aggressive_gc: bool = False


# =========================================================================
# BATCH CONFIG
# =========================================================================

@dataclass
class BatchConfig:
    num_variations: int = 1
    seed_algo: str = "Sequential"
    seed_start: int = -1
    seed_end: int = 0
    seed_step: int = 7919
    seed_ratio: float = 1.61803398875
    seed_walk_max: int = 1000000
    filename_pattern: str = "{date}_{time}_{prompt}_v{variation:04d}_s{seed}_{seed_algo}.wav"
    output_org_mode: str = "flat"
    output_org_pattern: str = "{date}/{prompt}"
    performance_mode: str = "Custom"
    auto_adjust: bool = True
    auto_adjust_aggressiveness: str = "moderate"
    skip_on_error: bool = True
    auto_retry_failed: bool = False
    auto_retry_count: int = 1
    early_preview: bool = True
    aggressive_gc: bool = False
    auto_enable_gc_threshold: int = 50
    notification_mode: str = "visual"
    tags: str = ""
    auto_export_zip: bool = False
    zip_name_pattern: str = "{date}_{prompt}_batch.zip"
    auto_filter_enabled: bool = False
    min_duration_s: float = 0.5
    min_rms_db: float = -60.0
    time_limit_enabled: bool = False
    time_limit_hours: float = 24.0
    auto_save_progress: bool = True
    track_stats: bool = True

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def save(self):
        try:
            with open(BATCH_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save batch config: {e}")

    @classmethod
    def load(cls):
        try:
            if os.path.exists(BATCH_CONFIG_FILE):
                with open(BATCH_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return cls.from_dict(json.load(f))
        except Exception:
            pass
        return cls()


# =========================================================================
# SEED STRATEGIES
# =========================================================================

def compute_batch_seeds(base_seed, num_variations, algo, config=None):
    if config is None:
        config = BatchConfig()
    seeds = []
    if algo == "Fixed":
        return [base_seed] * num_variations
    elif algo == "Sequential":
        for i in range(num_variations):
            seeds.append((base_seed + i * config.seed_step) % (2 ** 31))
    elif algo == "Random":
        for _ in range(num_variations):
            seeds.append(random.randint(0, 2 ** 31 - 1))
    elif algo == "Golden Ratio":
        phi = 1.61803398875
        for i in range(num_variations):
            seeds.append(int((base_seed + i * phi * 1000000) % (2 ** 31)))
    elif algo == "Interpolated":
        start = base_seed
        end = config.seed_end if config.seed_end > 0 else base_seed + num_variations * 7919
        for i in range(num_variations):
            t = i / max(num_variations - 1, 1)
            seeds.append(int(start + (end - start) * t) % (2 ** 31))
    elif algo == "Geometric":
        for i in range(num_variations):
            seeds.append(int((base_seed * (config.seed_ratio ** i)) % (2 ** 31)))
    elif algo == "Harmonic":
        for i in range(num_variations):
            h = sum(1 / (k + 1) for k in range(i + 1))
            seeds.append(int((base_seed + h * 10000000) % (2 ** 31)))
    elif algo == "Noise Walk":
        current = base_seed
        for i in range(num_variations):
            if i > 0:
                current = (current + random.randint(-config.seed_walk_max, config.seed_walk_max)) % (2 ** 31)
            seeds.append(current)
    elif algo == "Linspace":
        for i in range(num_variations):
            seeds.append(int((2 ** 31 - 1) * i / max(num_variations - 1, 1)) % (2 ** 31))
    elif algo == "Birthday":
        used = set()
        while len(seeds) < num_variations:
            s = random.randint(0, 2 ** 31 - 1)
            if s not in used:
                used.add(s)
                seeds.append(s)
    elif algo == "Perlin":
        for i in range(num_variations):
            t = i / max(num_variations - 1, 1)
            noise = (math.sin(t * 2 * math.pi * 3) * 0.3 + math.sin(t * 2 * math.pi * 7) * 0.2 + math.sin(t * 2 * math.pi * 13) * 0.1 + t * 0.4)
            noise = (noise + 1) / 2
            seeds.append(int(noise * (2 ** 31 - 1)))
    elif algo == "Fibonacci":
        fib = [1, 1]
        while len(fib) < num_variations + 2:
            fib.append(fib[-1] + fib[-2])
        for i in range(num_variations):
            seeds.append(int((base_seed + fib[i + 1] * 1000) % (2 ** 31)))
    else:
        for i in range(num_variations):
            seeds.append((base_seed + i * 7919) % (2 ** 31))
    return seeds


# =========================================================================
# FILENAME PATTERN ENGINE
# =========================================================================

def render_filename_pattern(pattern, params, variation, seed, config, batch_id="", index=0):
    now = datetime.now()
    replacements = {
        "{prompt}": sanitize_filename(params.prompt, 50),
        "{short_prompt}": sanitize_filename(params.prompt, 20),
        "{seed}": str(seed),
        "{variation}": str(variation + 1),
        "{variation:04d}": f"{variation + 1:04d}",
        "{date}": now.strftime("%Y-%m-%d"),
        "{time}": now.strftime("%H-%M-%S"),
        "{timestamp}": str(int(time.time())),
        "{iso}": now.strftime("%Y%m%dT%H%M%S"),
        "{year}": now.strftime("%Y"),
        "{month}": now.strftime("%m"),
        "{day}": now.strftime("%d"),
        "{hour}": now.strftime("%H"),
        "{minute}": now.strftime("%M"),
        "{second}": now.strftime("%S"),
        "{model}": params.model_name.replace("/", "_"),
        "{model_short}": params.model_name.split("/")[-1],
        "{device}": params.device,
        "{steps}": str(params.steps),
        "{guidance}": f"{params.guidance}",
        "{guidance:.1f}": f"{params.guidance:.1f}",
        "{tags}": sanitize_filename(config.tags, 30) if config.tags else "untagged",
        "{batch_id}": batch_id[:8] if batch_id else "nobatch",
        "{index}": str(index),
        "{counter}": f"{index:04d}",
        "{seed_algo}": sanitize_filename(params.seed_algo, 20).lower(),
    }
    for n in re.findall(r'\{prompt:(\d+)\}', pattern):
        replacements[f"{{prompt:{n}}}"] = sanitize_filename(params.prompt, int(n))
    result = pattern
    for key, val in replacements.items():
        result = result.replace(key, val)
    if not result.endswith('.wav'):
        result += '.wav'
    result = "".join(c for c in result if c.isalnum() or c in ('_', '-', '.')).rstrip()
    return result if result else f"audio_{variation + 1:04d}.wav"


def get_output_subfolder(org_mode, params, config, batch_id=""):
    now = datetime.now()
    if org_mode == "flat":
        return ""
    elif org_mode == "by_date":
        return now.strftime("%Y-%m-%d")
    elif org_mode == "by_prompt":
        return sanitize_filename(params.prompt, 30)
    elif org_mode == "by_model":
        return params.model_name.split("/")[-1]
    elif org_mode == "by_batch":
        return batch_id[:8] if batch_id else "nobatch"
    elif org_mode == "by_tags":
        return sanitize_filename(config.tags, 30) if config.tags else "untagged"
    elif org_mode == "custom":
        sub = render_filename_pattern(config.output_org_pattern, params, 0, 0, config, batch_id)
        return sub.replace(".wav", "")
    return ""


# =========================================================================
# BATCH TEMPLATE MANAGER
# =========================================================================

class BatchTemplateManager:
    def __init__(self):
        self.templates = self._load()

    def _load(self):
        try:
            if os.path.exists(BATCH_TEMPLATES_FILE):
                with open(BATCH_TEMPLATES_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save(self):
        try:
            with open(BATCH_TEMPLATES_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save batch templates: {e}")

    def save_template(self, name, config, params):
        self.templates[name] = {"config": config.to_dict(), "params": {"prompt": params.prompt, "negative_prompt": params.negative_prompt, "model_name": params.model_name, "duration": params.duration, "steps": params.steps, "guidance": params.guidance, "seed": params.seed}, "created": datetime.now().isoformat()}
        self._save()

    def load_template(self, name):
        if name not in self.templates:
            return None, None
        t = self.templates[name]
        return BatchConfig.from_dict(t["config"]), t["params"]

    def delete_template(self, name):
        if name in self.templates:
            del self.templates[name]
            self._save()

    def list_templates(self):
        return sorted(self.templates.keys())

    def get_template_info(self, name):
        if name not in self.templates:
            return {}
        t = self.templates[name]
        c = t["config"]
        p = t.get("params", {})
        return {"name": name, "variations": c.get("num_variations", 1), "mode": c.get("performance_mode", "Custom"), "duration": p.get("duration", 10.0), "steps": p.get("steps", 200), "seed_algo": c.get("seed_algo", "Sequential"), "created": t.get("created", "")}


batch_template_manager = BatchTemplateManager()


# =========================================================================
# BATCH STATS TRACKER
# =========================================================================

class BatchStatsTracker:
    def __init__(self):
        self.stats = self._load()

    def _load(self):
        try:
            if os.path.exists(BATCH_STATS_FILE):
                with open(BATCH_STATS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {"batches": [], "summary": {}}

    def _save(self):
        try:
            with open(BATCH_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save batch stats: {e}")

    def record_batch(self, batch_id, model, device, num_variations, num_completed, num_failed, num_skipped, total_time, avg_per_file, total_disk_mb, steps, duration, seed_algo, performance_mode, prompt):
        entry = {"batch_id": batch_id, "model": model, "device": device, "num_variations": num_variations, "num_completed": num_completed, "num_failed": num_failed, "num_skipped": num_skipped, "success_rate": (num_completed / max(num_variations, 1)) * 100, "total_time_s": total_time, "avg_per_file_s": avg_per_file, "total_disk_mb": total_disk_mb, "steps": steps, "duration": duration, "seed_algo": seed_algo, "performance_mode": performance_mode, "prompt": prompt[:100], "timestamp": datetime.now().isoformat()}
        self.stats["batches"].append(entry)
        if len(self.stats["batches"]) > 500:
            self.stats["batches"] = self.stats["batches"][-500:]
        self._update_summary()
        self._save()

    def _update_summary(self):
        batches = self.stats["batches"]
        if not batches:
            self.stats["summary"] = {}
            return
        total_files = sum(b["num_completed"] for b in batches)
        total_failed = sum(b["num_failed"] for b in batches)
        total_time = sum(b["total_time_s"] for b in batches)
        total_disk = sum(b["total_disk_mb"] for b in batches)
        self.stats["summary"] = {"total_batches": len(batches), "total_files_generated": total_files, "total_files_failed": total_failed, "overall_success_rate": (total_files / max(total_files + total_failed, 1)) * 100, "total_generation_time_s": total_time, "total_disk_usage_mb": total_disk, "avg_batch_size": total_files / max(len(batches), 1), "avg_time_per_file_s": total_time / max(total_files, 1)}

    def get_summary(self):
        return self.stats.get("summary", {})

    def get_recent_batches(self, n=20):
        return self.stats["batches"][-n:]

    def clear(self):
        self.stats = {"batches": [], "summary": {}}
        self._save()


batch_stats = BatchStatsTracker()


# =========================================================================
# AUTO-ADJUST
# =========================================================================

def auto_adjust_batch_settings(config, params, vram_mb=None):
    config = BatchConfig.from_dict(config.to_dict())
    changes = []
    if not config.auto_adjust:
        return config, params, []
    n = config.num_variations
    factor = {"conservative": 2, "moderate": 1, "aggressive": 0.5}.get(config.auto_adjust_aggressiveness, 1)
    if n >= AUTO_ADJUST_THRESHOLDS["skip_errors"] and not config.skip_on_error:
        config.skip_on_error = True
        changes.append("Auto-enabled Skip on Error")
    if n >= AUTO_ADJUST_THRESHOLDS["early_preview"] and not config.early_preview:
        config.early_preview = True
        changes.append("Auto-enabled Early Preview")
    if n >= AUTO_ADJUST_THRESHOLDS["aggressive_gc"] and not config.aggressive_gc:
        config.aggressive_gc = True
        changes.append("Auto-enabled Aggressive GC")
    if n >= AUTO_ADJUST_THRESHOLDS["reduce_steps"] * factor:
        if config.performance_mode == "Custom" and params.steps > 100:
            old = params.steps
            params.steps = max(50, int(params.steps * 0.5))
            changes.append(f"Reduced steps: {old} → {params.steps}")
    if n >= AUTO_ADJUST_THRESHOLDS["speed_mode"] * factor:
        if config.performance_mode in ("Custom", "Quality", "Balanced"):
            config.performance_mode = "Speed"
            p = PERFORMANCE_PRESETS["Speed"]
            params.steps = p["steps"]; params.guidance = p["guidance"]
            changes.append("Switched to Speed mode")
    if n >= AUTO_ADJUST_THRESHOLDS["draft_mode"] * factor:
        if config.performance_mode != "Draft":
            config.performance_mode = "Draft"
            p = PERFORMANCE_PRESETS["Draft"]
            params.steps = p["steps"]; params.guidance = p["guidance"]
            changes.append("Switched to Draft mode")
    if vram_mb is not None and vram_mb < VRAM_CRITICAL_MB:
        if not params.use_cpu_offload:
            params.use_cpu_offload = True
            changes.append("Enabled CPU Offload (low VRAM)")
    return config, params, changes


# =========================================================================
# DOWNLOAD TASK & THREAD
# =========================================================================

@dataclass
class DownloadTask:
    model_name: str
    cache_dir: str
    priority: int = 0
    status: str = "pending"


class EnhancedModelDownloadThread(QThread):
    progress_updated = Signal(int)
    speed_updated = Signal(str)
    eta_updated = Signal(str)
    status_updated = Signal(str)
    bytes_updated = Signal(int, int)
    file_started = Signal(str, str)
    file_completed = Signal(str)
    file_failed = Signal(str, str)
    download_completed = Signal(str, bool, str)

    def __init__(self, model_name, cache_dir, parent=None, max_retries=DOWNLOAD_MAX_RETRIES, skip_errors=True, use_hf_transfer=HF_TRANSFER_AVAILABLE, hf_endpoint=DEFAULT_HF_ENDPOINT, max_parallel=4):
        super().__init__(parent)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._cancel = False
        self._max_retries = max_retries
        self._skip_errors = skip_errors
        self._use_hf_transfer = use_hf_transfer
        self._hf_endpoint = hf_endpoint
        self._max_parallel = max_parallel
        self._speed_samples = deque(maxlen=DOWNLOAD_SPEED_WINDOW)
        self._total_bytes = 0
        self._downloaded_bytes = 0
        self._monitor_active = False
        self._start_time = 0.0

    def cancel(self):
        self._cancel = True
        self._monitor_active = False

    def run(self):
        self._start_time = time.time()
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["HF_HUB_CACHE"] = self.cache_dir
            if self._use_hf_transfer:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            if self._hf_endpoint and self._hf_endpoint != DEFAULT_HF_ENDPOINT:
                os.environ["HF_ENDPOINT"] = self._hf_endpoint

            if self._is_already_downloaded():
                self.progress_updated.emit(100)
                self.download_completed.emit(self.model_name, True, "Already downloaded")
                return

            self.status_updated.emit("Fetching model file list...")
            files_with_sizes = self._get_file_list()
            if not files_with_sizes:
                self.status_updated.emit("Using snapshot download...")
                self._download_with_snapshot()
                return

            self._total_bytes = sum(s for _, s in files_with_sizes)
            free_mb = get_free_disk_mb(self.cache_dir)
            if free_mb is not None and free_mb < (self._total_bytes / (1024*1024)) * 1.15:
                self.download_completed.emit(self.model_name, False, "Insufficient disk space")
                return

            self.bytes_updated.emit(0, self._total_bytes)
            total_files = len(files_with_sizes)
            completed = 0

            for filename, file_size in files_with_sizes:
                if self._cancel:
                    break
                self.file_started.emit(filename, format_bytes(file_size))
                self.status_updated.emit(f"Downloading {filename} ({format_bytes(file_size)})...")
                self._monitor_active = True
                monitor = threading.Thread(target=self._monitor_file, args=(filename, file_size), daemon=True)
                monitor.start()
                success = False
                for attempt in range(self._max_retries):
                    if self._cancel:
                        break
                    try:
                        from huggingface_hub import hf_hub_download
                        hf_hub_download(repo_id=self.model_name, filename=filename, cache_dir=self.cache_dir, repo_type="model")
                        success = True
                        break
                    except Exception as e:
                        if attempt < self._max_retries - 1 and not self._cancel:
                            wait = DOWNLOAD_RETRY_BASE_DELAY * (2 ** attempt)
                            self.status_updated.emit(f"Retry {attempt+1}/{self._max_retries} for {filename} in {wait}s...")
                            time.sleep(wait)
                        else:
                            self.file_failed.emit(filename, str(e))
                            if not self._skip_errors:
                                self._monitor_active = False
                                monitor.join(timeout=2)
                                self.download_completed.emit(self.model_name, False, f"Failed: {filename}")
                                return
                self._monitor_active = False
                monitor.join(timeout=2)
                if success:
                    completed += 1
                    self._downloaded_bytes += file_size
                    self.file_completed.emit(filename)
                    self.progress_updated.emit(int(completed / total_files * 100))
                    self.bytes_updated.emit(self._downloaded_bytes, self._total_bytes)

            if self._cancel:
                self.download_completed.emit(self.model_name, False, "Cancelled")
            else:
                self.progress_updated.emit(100)
                dur = time.time() - self._start_time
                add_to_download_log(self.model_name, "completed", completed, self._total_bytes, dur)
                self.download_completed.emit(self.model_name, True, f"Downloaded {completed}/{total_files} files")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            self.download_completed.emit(self.model_name, False, str(e))

    def _get_file_list(self):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.repo_info(repo_id=self.model_name, repo_type="model", files_metadata=True)
            files = [(s.rfilename, getattr(s, 'size', 0) or 0) for s in info.siblings if s.rfilename.endswith(DOWNLOAD_FILE_EXTENSIONS)]
            files.sort(key=lambda x: -x[1])
            return files
        except Exception as e:
            logger.warning(f"Could not get file list: {e}")
            return []

    def _monitor_file(self, filename, expected_size):
        if expected_size == 0:
            return
        cache_prefix = "models--" + self.model_name.replace("/", "--")
        blobs_dir = os.path.join(self.cache_dir, cache_prefix, "blobs")
        last_bytes = 0
        last_time = time.time()
        while self._monitor_active and not self._cancel:
            try:
                current_bytes = 0
                if os.path.exists(blobs_dir):
                    for f in os.listdir(blobs_dir):
                        if f.endswith('.incomplete'):
                            try:
                                current_bytes = max(current_bytes, os.path.getsize(os.path.join(blobs_dir, f)))
                            except OSError:
                                pass
                if current_bytes > 0:
                    overall = self._downloaded_bytes + min(current_bytes, expected_size)
                    self.progress_updated.emit(min(int(overall / max(self._total_bytes, 1) * 100), 99))
                    self.bytes_updated.emit(overall, self._total_bytes)
                    now = time.time()
                    dt = now - last_time
                    if dt >= 0.5 and current_bytes > last_bytes:
                        self._speed_samples.append((now, current_bytes))
                        if len(self._speed_samples) >= 2:
                            old_t, old_b = self._speed_samples[0]
                            window_dt = now - old_t
                            window_b = current_bytes - old_b
                            if window_dt > 0:
                                speed = window_b / window_dt / (1024*1024)
                                self.speed_updated.emit(f"{speed:.1f} MB/s")
                                remaining = self._total_bytes - overall
                                if speed > 0:
                                    self.eta_updated.emit(format_time(remaining / (speed * 1024*1024)))
                        last_bytes = current_bytes
                        last_time = now
            except Exception:
                pass
            time.sleep(DOWNLOAD_MONITOR_INTERVAL)

    def _is_already_downloaded(self):
        try:
            from huggingface_hub import try_to_load_from_cache
            p = try_to_load_from_cache(repo_id=self.model_name, filename="config.json", cache_dir=self.cache_dir)
            return p is not None and os.path.exists(str(p))
        except Exception:
            return False

    def _download_with_snapshot(self):
        try:
            from huggingface_hub import snapshot_download
            self._monitor_active = True
            monitor = threading.Thread(target=self._monitor_snapshot, daemon=True)
            monitor.start()
            snapshot_download(repo_id=self.model_name, cache_dir=self.cache_dir, max_workers=self._max_parallel)
            self._monitor_active = False
            monitor.join(timeout=2)
            self.progress_updated.emit(100)
            dur = time.time() - self._start_time
            add_to_download_log(self.model_name, "completed", 0, 0, dur)
            self.download_completed.emit(self.model_name, True, "Downloaded (snapshot)")
        except Exception as e:
            self._monitor_active = False
            self.download_completed.emit(self.model_name, False, str(e))

    def _monitor_snapshot(self):
        cache_prefix = "models--" + self.model_name.replace("/", "--")
        model_cache_path = os.path.join(self.cache_dir, cache_prefix)
        last_size = 0
        last_time = time.time()
        while self._monitor_active and not self._cancel:
            try:
                current_size = 0
                if os.path.exists(model_cache_path):
                    for dp, _, fns in os.walk(model_cache_path):
                        for f in fns:
                            fp = os.path.join(dp, f)
                            if not os.path.islink(fp):
                                try:
                                    current_size += os.path.getsize(fp)
                                except OSError:
                                    pass
                now = time.time()
                dt = now - last_time
                if dt >= 1.0 and current_size > last_size:
                    speed = (current_size - last_size) / dt / (1024*1024)
                    self.speed_updated.emit(f"{speed:.1f} MB/s")
                    self.bytes_updated.emit(current_size, 0)
                    last_size = current_size
                    last_time = now
            except Exception:
                pass
            time.sleep(DOWNLOAD_MONITOR_INTERVAL)


# =========================================================================
# DOWNLOAD MANAGER
# =========================================================================

class DownloadManager(QObject):
    queue_changed = Signal()
    download_started = Signal(str)
    download_progress = Signal(str, int)
    download_speed = Signal(str, str)
    download_eta = Signal(str, str)
    download_status = Signal(str, str)
    download_completed = Signal(str, bool, str)
    file_progress = Signal(str, str, str)
    bytes_updated = Signal(str, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = []
        self._current_thread = None
        self._lock = threading.Lock()
        self._settings = load_download_settings()

    @property
    def settings(self):
        return self._settings

    def save_settings(self):
        save_download_settings(self._settings)

    def add_to_queue(self, model_name, cache_dir, priority=0):
        with self._lock:
            if self._current_thread and self._current_thread.model_name == model_name:
                return False
            for t in self._queue:
                if t.model_name == model_name:
                    return False
            self._queue.append(DownloadTask(model_name=model_name, cache_dir=cache_dir, priority=priority))
            self._queue.sort(key=lambda t: -t.priority)
        self.queue_changed.emit()
        self._process_queue()
        return True

    def remove_from_queue(self, model_name):
        with self._lock:
            self._queue = [t for t in self._queue if t.model_name != model_name]
        self.queue_changed.emit()

    def cancel_current(self):
        if self._current_thread:
            self._current_thread.cancel()

    def clear_queue(self):
        with self._lock:
            self._queue.clear()
        self.queue_changed.emit()

    def get_queue(self):
        with self._lock:
            return list(self._queue)

    def get_current_task(self):
        if self._current_thread:
            return DownloadTask(model_name=self._current_thread.model_name, cache_dir=self._current_thread.cache_dir, status="downloading")
        return None

    def is_downloading(self):
        return self._current_thread is not None and self._current_thread.isRunning()

    def is_queued(self, model_name):
        with self._lock:
            return any(t.model_name == model_name for t in self._queue)

    def _process_queue(self):
        if self._current_thread and self._current_thread.isRunning():
            return
        with self._lock:
            if not self._queue:
                return
            task = self._queue.pop(0)
        self.download_started.emit(task.model_name)
        thread = EnhancedModelDownloadThread(task.model_name, task.cache_dir, parent=self.parent(),
            max_retries=self._settings.get("max_retries", DOWNLOAD_MAX_RETRIES),
            skip_errors=self._settings.get("skip_errors", True),
            use_hf_transfer=self._settings.get("use_hf_transfer", HF_TRANSFER_AVAILABLE),
            hf_endpoint=self._settings.get("hf_endpoint", DEFAULT_HF_ENDPOINT),
            max_parallel=self._settings.get("max_parallel", 4))
        thread.progress_updated.connect(lambda p, m=task.model_name: self.download_progress.emit(m, p), Qt.QueuedConnection)
        thread.speed_updated.connect(lambda s, m=task.model_name: self.download_speed.emit(m, s), Qt.QueuedConnection)
        thread.eta_updated.connect(lambda e, m=task.model_name: self.download_eta.emit(m, e), Qt.QueuedConnection)
        thread.status_updated.connect(lambda s, m=task.model_name: self.download_status.emit(m, s), Qt.QueuedConnection)
        thread.file_started.connect(lambda f, sz, m=task.model_name: self.file_progress.emit(m, f, sz), Qt.QueuedConnection)
        thread.bytes_updated.connect(lambda d, t, m=task.model_name: self.bytes_updated.emit(m, d, t), Qt.QueuedConnection)
        thread.download_completed.connect(self._on_completed, Qt.QueuedConnection)
        self._current_thread = thread
        thread.start()

    def _on_completed(self, model_name, success, message):
        self.download_completed.emit(model_name, success, message)
        thread = self._current_thread
        self._current_thread = None
        if thread:
            thread.deleteLater()
        self.queue_changed.emit()
        QTimer.singleShot(500, self._process_queue)


download_manager = DownloadManager()


# =========================================================================
# AUDIO GENERATION THREAD
# =========================================================================

class AudioGenerationThread(QThread):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    eta_updated = Signal(str)
    generation_completed = Signal(str, dict)
    error_occurred = Signal(str)
    cancelled = Signal()
    variation_completed = Signal(str, dict)
    batch_progress = Signal(int, int, str)
    performance_metrics = Signal(dict)
    early_preview_ready = Signal(str)
    model_status = Signal(str)

    _pipe_lock = threading.Lock()
    _uses_new_callback = True

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self._cancel_event = threading.Event()
        self.ema_step_time = None
        self.last_step_time = None
        self._last_progress_ts = 0.0
        self.pipe = None
        self.setObjectName("AudioGenerationThread")
        self._var_times = []
        self._skipped = []
        self._learned_sps = get_learned_steps_per_sec(params.model_name, params.device)

    @property
    def is_cancelled(self):
        return self._cancel_event.is_set()

    def cancel(self):
        self._cancel_event.set()

    def _smart_gc(self):
        gc.collect()
        if ML_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    def _load_pipeline(self, actual_device):
        if self.is_cancelled:
            raise InterruptedError("User cancelled")
        with AudioGenerationThread._pipe_lock:
            cached = model_cache.get(self.params.model_name, actual_device, self.params.use_cpu_offload)
            if cached is not None:
                self.pipe = cached
                self.model_status.emit(f"Using cached model: {self.params.model_name}")
                return
            self.model_status.emit(f"Loading model: {self.params.model_name}")
            self.status_updated.emit("Loading model weights...")
            self.progress_updated.emit(5)
            if self.is_cancelled:
                raise InterruptedError("User cancelled")
            dtype = torch.float16 if actual_device != "cpu" else torch.float32
            load_kwargs = {"torch_dtype": dtype, "cache_dir": self.params.cache_dir}
            if dtype == torch.float16:
                load_kwargs["variant"] = "fp16"
            pipe = None
            try:
                pipe = AudioLDM2Pipeline.from_pretrained(self.params.model_name, **load_kwargs, use_safetensors=True)
            except Exception as e_outer:
                if self.is_cancelled:
                    raise InterruptedError("User cancelled") from e_outer
                logger.warning(f"fp16 load failed, retrying: {e_outer}")
                load_kwargs.pop("variant", None)
                try:
                    pipe = AudioLDM2Pipeline.from_pretrained(self.params.model_name, **load_kwargs, use_safetensors=False)
                except Exception as e_inner:
                    raise RuntimeError(f"Failed to load model: {e_inner}") from e_inner
            if self.is_cancelled:
                del pipe
                raise InterruptedError("User cancelled")
            if self.params.use_cpu_offload and actual_device != "cpu":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(actual_device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            try:
                sig = inspect.signature(pipe.__call__)
                AudioGenerationThread._uses_new_callback = "callback_on_step_end" in sig.parameters
            except Exception:
                AudioGenerationThread._uses_new_callback = True
            vram_est = MODEL_VRAM_ESTIMATES.get(self.params.model_name, 3500)
            model_cache.put(self.params.model_name, actual_device, self.params.use_cpu_offload, pipe, vram_est)
            self.pipe = pipe
            self.model_status.emit(f"Model ready: {self.params.model_name}")

    def _on_step(self, *args, chunk_idx=0, **kwargs):
        if self.is_cancelled:
            raise InterruptedError("User cancelled")
        if len(args) >= 4:
            step = int(args[1])
            self._update_progress_and_eta(step, chunk_idx)
            return args[3]
        elif len(args) >= 3:
            step = int(args[0])
            self._update_progress_and_eta(step, chunk_idx)
        return None

    def _update_progress_and_eta(self, step, chunk_idx):
        p = self.params
        nci = int(self._num_chunks)
        steps_done = self._steps_done_before_variation + (chunk_idx * p.steps) + (step + 1)
        total_progress = 15 + int(80 * steps_done / max(self._total_steps_all, 1))
        now = time.time()
        is_final = (step + 1 >= p.steps and chunk_idx + 1 >= nci)
        if is_final or (now - self._last_progress_ts) >= PROGRESS_THROTTLE_S:
            self._last_progress_ts = now
            self.progress_updated.emit(min(total_progress, 95))
        if self.last_step_time is not None:
            dt = now - self.last_step_time
            if dt > 0:
                self.ema_step_time = dt if self.ema_step_time is None else ETA_SMOOTHING_ALPHA * dt + (1 - ETA_SMOOTHING_ALPHA) * self.ema_step_time
                rem_steps = max(0, p.steps - step - 1)
                rem_chunks = max(0, nci - chunk_idx - 1)
                rem_vars = max(0, self._total_variations - self._variation - 1)
                eta = (rem_steps + rem_chunks * p.steps + rem_vars * nci * p.steps) * self.ema_step_time
                if is_final or (now - self._last_progress_ts) >= PROGRESS_THROTTLE_S:
                    self.eta_updated.emit(format_time(eta))
        self.last_step_time = now
        self.status_updated.emit(f"Var {self._variation+1}/{self._total_variations} | Chunk {chunk_idx+1}/{nci} | Step {step+1}/{p.steps}")

    def run(self):
        wav_writer = None
        gen_start = time.time()
        completed_paths = []
        completed_meta = []
        failed_vars = []
        try:
            p = self.params
            cache_dir = os.path.normpath(p.cache_dir or DEFAULT_CACHE_DIR)
            os.makedirs(cache_dir, exist_ok=True)
            
            out_dir = p.output_dir
            
            # Add Section subfolder if specified
            if p.section:
                out_dir = os.path.join(out_dir, sanitize_filename(p.section, 50))
                
            os.makedirs(out_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir
            
            actual_device = p.device
            if actual_device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
            self._load_pipeline(actual_device)
            self.status_updated.emit("Generating audio...")
            self.progress_updated.emit(15)

            # Parse multi-prompt
            use_multi = p.use_multi_prompt
            prompt_configs = []
            if use_multi and p.multi_prompts:
                for line in p.multi_prompts.split('\n'):
                    line = line.strip()
                    if line:
                        prompt_configs.append(parse_multi_prompt_line(line, p.seed_algo, p.duration))
            if not prompt_configs:
                prompt_configs = [{"prompt": p.prompt, "seed_algo": p.seed_algo, "duration": p.duration}]
            has_custom_dur = any(abs(c["duration"] - p.duration) > 0.01 for c in prompt_configs)

            chunk_plan = []
            if has_custom_dur:
                for cfg in prompt_configs:
                    for cl in compute_chunks_for_duration(cfg["duration"]):
                        chunk_plan.append({"prompt": cfg["prompt"], "seed_algo": cfg["seed_algo"], "chunk_len": cl})
                total_duration = sum(c["duration"] for c in prompt_configs)
            else:
                total_duration = p.duration
                use_chunking = total_duration > LONG_FORM_THRESHOLD_S
                MAX_C = AUTO_CHUNK_SIZE_S if use_chunking else total_duration
                if use_chunking:
                    OLP = OVERLAP_S
                    nc = math.ceil((total_duration - OLP) / (MAX_C - OLP))
                    if total_duration - (nc - 1) * (MAX_C - OLP) < MIN_CHUNK_S and nc > 1:
                        nc -= 1
                else:
                    OLP = 0.0
                    nc = 1
                if len(prompt_configs) > 1:
                    cpp = max(1, nc // len(prompt_configs))
                    rem = nc % len(prompt_configs)
                    for i, cfg in enumerate(prompt_configs):
                        cnt = cpp + (1 if i < rem else 0)
                        for j in range(cnt):
                            is_last = (i == len(prompt_configs) - 1 and j == cnt - 1)
                            cl = max(MIN_CHUNK_S, total_duration - (nc-1)*(MAX_C-OLP)) if is_last else MAX_C
                            chunk_plan.append({"prompt": cfg["prompt"], "seed_algo": cfg["seed_algo"], "chunk_len": cl})
                    while len(chunk_plan) < nc:
                        cfg = prompt_configs[-1]
                        chunk_plan.append({"prompt": cfg["prompt"], "seed_algo": cfg["seed_algo"], "chunk_len": MAX_C})
                else:
                    cfg = prompt_configs[0]
                    for i in range(nc):
                        is_last = (i == nc - 1)
                        cl = max(MIN_CHUNK_S, total_duration - (nc-1)*(MAX_C-OLP)) if is_last and use_chunking else (MAX_C if use_chunking else total_duration)
                        chunk_plan.append({"prompt": cfg["prompt"], "seed_algo": cfg["seed_algo"], "chunk_len": cl})

            num_chunks = len(chunk_plan)
            use_chunking = num_chunks > 1
            OLP_L = OVERLAP_S if use_chunking else 0.0
            if use_chunking:
                self.status_updated.emit(f"Long-form: {num_chunks} chunks streaming to disk...")
            sr = getattr(getattr(getattr(self.pipe, "vae", None), "config", None), "sample_rate", 16000)

            # Precompute seeds
            base_seed = p.seed if p.seed >= 0 else int(time.time() * 1000) % (2**31)
            batch_cfg = BatchConfig.load()
            batch_cfg.seed_algo = p.seed_algo
            all_seeds = compute_batch_seeds(base_seed, p.num_variations, p.seed_algo, batch_cfg)

            start_var = max(0, p.resume_from_variation)
            if self._learned_sps and self._learned_sps > 0:
                est = (p.steps * num_chunks / self._learned_sps) * (p.num_variations - start_var)
                self.eta_updated.emit(format_time(est))

            batch_state = {"batch_id": p.batch_id, "model": p.model_name, "prompt": p.prompt, "duration": p.duration, "steps": p.steps, "num_variations": p.num_variations, "output_dir": out_dir, "started_at": gen_start, "completed": [], "failed": []}
            early_preview_done = False

            for variation in range(start_var, p.num_variations):
                if self.is_cancelled:
                    raise InterruptedError("User cancelled")
                var_start = time.time()
                base_seed_var = all_seeds[variation] if variation < len(all_seeds) else base_seed
                self.ema_step_time, self.last_step_time = None, None
                self._steps_done_before_variation = variation * num_chunks * p.steps
                self._total_steps_all = p.num_variations * num_chunks * p.steps
                self._num_chunks = num_chunks
                self._variation = variation
                self._total_variations = p.num_variations
                
                # --- NEW FILE NAMING & SUBFOLDER LOGIC ---
                # 1. Create subfolder based on prompt
                prompt_subfolder = sanitize_filename(p.prompt, 40) if p.prompt else "untitled"
                final_out_dir = os.path.join(out_dir, prompt_subfolder)
                os.makedirs(final_out_dir, exist_ok=True)
                
                # 2. Generate filename using the BatchConfig pattern engine
                filename = render_filename_pattern(
                    batch_cfg.filename_pattern, 
                    p, variation, 
                    base_seed_var, 
                    batch_cfg, 
                    p.batch_id, 
                    variation
                )
                out_path = os.path.join(final_out_dir, filename)
                
                self.batch_progress.emit(variation + 1, p.num_variations, filename)
                overlap_buffer = None
                wav_writer = wave.open(out_path, 'wb')
                wav_writer.setnchannels(1)
                wav_writer.setsampwidth(2)
                wav_writer.setframerate(sr)
                try:
                    for ci in range(num_chunks):
                        if self.is_cancelled:
                            raise InterruptedError("User cancelled")
                        is_last_c = (ci == num_chunks - 1)
                        cc = chunk_plan[ci]
                        cl = cc["chunk_len"]
                        cp = cc["prompt"]
                        ca = cc["seed_algo"]
                        if ca == "Random":
                            cs = random.randint(0, 2**31-1)
                        elif ca == "Fixed":
                            cs = base_seed_var
                        elif ca == "Golden Ratio":
                            cs = int((base_seed_var + ci * 1.61803398875 * 1000000) % (2**31))
                        else:
                            cs = (base_seed_var + ci * 7919) % (2**31)
                        gen_dev = "cpu" if p.use_cpu_offload else actual_device
                        gen = torch.Generator(device=gen_dev).manual_seed(cs)
                        cb = partial(self._on_step, chunk_idx=ci)
                        kwargs = dict(prompt=cp, negative_prompt=p.negative_prompt if p.negative_prompt else None, num_inference_steps=p.steps, audio_length_in_s=cl, guidance_scale=p.guidance, generator=gen)
                        if AudioGenerationThread._uses_new_callback:
                            kwargs["callback_on_step_end"] = cb
                            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                        else:
                            kwargs["callback"] = cb
                            kwargs["callback_steps"] = 1
                        with torch.inference_mode():
                            audio_chunk = self.pipe(**kwargs).audios[0]
                        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
                        if num_chunks > 1:
                            self._smart_gc()
                        if p.trim:
                            audio_chunk = trim_silence(audio_chunk)
                        if p.normalize:
                            audio_chunk = normalize_audio(audio_chunk)
                        if p.fade_in or p.fade_out:
                            audio_chunk = apply_fade(audio_chunk, sr, FADE_DURATION_S if p.fade_in else 0, FADE_DURATION_S if p.fade_out else 0)
                        ci16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                        olt = int(OLP_L * sr)
                        if overlap_buffer is not None and use_chunking and olt > 0 and len(ci16) >= olt:
                            os_ = min(len(overlap_buffer), olt)
                            t = np.linspace(0, np.pi/2, os_, dtype=np.float32)
                            prev_f = overlap_buffer[-os_:].astype(np.float32) / 32768.0
                            curr_f = ci16[:os_].astype(np.float32) / 32768.0
                            cf = prev_f * np.cos(t) + curr_f * np.sin(t)
                            ci16[:os_] = (np.clip(cf, -1.0, 1.0) * 32767).astype(np.int16)
                        if use_chunking and not is_last_c and len(ci16) > olt and olt > 0:
                            overlap_buffer = ci16[-olt:].copy()
                            wav_writer.writeframes(ci16[:-olt].tobytes())
                        else:
                            if not is_last_c:
                                overlap_buffer = None
                            wav_writer.writeframes(ci16.tobytes())
                except InterruptedError:
                    raise
                except Exception as ve:
                    if p.skip_on_error and variation < p.num_variations - 1:
                        logger.error(f"Var {variation+1} failed: {ve}")
                        self.status_updated.emit(f"Var {variation+1} failed, skipping...")
                        failed_vars.append({"variation": variation+1, "error": str(ve)})
                        self._skipped.append(variation+1)
                        try:
                            if wav_writer: wav_writer.close()
                            if os.path.exists(out_path): os.remove(out_path)
                        except Exception:
                            pass
                        wav_writer = None
                        self._smart_gc()
                        continue
                    else:
                        raise
                finally:
                    if wav_writer is not None:
                        try: wav_writer.close()
                        except Exception: pass
                        wav_writer = None
                vt = time.time() - var_start
                self._var_times.append(vt)
                meta = {'prompt': p.prompt, 'negative_prompt': p.negative_prompt, 'model': p.model_name, 'duration': total_duration, 'steps': p.steps, 'guidance': p.guidance, 'seed': base_seed_var, 'sample_rate': sr, 'device': actual_device, 'variation': variation+1, 'total_variations': p.num_variations, 'generation_time': vt, 'timestamp': datetime.now().isoformat(), 'chunked_generation': num_chunks > 1, 'num_chunks': num_chunks, 'path': out_path, 'filename': filename, 'tags': [], 'favorite': False, 'multi_prompt': p.use_multi_prompt, 'seed_algo': p.seed_algo, 'prompt_configs': prompt_configs if use_multi else None, 'performance_mode': p.performance_mode, 'batch_id': p.batch_id}
                completed_paths.append(out_path)
                completed_meta.append(meta)
                self.variation_completed.emit(out_path, meta)
                if p.early_preview and not early_preview_done and variation == 0:
                    early_preview_done = True
                    self.early_preview_ready.emit(out_path)
                batch_state["completed"].append({"variation": variation+1, "path": out_path})
                batch_state["failed"] = failed_vars
                save_batch_state(batch_state)
                if p.aggressive_gc or p.num_variations >= AGGRESSIVE_GC_THRESHOLD:
                    self._smart_gc()

            total_gen = time.time() - gen_start
            record_eta_point(p.model_name, actual_device, p.steps, total_duration, num_chunks, total_gen)
            clear_batch_state()
            self.progress_updated.emit(100)
            self.status_updated.emit("Generation complete")
            self.eta_updated.emit("")
            metrics = {"total_time": total_gen, "num_variations": len(completed_paths), "num_failed": len(failed_vars), "avg_per_variation": (sum(self._var_times)/len(self._var_times)) if self._var_times else 0, "steps_per_sec": (p.steps * num_chunks * len(completed_paths))/total_gen if total_gen > 0 else 0, "device": actual_device, "model": p.model_name, "skipped": self._skipped}
            self.performance_metrics.emit(metrics)
            self.generation_completed.emit(completed_paths[0] if completed_paths else "", {'paths': completed_paths, 'metadata': completed_meta, 'gen_time': total_gen, 'failed': failed_vars, 'metrics': metrics})
        except InterruptedError:
            if p.num_variations > 1 and completed_paths:
                batch_state["completed"] = [{"variation": i+1, "path": pt} for i, pt in enumerate(completed_paths)]
                save_batch_state(batch_state)
            self.status_updated.emit("Cancelled")
            self.progress_updated.emit(0)
            self.eta_updated.emit("")
            self.cancelled.emit()
        except Exception as e:
            err = traceback.format_exc()
            logger.error(f"Generation failed: {err}")
            if "out of memory" in str(e).lower():
                self.error_occurred.emit("GPU OOM! Try CPU Offload or fewer steps.\nFor large batches use Speed/Draft preset.")
            else:
                self.error_occurred.emit(f"Generation Failed:\n{err}")
        finally:
            if wav_writer is not None:
                try: wav_writer.close()
                except Exception: pass
            self.pipe = None
            gc.collect()
            if ML_AVAILABLE and torch.cuda.is_available():
                try: torch.cuda.empty_cache()
                except Exception: pass

    @classmethod
    def unload_pipeline(cls):
        model_cache.clear()


# =========================================================================
# MODEL PRELOAD THREAD
# =========================================================================

class ModelPreloadThread(QThread):
    status_updated = Signal(str)
    component_loaded = Signal(str)
    progress_updated = Signal(int)
    finished_preload = Signal(str, bool, str)

    def __init__(self, model_name, cache_dir, device, use_cpu_offload=False, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.use_cpu_offload = use_cpu_offload
        self.setObjectName("ModelPreloadThread")

    def run(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["HF_HUB_CACHE"] = self.cache_dir
            cached = model_cache.get(self.model_name, self.device, self.use_cpu_offload)
            if cached is not None:
                self.status_updated.emit(f"Already loaded (cached)")
                self.progress_updated.emit(100)
                self.finished_preload.emit(self.model_name, True, "cached")
                return
            self.status_updated.emit(f"Loading {self.model_name}...")
            self.progress_updated.emit(15)
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            lk = {"torch_dtype": dtype, "cache_dir": self.cache_dir}
            if dtype == torch.float16:
                lk["variant"] = "fp16"
            pipe = None
            try:
                pipe = AudioLDM2Pipeline.from_pretrained(self.model_name, **lk, use_safetensors=True)
            except Exception as e:
                logger.warning(f"fp16 failed: {e}")
                lk.pop("variant", None)
                pipe = AudioLDM2Pipeline.from_pretrained(self.model_name, **lk, use_safetensors=False)
            self.progress_updated.emit(60)
            self.component_loaded.emit("weights")
            self.status_updated.emit("Applying optimizations...")
            self.progress_updated.emit(70)
            if self.use_cpu_offload and self.device != "cpu":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            self.progress_updated.emit(85)
            try:
                sig = inspect.signature(pipe.__call__)
                AudioGenerationThread._uses_new_callback = "callback_on_step_end" in sig.parameters
            except Exception:
                pass
            vram_est = MODEL_VRAM_ESTIMATES.get(self.model_name, 3500)
            model_cache.put(self.model_name, self.device, self.use_cpu_offload, pipe, vram_est)
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Ready")
            self.finished_preload.emit(self.model_name, True, "loaded")
        except Exception as e:
            logger.error(f"Preload failed: {e}")
            self.finished_preload.emit(self.model_name, False, str(e))


# =========================================================================
# UI WIDGETS
# =========================================================================

class SFXPromptBuilderTab(QWidget):
    insertPromptRequested = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QGroupBox { font-weight: bold; color: #ccc; } QLabel { color: #ddd; }")
        layout = QVBoxLayout(self)
        info = QLabel("Formula: [Category]: [Action], [Environment], [Quality]")
        info.setStyleSheet("color: #2A82DA; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(info)
        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignRight)
        self.category_combo = QComboBox(); self.category_combo.setEditable(True)
        self.category_combo.addItems(["Foley","Cinematic","Ambient","Game UI","Nature","Sci-Fi","Mechanical","Weapon","Vehicle","Creature"])
        form.addRow("1. Category:", self.category_combo)
        self.action_combo = QComboBox(); self.action_combo.setEditable(True)
        self.action_combo.addItems(["footsteps on gravel","heavy door slamming","rain on a tin roof","sword draw metallic scrape","massive sub-bass impact","clean digital confirmation beep","distant thunder rolling","fire crackling","glass breaking","engine revving"])
        form.addRow("2. Action:", self.action_combo)
        self.env_combo = QComboBox(); self.env_combo.setEditable(True)
        self.env_combo.addItems(["close microphone","distant","large reverberant hall","small dry room","dense forest","busy city street","professional studio","underwater","vast open space","inside a car"])
        form.addRow("3. Environment:", self.env_combo)
        self.quality_combo = QComboBox(); self.quality_combo.setEditable(True)
        self.quality_combo.addItems(["high quality, professional recording","ASMR, highly detailed","cinematic, epic","crisp transients, clean","deep bass response","no reverb, dry","immersive, 3D audio","lo-fi, vintage"])
        form.addRow("4. Quality:", self.quality_combo)
        layout.addLayout(form)
        pg = QGroupBox("Live Preview")
        pl = QVBoxLayout(pg)
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color: #2A82DA; font-style: italic; padding: 10px; background-color: #1e1e1e; border: 1px solid #3c3c3c; border-radius: 4px; min-height: 40px;")
        pl.addWidget(self.preview_label)
        layout.addWidget(pg)
        for c in [self.category_combo, self.action_combo, self.env_combo, self.quality_combo]:
            c.currentTextChanged.connect(self.update_preview)
        layout.addStretch()
        br = QHBoxLayout(); br.addStretch()
        self.insert_btn = QPushButton("Insert into Prompt")
        self.insert_btn.setStyleSheet("QPushButton { background-color: #2A82DA; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #3A92EA; }")
        self.insert_btn.clicked.connect(self._on_insert)
        br.addWidget(self.insert_btn)
        layout.addLayout(br)
        self.update_preview()
    def _on_insert(self):
        t = self.get_formatted_prompt()
        if t: self.insertPromptRequested.emit(t)
    def update_preview(self):
        cat = self.category_combo.currentText().strip()
        act = self.action_combo.currentText().strip()
        env = self.env_combo.currentText().strip()
        qual = self.quality_combo.currentText().strip()
        parts = []
        if cat and act: parts.append(f"{cat}: {act}")
        elif act: parts.append(act)
        if env: parts.append(env)
        if qual: parts.append(qual)
        self.preview_label.setText(", ".join(parts) if parts else "Fill in fields above...")
    def get_formatted_prompt(self):
        t = self.preview_label.text()
        return t if t != "Fill in fields above..." else ""


class MultiPromptEditorTab(QWidget):
    applyRequested = Signal(str, float)
    SEED_ALGOS = list(SEED_ALGORITHMS_ADVANCED.keys())
    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_seed_algo = "Sequential"
        self.default_duration = 10.0
        self._loading = True
        layout = QVBoxLayout(self); layout.setSpacing(10)
        info = QLabel("Format: prompt | seed_algo | duration\nExample: Cinematic impact | Random | 30")
        info.setWordWrap(True)
        info.setStyleSheet("color: #2A82DA; font-style: italic; background-color: #1e1e1e; padding: 10px; border: 1px solid #3c3c3c; border-radius: 4px;")
        layout.addWidget(info)
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        layout.addWidget(self.summary_label)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["#","Prompt","Seed Algorithm","Duration (s)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table)
        br = QHBoxLayout()
        self.add_btn = QPushButton("Add"); self.add_btn.clicked.connect(lambda: self._add_row())
        br.addWidget(self.add_btn)
        self.dup_btn = QPushButton("Duplicate"); self.dup_btn.clicked.connect(self._duplicate_row)
        br.addWidget(self.dup_btn)
        self.rm_btn = QPushButton("Remove"); self.rm_btn.clicked.connect(self._remove_selected)
        br.addWidget(self.rm_btn)
        br.addStretch()
        self.up_btn = QPushButton("Up"); self.up_btn.clicked.connect(lambda: self._move_row(-1))
        br.addWidget(self.up_btn)
        self.dn_btn = QPushButton("Down"); self.dn_btn.clicked.connect(lambda: self._move_row(1))
        br.addWidget(self.dn_btn)
        layout.addLayout(br)
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Quick fill:"))
        self.pdur = QDoubleSpinBox(); self.pdur.setRange(1.0, MAX_DURATION_S); self.pdur.setValue(self.default_duration); self.pdur.setSuffix(" s")
        pr.addWidget(self.pdur)
        self.adur = QPushButton("Apply Dur"); self.adur.clicked.connect(self._apply_dur_all)
        pr.addWidget(self.adur)
        self.palgo = QComboBox(); self.palgo.addItems(self.SEED_ALGOS); self.palgo.setCurrentText(self.default_seed_algo)
        pr.addWidget(self.palgo)
        self.aalgo = QPushButton("Apply Algo"); self.aalgo.clicked.connect(self._apply_algo_all)
        pr.addWidget(self.aalgo)
        pr.addStretch()
        layout.addLayout(pr)
        ok = QHBoxLayout(); ok.addStretch()
        self.apply_btn = QPushButton("Apply to Generate Tab")
        self.apply_btn.setStyleSheet("QPushButton { background-color: #2A82DA; color: white; font-weight: bold; padding: 8px 20px; border-radius: 4px; } QPushButton:hover { background-color: #3A92EA; }")
        self.apply_btn.clicked.connect(self._on_apply)
        ok.addWidget(self.apply_btn)
        layout.addLayout(ok)
    def update_defaults(self, algo, dur):
        self.default_seed_algo = algo; self.default_duration = dur
        self.palgo.setCurrentText(algo); self.pdur.setValue(dur)
    def load_from_text(self, text):
        self._loading = True
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        self.table.setRowCount(0)
        for line in lines:
            cfg = parse_multi_prompt_line(line, self.default_seed_algo, self.default_duration)
            self._add_row(cfg["prompt"], cfg["seed_algo"], cfg["duration"])
        self._loading = False; self._update_summary()
    def _add_row(self, prompt="", seed_algo=None, duration=None):
        self._loading = True
        row = self.table.rowCount(); self.table.insertRow(row)
        ni = QTableWidgetItem(str(row+1)); ni.setFlags(Qt.ItemIsEnabled); ni.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, ni)
        self.table.setItem(row, 1, QTableWidgetItem(prompt))
        combo = QComboBox(); combo.addItems(self.SEED_ALGOS); combo.setCurrentText(seed_algo or self.default_seed_algo)
        self.table.setCellWidget(row, 2, combo)
        spin = QDoubleSpinBox(); spin.setRange(1.0, MAX_DURATION_S); spin.setDecimals(1)
        spin.setValue(duration if duration is not None else self.default_duration)
        spin.valueChanged.connect(self._update_summary)
        self.table.setCellWidget(row, 3, spin)
        combo.currentTextChanged.connect(self._update_summary)
        self._loading = False; self._renumber(); self._update_summary()
    def _duplicate_row(self):
        r = self.table.currentRow()
        if r < 0: return
        self._add_row(self.table.item(r,1).text(), self.table.cellWidget(r,2).currentText(), self.table.cellWidget(r,3).value())
    def _remove_selected(self):
        rows = sorted(set(i.row() for i in self.table.selectedItems()), reverse=True)
        for r in rows: self.table.removeRow(r)
        self._renumber(); self._update_summary()
    def _move_row(self, d):
        r = self.table.currentRow()
        if r < 0: return
        t = r + d
        if t < 0 or t >= self.table.rowCount(): return
        pa = self.table.item(r,1).text(); pb = self.table.item(t,1).text()
        self.table.item(r,1).setText(pb); self.table.item(t,1).setText(pa)
        ca = self.table.cellWidget(r,2); cb = self.table.cellWidget(t,2)
        ca.setCurrentText(cb.currentText()); cb.setCurrentText(ca.currentText())
        sa = self.table.cellWidget(r,3); sb = self.table.cellWidget(t,3)
        sa.setValue(sb.value()); sb.setValue(sa.value())
        self.table.setCurrentCell(t, 1)
    def _renumber(self):
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 0)
            if it: it.setText(str(i+1))
    def _on_cell_changed(self, r, c):
        if not self._loading and c == 1: self._update_summary()
    def _apply_dur_all(self):
        v = self.pdur.value()
        for r in range(self.table.rowCount()):
            s = self.table.cellWidget(r, 3)
            if s: s.setValue(v)
    def _apply_algo_all(self):
        a = self.palgo.currentText()
        for r in range(self.table.rowCount()):
            c = self.table.cellWidget(r, 2)
            if c: c.setCurrentText(a)
    def _update_summary(self):
        total = 0.0; cnt = self.table.rowCount(); chunks = 0
        for r in range(cnt):
            s = self.table.cellWidget(r, 3)
            if s: total += s.value(); chunks += len(compute_chunks_for_duration(s.value()))
        self.summary_label.setText(f"{cnt} prompt(s) | Total: {format_time(total)} | ~{chunks} chunks")
    def _on_apply(self):
        self.applyRequested.emit(self.get_serialized_text(), self.get_total_duration())
    def get_total_duration(self):
        t = 0.0
        for r in range(self.table.rowCount()):
            s = self.table.cellWidget(r, 3)
            if s: t += s.value()
        return t
    def get_serialized_text(self):
        lines = []
        for r in range(self.table.rowCount()):
            pi = self.table.item(r, 1)
            if not pi: continue
            p = pi.text().strip()
            if not p: continue
            c = self.table.cellWidget(r, 2); s = self.table.cellWidget(r, 3)
            a = c.currentText() if c else self.default_seed_algo
            d = s.value() if s else self.default_duration
            hca = a != self.default_seed_algo
            hcd = abs(d - self.default_duration) > 0.01
            if hca and hcd: lines.append(f"{p} | {a} | {d}")
            elif hca: lines.append(f"{p} | {a}")
            elif hcd: lines.append(f"{p} | {d}")
            else: lines.append(p)
        return '\n'.join(lines)


class WaveformWidget(QWidget):
    seekRequested = Signal(float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_audio = None; self._peaks = None; self._peaks_width = -1; self._poly_peak = None
        self.sample_rate = 16000; self.duration_s = 0.0; self._audio_loaded = False
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.playback_position = 0.0
        self.setMouseTracking(True)
    def set_audio(self, path):
        if not path or not os.path.exists(path):
            self._raw_audio = None; self._audio_loaded = False; self._peaks = None; self._poly_peak = None
            self.duration_s = 0.0; self.update(); return
        try:
            with wave.open(path, 'rb') as wf:
                self.sample_rate = wf.getframerate()
                fr = min(wf.getnframes(), self.sample_rate * 300)
                raw = wf.readframes(fr)
                self._raw_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                self.duration_s = wf.getnframes() / self.sample_rate
                self._audio_loaded = True; self._peaks = None; self._peaks_width = -1
                self._ensure_peaks(self.width())
        except Exception as e:
            logger.error(f"Waveform load failed: {e}")
            self._audio_loaded = False; self._raw_audio = None; self.update()
    def resizeEvent(self, e):
        super().resizeEvent(e); self._peaks = None; self._peaks_width = -1; self._ensure_peaks(self.width())
    def _ensure_peaks(self, w):
        if w <= 0: return
        if self._peaks is not None and self._peaks_width == w: return
        if self._raw_audio is None: self._peaks = None; self._poly_peak = None; return
        n = len(self._raw_audio); ab = np.abs(self._raw_audio)
        if n <= w:
            self._peaks = np.pad(ab, (0, max(0, w-n)), 'constant').astype(np.float32)
        else:
            cs = max(1, n // w); nf = (n // cs) * cs
            self._peaks = ab[:nf].reshape(-1, cs).max(axis=1).astype(np.float32)
            if len(self._peaks) < w: self._peaks = np.pad(self._peaks, (0, w-len(self._peaks)), 'constant')
        self._peaks_width = w; self._poly_peak = self._build_poly(self._peaks, w); self.update()
    def _build_poly(self, heights, w):
        mid = self.height() / 2.0; mh = self.height() * 0.42
        poly = QPolygonF(); poly << QPointF(0, mid)
        for x, h in enumerate(heights): poly << QPointF(x, mid - (h * mh))
        poly << QPointF(w, mid)
        for x in range(w-1, -1, -1): poly << QPointF(x, mid + (heights[x] * mh))
        return poly
    def set_playback_position(self, pos):
        self.playback_position = max(0.0, min(1.0, pos)); self.update()
    def mousePressEvent(self, e):
        if self._audio_loaded and e.button() == Qt.LeftButton:
            pos = e.position().x() / max(self.width(), 1)
            self.seekRequested.emit(pos); self.set_playback_position(pos)
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(22, 22, 26))
        if self._audio_loaded and (self._poly_peak is None or self._peaks_width != w):
            self._ensure_peaks(w)
        if not self._audio_loaded or self._poly_peak is None or w <= 0:
            p.setPen(QColor(100, 100, 100))
            p.drawText(self.rect(), Qt.AlignCenter, "No audio loaded" if not self._audio_loaded else "Loading...")
            return
        mid = h / 2.0; px = int(self.playback_position * (w - 1))
        p.setPen(QPen(QColor(50,50,55), 1, Qt.DashLine)); p.setFont(QFont("Segoe UI", 8))
        if self.duration_s > 0:
            ss = 60.0 if self.duration_s > 600 else (15.0 if self.duration_s > 60 else 2.0)
            t = 0.0
            while t < self.duration_s:
                gx = int((t / self.duration_s) * w)
                p.drawLine(gx, 10, gx, h-10)
                p.drawText(gx+3, 12, 40, 15, Qt.AlignLeft, format_time(t))
                t += ss
        p.setClipRect(px, 0, w-px, h); p.setBrush(QColor(60,65,75)); p.setPen(Qt.NoPen); p.drawPolygon(self._poly_peak)
        p.setClipRect(0, 0, px+1, h)
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(40,160,240)); grad.setColorAt(0.5, QColor(80,220,255)); grad.setColorAt(1.0, QColor(40,160,240))
        p.setBrush(grad); p.drawPolygon(self._poly_peak); p.setClipping(False)
        if self.playback_position >= 0:
            p.setPen(QPen(QColor(255,255,255,220), 2)); p.drawLine(px, 0, px, h)


class PromptHistoryLineEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.setPlaceholderText("Describe the audio... (Ctrl+Enter to generate)")
        self.setMaximumHeight(70)
        self.setTabChangesFocus(True)
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Return and (e.modifiers() & Qt.ControlModifier):
            self.window().start_generation(); return
        super().keyPressEvent(e)
    def add_to_history(self, text):
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            if len(self.history) > MAX_PROMPT_HISTORY: self.history.pop(0)


# =========================================================================
# MODEL MANAGER DIALOG
# =========================================================================

class ModelManagerDialog(QDialog):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("Model Manager"); self.setMinimumSize(600, 500)
        self.setStyleSheet(STYLESHEET)
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Model","Status","VRAM","Description","Action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        br = QHBoxLayout()
        self.add_btn = QPushButton("Add Custom Model"); self.add_btn.clicked.connect(self._add_custom)
        br.addWidget(self.add_btn)
        self.dl_btn = QPushButton("Download Selected"); self.dl_btn.clicked.connect(self._dl_selected)
        br.addWidget(self.dl_btn)
        self.evict_btn = QPushButton("Evict from Cache"); self.evict_btn.clicked.connect(self._evict_selected)
        br.addWidget(self.evict_btn)
        self.use_btn = QPushButton("Use Selected")
        self.use_btn.setStyleSheet("QPushButton { background-color: #2A82DA; color: white; font-weight: bold; }")
        self.use_btn.clicked.connect(self._use_selected)
        br.addWidget(self.use_btn)
        layout.addLayout(br)
        cr = QHBoxLayout(); cr.addStretch()
        cb = QPushButton("Close"); cb.setMinimumHeight(35); cb.clicked.connect(self.accept)
        cr.addWidget(cb); layout.addLayout(cr)
        self._refresh()
    def _refresh(self):
        models = self.window.settings.get("custom_models", [])
        all_m = list(set(DEFAULT_MODELS + models))
        loaded = model_cache.list_loaded()
        loaded_names = [m["model"] for m in loaded]
        cd = self.window.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        self.table.setRowCount(0)
        for m in sorted(all_m):
            row = self.table.rowCount(); self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(m))
            if m in loaded_names: st = "Loaded"; sc = QColor(46,204,113)
            elif self._is_dl(m, cd): st = "Cached"; sc = QColor(42,130,218)
            else: st = "Not downloaded"; sc = QColor(149,165,166)
            si = QTableWidgetItem(st); si.setForeground(sc)
            self.table.setItem(row, 1, si)
            self.table.setItem(row, 2, QTableWidgetItem(f"~{MODEL_VRAM_ESTIMATES.get(m, 3500)} MB"))
            self.table.setItem(row, 3, QTableWidgetItem(MODEL_DESCRIPTIONS.get(m, "Custom")))
            btn = QPushButton("Download" if not self._is_dl(m, cd) else "Re-download")
            btn.setFixedHeight(28); btn.clicked.connect(lambda c=False, mm=m: download_manager.add_to_queue(mm, self.window.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR))
            self.table.setCellWidget(row, 4, btn)
    def _is_dl(self, m, cd):
        try:
            from huggingface_hub import try_to_load_from_cache
            p = try_to_load_from_cache(repo_id=m, filename="config.json", cache_dir=cd)
            return p is not None and os.path.exists(str(p))
        except Exception: return False
    def _add_custom(self):
        t, ok = QInputDialog.getText(self, "Add Custom Model", "Enter HF model ID or local path:")
        if ok and t:
            t = t.strip()
            if '/' not in t and not os.path.isdir(t):
                QMessageBox.warning(self, "Invalid", "Must be 'org/model' or local path."); return
            custom = self.window.settings.get("custom_models", [])
            if t not in custom:
                custom.append(t); self.window.settings["custom_models"] = custom
                self.window.save_json_settings(); self.window.model_combo.addItem(t)
            self._refresh()
    def _dl_selected(self):
        r = self.table.currentRow()
        if r >= 0:
            download_manager.add_to_queue(self.table.item(r,0).text(), self.window.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR)
    def _evict_selected(self):
        r = self.table.currentRow()
        if r >= 0:
            model_cache.evict(self.table.item(r,0).text()); self._refresh()
    def _use_selected(self):
        r = self.table.currentRow()
        if r >= 0:
            m = self.table.item(r,0).text()
            self.window.model_combo.setCurrentText(m)
            profile_manager.apply_profile(m, self.window)
            self.accept()


# =========================================================================
# DOWNLOAD MANAGER DIALOG
# =========================================================================

class DownloadManagerDialog(QDialog):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("Download Manager"); self.setMinimumSize(700, 600)
        self.setStyleSheet(STYLESHEET)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("📦 Model Download Manager"))
        ag = QGroupBox("Active Download"); al = QVBoxLayout(ag)
        self.active_label = QLabel("No active download"); self.active_label.setStyleSheet("color: #888;")
        al.addWidget(self.active_label)
        self.active_progress = QProgressBar(); self.active_progress.setVisible(False); al.addWidget(self.active_progress)
        self.file_detail = QLabel(""); self.file_detail.setStyleSheet("color: #2A82DA; font-size: 10px;"); self.file_detail.setVisible(False); al.addWidget(self.file_detail)
        sr = QHBoxLayout()
        self.speed_label = QLabel(""); self.speed_label.setStyleSheet("color: #2ecc71;")
        sr.addWidget(self.speed_label); sr.addStretch()
        self.eta_label = QLabel(""); self.eta_label.setStyleSheet("color: #f39c12;")
        sr.addWidget(self.eta_label); sr.addStretch()
        self.bytes_label = QLabel(""); self.bytes_label.setStyleSheet("color: #888; font-size: 10px;")
        sr.addWidget(self.bytes_label); al.addLayout(sr)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(lambda: download_manager.cancel_current())
        al.addWidget(self.cancel_btn)
        layout.addWidget(ag)
        qg = QGroupBox("Queue"); ql = QVBoxLayout(qg)
        self.queue_list = QListWidget(); self.queue_list.setMaximumHeight(100); ql.addWidget(self.queue_list)
        self.clear_q_btn = QPushButton("Clear Queue"); self.clear_q_btn.clicked.connect(lambda: download_manager.clear_queue())
        ql.addWidget(self.clear_q_btn)
        layout.addWidget(qg)
        mg = QGroupBox("Models"); ml = QVBoxLayout(mg)
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(5)
        self.model_table.setHorizontalHeaderLabels(["Model","Status","Size","Description","Action"])
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ml.addWidget(self.model_table)
        br = QHBoxLayout()
        self.dl_all_btn = QPushButton("Download All"); self.dl_all_btn.clicked.connect(self._dl_all)
        br.addWidget(self.dl_all_btn)
        self.dl_sel_btn = QPushButton("Download Selected"); self.dl_sel_btn.clicked.connect(self._dl_sel)
        br.addWidget(self.dl_sel_btn)
        self.del_cache_btn = QPushButton("Delete Cache"); self.del_cache_btn.clicked.connect(self._del_cache)
        br.addWidget(self.del_cache_btn)
        br.addStretch()
        ml.addLayout(br)
        layout.addWidget(mg)
        sg = QGroupBox("Settings"); sl = QFormLayout(sg)
        self.hf_check = QCheckBox("Use hf_transfer"); self.hf_check.setChecked(download_manager.settings.get("use_hf_transfer", False))
        self.hf_check.setEnabled(HF_TRANSFER_AVAILABLE)
        sl.addRow("Acceleration:", self.hf_check)
        self.auto_dl_check = QCheckBox("Auto-download on select"); self.auto_dl_check.setChecked(download_manager.settings.get("auto_download", False))
        sl.addRow("Auto-download:", self.auto_dl_check)
        self.mirror_edit = QLineEdit(); self.mirror_edit.setPlaceholderText(DEFAULT_HF_ENDPOINT)
        self.mirror_edit.setText(download_manager.settings.get("hf_endpoint", DEFAULT_HF_ENDPOINT))
        sl.addRow("Mirror:", self.mirror_edit)
        layout.addWidget(sg)
        cr = QHBoxLayout(); cr.addStretch()
        cb = QPushButton("Close"); cb.setMinimumHeight(35); cb.clicked.connect(self._save_and_close)
        cr.addWidget(cb); layout.addLayout(cr)
        download_manager.queue_changed.connect(self._refresh, Qt.QueuedConnection)
        download_manager.download_started.connect(self._on_started, Qt.QueuedConnection)
        download_manager.download_progress.connect(lambda m, p: (self.active_progress.setValue(p),), Qt.QueuedConnection)
        download_manager.download_speed.connect(lambda m, s: self.speed_label.setText(s), Qt.QueuedConnection)
        download_manager.download_eta.connect(lambda m, e: self.eta_label.setText(e), Qt.QueuedConnection)
        download_manager.download_status.connect(lambda m, s: self.active_label.setText(s), Qt.QueuedConnection)
        download_manager.file_progress.connect(lambda m, f, sz: self.file_detail.setText(f"{f} ({sz})"), Qt.QueuedConnection)
        download_manager.bytes_updated.connect(lambda m, d, t: self.bytes_label.setText(f"{format_bytes(d)} / {format_bytes(t)}" if t > 0 else f"{format_bytes(d)}"), Qt.QueuedConnection)
        download_manager.download_completed.connect(self._on_completed, Qt.QueuedConnection)
        self._refresh()
    def _refresh(self):
        q = download_manager.get_queue()
        self.queue_list.clear()
        for t in q: self.queue_list.addItem(t.model_name)
        models = self.window.settings.get("custom_models", [])
        all_m = list(set(DEFAULT_MODELS + models))
        cd = self.window.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        self.model_table.setRowCount(0)
        for m in sorted(all_m):
            row = self.model_table.rowCount(); self.model_table.insertRow(row)
            self.model_table.setItem(row, 0, QTableWidgetItem(m))
            if download_manager.is_queued(m): st = "Queued"; sc = QColor(155,89,182)
            elif self._is_dl(m, cd): st = "Downloaded"; sc = QColor(46,204,113)
            else: st = "Not downloaded"; sc = QColor(231,76,60)
            si = QTableWidgetItem(st); si.setForeground(sc)
            self.model_table.setItem(row, 1, si)
            self.model_table.setItem(row, 2, QTableWidgetItem(f"~{MODEL_VRAM_ESTIMATES.get(m, 3500)} MB"))
            self.model_table.setItem(row, 3, QTableWidgetItem(MODEL_DESCRIPTIONS.get(m, "Custom")))
            btn = QPushButton("Download"); btn.setFixedHeight(28)
            btn.clicked.connect(lambda c=False, mm=m: download_manager.add_to_queue(mm, cd))
            self.model_table.setCellWidget(row, 4, btn)
        if not download_manager.is_downloading() and not q:
            self.active_label.setText("No active download")
            self.active_progress.setVisible(False); self.file_detail.setVisible(False)
            self.speed_label.setText(""); self.eta_label.setText(""); self.bytes_label.setText("")
            self.cancel_btn.setEnabled(False)
    def _is_dl(self, m, cd):
        try:
            from huggingface_hub import try_to_load_from_cache
            p = try_to_load_from_cache(repo_id=m, filename="config.json", cache_dir=cd)
            return p is not None and os.path.exists(str(p))
        except: return False
    def _dl_all(self):
        cd = self.window.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        for r in range(self.model_table.rowCount()):
            m = self.model_table.item(r, 0).text()
            if not self._is_dl(m, cd): download_manager.add_to_queue(m, cd)
    def _dl_sel(self):
        r = self.model_table.currentRow()
        if r >= 0:
            cd = self.window.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
            download_manager.add_to_queue(self.model_table.item(r, 0).text(), cd)
    def _del_cache(self):
        r = self.model_table.currentRow()
        if r < 0: return
        m = self.model_table.item(r, 0).text()
        cd = self.window.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        sz = get_model_cache_size(cd, m)
        if sz == 0: QMessageBox.information(self, "No Cache", f"No cached files for {m}"); return
        if QMessageBox.question(self, "Delete", f"Delete {format_bytes(sz)} for {m}?") == QMessageBox.Yes:
            model_cache.evict(m); delete_model_cache(cd, m); self._refresh()
    def _on_started(self, m):
        self.active_label.setText(f"Downloading: {m}")
        self.active_progress.setVisible(True); self.active_progress.setValue(0)
        self.file_detail.setVisible(True); self.cancel_btn.setEnabled(True)
        self._refresh()
    def _on_completed(self, m, success, msg):
        if success:
            self.active_label.setText(f"Done: {m}")
        else:
            self.active_label.setText(f"Failed: {m}")
        self.active_progress.setVisible(False); self.file_detail.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self._refresh()
        if hasattr(self.window, '_on_download_complete'):
            self.window._on_download_complete(m, success)
    def _save_and_close(self):
        download_manager.settings["use_hf_transfer"] = self.hf_check.isChecked()
        download_manager.settings["auto_download"] = self.auto_dl_check.isChecked()
        download_manager.settings["hf_endpoint"] = self.mirror_edit.text().strip() or DEFAULT_HF_ENDPOINT
        download_manager.save_settings()
        self.accept()


# =========================================================================
# BATCH SETTINGS DIALOG
# =========================================================================

class BatchSettingsDialog(QDialog):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("Batch & Variation Settings"); self.setMinimumSize(700, 700)
        self.setStyleSheet(STYLESHEET)
        self.config = BatchConfig.load()
        layout = QVBoxLayout(self)
        pg = QGroupBox("Quick Presets"); pl = QHBoxLayout(pg)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([""] + list(BATCH_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        pl.addWidget(QLabel("Preset:")); pl.addWidget(self.preset_combo, 1)
        pl.addStretch()
        self.save_tpl_btn = QPushButton("Save Template"); self.save_tpl_btn.clicked.connect(self._save_template)
        pl.addWidget(self.save_tpl_btn)
        self.load_tpl_btn = QPushButton("Load Template"); self.load_tpl_btn.clicked.connect(self._load_template)
        pl.addWidget(self.load_tpl_btn)
        layout.addWidget(pg)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        sw = QWidget(); sl = QVBoxLayout(sw)
        vg = QGroupBox("Variation Settings"); vf = QFormLayout(vg)
        self.var_spin = QSpinBox(); self.var_spin.setRange(1, 1000); self.var_spin.setValue(self.config.num_variations)
        vf.addRow("Variations:", self.var_spin)
        self.algo_combo = QComboBox(); self.algo_combo.addItems(list(SEED_ALGORITHMS_ADVANCED.keys()))
        self.algo_combo.setCurrentText(self.config.seed_algo)
        vf.addRow("Seed algorithm:", self.algo_combo)
        self.algo_desc = QLabel(""); self.algo_desc.setStyleSheet("color: #888; font-size: 10px;"); self.algo_desc.setWordWrap(True)
        vf.addRow("", self.algo_desc)
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        self.seed_start = QSpinBox(); self.seed_start.setRange(-1, 999999999); self.seed_start.setValue(self.config.seed_start)
        vf.addRow("Base seed:", self.seed_start)
        self.seed_end = QSpinBox(); self.seed_end.setRange(0, 999999999); self.seed_end.setValue(self.config.seed_end)
        vf.addRow("End seed (Interpolated):", self.seed_end)
        self.seed_step = QSpinBox(); self.seed_step.setRange(1, 999999999); self.seed_step.setValue(self.config.seed_step)
        vf.addRow("Step (Sequential):", self.seed_step)
        self.seed_ratio = QDoubleSpinBox(); self.seed_ratio.setRange(1.0, 10.0); self.seed_ratio.setValue(self.config.seed_ratio)
        vf.addRow("Ratio (Geometric):", self.seed_ratio)
        self.seed_walk = QSpinBox(); self.seed_walk.setRange(1000, 100000000); self.seed_walk.setSingleStep(10000); self.seed_walk.setValue(self.config.seed_walk_max)
        vf.addRow("Max walk (Noise Walk):", self.seed_walk)
        self.preview_btn = QPushButton("Preview Seeds"); self.preview_btn.clicked.connect(self._preview_seeds)
        vf.addRow("", self.preview_btn)
        self.seed_preview = QLabel(""); self.seed_preview.setStyleSheet("color: #2A82DA; font-size: 10px; font-family: monospace;"); self.seed_preview.setWordWrap(True)
        vf.addRow("", self.seed_preview)
        sl.addWidget(vg)
        fg = QGroupBox("Filename & Output"); ff = QFormLayout(fg)
        self.fn_pattern = QLineEdit(); self.fn_pattern.setText(self.config.filename_pattern)
        ff.addRow("Filename pattern:", self.fn_pattern)
        self.fn_preview = QLabel(""); self.fn_preview.setStyleSheet("color: #2ecc71; font-size: 10px; font-family: monospace;")
        ff.addRow("Preview:", self.fn_preview)
        self.fn_pattern.textChanged.connect(self._update_fn_preview)
        self.org_combo = QComboBox(); self.org_combo.addItems(list(OUTPUT_ORG_MODES.keys()))
        self.org_combo.setCurrentText(self.config.output_org_mode)
        ff.addRow("Organization:", self.org_combo)
        sl.addWidget(fg)
        qg = QGroupBox("Quality & Auto-Adjust"); qf = QFormLayout(qg)
        self.auto_adj = QCheckBox("Auto-adjust based on batch size"); self.auto_adj.setChecked(self.config.auto_adjust)
        qf.addRow("", self.auto_adj)
        self.adj_agg = QComboBox(); self.adj_agg.addItems(["conservative","moderate","aggressive"])
        self.adj_agg.setCurrentText(self.config.auto_adjust_aggressiveness)
        qf.addRow("Adjust level:", self.adj_agg)
        sl.addWidget(qg)
        eg = QGroupBox("Error Handling"); el = QVBoxLayout(eg)
        self.skip_err = QCheckBox("Skip failed variations"); self.skip_err.setChecked(self.config.skip_on_error)
        el.addWidget(self.skip_err)
        self.retry_chk = QCheckBox("Auto-retry failed with new seed"); self.retry_chk.setChecked(self.config.auto_retry_failed)
        el.addWidget(self.retry_chk)
        sl.addWidget(eg)
        mg = QGroupBox("Memory"); ml = QVBoxLayout(mg)
        self.early_pv = QCheckBox("Early preview"); self.early_pv.setChecked(self.config.early_preview)
        ml.addWidget(self.early_pv)
        self.aggr_gc = QCheckBox("Aggressive GC"); self.aggr_gc.setChecked(self.config.aggressive_gc)
        ml.addWidget(self.aggr_gc)
        sl.addWidget(mg)
        ng = QGroupBox("Notifications"); nf = QFormLayout(ng)
        self.notif = QComboBox(); self.notif.addItems(list(NOTIFICATION_OPTIONS.keys()))
        self.notif.setCurrentText(self.config.notification_mode)
        nf.addRow("Type:", self.notif)
        sl.addWidget(ng)
        tg = QGroupBox("Tags & Export"); tf = QFormLayout(tg)
        self.tags = QLineEdit(); self.tags.setText(self.config.tags)
        tf.addRow("Tags:", self.tags)
        self.zip_chk = QCheckBox("Auto-export ZIP"); self.zip_chk.setChecked(self.config.auto_export_zip)
        tf.addRow("", self.zip_chk)
        sl.addWidget(tg)
        sg = QGroupBox("Statistics"); sgl = QVBoxLayout(sg)
        summ = batch_stats.get_summary()
        if summ:
            st = (f"Total batches: {summ.get('total_batches',0)}\n"
                  f"Files generated: {summ.get('total_files_generated',0)}\n"
                  f"Success rate: {summ.get('overall_success_rate',0):.1f}%\n"
                  f"Total time: {format_time(summ.get('total_generation_time_s',0))}\n"
                  f"Disk usage: {summ.get('total_disk_usage_mb',0):.0f} MB")
        else: st = "No stats yet."
        self.stats_lbl = QLabel(st); self.stats_lbl.setStyleSheet("color: #2A82DA; font-size: 10px;"); self.stats_lbl.setWordWrap(True)
        sgl.addWidget(self.stats_lbl)
        sl.addWidget(sg)
        sl.addStretch()
        scroll.setWidget(sw); layout.addWidget(scroll)
        br = QHBoxLayout()
        self.reset_btn = QPushButton("Reset"); self.reset_btn.clicked.connect(self._reset)
        br.addWidget(self.reset_btn); br.addStretch()
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.clicked.connect(self.reject)
        br.addWidget(self.cancel_btn)
        self.apply_btn = QPushButton("Apply & Save")
        self.apply_btn.setStyleSheet("QPushButton { background-color: #2A82DA; color: white; font-weight: bold; } QPushButton:hover { background-color: #3A92EA; }")
        self.apply_btn.clicked.connect(self._apply)
        br.addWidget(self.apply_btn)
        layout.addLayout(br)
        self._on_algo_changed(self.config.seed_algo)
        self._update_fn_preview()
    def _on_algo_changed(self, a):
        self.algo_desc.setText(SEED_ALGORITHMS_ADVANCED.get(a, ""))
        self.seed_end.setEnabled(a == "Interpolated")
        self.seed_step.setEnabled(a == "Sequential")
        self.seed_ratio.setEnabled(a == "Geometric")
        self.seed_walk.setEnabled(a == "Noise Walk")
    def _update_fn_preview(self):
        p = self.fn_pattern.text()
        if not p: self.fn_preview.setText("(empty)"); return
        dp = GenerationParams(prompt="Test", negative_prompt="", model_name="cvssp/audioldm2", duration=10.0, steps=200, guidance=3.5, seed=42, device="cuda", cache_dir=".", output_dir=".")
        dc = BatchConfig(filename_pattern=p, tags=self.tags.text())
        try:
            self.fn_preview.setText("→ " + render_filename_pattern(p, dp, 0, 42, dc, "batch123", 0))
        except Exception as e:
            self.fn_preview.setText(f"Error: {e}")
    def _preview_seeds(self):
        n = min(self.var_spin.value(), 50)
        bc = BatchConfig(num_variations=n, seed_algo=self.algo_combo.currentText(), seed_start=self.seed_start.value(), seed_end=self.seed_end.value(), seed_step=self.seed_step.value(), seed_ratio=self.seed_ratio.value(), seed_walk_max=self.seed_walk.value())
        base = self.seed_start.value() if self.seed_start.value() >= 0 else 12345
        seeds = compute_batch_seeds(base, n, self.algo_combo.currentText(), bc)
        prev = ", ".join(str(s) for s in seeds[:20])
        if len(seeds) > 20: prev += f" ... (+{len(seeds)-20} more)"
        self.seed_preview.setText(f"Seeds: {prev}")
    def _apply_preset(self, name):
        if not name: return
        p = BATCH_PRESETS.get(name)
        if not p: return
        self.var_spin.setValue(p["num_variations"])
        self.algo_combo.setCurrentText(p["seed_algo"])
        self.skip_err.setChecked(p["skip_on_error"])
        self.early_pv.setChecked(p["early_preview"])
        self.aggr_gc.setChecked(p["aggressive_gc"])
        self.window.perf_mode_combo.setCurrentText(p["performance_mode"])
        self.window.duration_spin.setValue(p["duration"])
    def _save_template(self):
        name, ok = QInputDialog.getText(self, "Save Template", "Name:")
        if not ok or not name: return
        self._collect()
        dp = GenerationParams(prompt=self.window.prompt_input.toPlainText(), negative_prompt=self.window.neg_prompt_input.text(), model_name=self.window.model_combo.currentText(), duration=self.window.duration_spin.value(), steps=self.window.steps_spin.value(), guidance=self.window.guidance_spin.value(), seed=self.window.seed_spin.value(), device="", cache_dir="", output_dir="")
        batch_template_manager.save_template(name, self.config, dp)
        QMessageBox.information(self, "Saved", f"Template '{name}' saved!")
    def _load_template(self):
        templates = batch_template_manager.list_templates()
        if not templates:
            QMessageBox.information(self, "No Templates", "No saved templates."); return
        name, ok = QInputDialog.getItem(self, "Load Template", "Select:", templates, 0, False)
        if not ok or not name: return
        config, params = batch_template_manager.load_template(name)
        if config:
            self.config = config; self._populate()
            if params:
                self.window.prompt_input.setPlainText(params.get("prompt", ""))
                self.window.neg_prompt_input.setText(params.get("negative_prompt", ""))
                self.window.model_combo.setCurrentText(params.get("model_name", ""))
                self.window.duration_spin.setValue(params.get("duration", 10.0))
                self.window.steps_spin.setValue(params.get("steps", 200))
                self.window.guidance_spin.setValue(params.get("guidance", 3.5))
                self.window.seed_spin.setValue(params.get("seed", -1))
    def _collect(self):
        c = self.config
        c.num_variations = self.var_spin.value()
        c.seed_algo = self.algo_combo.currentText()
        c.seed_start = self.seed_start.value()
        c.seed_end = self.seed_end.value()
        c.seed_step = self.seed_step.value()
        c.seed_ratio = self.seed_ratio.value()
        c.seed_walk_max = self.seed_walk.value()
        c.filename_pattern = self.fn_pattern.text()
        c.output_org_mode = self.org_combo.currentText()
        c.auto_adjust = self.auto_adj.isChecked()
        c.auto_adjust_aggressiveness = self.adj_agg.currentText()
        c.skip_on_error = self.skip_err.isChecked()
        c.auto_retry_failed = self.retry_chk.isChecked()
        c.early_preview = self.early_pv.isChecked()
        c.aggressive_gc = self.aggr_gc.isChecked()
        c.notification_mode = self.notif.currentText()
        c.tags = self.tags.text()
        c.auto_export_zip = self.zip_chk.isChecked()
        return c
    def _populate(self):
        c = self.config
        self.var_spin.setValue(c.num_variations)
        self.algo_combo.setCurrentText(c.seed_algo)
        self.seed_start.setValue(c.seed_start)
        self.seed_end.setValue(c.seed_end)
        self.seed_step.setValue(c.seed_step)
        self.seed_ratio.setValue(c.seed_ratio)
        self.seed_walk.setValue(c.seed_walk_max)
        self.fn_pattern.setText(c.filename_pattern)
        self.org_combo.setCurrentText(c.output_org_mode)
        self.auto_adj.setChecked(c.auto_adjust)
        self.adj_agg.setCurrentText(c.auto_adjust_aggressiveness)
        self.skip_err.setChecked(c.skip_on_error)
        self.retry_chk.setChecked(c.auto_retry_failed)
        self.early_pv.setChecked(c.early_preview)
        self.aggr_gc.setChecked(c.aggressive_gc)
        self.notif.setCurrentText(c.notification_mode)
        self.tags.setText(c.tags)
        self.zip_chk.setChecked(c.auto_export_zip)
        self._on_algo_changed(c.seed_algo)
        self._update_fn_preview()
    def _reset(self):
        self.config = BatchConfig(); self._populate()
    def _apply(self):
        self._collect(); self.config.save()
        self.window.variations_spin.setValue(self.config.num_variations)
        self.window.seed_algo_combo.setCurrentText(self.config.seed_algo)
        self.window.seed_spin.setValue(self.config.seed_start)
        self.window.skip_error_check.setChecked(self.config.skip_on_error)
        self.window.early_preview_check.setChecked(self.config.early_preview)
        self.window.aggressive_gc_check.setChecked(self.config.aggressive_gc)
        self.window.batch_config = self.config
        self.accept()


# =========================================================================
# MAIN WINDOW
# =========================================================================

class AudioLDM2Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)
        if not ML_AVAILABLE:
            QMessageBox.critical(self, "Missing Dependencies", "Required ML libraries (torch/diffusers/transformers) not installed.")
            sys.exit(1)
        self.settings_file = os.path.join(CONFIG_DIR, SETTINGS_FILE_NAME)
        self.settings = self.load_json_settings()
        cd = self.settings.get("cache_dir", "")
        if cd:
            try: os.makedirs(cd, exist_ok=True)
            except: pass
            os.environ["HF_HOME"] = cd
            os.environ["HF_HUB_CACHE"] = cd
        self.worker_thread = None
        self._preload_thread = None
        self._gen_queue = deque()
        self.base_audio_path = None
        self.current_audio_path = None
        self.batch_config = BatchConfig.load()
        self.setAcceptDrops(True)
        self.setup_ui()
        self.setup_audio()
        self.setup_statusbar()
        self.load_settings()
        self.setup_shortcuts()
        QTimer.singleShot(500, self._maybe_preload_model)
        QTimer.singleShot(1000, self._check_for_interrupted_batch)
        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._update_vram_info)
        self._vram_timer.start(5000)
        QTimer.singleShot(2000, self._update_vram_info)
        self.variations_spin.valueChanged.connect(self._update_batch_estimate)
        self.duration_spin.valueChanged.connect(self._update_batch_estimate)
        self.steps_spin.valueChanged.connect(self._update_batch_estimate)
        download_manager.download_started.connect(lambda m: self.status_bar.showMessage(f"Downloading {m}...", 3000), Qt.QueuedConnection)
        download_manager.download_completed.connect(self._on_download_complete, Qt.QueuedConnection)

    def load_json_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def save_json_settings(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def setup_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        ml = QHBoxLayout(cw); ml.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal); ml.addWidget(splitter)
        self.create_left_panel(splitter)
        self.create_right_panel(splitter)
        splitter.setSizes([450, 950])

    def create_left_panel(self, parent):
        lp = QWidget(); ll = QVBoxLayout(lp); ll.setContentsMargins(0, 0, 0, 0)
        self.tab_widget = QTabWidget(); ll.addWidget(self.tab_widget)
        gen_tab = QWidget(); layout = QVBoxLayout(gen_tab)
        layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(8)
        pg = QGroupBox("Prompt"); pl = QVBoxLayout(pg)
        self.prompt_input = PromptHistoryLineEdit(); pl.addWidget(self.prompt_input)
        pr = QHBoxLayout()
        self.preset_combo = QComboBox(); self.preset_combo.addItems([""] + DEFAULT_PROMPT_PRESETS)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        pr.addWidget(self.preset_combo, 1)
        self.sfx_builder_btn = QPushButton("SFX"); self.sfx_builder_btn.setFixedWidth(60)
        self.sfx_builder_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        pr.addWidget(self.sfx_builder_btn)
        pl.addLayout(pr)
        self.multi_prompt_check = QCheckBox("Use Prompt Sequence")
        self.multi_prompt_check.toggled.connect(self.toggle_multi_prompt)
        pl.addWidget(self.multi_prompt_check)
        er = QHBoxLayout()
        self.multi_prompt_edit_btn = QPushButton("Open Sequence Editor")
        self.multi_prompt_edit_btn.setFixedHeight(28)
        self.multi_prompt_edit_btn.clicked.connect(self._focus_multi_prompt_tab)
        er.addWidget(self.multi_prompt_edit_btn); er.addStretch()
        pl.addLayout(er)
        self.multi_prompt_edit = QPlainTextEdit()
        self.multi_prompt_edit.setPlaceholderText("One prompt per line...\nFormat: prompt | seed_algo | duration")
        self.multi_prompt_edit.setVisible(False); self.multi_prompt_edit.setMaximumHeight(120)
        pl.addWidget(self.multi_prompt_edit)
        self.neg_prompt_input = QLineEdit(); self.neg_prompt_input.setPlaceholderText("Negative prompt (optional)...")
        pl.addWidget(self.neg_prompt_input)
        layout.addWidget(pg)
        parg = QGroupBox("Parameters"); fl = QFormLayout(parg); fl.setSpacing(6)
        mr = QHBoxLayout(); mr.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox(); self.model_combo.setEditable(True)
        self.model_combo.addItems(DEFAULT_MODELS)
        for cm in self.settings.get("custom_models", []):
            if cm not in DEFAULT_MODELS: self.model_combo.addItem(cm)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        mr.addWidget(self.model_combo, 1)
        self.manage_models_btn = QPushButton("⚙"); self.manage_models_btn.setFixedWidth(30)
        self.manage_models_btn.setToolTip("Model Manager")
        self.manage_models_btn.clicked.connect(self._open_model_manager)
        mr.addWidget(self.manage_models_btn)
        self.download_model_btn = QPushButton("⬇"); self.download_model_btn.setFixedWidth(30)
        self.download_model_btn.setToolTip("Download Model")
        self.download_model_btn.clicked.connect(self._download_current_model)
        mr.addWidget(self.download_model_btn)
        self.dl_manager_btn = QPushButton("📦"); self.dl_manager_btn.setFixedWidth(30)
        self.dl_manager_btn.setToolTip("Download Manager")
        self.dl_manager_btn.clicked.connect(self._open_download_manager)
        mr.addWidget(self.dl_manager_btn)
        fl.addRow(mr)
        self.model_status_label = QLabel(""); self.model_status_label.setStyleSheet("color: #888; font-size: 10px;")
        fl.addRow("", self.model_status_label)
        self.model_info_label = QLabel(""); self.model_info_label.setStyleSheet("color: #2A82DA; font-size: 10px;"); self.model_info_label.setWordWrap(True)
        fl.addRow("", self.model_info_label)
        self.duration_spin = QDoubleSpinBox(); self.duration_spin.setRange(1.0, MAX_DURATION_S); self.duration_spin.setValue(10.0); self.duration_spin.setSuffix(" s"); self.duration_spin.setDecimals(1)
        fl.addRow("Duration:", self.duration_spin)
        self.duration_info = QLabel("→ Single chunk."); self.duration_info.setStyleSheet("color: #888; font-size: 10px;")
        fl.addRow("", self.duration_info)
        self.duration_spin.valueChanged.connect(self.update_duration_info)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(10, 500); self.steps_spin.setValue(200)
        fl.addRow("Steps:", self.steps_spin)
        self.guidance_spin = QDoubleSpinBox(); self.guidance_spin.setRange(1.0, 20.0); self.guidance_spin.setValue(3.5); self.guidance_spin.setSingleStep(0.5)
        fl.addRow("Guidance:", self.guidance_spin)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 999999999); self.seed_spin.setValue(-1)
        fl.addRow("Seed:", self.seed_spin)
        self.seed_algo_combo = QComboBox(); self.seed_algo_combo.addItems(list(SEED_ALGORITHMS_ADVANCED.keys()))
        fl.addRow("Seed Algo:", self.seed_algo_combo)
        self.variations_spin = QSpinBox(); self.variations_spin.setRange(1, 1000); self.variations_spin.setValue(1)
        self.variations_spin.setToolTip("1-1000 variations")
        fl.addRow("Variations:", self.variations_spin)
        bsr = QHBoxLayout()
        self.batch_settings_btn = QPushButton("⚙ Batch Settings")
        self.batch_settings_btn.setStyleSheet("QPushButton { background-color: #8e44ad; color: white; font-weight: bold; padding: 6px; border-radius: 3px; } QPushButton:hover { background-color: #9e54bd; }")
        self.batch_settings_btn.clicked.connect(self._open_batch_settings)
        bsr.addWidget(self.batch_settings_btn)
        fl.addRow("", bsr)
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto","cpu","cuda","mps"])
        if torch.cuda.is_available(): self.device_combo.setCurrentText("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): self.device_combo.setCurrentText("mps")
        self.device_combo.currentTextChanged.connect(lambda _: self._maybe_preload_model())
        fl.addRow("Device:", self.device_combo)
        self.cpu_offload_check = QCheckBox("CPU Offload (Low VRAM)")
        fl.addRow("", self.cpu_offload_check)
        cr = QHBoxLayout()
        self.cache_dir_edit = QLineEdit(); self.cache_dir_edit.setPlaceholderText(DEFAULT_CACHE_DIR)
        cb = QPushButton("…"); cb.setFixedWidth(30); cb.clicked.connect(self.browse_cache_dir)
        cr.addWidget(self.cache_dir_edit); cr.addWidget(cb)
        fl.addRow("Cache:", cr)
        orr = QHBoxLayout()
        self.output_dir_edit = QLineEdit(); self.output_dir_edit.setPlaceholderText("./generated_audio")
        ob = QPushButton("…"); ob.setFixedWidth(30); ob.clicked.connect(self.browse_output_dir)
        orr.addWidget(self.output_dir_edit); orr.addWidget(ob)
        fl.addRow("Output:", orr)
        self.section_edit = QLineEdit(); self.section_edit.setPlaceholderText("Subfolder (optional)")
        fl.addRow("Section:", self.section_edit)
        layout.addWidget(parg)
        perfg = QGroupBox("Smart Performance"); perfl = QVBoxLayout(perfg)
        pfrow = QHBoxLayout(); pfrow.addWidget(QLabel("Mode:"))
        self.perf_mode_combo = QComboBox(); self.perf_mode_combo.addItems(list(PERFORMANCE_PRESETS.keys()))
        self.perf_mode_combo.setCurrentText("Custom")
        self.perf_mode_combo.currentTextChanged.connect(self._on_perf_mode_changed)
        pfrow.addWidget(self.perf_mode_combo, 1); perfl.addLayout(pfrow)
        self.perf_note_label = QLabel("Manual control"); self.perf_note_label.setStyleSheet("color: #888; font-size: 10px;"); self.perf_note_label.setWordWrap(True)
        perfl.addWidget(self.perf_note_label)
        self.skip_error_check = QCheckBox("Auto-skip failed variations"); self.skip_error_check.setChecked(True)
        perfl.addWidget(self.skip_error_check)
        self.early_preview_check = QCheckBox("Early preview"); self.early_preview_check.setChecked(True)
        perfl.addWidget(self.early_preview_check)
        self.aggressive_gc_check = QCheckBox("Aggressive GC"); self.aggressive_gc_check.setChecked(False)
        perfl.addWidget(self.aggressive_gc_check)
        self.vram_label = QLabel(""); self.vram_label.setStyleSheet("color: #2ecc71; font-size: 10px;")
        perfl.addWidget(self.vram_label)
        self.batch_estimate_label = QLabel(""); self.batch_estimate_label.setStyleSheet("color: #2A82DA; font-size: 10px; font-weight: bold;"); self.batch_estimate_label.setWordWrap(True)
        perfl.addWidget(self.batch_estimate_label)
        layout.addWidget(perfg)
        postg = QGroupBox("Post-Processing"); postl = QVBoxLayout(postg)
        self.normalize_check = QCheckBox("Normalize"); self.normalize_check.setChecked(True); postl.addWidget(self.normalize_check)
        self.trim_check = QCheckBox("Trim silence"); self.trim_check.setChecked(True); postl.addWidget(self.trim_check)
        fr = QHBoxLayout()
        self.fade_in_check = QCheckBox("Fade in"); self.fade_in_check.setChecked(True); fr.addWidget(self.fade_in_check)
        self.fade_out_check = QCheckBox("Fade out"); self.fade_out_check.setChecked(True); fr.addWidget(self.fade_out_check)
        postl.addLayout(fr)
        layout.addWidget(postg)
        gbr = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Audio"); self.generate_btn.setMinimumHeight(45)
        self.generate_btn.setStyleSheet("QPushButton { background-color: #2A82DA; color: white; font-size: 14px; font-weight: bold; border: none; border-radius: 4px; } QPushButton:hover { background-color: #3A92EA; }")
        self.generate_btn.clicked.connect(self.start_generation)
        gbr.addWidget(self.generate_btn)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setMinimumHeight(45); self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton { background-color: #c0392b; color: white; font-size: 14px; font-weight: bold; border: none; border-radius: 4px; } QPushButton:hover { background-color: #d0493b; }")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        gbr.addWidget(self.cancel_btn)
        layout.addLayout(gbr)
        stg = QGroupBox("Status"); stl = QVBoxLayout(stg)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); stl.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready"); self.status_label.setStyleSheet("font-weight: bold;"); stl.addWidget(self.status_label)
        self.eta_label = QLabel(""); self.eta_label.setAlignment(Qt.AlignRight); stl.addWidget(self.eta_label)
        layout.addWidget(stg)
        qg = QGroupBox("Queue"); ql = QVBoxLayout(qg)
        self.queue_list = QListWidget(); self.queue_list.setMaximumHeight(60); self.queue_list.setAlternatingRowColors(True)
        ql.addWidget(self.queue_list)
        self.queue_clear_btn = QPushButton("Clear Queue"); self.queue_clear_btn.clicked.connect(self.clear_queue)
        ql.addWidget(self.queue_clear_btn)
        layout.addWidget(qg)
        layout.addStretch()
        gs = QScrollArea(); gs.setWidgetResizable(True); gs.setFrameShape(QFrame.NoFrame); gs.setWidget(gen_tab)
        self.tab_widget.addTab(gs, "Generate")
        self.sfx_builder_tab = SFXPromptBuilderTab()
        self.sfx_builder_tab.insertPromptRequested.connect(self._insert_sfx_prompt)
        self.tab_widget.addTab(self.sfx_builder_tab, "SFX Builder")
        self.multi_prompt_tab = MultiPromptEditorTab()
        self.multi_prompt_tab.applyRequested.connect(self._apply_multi_prompt_tab)
        self.tab_widget.addTab(self.multi_prompt_tab, "Sequence Editor")
        parent.addWidget(lp)

    def create_right_panel(self, parent):
        rp = QWidget(); rl = QVBoxLayout(rp); rl.setContentsMargins(0, 0, 0, 0)
        vg = QGroupBox("Waveform"); vl = QVBoxLayout(vg)
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.seekRequested.connect(self._seek_to)
        vl.addWidget(self.waveform_widget)
        rl.addWidget(vg)
        plg = QGroupBox("Playback"); pll = QVBoxLayout(plg)
        prow = QHBoxLayout()
        self.load_btn = QPushButton("Load WAV"); self.load_btn.setFixedHeight(36); self.load_btn.clicked.connect(self.open_audio)
        prow.addWidget(self.load_btn)
        self.play_btn = QPushButton("Play"); self.play_btn.setFixedHeight(36); self.play_btn.setMinimumWidth(80); self.play_btn.setEnabled(False); self.play_btn.clicked.connect(self.toggle_playback)
        prow.addWidget(self.play_btn)
        self.save_btn = QPushButton("Save As..."); self.save_btn.setFixedHeight(36); self.save_btn.setEnabled(False); self.save_btn.clicked.connect(self.save_audio)
        prow.addWidget(self.save_btn)
        pll.addLayout(prow)
        vol = QHBoxLayout(); vol.addWidget(QLabel("Vol:"))
        self.vol_slider = QSlider(Qt.Horizontal); self.vol_slider.setRange(0, 100); self.vol_slider.setValue(50); self.vol_slider.valueChanged.connect(self.set_volume)
        vol.addWidget(self.vol_slider)
        self.vol_label = QLabel("50%"); self.vol_label.setFixedWidth(35); vol.addWidget(self.vol_label)
        pll.addLayout(vol)
        self.time_label = QLabel("00:00 / 00:00"); self.time_label.setAlignment(Qt.AlignCenter); self.time_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2A82DA;")
        pll.addWidget(self.time_label)
        rl.addWidget(plg)
        hg = QGroupBox("History"); hl = QVBoxLayout(hg)
        self.playlist_table = QTableWidget()
        self.playlist_table.setColumnCount(4)
        self.playlist_table.setHorizontalHeaderLabels(["Prompt","Duration","Date","Seed"])
        self.playlist_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.playlist_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.playlist_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.playlist_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.playlist_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.playlist_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.playlist_table.setAlternatingRowColors(True)
        self.playlist_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.playlist_table.customContextMenuRequested.connect(self.show_history_context_menu)
        self.playlist_table.itemDoubleClicked.connect(self.play_from_playlist)
        hl.addWidget(self.playlist_table)
        rl.addWidget(hg)
        parent.addWidget(rp)

    def _on_preset_changed(self, text):
        if not text: return
        if self.multi_prompt_check.isChecked():
            self.multi_prompt_edit.setPlainText(text)
        else:
            self.prompt_input.setPlainText(text)

    def show_history_context_menu(self, pos):
        item = self.playlist_table.itemAt(pos)
        if not item: return
        row = item.row()
        pi = self.playlist_table.item(row, 0)
        if not pi: return
        prompt = pi.text()
        menu = QMenu(self)
        ca = QAction("Copy Prompt", self); ca.triggered.connect(lambda: QApplication.clipboard().setText(prompt))
        menu.addAction(ca)
        ua = QAction("Use Prompt", self); ua.triggered.connect(lambda: self.use_prompt_from_history(prompt))
        menu.addAction(ua)
        menu.exec(self.playlist_table.viewport().mapToGlobal(pos))

    def use_prompt_from_history(self, prompt):
        if self.multi_prompt_check.isChecked():
            self.multi_prompt_edit.setPlainText(prompt)
        else:
            self.prompt_input.setPlainText(prompt)

    def _insert_sfx_prompt(self, text):
        if not text: return
        if self.multi_prompt_check.isChecked():
            c = self.multi_prompt_edit.toPlainText().strip()
            self.multi_prompt_edit.setPlainText(f"{c}, {text}" if c else text)
        else:
            c = self.prompt_input.toPlainText().strip()
            self.prompt_input.setPlainText(f"{c}, {text}" if c else text)
        self.tab_widget.setCurrentIndex(0)

    def _focus_multi_prompt_tab(self):
        self.multi_prompt_tab.update_defaults(self.seed_algo_combo.currentText(), self.duration_spin.value())
        self.multi_prompt_tab.load_from_text(self.multi_prompt_edit.toPlainText())
        self.tab_widget.setCurrentIndex(2)

    def _apply_multi_prompt_tab(self, text, total_duration):
        self.multi_prompt_edit.setPlainText(text)
        self.multi_prompt_check.setChecked(True)
        self.duration_spin.setValue(min(total_duration, MAX_DURATION_S))
        self.tab_widget.setCurrentIndex(0)

    def toggle_multi_prompt(self, checked):
        self.multi_prompt_edit.setVisible(checked)
        self.multi_prompt_edit_btn.setVisible(checked)
        self.prompt_input.setVisible(not checked)
        self.preset_combo.setVisible(not checked)
        self.sfx_builder_btn.setVisible(not checked)

    def update_duration_info(self):
        d = self.duration_spin.value()
        if d > LONG_FORM_THRESHOLD_S:
            self.duration_info.setText(f"Long-form: {int(AUTO_CHUNK_SIZE_S)}s chunks, {OVERLAP_S}s crossfade.")
            self.duration_info.setStyleSheet("color: #2ecc71; font-size: 10px;")
        else:
            self.duration_info.setText("→ Single chunk.")
            self.duration_info.setStyleSheet("color: #888; font-size: 10px;")
        self._update_batch_estimate()

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
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.status_bar.showMessage("Ready")

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.start_generation)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.toggle_playback)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.cancel_generation)

    def browse_cache_dir(self):
        cur = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        path = QFileDialog.getExistingDirectory(self, "Cache Folder", cur or "")
        if path:
            self.cache_dir_edit.setText(path)
            self.settings["cache_dir"] = path; self.save_json_settings()
            self._maybe_preload_model()

    def browse_output_dir(self):
        cur = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        path = QFileDialog.getExistingDirectory(self, "Output Folder", cur)
        if path: self.output_dir_edit.setText(path)

    def _resolve_device(self):
        d = self.device_combo.currentText()
        if d == "auto":
            if torch.cuda.is_available(): return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
            return "cpu"
        return d

    def _on_model_changed(self, model_name):
        if not model_name or '/' not in model_name:
            self.model_status_label.setText(""); self.model_info_label.setText(""); return
        vram = MODEL_VRAM_ESTIMATES.get(model_name, 3500)
        desc = MODEL_DESCRIPTIONS.get(model_name, "Custom model")
        cached = model_cache.is_cached(model_name)
        downloaded = self._is_model_downloaded(model_name)
        queued = download_manager.is_queued(model_name)
        downloading = download_manager.is_downloading() and download_manager.get_current_task() and download_manager.get_current_task().model_name == model_name
        if cached: st = "✓ Loaded in VRAM"; color = "#2ecc71"
        elif downloading: st = "⏳ Downloading..."; color = "#f39c12"
        elif queued: st = "📋 Queued"; color = "#9b59b6"
        elif downloaded: st = "Cached on disk"; color = "#2A82DA"
        else: st = "⚠ Not downloaded"; color = "#e74c3c"
        self.model_status_label.setText(st); self.model_status_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.model_info_label.setText(f"💾 {vram} MB VRAM | {desc}")
        profile_manager.apply_profile(model_name, self)
        if downloaded and not cached: self._maybe_preload_model()

    def _is_model_downloaded(self, model_name):
        try:
            from huggingface_hub import try_to_load_from_cache
            cd = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
            p = try_to_load_from_cache(repo_id=model_name, filename="config.json", cache_dir=cd)
            return p is not None and os.path.exists(str(p))
        except: return False

    def _open_model_manager(self):
        ModelManagerDialog(self, parent=self).exec()
        self._on_model_changed(self.model_combo.currentText())

    def _open_download_manager(self):
        DownloadManagerDialog(self, parent=self).exec()
        self._on_model_changed(self.model_combo.currentText())

    def _download_current_model(self):
        model = self.model_combo.currentText().strip()
        if not model or '/' not in model:
            QMessageBox.warning(self, "Invalid", "Select a valid model."); return
        cd = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        if download_manager.is_downloading() or download_manager.is_queued(model):
            QMessageBox.information(self, "Already Downloading", f"{model} is already downloading/queued."); return
        if self._is_model_downloaded(model):
            if QMessageBox.question(self, "Downloaded", f"{model} already downloaded. Re-download?") != QMessageBox.Yes: return
        download_manager.add_to_queue(model, cd)
        self.status_bar.showMessage(f"Added {model} to download queue", 3000)

    def _maybe_preload_model(self):
        if not ML_AVAILABLE: return
        if self._preload_thread and self._preload_thread.isRunning(): return
        model = self.model_combo.currentText().strip()
        if not model or '/' not in model: return
        device = self._resolve_device()
        cd = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        os.makedirs(cd, exist_ok=True)
        if model_cache.is_cached(model, device, self.cpu_offload_check.isChecked()):
            self.model_status_label.setText("✓ Already loaded"); self.model_status_label.setStyleSheet("color: #2ecc71; font-size: 10px;"); return
        if not self._is_model_downloaded(model):
            self.model_status_label.setText("⚠ Not downloaded - click ⬇"); self.model_status_label.setStyleSheet("color: #e74c3c; font-size: 10px;"); return
        self._start_preload(model, device, cd)

    def _start_preload(self, model_name, device, cache_dir):
        if self._preload_thread and self._preload_thread.isRunning(): return
        self._preload_thread = ModelPreloadThread(model_name, cache_dir, device, self.cpu_offload_check.isChecked(), parent=self)
        self._preload_thread.status_updated.connect(self.model_status_label.setText, Qt.QueuedConnection)
        self._preload_thread.progress_updated.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        self._preload_thread.finished_preload.connect(self._on_preload_done, Qt.QueuedConnection)
        self._preload_thread.start()

    def _on_preload_done(self, model_name, success, message):
        if success:
            self.model_status_label.setText(f"✓ {model_name} ready" if message != "cached" else f"✓ {model_name} cached")
            self.model_status_label.setStyleSheet("color: #2ecc71; font-size: 10px;")
            self.status_bar.showMessage(f"Model '{model_name}' ready", 5000)
        else:
            self.model_status_label.setText(f"✗ Failed: {message[:50]}")
            self.model_status_label.setStyleSheet("color: #e74c3c; font-size: 10px;")
        t = self._preload_thread; self._preload_thread = None
        if t: t.deleteLater()

    def _refresh_queue_view(self):
        self.queue_list.clear()
        for i, p in enumerate(self._gen_queue, 1):
            self.queue_list.addItem(f"{i}. [{sanitize_filename(p.model_name, 20)}] {p.prompt[:60]}")
        if self._gen_queue: self.status_bar.showMessage(f"Queue: {len(self._gen_queue)} pending", 3000)

    def clear_queue(self):
        n = len(self._gen_queue); self._gen_queue.clear(); self._refresh_queue_view()
        if n: self.status_bar.showMessage(f"Cleared {n} queued item(s)", 3000)

    def _on_perf_mode_changed(self, mode):
        preset = PERFORMANCE_PRESETS.get(mode, {})
        self.perf_note_label.setText(preset.get("note", ""))
        steps = preset.get("steps"); guidance = preset.get("guidance")
        if steps is not None: self.steps_spin.setValue(steps); self.steps_spin.setEnabled(False)
        else: self.steps_spin.setEnabled(True)
        if guidance is not None: self.guidance_spin.setValue(guidance); self.guidance_spin.setEnabled(False)
        else: self.guidance_spin.setEnabled(True)
        self._update_batch_estimate()

    def _update_vram_info(self):
        gpu = get_gpu_info()
        if gpu:
            color = "#2ecc71" if gpu["free_mb"] > VRAM_WARNING_MB else "#e74c3c"
            if gpu["free_mb"] < VRAM_CRITICAL_MB:
                color = "#e74c3c"; self.cpu_offload_check.setChecked(True)
            self.vram_label.setText(f"GPU: {gpu['name'][:30]} | VRAM: {gpu['free_mb']:.0f}/{gpu['total_mb']:.0f} MB")
            self.vram_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        else:
            self.vram_label.setText("GPU: N/A (CPU mode)")
            self.vram_label.setStyleSheet("color: #888; font-size: 10px;")

    def _update_batch_estimate(self):
        nv = self.variations_spin.value()
        if nv <= 1: self.batch_estimate_label.setText(""); return
        d = self.duration_spin.value(); s = self.steps_spin.value(); m = self.model_combo.currentText().strip()
        chunks = len(compute_chunks_for_duration(d))
        lsps = get_learned_steps_per_sec(m, self._resolve_device())
        est = estimate_batch_time(nv, s, chunks, lsps)
        disk = estimate_disk_usage_mb(d, nv)
        src = "learned" if lsps else "estimated"
        self.batch_estimate_label.setText(f"📦 {nv} files | ⏱️ {format_time(est)} ({src}) | 💾 {disk:.0f} MB")

    def _open_batch_settings(self):
        BatchSettingsDialog(self, parent=self).exec()
        self._update_batch_estimate()

    def _check_for_interrupted_batch(self):
        state = load_batch_state()
        if state and state.get("completed"):
            self.status_bar.showMessage(f"Interrupted batch: {len(state['completed'])}/{state.get('num_variations',0)} done. Check Batch Settings to resume.", 10000)

    def _on_download_complete(self, model, success):
        if success:
            self.status_bar.showMessage(f"✓ Downloaded {model}", 5000)
            if model == self.model_combo.currentText(): self._on_model_changed(model)
        else:
            self.status_bar.showMessage(f"✗ Download failed: {model}", 5000)
        self._on_model_changed(self.model_combo.currentText())

    def _build_params(self):
        use_multi = self.multi_prompt_check.isChecked()
        multi_text = self.multi_prompt_edit.toPlainText().strip() if use_multi else ""
        if use_multi:
            if not multi_text:
                QMessageBox.warning(self, "Input Error", "Enter at least one prompt in the sequence."); return None
            prompt = multi_text.split('\n')[0].strip()
        else:
            prompt = self.prompt_input.toPlainText().strip()
            if not prompt:
                QMessageBox.warning(self, "Input Error", "Enter a prompt."); return None
        model = self.model_combo.currentText().strip()
        if not model or '/' not in model:
            QMessageBox.warning(self, "Invalid Model", "Model must be 'org/model'."); return None
        out_dir = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        os.makedirs(out_dir, exist_ok=True)
        cache_dir = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        if not use_multi: self.prompt_input.add_to_history(prompt)
        profile_manager.save_profile(model, self.steps_spin.value(), self.guidance_spin.value(), self.duration_spin.value(), self.seed_algo_combo.currentText(), self.perf_mode_combo.currentText())
        return GenerationParams(
            prompt=prompt, negative_prompt=self.neg_prompt_input.text().strip(),
            model_name=model, duration=self.duration_spin.value(),
            steps=self.steps_spin.value(), guidance=self.guidance_spin.value(),
            seed=self.seed_spin.value(), device=self.device_combo.currentText(),
            cache_dir=cache_dir, output_dir=out_dir,
            num_variations=self.variations_spin.value(),
            normalize=self.normalize_check.isChecked(), fade_in=self.fade_in_check.isChecked(),
            fade_out=self.fade_out_check.isChecked(), trim=self.trim_check.isChecked(),
            use_cpu_offload=self.cpu_offload_check.isChecked(),
            section=self.section_edit.text().strip(),
            use_multi_prompt=use_multi, multi_prompts=multi_text,
            seed_algo=self.seed_algo_combo.currentText(),
            performance_mode=self.perf_mode_combo.currentText(),
            skip_on_error=self.skip_error_check.isChecked(),
            early_preview=self.early_preview_check.isChecked(),
            aggressive_gc=self.aggressive_gc_check.isChecked(),
            batch_id=str(time.time()),
        )

    def start_generation(self):
        if self.worker_thread and self.worker_thread.isRunning():
            params = self._build_params()
            if params is None: return False
            self._gen_queue.append(params); self._refresh_queue_view()
            self.status_bar.showMessage(f"Queued (position {len(self._gen_queue)})", 3000); return True
        params = self._build_params()
        if params is None: return False
        free_mb = get_free_disk_mb(params.output_dir)
        if free_mb is not None:
            needed = estimate_disk_usage_mb(params.duration, params.num_variations)
            if free_mb < needed:
                if QMessageBox.critical(self, "Disk Space", f"Need ~{needed:.0f} MB, have {free_mb:.0f} MB. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                    return False
        if params.num_variations >= 50:
            lsps = get_learned_steps_per_sec(params.model_name, params.device)
            chunks = len(compute_chunks_for_duration(params.duration))
            est = estimate_batch_time(params.num_variations, params.steps, chunks, lsps)
            disk = estimate_disk_usage_mb(params.duration, params.num_variations)
            if QMessageBox.warning(self, "Large Batch", f"{params.num_variations} files | ~{format_time(est)} | ~{disk:.0f} MB\nContinue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return False
        if params.num_variations >= AGGRESSIVE_GC_THRESHOLD:
            params.aggressive_gc = True
        return self._start_with_params(params)

    def _start_with_params(self, params):
        self.worker_thread = AudioGenerationThread(params, parent=self)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        self.worker_thread.status_updated.connect(self.status_label.setText, Qt.QueuedConnection)
        self.worker_thread.eta_updated.connect(self.eta_label.setText, Qt.QueuedConnection)
        self.worker_thread.generation_completed.connect(self.generation_finished, Qt.QueuedConnection)
        self.worker_thread.error_occurred.connect(self.generation_error, Qt.QueuedConnection)
        self.worker_thread.cancelled.connect(self.generation_cancelled, Qt.QueuedConnection)
        self.worker_thread.finished.connect(self.on_worker_thread_finished, Qt.QueuedConnection)
        self.worker_thread.variation_completed.connect(self._on_variation_completed, Qt.QueuedConnection)
        self.worker_thread.batch_progress.connect(self._on_batch_progress, Qt.QueuedConnection)
        self.worker_thread.performance_metrics.connect(self._on_performance_metrics, Qt.QueuedConnection)
        self.worker_thread.early_preview_ready.connect(self._on_early_preview, Qt.QueuedConnection)
        self.worker_thread.model_status.connect(self.model_status_label.setText, Qt.QueuedConnection)
        self.generate_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.status_label.setText("Starting..."); self.worker_thread.start()
        return True

    def cancel_generation(self):
        if self.worker_thread:
            self.worker_thread.cancel(); self.status_label.setText("Cancelling..."); self.cancel_btn.setEnabled(False)

    def generation_cancelled(self):
        self.status_bar.showMessage("Generation cancelled", 3000)

    def generation_finished(self, path, metadata):
        self.base_audio_path = path; self.current_audio_path = path
        self.progress_bar.setVisible(False); self.eta_label.setText("")
        self.play_btn.setEnabled(True); self.save_btn.setEnabled(True); self.play_btn.setText("Play")
        self._load_audio_file(path)
        meta_list = metadata['metadata']
        history_data = self.settings.get("history", [])
        row = self.playlist_table.rowCount()
        for meta in meta_list:
            history_data.append(meta)
            self.playlist_table.insertRow(row)
            pi = QTableWidgetItem(meta.get('prompt', '')); pi.setToolTip(meta.get('path', ''))
            self.playlist_table.setItem(row, 0, pi)
            di = QTableWidgetItem(format_time(meta.get('duration', 0.0))); di.setTextAlignment(Qt.AlignCenter); di.setToolTip(meta.get('path', ''))
            self.playlist_table.setItem(row, 1, di)
            self.playlist_table.setItem(row, 2, QTableWidgetItem(meta.get('timestamp', '')[:10]))
            self.playlist_table.setItem(row, 3, QTableWidgetItem(str(meta.get('seed', ''))))
            row += 1
        if len(history_data) > MAX_HISTORY: history_data = history_data[-MAX_HISTORY:]
        self.settings["history"] = history_data; self.save_json_settings()
        self.status_bar.showMessage(f"Generated {len(metadata['paths'])} file(s)", 5000)
        # Record stats
        if self.batch_config.track_stats:
            completed = len(meta_list)
            failed = len(metadata.get('failed', []))
            gen_time = metadata.get('gen_time', 0)
            avg = gen_time / max(completed, 1) if completed > 0 else 0
            disk = 0
            for m in meta_list:
                try:
                    if os.path.exists(m.get('path', '')): disk += os.path.getsize(m.get('path', ''))
                except: pass
            batch_stats.record_batch(
                meta_list[0].get('batch_id', '') if meta_list else '',
                meta_list[0].get('model', '') if meta_list else '',
                meta_list[0].get('device', '') if meta_list else '',
                meta_list[0].get('total_variations', 1) if meta_list else 1,
                completed, failed, 0, gen_time, avg, disk / (1024*1024),
                meta_list[0].get('steps', 200) if meta_list else 200,
                meta_list[0].get('duration', 10.0) if meta_list else 10.0,
                meta_list[0].get('seed_algo', 'Sequential') if meta_list else 'Sequential',
                meta_list[0].get('performance_mode', 'Custom') if meta_list else 'Custom',
                meta_list[0].get('prompt', '') if meta_list else '',
            )
        # Notification
        mode = self.batch_config.notification_mode
        if mode in ("visual", "all"):
            self.activateWindow(); self.raise_()
            if completed >= 5:
                metrics = metadata.get('metrics', {})
                QMessageBox.information(self, "Batch Complete",
                    f"Files: {completed}\nTime: {format_time(gen_time)}\nAvg: {format_time(avg)}\nSpeed: {metrics.get('steps_per_sec', 0):.1f} steps/sec")

    def generation_error(self, msg):
        self.progress_bar.setVisible(False); self.eta_label.setText("")
        QMessageBox.critical(self, "Generation Error", msg)

    def on_worker_thread_finished(self):
        self.generate_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        self.status_label.setText("Ready")
        t = self.worker_thread; self.worker_thread = None
        if t: t.deleteLater()
        gc.collect()
        if self._gen_queue:
            next_params = self._gen_queue.popleft()
            self._refresh_queue_view()
            self.status_bar.showMessage("Starting next queued job...", 3000)
            self._start_with_params(next_params)

    def _on_variation_completed(self, path, metadata):
        history_data = self.settings.get("history", [])
        history_data.append(metadata)
        if len(history_data) > MAX_HISTORY: history_data = history_data[-MAX_HISTORY:]
        self.settings["history"] = history_data

    def _on_batch_progress(self, current, total, filename):
        self.status_bar.showMessage(f"Batch: {current}/{total} - {filename[:40]}", 3000)

    def _on_early_preview(self, path):
        if path and os.path.exists(path):
            self._load_audio_file(path)
            self.status_bar.showMessage(" Early preview loaded! Batch continues...", 5000)

    def _on_performance_metrics(self, metrics):
        skip = f" | Skipped: {len(metrics.get('skipped', []))}" if metrics.get('skipped') else ""
        self.status_bar.showMessage(
            f"Done in {format_time(metrics['total_time'])} | {metrics['num_variations']} files | "
            f"Avg: {format_time(metrics['avg_per_variation'])}/file | {metrics['steps_per_sec']:.1f} steps/sec{skip}", 10000)

    def toggle_playback(self):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path): return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.stop()
        else:
            cs = self.media_player.source().toLocalFile()
            ts = os.path.abspath(self.current_audio_path)
            if cs != ts: self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_path))
            self.media_player.play()

    def on_playback_state_changed(self, state):
        self.play_btn.setText("Stop" if state == QMediaPlayer.PlayingState else "Play")

    def play_from_playlist(self, item):
        pi = self.playlist_table.item(item.row(), 1)
        path = pi.toolTip() if pi else ""
        if path and os.path.exists(path):
            self._load_audio_file(path); self.toggle_playback()

    def _load_audio_file(self, path):
        if not path or not os.path.exists(path): return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState: self.media_player.stop()
        self.base_audio_path = path; self.current_audio_path = path
        self.play_btn.setEnabled(True); self.save_btn.setEnabled(True); self.play_btn.setText("Play")
        self.waveform_widget.set_audio(path)
        self.media_player.setSource(QUrl.fromLocalFile(path))

    def on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.play_btn.setText("Play"); self.waveform_widget.set_playback_position(0.0)

    def on_position_changed(self, position):
        duration = self.media_player.duration()
        if duration > 0:
            self.time_label.setText(f"{format_time(position/1000)} / {format_time(duration/1000)}")
            self.waveform_widget.set_playback_position(position / duration)

    def on_duration_changed(self, duration):
        if duration > 0: self.time_label.setText(f"00:00 / {format_time(duration/1000)}")

    def _seek_to(self, pos):
        if self.media_player.duration() > 0:
            self.media_player.setPosition(int(pos * self.media_player.duration()))

    def set_volume(self, val):
        self.audio_output.setVolume(val / 100.0); self.vol_label.setText(f"{val}%")

    def open_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "WAV Files (*.wav)")
        if path: self._load_audio_file(path)

    def save_audio(self):
        if not self.base_audio_path: return
        default_name = os.path.basename(self.base_audio_path)
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", default_name, "WAV Audio (*.wav)")
        if path:
            try:
                shutil.copy2(self.base_audio_path, path)
                self.status_bar.showMessage("Audio saved!", 3000)
            except Exception as e:
                logger.error(f"Save failed: {e}")
                QMessageBox.warning(self, "Save Failed", f"Could not save:\n{e}")

    def load_settings(self):
        self.prompt_input.setPlainText(self.settings.get("prompt", ""))
        self.neg_prompt_input.setText(self.settings.get("negative_prompt", ""))
        self.model_combo.setCurrentText(self.settings.get("model", "cvssp/audioldm2"))
        self.duration_spin.setValue(float(self.settings.get("duration", 10.0)))
        self.steps_spin.setValue(int(self.settings.get("steps", 200)))
        self.guidance_spin.setValue(float(self.settings.get("guidance", 3.5)))
        self.seed_spin.setValue(int(self.settings.get("seed", -1)))
        self.variations_spin.setValue(int(self.settings.get("variations", 1)))
        self.output_dir_edit.setText(self.settings.get("output_dir", os.path.join(os.getcwd(), "generated_audio")))
        self.cache_dir_edit.setText(self.settings.get("cache_dir", ""))
        self.cpu_offload_check.setChecked(self.settings.get("cpu_offload", False))
        self.multi_prompt_check.setChecked(self.settings.get("multi_prompt_check", False))
        self.multi_prompt_edit.setPlainText(self.settings.get("multi_prompts", ""))
        self.toggle_multi_prompt(self.multi_prompt_check.isChecked())
        self.seed_algo_combo.setCurrentText(self.settings.get("seed_algo", "Sequential"))
        self.perf_mode_combo.setCurrentText(self.settings.get("performance_mode", "Custom"))
        self.skip_error_check.setChecked(self.settings.get("skip_on_error", True))
        self.early_preview_check.setChecked(self.settings.get("early_preview", True))
        self.aggressive_gc_check.setChecked(self.settings.get("aggressive_gc", False))
        self.update_duration_info()
        for cm in self.settings.get("custom_models", []):
            if self.model_combo.findText(cm) < 0: self.model_combo.addItem(cm)
        history_data = self.settings.get("history", [])
        for meta in history_data:
            row = self.playlist_table.rowCount(); self.playlist_table.insertRow(row)
            pi = QTableWidgetItem(meta.get('prompt', '')); pi.setToolTip(meta.get('path', ''))
            self.playlist_table.setItem(row, 0, pi)
            di = QTableWidgetItem(format_time(meta.get('duration', 0.0))); di.setTextAlignment(Qt.AlignCenter); di.setToolTip(meta.get('path', ''))
            self.playlist_table.setItem(row, 1, di)
            self.playlist_table.setItem(row, 2, QTableWidgetItem(meta.get('timestamp', '')[:10]))
            self.playlist_table.setItem(row, 3, QTableWidgetItem(str(meta.get('seed', ''))))
        QTimer.singleShot(500, lambda: self._on_model_changed(self.model_combo.currentText()))
        QTimer.singleShot(100, self._update_batch_estimate)

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel(); self.status_bar.showMessage("Waiting for worker...")
            self.worker_thread.wait(2000)
            if self.worker_thread.isRunning():
                self.worker_thread.terminate(); self.worker_thread.wait(1000)
        if self._preload_thread and self._preload_thread.isRunning():
            self._preload_thread.wait(2000)
            if self._preload_thread.isRunning():
                self._preload_thread.terminate(); self._preload_thread.wait(1000)
        AudioGenerationThread.unload_pipeline()
        download_manager.cancel_current(); download_manager.clear_queue()
        download_manager.save_settings()
        if hasattr(self, '_vram_timer'): self._vram_timer.stop()
        current_model = self.model_combo.currentText().strip()
        if current_model and '/' in current_model:
            profile_manager.save_profile(current_model, self.steps_spin.value(), self.guidance_spin.value(), self.duration_spin.value(), self.seed_algo_combo.currentText(), self.perf_mode_combo.currentText())
        self.batch_config.num_variations = self.variations_spin.value()
        self.batch_config.seed_algo = self.seed_algo_combo.currentText()
        self.batch_config.save()
        self.settings["prompt"] = self.prompt_input.toPlainText()
        self.settings["negative_prompt"] = self.neg_prompt_input.text()
        self.settings["model"] = self.model_combo.currentText()
        self.settings["duration"] = self.duration_spin.value()
        self.settings["steps"] = self.steps_spin.value()
        self.settings["guidance"] = self.guidance_spin.value()
        self.settings["seed"] = self.seed_spin.value()
        self.settings["variations"] = self.variations_spin.value()
        self.settings["output_dir"] = self.output_dir_edit.text().strip()
        self.settings["cpu_offload"] = self.cpu_offload_check.isChecked()
        self.settings["multi_prompt_check"] = self.multi_prompt_check.isChecked()
        self.settings["multi_prompts"] = self.multi_prompt_edit.toPlainText()
        self.settings["seed_algo"] = self.seed_algo_combo.currentText()
        self.settings["performance_mode"] = self.perf_mode_combo.currentText()
        self.settings["skip_on_error"] = self.skip_error_check.isChecked()
        self.settings["early_preview"] = self.early_preview_check.isChecked()
        self.settings["aggressive_gc"] = self.aggressive_gc_check.isChecked()
        self.settings["cache_dir"] = self.cache_dir_edit.text().strip()
        self.save_json_settings()
        event.accept()


# =========================================================================
# ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setApplicationName(APP_NAME)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)
        app.setStyleSheet(STYLESHEET)
        app.setFont(QFont("Segoe UI", 9))
        window = AudioLDM2Studio()
        window.show()
        def signal_handler(sig, frame):
            logger.info("SIGINT received, quitting...")
            if window: window.close()
        signal.signal(signal.SIGINT, signal_handler)
        sig_timer = QTimer(); sig_timer.start(250); sig_timer.timeout.connect(lambda: None)
        sys.exit(app.exec())
    except Exception:
        logger.critical("Fatal Startup Error", exc_info=True)
        sys.exit(1)