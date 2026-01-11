"""
Configuration file containing all main addresses and paths used across the project.
"""
from pathlib import Path


# Project root directory
REPO_ROOT = Path(__file__).resolve().parent


# Data directories
DATA_ROOT = REPO_ROOT / "data"
LANGUAGE = "hi"  # Language code for Common Voice dataset
AUDIO_ROOT = DATA_ROOT / LANGUAGE / "clips"


# Common Voice file patterns
AUDIO_FILE_PATTERN = "common_voice_*.mp3"  # NEW
CLIPS_SUBDIR = "clips"  # NEW


# Common Voice TSV files
VALIDATED_TSV = "validated.tsv"
TRAIN_TSV = "train.tsv"
DEV_TSV = "dev.tsv"
TEST_TSV = "test.tsv"


# Transcription files
TRANSCRIPTIONS_FILE = "transcriptions.txt"
TRANSCRIPTIONS_UROMAN_FILE = "transcriptions_uroman.txt"


# Results directories
RESULTS_ROOT = REPO_ROOT / "results"
BASELINES_RESULTS_DIR = RESULTS_ROOT / "baselines"
QWEN_AGENT_RESULTS_DIR = RESULTS_ROOT / "qwen_agent"


# Audio processing
ASR_SAMPLING_RATE = 16_000


# Model configurations
MMS_MODEL_ID = "facebook/mms-1b-all"
MMS_TARGET_LANG = "hin"  # MMS language code for Hindi


# Evaluation defaults
DEFAULT_MAX_SAMPLES = None
DEFAULT_START_IDX = 0
DEFAULT_QUIET = False


print(REPO_ROOT)
