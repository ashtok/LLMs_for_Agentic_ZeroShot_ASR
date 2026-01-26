from pathlib import Path


# Project root directory
REPO_ROOT = Path(__file__).resolve().parent


# Data directories
DATA_ROOT = REPO_ROOT / "data"
LANGUAGE = "bg"
AUDIO_ROOT = DATA_ROOT / LANGUAGE / "clips"


# Common Voice file patterns
AUDIO_FILE_PATTERN = "common_voice_*.mp3"
CLIPS_SUBDIR = "clips"


# Common Voice TSV files
VALIDATED_TSV = "validated.tsv"
TRAIN_TSV = "train.tsv"
DEV_TSV = "dev.tsv"
TEST_TSV = "test.tsv"
INVALIDATED_TSV = "invalidated.tsv"
OTHER_TSV = "other.tsv"


# Transcription files
TRANSCRIPTIONS_FILE = "transcriptions.txt"
TRANSCRIPTIONS_UROMAN_FILE = "transcriptions_uroman.txt"
WORDS_FILE = "words.txt" 
LEXICON_FILE = "lexicon.txt" 


# Uroman configuration
UROMAN_DIR = REPO_ROOT / "uroman" / "uroman"
UROMAN_LANG_CODE = "bul"


# Results directories
RESULTS_ROOT = REPO_ROOT / "results"
BASELINES_RESULTS_DIR = RESULTS_ROOT / "baselines"
QWEN_AGENT_RESULTS_DIR = RESULTS_ROOT / "qwen_agent"
LLAMA_AGENT_RESULTS_DIR = RESULTS_ROOT / "llama_agent"


# Audio processing
ASR_SAMPLING_RATE = 16_000


# MMS
MMS_MODEL_ID = "facebook/mms-1b-all"
MMS_TARGET_LANG = "bul"
MMS_ZEROSHOT_MODEL_ID = "mms-meta/mms-zeroshot-300m"


# OmniASR
OMNI_MODEL_CARD = "omniASR_CTC_300M"
OMNI_LANG_TAG = "bul_Cyrl"


# Whisper
WHISPER_MODEL_NAME = "small"
WHISPER_LANG_CODE = "bg"


# Qwen configuration
QWEN_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
#QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


# Llama configuration
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"



# ---- CURRENT SELECTION SWITCHES ----


# Choose which LLM to use for the agent: "qwen" or "llama"
CURRENT_BACKBONE = "llama"  # llama or "qwen"


if CURRENT_BACKBONE == "qwen":
    CURRENT_MODEL_NAME = QWEN_MODEL_NAME
    CURRENT_RESULTS_DIR = QWEN_AGENT_RESULTS_DIR
    CURRENT_ENGINE_LABEL = "QWEN"
elif CURRENT_BACKBONE == "llama":
    CURRENT_MODEL_NAME = LLAMA_MODEL_NAME
    CURRENT_RESULTS_DIR = LLAMA_AGENT_RESULTS_DIR
    CURRENT_ENGINE_LABEL = "LLAMA"
else:
    raise ValueError(f"Unknown CURRENT_BACKBONE: {CURRENT_BACKBONE}")


# Common generation settings
LOAD_8BIT = False
MAX_NEW_TOKENS = 256
BATCH_SIZE = 2
USE_FLASH_ATTENTION = True


# Agent evaluation defaults
AGENT_MAX_FILES = 2000
AGENT_START_IDX = 0


# Evaluation defaults
DEFAULT_MAX_SAMPLES = None
DEFAULT_START_IDX = 0
DEFAULT_QUIET = False


print(REPO_ROOT)
