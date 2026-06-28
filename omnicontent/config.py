import os
import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path="omnicontent/.env")

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
PEXELS_KEY     = os.environ.get("PEXELS_KEY", "")
NGROK_TOKEN    = os.environ.get("NGROK_TOKEN", "")

PORT           = int(os.environ.get("PORT", 5005))

OUTPUT_VIDEO   = "omnicontent_output.mp4"
TEMP_AUDIO     = "temp_audio.mp3"
TEMP_VIDEO     = "temp_video.mp4"

VIRAL_THRESHOLD = 0.65
MODEL_PATH      = "omnicontent/model/viral_mlp.keras"

LLM_MODEL       = "openrouter/free"
LLM_BASE_URL    = "https://openrouter.ai/api/v1"
MAX_RETRIES     = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s",
    datefmt="%H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
