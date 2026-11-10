"""Lists the Gemini models your API key can use for text generation.

Run from the project folder:   python scripts/list_models.py
Then put one name in .env, for example:   GEMINI_MODEL=gemini-2.5-flash
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from app.config import get_settings  # noqa: E402

settings = get_settings()
if not settings.gemini_api_key or settings.gemini_api_key == "your_key_here":
    sys.exit("GEMINI_API_KEY is not set in .env. Get one at https://aistudio.google.com/apikey")

http_options = types.HttpOptions(api_version=settings.gemini_api_version) if settings.gemini_api_version else None
client = genai.Client(api_key=settings.gemini_api_key, http_options=http_options)

names = sorted(
    m.name.split("/", 1)[-1] for m in client.models.list() if "generateContent" in (m.supported_actions or [])
)
print("Models your key can use (put one in GEMINI_MODEL in .env):\n")
for name in names:
    print("  ", name, "   <- good choice" if "flash" in name and not any(
        w in name for w in ("lite", "image", "live", "tts", "preview", "exp")) else "")
