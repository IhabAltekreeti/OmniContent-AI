import requests
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
from gtts import gTTS

from omnicontent.config import PEXELS_KEY, OUTPUT_VIDEO, TEMP_AUDIO, TEMP_VIDEO, get_logger

log = get_logger("pipeline.video")


def generate_voiceover(text: str) -> None:
    log.info("Generating voiceover (gTTS)...")
    tts = gTTS(text=text, lang="en")
    tts.save(TEMP_AUDIO)
    log.info(f"Voiceover saved -> {TEMP_AUDIO}")


def fetch_stock_video(query: str, max_attempts: int = 3) -> str:
    fallback_queries = [query, "lifestyle", "nature", "city"]
    for attempt, q in enumerate(fallback_queries[:max_attempts], start=1):
        log.info(f"Pexels query (attempt {attempt}/{max_attempts}): {q!r}")
        try:
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": PEXELS_KEY},
                params={"query": q, "per_page": 1, "orientation": "portrait"},
                timeout=15,
            )
            resp.raise_for_status()
            videos = resp.json().get("videos", [])
            if videos:
                video_url = videos[0]["video_files"][0]["link"]
                log.info(f"Video found: {q!r}")
                return video_url
            log.warning(f"No results found for '{q}', trying fallback.")
        except requests.RequestException as e:
            log.warning(f"Pexels request failed: {e}")
    raise RuntimeError("Could not find a video from Pexels in any query.")


def download_video(video_url: str) -> None:
    log.info("Downloading stock video...")
    resp = requests.get(video_url, timeout=60)
    resp.raise_for_status()
    with open(TEMP_VIDEO, "wb") as f:
        f.write(resp.content)
    log.info(f"Video downloaded -> {TEMP_VIDEO}")


def render_final_video() -> str:
    log.info("Rendering final video...")
    bg  = VideoFileClip(TEMP_VIDEO)
    sfx = AudioFileClip(TEMP_AUDIO)
    final = bg.fx(vfx.loop, duration=sfx.duration).resize((1080, 1920)).set_audio(sfx)
    final.write_videofile(OUTPUT_VIDEO, fps=24, logger=None)
    bg.close()
    sfx.close()
    final.close()
    log.info(f"Final video ready -> {OUTPUT_VIDEO}")
    return OUTPUT_VIDEO


def run_video_pipeline(script: dict, keyword: str = "") -> str:
    generate_voiceover(script["voiceover"])

    if keyword.strip():
        search_query = keyword.strip()
    else:
        first_scene = script["scenes"][0]["description"]
        search_query = " ".join(first_scene.split()[:3])

    video_url = fetch_stock_video(search_query)
    download_video(video_url)
    return render_final_video()
