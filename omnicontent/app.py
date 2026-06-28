import os
import threading
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from omnicontent.config import PORT, OUTPUT_VIDEO, get_logger
from omnicontent.model.predictor import predict_viral_score, score_script_quality
from omnicontent.pipeline.agents import run_agent_pipeline
from omnicontent.pipeline.video import run_video_pipeline

log = get_logger("app")

app = Flask(__name__)
CORS(app)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>OmniContent AI</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { font-family: 'Inter', system-ui, sans-serif; background: #0B0B12; }
  .gradient-text {
    background: linear-gradient(135deg, #A855F7, #EC4899);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .gradient-bg {
    background: linear-gradient(135deg, #A855F7, #EC4899);
  }
  .glow {
    box-shadow: 0 0 40px rgba(168, 85, 247, 0.25);
  }
  .card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
  }
  @keyframes pulse-bar {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }
  .pulse { animation: pulse-bar 1.5s ease-in-out infinite; }
</style>
</head>
<body class="min-h-screen text-white flex items-center justify-center px-4 py-10">

  <div class="w-full max-w-md">

    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold mb-1">
        <span class="gradient-text">OmniContent</span> AI
      </h1>
      <p class="text-gray-500 text-sm">Keyword in. Viral video out.</p>
    </div>

    <div class="card rounded-2xl p-6 glow">

      <label class="text-xs text-gray-400 uppercase tracking-wide mb-2 block">Keyword</label>
      <input id="keyword" type="text" placeholder="e.g. fitness motivation"
        class="w-full p-3 rounded-xl bg-black/30 border border-white/10 text-white mb-4 outline-none focus:border-purple-400 transition" />

      <button id="generateBtn" onclick="generate()"
        class="w-full gradient-bg p-3 rounded-xl font-semibold transition hover:opacity-90 active:scale-[0.99]">
        Generate Video
      </button>

      <div id="progress" class="mt-5 hidden">
        <div class="w-full bg-white/10 rounded-full h-1.5 overflow-hidden">
          <div id="bar" class="gradient-bg h-1.5 rounded-full transition-all duration-700 pulse" style="width:5%"></div>
        </div>
        <p id="step" class="text-xs text-gray-400 mt-2 text-center"></p>
      </div>

      <div id="errorBox" class="mt-4 hidden text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl p-3"></div>

      <div id="result" class="mt-5 hidden">

        <div class="flex items-center justify-between mb-4">
          <span class="text-xs text-gray-400 uppercase tracking-wide">Viral Score</span>
          <span id="scoreValue" class="text-lg font-bold gradient-text">--</span>
        </div>
        <div class="w-full bg-white/10 rounded-full h-2 mb-5 overflow-hidden">
          <div id="scoreBar" class="gradient-bg h-2 rounded-full transition-all duration-1000" style="width:0%"></div>
        </div>

        <details class="mb-4 group">
          <summary class="text-xs text-gray-400 uppercase tracking-wide cursor-pointer select-none flex items-center justify-between">
            Script preview
            <span class="group-open:rotate-180 transition">&#9662;</span>
          </summary>
          <div id="scriptPreview" class="mt-3 space-y-2 text-sm text-gray-300"></div>
        </details>

        <video id="videoPlayer" class="w-full rounded-xl mb-4 hidden" controls></video>

        <a href="/download" download id="downloadBtn"
          class="block w-full text-center bg-white/10 hover:bg-white/20 border border-white/10 p-3 rounded-xl font-semibold transition">
          Download MP4
        </a>
      </div>

    </div>

    <p class="text-center text-xs text-gray-600 mt-6">
      Built by
      <a href="https://github.com/IhabAltekreeti" target="_blank"
         class="text-gray-400 hover:text-purple-400 transition underline underline-offset-2">
         Ihab Altekreeti
      </a>
    </p>

  </div>

<script>
const STEPS = [
  [8,  "Predicting viral score..."],
  [22, "Researching the topic..."],
  [40, "Writing the script..."],
  [58, "Brand guard review..."],
  [72, "Generating voiceover..."],
  [85, "Fetching footage..."],
  [95, "Rendering MP4..."],
];

async function generate() {
  const keyword = document.getElementById("keyword").value.trim();
  if (!keyword) return;

  const btn = document.getElementById("generateBtn");
  btn.disabled = true;
  btn.classList.add("opacity-50");

  document.getElementById("progress").classList.remove("hidden");
  document.getElementById("result").classList.add("hidden");
  document.getElementById("errorBox").classList.add("hidden");
  document.getElementById("videoPlayer").classList.add("hidden");

  let i = 0;
  const ticker = setInterval(() => {
    if (i < STEPS.length) {
      document.getElementById("bar").style.width = STEPS[i][0] + "%";
      document.getElementById("step").innerText = STEPS[i][1];
      i++;
    }
  }, 9000);

  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keyword }),
    });
    const data = await res.json();
    clearInterval(ticker);

    if (data.success) {
      document.getElementById("bar").style.width = "100%";
      setTimeout(() => {
        document.getElementById("progress").classList.add("hidden");
        document.getElementById("result").classList.remove("hidden");

        const score = Math.round(data.viral_score * 100);
        document.getElementById("scoreValue").innerText = score + "%";
        setTimeout(() => {
          document.getElementById("scoreBar").style.width = score + "%";
        }, 100);

        const preview = document.getElementById("scriptPreview");
        preview.innerHTML = data.script.scenes.map((s, idx) =>
          `<div class="border-l-2 border-purple-500/40 pl-3"><span class="text-gray-500">Scene ${idx + 1}</span><br/>${s.description}</div>`
        ).join("");

        const video = document.getElementById("videoPlayer");
        video.src = "/download?t=" + Date.now();
        video.classList.remove("hidden");
      }, 400);
    } else {
      document.getElementById("progress").classList.add("hidden");
      const errBox = document.getElementById("errorBox");
      errBox.innerText = "Something went wrong: " + data.error;
      errBox.classList.remove("hidden");
    }
  } catch (err) {
    clearInterval(ticker);
    document.getElementById("progress").classList.add("hidden");
    const errBox = document.getElementById("errorBox");
    errBox.innerText = "Network error. Please try again.";
    errBox.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.classList.remove("opacity-50");
  }
}
</script>
</body>
</html>"""


@app.get("/")
def index():
    return HTML_PAGE


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/generate")
def generate():
    data = request.json or {}
    keyword = data.get("keyword", "").strip()

    if not keyword:
        return jsonify({"success": False, "error": "Keyword cannot be empty."}), 400

    log.info(f"Generation started -> keyword={keyword!r}")
    try:
        viral_score = predict_viral_score(keyword)
        script = run_agent_pipeline(keyword)
        quality = score_script_quality(script)
        log.info(f"Script quality score: {quality}")

        run_video_pipeline(script, keyword=keyword)

        return jsonify({
            "success": True,
            "viral_score": viral_score,
            "script": script,
            "quality": quality,
        })
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/download")
def download():
    abs_path = os.path.abspath(OUTPUT_VIDEO)
    if not os.path.exists(abs_path):
        return jsonify({"error": "Video has not been generated yet."}), 404

    return send_file(
        abs_path,
        as_attachment=True,
        download_name="omnicontent_viral_video.mp4",
        mimetype="video/mp4"
    )

def run_app():
    app.run(host="0.0.0.0", port=PORT, use_reloader=False, threaded=True)
