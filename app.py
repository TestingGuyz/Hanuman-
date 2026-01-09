import os
import json
import logging
import random
import time
import difflib
import warnings
import asyncio
import requests
import shutil
import subprocess
import stat
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from youtube_search import YoutubeSearch
from pydub import AudioSegment
from gtts import gTTS
import edge_tts

# Suppress warnings
warnings.filterwarnings("ignore")
load_dotenv()

# --- VERCEL & FILESYSTEM SETUP ---
IS_VERCEL = os.environ.get("VERCEL") == "1"

if IS_VERCEL:
    BASE_DIR = Path("/tmp")
else:
    BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "audio"
CACHE_DIR = BASE_DIR / "cache"
AUDIO_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# --- FFmpeg AUTO-INSTALLER ---
def install_ffmpeg_if_needed():
    if not IS_VERCEL: return
    ffmpeg_path = BASE_DIR / "ffmpeg"
    if not ffmpeg_path.exists():
        try:
            print("📥 Downloading FFmpeg...")
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
            tar_path = BASE_DIR / "ffmpeg.tar.xz"
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(tar_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("📦 Extracting FFmpeg...")
            subprocess.run(["tar", "-xJf", str(tar_path), "-C", str(BASE_DIR)], check=True)
            for root, dirs, files in os.walk(BASE_DIR):
                if "ffmpeg" in files and "ffmpeg" not in root:
                    shutil.move(Path(root) / "ffmpeg", ffmpeg_path)
                    break
            if tar_path.exists(): tar_path.unlink()
        except Exception as e:
            print(f"FFmpeg Error: {e}")
    
    if ffmpeg_path.exists():
        st = os.stat(ffmpeg_path)
        os.chmod(ffmpeg_path, st.st_mode | stat.S_IEXEC)
        AudioSegment.converter = str(ffmpeg_path)
        AudioSegment.ffmpeg = str(ffmpeg_path)
        AudioSegment.ffprobe = str(ffmpeg_path)

install_ffmpeg_if_needed()

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM" 

app = Flask(__name__)

# --- LOGGER HELPER ---
def log_local(logs_list, msg, type="info"):
    """Appends logs to the list instead of emitting them"""
    print(f"[{type.upper()}] {msg}")
    logs_list.append({'msg': msg, 'type': type})

# === 1. INTELLIGENT COMMAND DICTIONARY ===
VALID_COMMANDS = {
    "wake": ["hanuman", "anuman", "human", "naman", "hello hanuman", "hey hanuman", "हनुमान", "jai shri ram"],
    "aagya": ["aagya", "chat", "talk", "anya", "bat", "आज्ञा", "baat"],
    "hasya": ["hasya", "joke", "funny", "laugh", "chutkule", "hasa", "हास्य"],
    "yudha": ["yudha", "game", "play", "fight", "war", "yuda", "युद्ध", "khel"],
    "gandharva": ["gandharva", "music", "song", "gana", "play song", "dj", "गंधर्व"],
    "khoj": ["khoj", "search", "find", "google", "dhoondo", "खोज"],
    "exit": ["exit", "stop", "back", "bye", "band", "ruk", "बंद"]
}

def identify_command(text):
    if not text: return None, 0.0
    text_lower = text.lower().strip()
    for cmd, keywords in VALID_COMMANDS.items():
        if any(k in text_lower for k in keywords): return cmd, 1.0
    best_cmd, best_score = None, 0.0
    for cmd, keywords in VALID_COMMANDS.items():
        for kw in keywords:
            score = difflib.SequenceMatcher(None, kw, text_lower).ratio()
            if score > 0.65 and score > best_score:
                best_score = score
                best_cmd = cmd
    return best_cmd, best_score

# === 2. STT ===
def call_groq_whisper(audio_path, model, logs):
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"model": model, "language": "en", "temperature": 0.0},
                timeout=15
            )
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
    except Exception as e:
        log_local(logs, f"STT Error ({model}): {e}", "error")
    return ""

def transcribe_smart(audio_path, logs):
    if not GROQ_API_KEY:
        log_local(logs, "Groq Key Missing!", "error")
        return ""

    # We use threading to speed up Vercel execution
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_turbo = executor.submit(call_groq_whisper, audio_path, "whisper-large-v3-turbo", logs)
        future_large = executor.submit(call_groq_whisper, audio_path, "whisper-large-v3", logs)
        
        turbo_text = future_turbo.result()
        log_local(logs, f"⚡ Turbo heard: '{turbo_text}'", "debug")
        
        cmd, score = identify_command(turbo_text)
        if cmd:
            log_local(logs, f"✅ Fast match: {cmd.upper()}", "success")
            return turbo_text
            
        large_text = future_large.result()
        log_local(logs, f"🧠 Large heard: '{large_text}'", "debug")
        return large_text if len(large_text) > len(turbo_text) else turbo_text

# === 3. TTS ===
def generate_tts(text, logs):
    # ElevenLabs
    if ELEVENLABS_API_KEY:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
            data = {"text": text, "model_id": "eleven_turbo_v2_5"}
            response = requests.post(url, json=data, headers=headers, timeout=5)
            if response.status_code == 200:
                fname = f"eleven_{int(time.time())}.mp3"
                with open(CACHE_DIR / fname, "wb") as f: f.write(response.content)
                log_local(logs, "🔊 Generated with ElevenLabs", "success")
                return f"/audio/{fname}"
        except: pass

    # EdgeTTS Fallback
    try:
        fname = f"edge_{int(time.time())}.mp3"
        async def _edge():
            comm = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
            await comm.save(str(CACHE_DIR / fname))
        asyncio.run(_edge())
        log_local(logs, "🔊 Generated with EdgeTTS", "info")
        return f"/audio/{fname}"
    except Exception as e:
        log_local(logs, f"TTS Error: {e}", "error")
    return None

# === 4. LOGIC ===
def chat_llm(user_text, sys_prompt):
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}],
                "max_tokens": 100
            },
            timeout=5
        )
        return resp.json()['choices'][0]['message']['content'].strip()
    except:
        return "System offline."

# Simple state tracking (NOTE: This resets per request on Vercel, so persistent conversational state requires a DB, but basic commands work fine)
def process_input(text, logs):
    cmd, score = identify_command(text)
    
    if cmd == "exit": return "Stopping."
    if cmd == "wake": return "Jai Shri Ram. I am ready."

    # Logic
    sys_prompt = "You are Hanuman. Be concise, wise, and helpful. Use 1 sentence."
    
    if cmd == "aagya": 
        log_local(logs, "Mode: Aagya (Chat)", "info")
        return chat_llm(text, sys_prompt)
    if cmd == "hasya": 
        log_local(logs, "Mode: Hasya (Joke)", "info")
        return chat_llm(text, sys_prompt + " Tell a joke.")
    if cmd == "khoj": 
        log_local(logs, "Mode: Khoj (Search)", "info")
        return chat_llm(text, sys_prompt + " Search and answer.")
    
    return chat_llm(text, sys_prompt)

# === FRONTEND ===
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Hanuman Pro</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <style>
        body { background: #050505; color: #0f0; font-family: 'Courier New', monospace; display: flex; flex-direction: column; height: 100vh; margin: 0; overflow: hidden; }
        #header { padding: 15px; text-align: center; border-bottom: 1px solid #333; }
        #status { font-size: 1.2rem; font-weight: bold; color: #fff; }
        #visualizer { flex: 1; display: flex; justify-content: center; align-items: center; position: relative; }
        .mic-btn {
            width: 100px; height: 100px; border-radius: 50%; background: #111; border: 2px solid #333;
            color: #555; font-size: 3rem; cursor: pointer; z-index: 2; transition: all 0.2s;
            display: flex; justify-content: center; align-items: center;
        }
        .mic-btn.active { border-color: #ff9800; color: #ff9800; box-shadow: 0 0 30px rgba(255, 152, 0, 0.2); }
        .ring { position: absolute; width: 100px; height: 100px; border-radius: 50%; border: 2px solid #ff9800; opacity: 0; z-index: 1; transition: 0.1s; }
        #console { 
            height: 35%; background: #000; border-top: 1px solid #333; padding: 10px; 
            overflow-y: auto; font-size: 0.85rem; display: flex; flex-direction: column;
        }
        .log-entry { margin-bottom: 4px; border-left: 3px solid #333; padding-left: 8px; }
        .log-info { border-color: #2196f3; color: #badbf9; }
        .log-success { border-color: #4caf50; color: #a5d6a7; }
        .log-warn { border-color: #ff9800; color: #ffcc80; }
        .log-error { border-color: #f44336; color: #ef9a9a; }
        .log-debug { border-color: #555; color: #777; font-size: 0.75rem; }
    </style>
</head>
<body>
    <div id="header"><div id="status">TAP TO START</div></div>
    <div id="visualizer">
        <div id="ring" class="ring"></div>
        <div id="mic" class="mic-btn" onclick="toggleListen()">🎙️</div>
    </div>
    <div id="console">
        <div class="log-entry log-info">> System Initialized.</div>
    </div>
    <audio id="audio" autoplay></audio>

    <script>
        let isListening = false;
        let audioCtx, analyser, micStream, recorder, chunks = [];
        let silenceStart = 0;
        let hasSpoken = false;

        function addLog(msg, type='info') {
            const c = document.getElementById('console');
            const div = document.createElement('div');
            div.className = `log-entry log-${type}`;
            div.innerText = `> ${msg}`;
            c.appendChild(div);
            c.scrollTop = c.scrollHeight;
        }

        async function toggleListen() {
            if (isListening) stopListen(); else startListen();
        }

        async function startListen() {
            try {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 256;
                const src = audioCtx.createMediaStreamSource(micStream);
                src.connect(analyser);

                recorder = new MediaRecorder(micStream);
                chunks = [];
                recorder.ondataavailable = e => chunks.push(e.data);
                recorder.onstop = uploadAudio;
                recorder.start();

                isListening = true;
                hasSpoken = false;
                silenceStart = Date.now();
                
                document.getElementById('mic').classList.add('active');
                document.getElementById('status').innerText = "LISTENING...";
                document.getElementById('status').style.color = "#ff9800";
                
                detectSilence();
            } catch(e) { alert("Mic Error: " + e); }
        }

        function stopListen() {
            if (!isListening) return;
            isListening = false;
            document.getElementById('mic').classList.remove('active');
            document.getElementById('status').innerText = "PROCESSING...";
            document.getElementById('status').style.color = "#2196f3";
            document.getElementById('ring').style.opacity = 0;
            
            if (recorder && recorder.state !== 'inactive') recorder.stop();
            if (micStream) micStream.getTracks().forEach(t => t.stop());
            if (audioCtx) audioCtx.close();
        }

        function detectSilence() {
            if (!isListening) return;
            const data = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(data);
            
            let sum = 0;
            for(let i=0; i<data.length; i++) sum += data[i];
            let vol = sum / data.length;

            const ring = document.getElementById('ring');
            ring.style.transform = `scale(${1 + vol/50})`;
            ring.style.opacity = vol > 10 ? 1 : 0.2;

            if (vol > 20) {
                hasSpoken = true;
                silenceStart = Date.now();
            } else {
                if (hasSpoken && (Date.now() - silenceStart > 1200)) stopListen();
                if (!hasSpoken && (Date.now() - silenceStart > 6000)) stopListen();
            }
            requestAnimationFrame(detectSilence);
        }

        async function uploadAudio() {
            if (!hasSpoken && chunks.length < 5) {
                document.getElementById('status').innerText = "TAP TO START";
                return;
            }
            const blob = new Blob(chunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', blob);

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                
                // Display Logs
                if (data.logs) {
                    data.logs.forEach(log => addLog(log.msg, log.type));
                }

                if (data.status === "ignored") {
                    document.getElementById('status').innerText = "IGNORED";
                } else if (data.reply) {
                    addLog("AI: " + data.reply, "success");
                    document.getElementById('status').innerText = "READY";
                    document.getElementById('status').style.color = "#fff";
                    
                    if (data.audio) {
                        const audio = document.getElementById('audio');
                        audio.src = data.audio;
                        audio.play();
                    }
                }
            } catch (e) {
                document.getElementById('status').innerText = "ERROR";
                addLog("Server Error: " + e, "error");
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML_CONTENT

@app.route("/upload", methods=["POST"])
def upload():
    logs = []  # Init Logs for this request
    if 'audio' not in request.files: return jsonify({"error": "No audio"}), 400
    
    f = request.files['audio']
    fname = f"rec_{int(time.time())}_{random.randint(100,999)}.wav"
    path = AUDIO_DIR / fname
    f.save(path)
    
    try:
        log_local(logs, "Processing audio...", "info")
        text = transcribe_smart(path, logs)
        
        if path.exists(): path.unlink()

        if not text:
            return jsonify({"status": "ignored", "logs": logs})
            
        log_local(logs, f"FINAL INPUT: {text}", "success")
        
        reply = process_input(text, logs)
        if not reply: return jsonify({"status": "ignored", "logs": logs})
        
        audio_url = generate_tts(reply, logs)
        return jsonify({"text": text, "reply": reply, "audio": audio_url, "logs": logs})

    except Exception as e:
        if path.exists(): path.unlink()
        log_local(logs, f"Critical Error: {e}", "error")
        return jsonify({"error": str(e), "logs": logs})

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_file(CACHE_DIR / filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
