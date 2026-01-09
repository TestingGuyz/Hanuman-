import os
import json
import logging
import random
import time
import difflib
import warnings
import asyncio
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from youtube_search import YoutubeSearch
from pydub import AudioSegment
from gtts import gTTS
import edge_tts

# Suppress warnings
warnings.filterwarnings("ignore")
load_dotenv()

# --- SETUP ---
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
CACHE_DIR = BASE_DIR / "cache"
AUDIO_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# Default Voice ID (Rachel). Change this to your preferred Voice ID.
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM" 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hanuman_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Custom Logger that sends to UI
def log_ui(msg, type="info"):
    print(f"[{type.upper()}] {msg}")
    socketio.emit('log_update', {'msg': msg, 'type': type})

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
    
    # 1. Direct Keyword Check (Fastest)
    for cmd, keywords in VALID_COMMANDS.items():
        if any(k in text_lower for k in keywords):
            return cmd, 1.0

    # 2. Fuzzy Match (Slower fallback)
    best_cmd, best_score = None, 0.0
    for cmd, keywords in VALID_COMMANDS.items():
        for kw in keywords:
            score = difflib.SequenceMatcher(None, kw, text_lower).ratio()
            if score > 0.65 and score > best_score:  # Stricter threshold
                best_score = score
                best_cmd = cmd
                
    return best_cmd, best_score

# === 2. DUAL-ENGINE STT (Turbo + Large) ===
def call_groq_whisper(audio_path, model):
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={
                    "model": model,
                    "language": "en", # 'en' works best for Hinglish if prompt is good
                    "prompt": "Hanuman, Aagya, Hasya, Yudha, Gandharva, Khoj, Hindi, Jai Shri Ram",
                    "temperature": 0.0
                },
                timeout=8
            )
        if resp.status_code == 200:
            text = resp.json().get("text", "").strip()
            # Basic Hallucination Filter
            if text.lower() in ["you", "thank you", "bye", "am i", "mbc news"]: return ""
            return text
    except Exception as e:
        log_ui(f"STT Error ({model}): {e}", "error")
    return ""

def transcribe_smart(audio_path):
    """Runs Turbo and Large simultaneously. Returns Turbo if it finds a command, else waits for Large."""
    if not GROQ_API_KEY:
        log_ui("❌ Groq API Key Missing!", "error")
        return ""

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Start both models
        future_turbo = executor.submit(call_groq_whisper, audio_path, "whisper-large-v3-turbo")
        future_large = executor.submit(call_groq_whisper, audio_path, "whisper-large-v3")
        
        # Check Turbo First (Fast)
        turbo_text = future_turbo.result()
        log_ui(f"⚡ Turbo heard: '{turbo_text}'", "debug")
        
        cmd, score = identify_command(turbo_text)
        if cmd:
            log_ui(f"✅ Fast match detected: {cmd.upper()}", "success")
            return turbo_text # Trust Turbo if it found a command
            
        # If Turbo was gibberish/uncertain, wait for Large (Accurate)
        large_text = future_large.result()
        log_ui(f"🧠 Large heard: '{large_text}'", "debug")
        
        # Return the longer/more coherent one
        return large_text if len(large_text) > len(turbo_text) else turbo_text

# === 3. ROBUST TTS (ElevenLabs -> Edge -> gTTS) ===
def generate_tts_elevenlabs(text):
    if not ELEVENLABS_API_KEY: return None
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2_5", # Turbo model for speed
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}
        }
        response = requests.post(url, json=data, headers=headers, timeout=5)
        
        if response.status_code == 200:
            fname = f"eleven_{int(time.time())}.mp3"
            path = CACHE_DIR / fname
            with open(path, "wb") as f:
                f.write(response.content)
            log_ui("🔊 Generated with ElevenLabs", "success")
            return f"/audio/{fname}"
        else:
            log_ui(f"⚠️ ElevenLabs Failed: {response.status_code}", "warn")
    except Exception as e:
        log_ui(f"ElevenLabs Error: {e}", "error")
    return None

def generate_tts(text):
    # 1. Try ElevenLabs
    audio_url = generate_tts_elevenlabs(text)
    if audio_url: return audio_url

    # 2. Try Edge TTS (Fallback)
    try:
        fname = f"edge_{int(time.time())}.mp3"
        path = CACHE_DIR / fname
        async def _edge():
            comm = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
            await comm.save(str(path))
        asyncio.run(_edge())
        log_ui("🔊 Generated with EdgeTTS (Fallback)", "info")
        return f"/audio/{fname}"
    except:
        pass
    
    return None

# === 4. LOGIC PROCESSOR ===
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

user_state = {"mode": "idle"}

def process_input(text):
    mode = user_state["mode"]
    cmd, score = identify_command(text)
    
    # Global Commands
    if cmd == "exit":
        user_state["mode"] = "idle"
        return "Stopping."
    if cmd == "wake":
        user_state["mode"] = "active"
        return "Jai Shri Ram. I am ready."

    if mode == "idle": return None

    # Mode Switch
    if cmd in ["aagya", "hasya", "yudha", "gandharva", "khoj"]:
        user_state["mode"] = cmd
        return f"{cmd.capitalize()} mode."

    # Mode Logic
    sys_prompt = "You are Hanuman. Be concise, wise, and helpful. Use 1 sentence."
    if mode == "aagya": return chat_llm(text, sys_prompt)
    if mode == "hasya": return chat_llm(text, sys_prompt + " Tell a joke.")
    if mode == "khoj": return chat_llm(text, sys_prompt + " Search and answer.")
    
    # Default Chat if active
    return chat_llm(text, sys_prompt)

# === FRONTEND ===
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Hanuman Pro</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { background: #050505; color: #0f0; font-family: 'Courier New', monospace; display: flex; flex-direction: column; height: 100vh; margin: 0; overflow: hidden; }
        
        /* Top Status Bar */
        #header { padding: 15px; text-align: center; border-bottom: 1px solid #333; }
        #status { font-size: 1.2rem; font-weight: bold; color: #fff; }
        
        /* Mic & Visuals */
        #visualizer { flex: 1; display: flex; justify-content: center; align-items: center; position: relative; }
        .mic-btn {
            width: 100px; height: 100px; border-radius: 50%; background: #111; border: 2px solid #333;
            color: #555; font-size: 3rem; cursor: pointer; z-index: 2; transition: all 0.2s;
            display: flex; justify-content: center; align-items: center;
        }
        .mic-btn.active { border-color: #ff9800; color: #ff9800; box-shadow: 0 0 30px rgba(255, 152, 0, 0.2); }
        .ring { position: absolute; width: 100px; height: 100px; border-radius: 50%; border: 2px solid #ff9800; opacity: 0; z-index: 1; transition: 0.1s; }

        /* Live Console */
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

    <div id="console"></div>
    <audio id="audio" autoplay></audio>

    <script>
        const socket = io();
        let isListening = false;
        let audioCtx, analyser, micStream, recorder, chunks = [];
        let silenceStart = 0;
        let hasSpoken = false;
        
        // --- LIVE LOGGING ---
        socket.on('log_update', function(data) {
            const c = document.getElementById('console');
            const div = document.createElement('div');
            div.className = `log-entry log-${data.type}`;
            div.innerText = `> ${data.msg}`;
            c.appendChild(div);
            c.scrollTop = c.scrollHeight;
        });

        // --- AUDIO LOGIC ---
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
            
            // Calculate Volume
            let sum = 0;
            for(let i=0; i<data.length; i++) sum += data[i];
            let vol = sum / data.length;

            // Visuals
            const ring = document.getElementById('ring');
            ring.style.transform = `scale(${1 + vol/50})`;
            ring.style.opacity = vol > 10 ? 1 : 0.2;

            // Silence Logic
            if (vol > 20) { // Threshold
                hasSpoken = true;
                silenceStart = Date.now();
            } else {
                // If quiet...
                if (hasSpoken && (Date.now() - silenceStart > 1200)) {
                    stopListen(); // Stop after 1.2s silence
                    return;
                }
                if (!hasSpoken && (Date.now() - silenceStart > 6000)) {
                    stopListen(); // Stop after 6s idle
                    return;
                }
            }
            requestAnimationFrame(detectSilence);
        }

        async function uploadAudio() {
            if (!hasSpoken && chunks.length < 5) {
                document.getElementById('status').innerText = "IGNORED (SILENCE)";
                setTimeout(() => document.getElementById('status').innerText = "TAP TO START", 1000);
                return;
            }

            const blob = new Blob(chunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', blob);

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                
                document.getElementById('status').innerText = "READY";
                document.getElementById('status').style.color = "#fff";

                if (data.audio) {
                    const audio = document.getElementById('audio');
                    audio.src = data.audio;
                    audio.play();
                }
            } catch (e) {
                document.getElementById('status').innerText = "ERROR";
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
    if 'audio' not in request.files: return jsonify({"error": "No audio"}), 400
    
    # Save Temp File
    f = request.files['audio']
    path = AUDIO_DIR / f"rec_{int(time.time())}.wav"
    f.save(path)
    
    try:
        # 1. Transcribe (Dual Model)
        log_ui("Processing audio...", "info")
        text = transcribe_smart(path)
        
        # Cleanup
        if path.exists(): path.unlink()

        if not text:
            log_ui("No clear speech detected.", "warn")
            return jsonify({"status": "ignored"})
            
        log_ui(f"FINAL INPUT: {text}", "success")
        
        # 2. Process
        reply = process_input(text)
        if not reply: return jsonify({"status": "ignored"})
        
        log_ui(f"AI: {reply}", "info")

        # 3. TTS
        audio_url = generate_tts(reply)
        return jsonify({"text": text, "reply": reply, "audio": audio_url})

    except Exception as e:
        log_ui(f"Critical Error: {e}", "error")
        return jsonify({"error": str(e)})

@app.route("/audio/<path:filename>")
def serve_audio(filename): return send_file(CACHE_DIR / filename)

if __name__ == "__main__":
    log_ui("🚀 System Starting...", "success")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)

