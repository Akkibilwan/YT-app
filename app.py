import os
import time
import json
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.signal import find_peaks
import requests
import sqlite3
import hashlib
import re
from datetime import datetime, timedelta, timezone
import openai
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict
import atexit
import subprocess
import tempfile
import shutil
import pandas as pd
import isodate

# Selenium & related imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import imageio_ffmpeg
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# =============================================================================
# 1. Logging Setup
# =============================================================================
def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_file = "logs/youtube_finance_search.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

# =============================================================================
# 2. API Keys from Streamlit Secrets (single key, no proxies)
# =============================================================================
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]["key"]
except Exception as e:
    st.error("No YouTube API key provided in secrets!")
    YOUTUBE_API_KEY = None

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]["key"]
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key loaded successfully")
except Exception as e:
    logger.error(f"Failed to load OpenAI API key: {str(e)}")
    st.error("Failed to load OpenAI API key. Please check your secrets configuration.")

def get_youtube_api_key():
    if not YOUTUBE_API_KEY:
        raise Exception("No YouTube API key available.")
    return YOUTUBE_API_KEY

# =============================================================================
# 3. SQLite DB Setup (Caching)
# =============================================================================
DB_PATH = "cache.db"

def init_db(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS youtube_cache (
            cache_key TEXT PRIMARY KEY,
            json_data TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
        """)

def get_cached_result(cache_key, ttl=600, db_path=DB_PATH):
    now = time.time()
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT json_data, timestamp FROM youtube_cache WHERE cache_key = ?", (cache_key,)).fetchone()
        if row:
            json_data, cached_time = row
            if now - cached_time < ttl:
                return json.loads(json_data)
            else:
                delete_cache_key(cache_key, db_path)
    except Exception as e:
        logger.error(f"get_cached_result DB error: {str(e)}")
    return None

def set_cached_result(cache_key, data_obj, db_path=DB_PATH):
    now = time.time()
    json_str = json.dumps(data_obj, default=str)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO youtube_cache (cache_key, json_data, timestamp) VALUES (?, ?, ?)",
                         (cache_key, json_str, now))
    except Exception as e:
        logger.error(f"set_cached_result DB error: {str(e)}")

def delete_cache_key(cache_key, db_path=DB_PATH):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM youtube_cache WHERE cache_key = ?", (cache_key,))
    except Exception as e:
        logger.error(f"delete_cache_key DB error: {str(e)}")

# =============================================================================
# 4. Utility Helpers
# =============================================================================
def format_date(date_string):
    try:
        date_obj = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return date_obj.strftime("%d-%m-%y")
    except Exception:
        return "Unknown"

def format_number(num):
    try:
        n = int(num)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n/1_000:.1f}K"
        return str(n)
    except:
        return str(num)

def build_cache_key(*args):
    raw_str = "-".join(str(a) for a in args)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

def parse_iso8601_duration(duration_str):
    try:
        duration = isodate.parse_duration(duration_str)
        return int(duration.total_seconds())
    except Exception:
        return 0

# =============================================================================
# 5. Channel Folders
# =============================================================================
CHANNELS_FILE = "channels.json"
FOLDERS_FILE = "channel_folders.json"

def load_channels_json():
    if os.path.exists(CHANNELS_FILE):
        with open(CHANNELS_FILE, "r") as f:
            return json.load(f)
    return {}

channels_data_json = load_channels_json()

def load_channel_folders():
    if os.path.exists(FOLDERS_FILE):
        with open(FOLDERS_FILE, "r") as f:
            return json.load(f)
    else:
        finance = channels_data_json.get("finance", {})
        default_folders = {}
        if "USA" in finance:
            default_folders["USA Finance niche"] = [
                {"channel_name": name, "channel_id": cid}
                for name, cid in finance["USA"].items()
            ]
        else:
            default_folders["USA Finance niche"] = []
        if "India" in finance:
            default_folders["India Finance niche"] = [
                {"channel_name": name, "channel_id": cid}
                for name, cid in finance["India"].items()
            ]
        else:
            default_folders["India Finance niche"] = []
        save_channel_folders(default_folders)
        return default_folders

def save_channel_folders(folders):
    with open(FOLDERS_FILE, "w") as f:
        json.dump(folders, f, indent=4)

def get_channel_id(channel_name_or_url):
    if "youtube.com/channel/" in channel_name_or_url:
        return channel_name_or_url.split("youtube.com/channel/")[-1].split("/")[0].split("?")[0]
    key = get_youtube_api_key()
    url = f"https://www.googleapis.com/youtube/v3/search?part=id&type=channel&q={channel_name_or_url}&key={key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        items = response.json().get("items", [])
        if items:
            return items[0]["id"]["channelId"]
    except requests.exceptions.HTTPError as e:
        logger.error(f"Channel search error: {e}")
    return None

def show_channel_folder_manager():
    st.write("### Manage Channel Folders")
    folders = load_channel_folders()
    action = st.selectbox("Action", ["Create New Folder", "Modify Folder", "Delete Folder"], key="folder_action")
    
    if action == "Create New Folder":
        folder_name = st.text_input("Folder Name", key="new_folder_name")
        st.write("Enter at least one channel name or URL (one per line):")
        channels_text = st.text_area("Channels", key="folder_channels")
        if st.button("Create Folder", key="create_folder_btn"):
            if folder_name.strip() == "":
                st.error("Folder name cannot be empty.")
                return
            if folder_name in folders:
                st.error(f"Folder '{folder_name}' already exists.")
                return
            lines = channels_text.strip().split("\n")
            channel_list = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                ch_id = get_channel_id(line)
                if ch_id:
                    channel_list.append({"channel_name": line, "channel_id": ch_id})
            if not channel_list:
                st.error("At least one valid channel is required.")
                return
            folders[folder_name] = channel_list
            save_channel_folders(folders)
            st.success(f"Folder '{folder_name}' created with {len(channel_list)} channel(s).")
            for ch in channel_list:
                search_youtube("", [ch["channel_id"]], "3 months", "Both", ttl=7776000)

    elif action == "Modify Folder":
        if not folders:
            st.info("No folders available.")
            return
        selected_folder = st.selectbox("Select Folder to Modify", list(folders.keys()))
        if selected_folder:
            st.write("Channels in this folder:")
            if folders[selected_folder]:
                for ch in folders[selected_folder]:
                    st.write(f"- {ch['channel_name']}")
            else:
                st.write("(No channels yet)")
            modify_action = st.radio("Modify Action", ["Add Channels", "Remove Channel"], key="modify_folder_action")
            if modify_action == "Add Channels":
                st.write("Enter channel name(s) or URL(s) (one per line):")
                new_ch_text = st.text_area("Add Channels", key="add_channels_text")
                if st.button("Add to Folder", key="add_to_folder_btn"):
                    lines = new_ch_text.strip().split("\n")
                    added = 0
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        ch_id = get_channel_id(line)
                        if ch_id:
                            folders[selected_folder].append({"channel_name": line, "channel_id": ch_id})
                            added += 1
                            search_youtube("", [ch_id], "3 months", "Both", ttl=7776000)
                    save_channel_folders(folders)
                    st.success(f"Added {added} channel(s) to folder '{selected_folder}'.")
            else:
                if folders[selected_folder]:
                    channel_options = [ch["channel_name"] for ch in folders[selected_folder]]
                    rem_choice = st.selectbox("Select channel to remove", channel_options, key="remove_channel_choice")
                    if st.button("Remove Channel", key="remove_channel_btn"):
                        folders[selected_folder] = [c for c in folders[selected_folder] if c["channel_name"] != rem_choice]
                        save_channel_folders(folders)
                        st.success(f"Channel '{rem_choice}' removed from '{selected_folder}'.")
                else:
                    st.info("No channels in this folder to remove.")

    else:
        if not folders:
            st.info("No folders available.")
            return
        selected_folder = st.selectbox("Select Folder to Delete", list(folders.keys()))
        if st.button("Delete Folder", key="delete_folder_btn"):
            del folders[selected_folder]
            save_channel_folders(folders)
            st.success(f"Folder '{selected_folder}' deleted.")

# =============================================================================
# 6. Transcript & Fallback
# =============================================================================
def get_transcript(video_id):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        logger.error(f"Transcript error for {video_id}: {e}")
        return None

def download_audio(video_id):
    if not shutil.which("ffmpeg"):
        st.error("ffmpeg not found. Please install ffmpeg and ensure it is in your PATH.")
        return None
    temp_dir = tempfile.gettempdir()
    output_template = os.path.join(temp_dir, f"{video_id}.%(ext)s")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True
    }
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
        return audio_path if os.path.exists(audio_path) else None
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None

def generate_transcript_with_openai(audio_file):
    max_size = 26214400
    actual_size = os.path.getsize(audio_file)
    if actual_size > max_size:
        snippet_file = audio_file.replace(".mp3", "_snippet.mp3")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, "-i", audio_file, "-t", "60", "-y", snippet_file]
        subprocess.run(cmd, check=True, capture_output=True)
        audio_file = snippet_file
    try:
        with open(audio_file, "rb") as file:
            transcript_response = openai.Audio.transcribe(
                model="whisper-1",
                file=file
            )
        text = transcript_response["text"]
        words = text.split()
        segments = []
        chunk_size = 10
        avg_duration = 2.5
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            segments.append({
                "start": i / chunk_size * avg_duration,
                "duration": avg_duration,
                "text": chunk
            })
        if "_snippet.mp3" in audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        return segments, "openai"
    except Exception as e:
        st.error(f"Error with OpenAI Whisper: {e}")
        return None, None

def get_transcript_with_fallback(video_id):
    key = f"transcript_data_{video_id}"
    if key in st.session_state:
        return st.session_state[key], st.session_state.get(f"transcript_source_{video_id}")
    direct = get_transcript(video_id)
    if direct:
        st.session_state[key] = direct
        st.session_state[f"transcript_source_{video_id}"] = "youtube"
        return direct, "youtube"
    audio_file = download_audio(video_id)
    if audio_file:
        fallback, fallback_source = generate_transcript_with_openai(audio_file)
        try:
            os.remove(audio_file)
        except:
            pass
        if fallback:
            st.session_state[key] = fallback
            st.session_state[f"transcript_source_{video_id}"] = fallback_source
            return fallback, fallback_source
        else:
            st.session_state[key] = None
            st.session_state[f"transcript_source_{video_id}"] = None
            return None, None
    else:
        st.session_state[key] = None
        st.session_state[f"transcript_source_{video_id}"] = None
        return None, None

def get_intro_outro_transcript(video_id, total_duration):
    transcript, source = get_transcript_with_fallback(video_id)
    if not transcript:
        return (None, None)
    end_intro = min(60, total_duration)
    start_outro = max(total_duration - 60, 0)
    intro_lines = []
    outro_lines = []
    for item in transcript:
        start_sec = float(item["start"])
        end_sec = start_sec + float(item.get("duration", 0))
        if end_sec > 0 and start_sec < end_intro:
            intro_lines.append(item)
        if end_sec > start_outro and start_sec < total_duration:
            outro_lines.append(item)
    intro_text = " ".join(seg["text"] for seg in intro_lines) if intro_lines else None
    outro_text = " ".join(seg["text"] for seg in outro_lines) if outro_lines else None
    return (intro_text, outro_text)

def summarize_intro_outro(intro_text, outro_text):
    if not intro_text and not outro_text:
        return (None, None)
    cache_key = f"intro_outro_summary_{hashlib.sha256((intro_text + outro_text).encode()).hexdigest()}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    prompt_str = ""
    if intro_text:
        prompt_str += f"Intro snippet:\n{intro_text}\n\n"
    if outro_text:
        prompt_str += f"Outro snippet:\n{outro_text}\n\n"
    prompt_str += (
        "Please produce two short bullet-point summaries:\n"
        "1) For the intro snippet\n"
        "2) For the outro snippet.\n"
        "If one snippet is missing, skip it.\n"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_str}]
        )
        result_txt = response.choices[0].message.content
        st.session_state[cache_key] = (result_txt, result_txt)
        return (result_txt, result_txt)
    except Exception as e:
        return (None, None)

def summarize_script(script_text):
    if not script_text.strip():
        return "No script text to summarize."
    hashed = hashlib.sha256(script_text.encode("utf-8")).hexdigest()
    if "script_summary_cache" not in st.session_state:
        st.session_state["script_summary_cache"] = {}
    if hashed in st.session_state["script_summary_cache"]:
        return st.session_state["script_summary_cache"][hashed]
    prompt_content = (
        "Please provide a short, plain-English summary of the following text:\n\n"
        f"{script_text}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_content}]
        )
        summary = response.choices[0].message.content
        st.session_state["script_summary_cache"][hashed] = summary
        return summary
    except:
        return "Script summary failed."

# =============================================================================
# 8. Searching & Calculating Outliers
# =============================================================================
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def calculate_metrics(df):
    now = datetime.now()
    df["published_date"] = pd.to_datetime(df["published_at"]).dt.tz_localize(None)
    df["hours_since_published"] = ((now - df["published_date"]).dt.total_seconds() / 3600).round(1)
    df["days_since_published"] = df["hours_since_published"] / 24

    df["raw_vph"] = df["views"] / np.maximum(df["hours_since_published"], 1)
    peak_hours = np.minimum(df["hours_since_published"], 30 * 24)
    df["peak_vph"] = df["views"] / np.maximum(peak_hours, 1)
    df["effective_vph"] = df.apply(lambda row: row["raw_vph"] if row["days_since_published"] < 90 else row["peak_vph"], axis=1)

    df["engagement_rate"] = (df["like_count"] + 5 * df["comment_count"]) / np.maximum(df["views"], 1)
    df["cvr_float"] = df["comment_count"] / np.maximum(df["views"], 1)
    df["clr_float"] = df["like_count"] / np.maximum(df["views"], 1)

    recent_videos = df[df["days_since_published"] <= 30]
    if recent_videos.empty:
        recent_videos = df
    recent_videos = recent_videos.copy()
    recent_videos["effective_hours"] = recent_videos["hours_since_published"].apply(lambda x: min(x, 30*24))

    channel_total_views = recent_videos["views"].sum()
    channel_total_hours = recent_videos["effective_hours"].sum()
    channel_avg_vph_last30 = channel_total_views / np.maximum(channel_total_hours, 1)

    channel_total_engagement = (recent_videos["like_count"] + 5 * recent_videos["comment_count"]).sum()
    channel_total_views_for_eng = recent_videos["views"].sum()
    channel_avg_engagement = channel_total_engagement / np.maximum(channel_total_views_for_eng, 1)

    channel_avg_cvr = recent_videos["cvr_float"].mean()
    channel_avg_clr = recent_videos["clr_float"].mean()

    df["channel_avg_vph"] = channel_avg_vph_last30
    df["channel_avg_engagement"] = channel_avg_engagement
    df["vph_ratio"] = df["effective_vph"] / np.maximum(channel_avg_vph_last30, 0.1)
    df["engagement_ratio"] = df["engagement_rate"] / np.maximum(channel_avg_engagement, 0.001)

    df["combined_performance"] = (0.9 * df["vph_ratio"]) + (0.1 * df["engagement_ratio"])
    df["log_performance"] = np.log1p(df["combined_performance"])

    def recency_factor(days):
        if days <= 30:
            return 1.5
        elif days < 90:
            return 1.5 - 0.5 * ((days - 30) / 60)
        else:
            return 1.0

    df["recency_factor"] = df["days_since_published"].apply(recency_factor)

# 9. Comments & Analysis
# =============================================================================
def analyze_comments(comments):
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}
    lines = [c["text"] for c in comments]
    hashed = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    if hashed in st.session_state["analysis_cache"]:
        return st.session_state["analysis_cache"][hashed]
    prompt_content = (
        "Analyze the following YouTube comments and provide a short summary in three sections:\n"
        "1) Positive Comments Summary\n"
        "2) Negative Comments Summary\n"
        "3) Top 5 Suggested Topics\n\n"
        "Comments:\n" + "\n".join(lines)
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_content}]
        )
        txt = response.choices[0].message.content
        st.session_state["analysis_cache"][hashed] = txt
        return txt
    except:
        return "Analysis failed."

def get_video_comments(video_id):
    url = (
        "https://www.googleapis.com/youtube/v3/commentThreads"
        f"?part=snippet&videoId={video_id}&maxResults=50&order=relevance&key={get_youtube_api_key()}"
    )
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        comments = []
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]
            like_count = int(snippet.get("likeCount", 0))
            comments.append({"text": text, "likeCount": like_count})
        return comments
    except:
        return []

# =============================================================================
# 10. Retention Analysis
# =============================================================================
def load_cookies(driver, cookie_file="youtube_cookies.json"):
    if not os.path.exists(cookie_file):
        return
    with open(cookie_file, "r", encoding="utf-8") as f:
        cookies = json.load(f)
    for cookie in cookies:
        cookie.pop("sameSite", None)
        if "expiry" in cookie:
            cookie["expires"] = cookie.pop("expiry")
        driver.add_cookie(cookie)

def capture_player_screenshot_with_hover(video_url, timestamp, output_path="player_retention.png", use_cookies=True):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.binary_location = "/usr/bin/chromium"
    driver_version = "120.0.6099.224"
    service = Service(ChromeDriverManager(driver_version=driver_version).install())
    driver = webdriver.Chrome(service=service, options=options)
    try:
        if use_cookies:
            driver.get("https://youtube.com")
            time.sleep(3)
            load_cookies(driver, "youtube_cookies.json")
            driver.refresh()
            time.sleep(3)
        driver.get(video_url)
        time.sleep(5)
        driver.execute_script(f"document.querySelector('video').currentTime = {timestamp};")
        time.sleep(3)
        player = driver.find_element(By.ID, "player")
        progress_bar = player.find_element(By.XPATH, "//div[@class='ytp-progress-bar-padding']")
        ActionChains(driver).move_to_element(progress_bar).perform()
        time.sleep(2)
        player.screenshot(output_path)
        duration = driver.execute_script("return document.querySelector('video').duration")
    finally:
        driver.quit()
    return duration

def detect_retention_peaks(image_path, crop_ratio=0.2, height_threshold=200, distance=20, top_n=5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File {image_path} not found.")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from {image_path}.")
    height, width, _ = img.shape
    roi = img[int(height * (1 - crop_ratio)):height, 0:width]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
    col_sums = np.sum(binary_roi, axis=0)
    peaks, properties = find_peaks(col_sums, height=height_threshold, distance=distance)
    if len(peaks) > top_n:
        peak_heights = properties["peak_heights"]
        top_indices = np.argsort(peak_heights)[-top_n:]
        top_peaks = peaks[top_indices]
        top_peaks = np.sort(top_peaks)
    else:
        top_peaks = peaks
    return top_peaks, roi, binary_roi, width, col_sums

def capture_frame_at_time(video_url, target_time, output_path="frame_retention.png", use_cookies=True):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.binary_location = "/usr/bin/chromium"
    driver_version = "120.0.6099.224"
    service = Service(ChromeDriverManager(driver_version=driver_version).install())
    driver = webdriver.Chrome(service=service, options=options)
    try:
        if use_cookies:
            driver.get("https://youtube.com")
            time.sleep(3)
            load_cookies(driver, "youtube_cookies.json")
            driver.refresh()
            time.sleep(3)
        driver.get(video_url)
        time.sleep(5)
        driver.execute_script(f"document.querySelector('video').currentTime = {target_time};")
        time.sleep(1)
        driver.execute_script("document.querySelector('video').play();")
        time.sleep(5)
        driver.execute_script("document.querySelector('video').pause();")
        time.sleep(1)
        player = driver.find_element(By.ID, "player")
        player.screenshot(output_path)
    finally:
        driver.quit()
    return output_path

def plot_brightness_profile(col_sums, peaks):
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(col_sums, label="Brightness Sum")
    ax.plot(peaks, col_sums[peaks], "x", label="Detected Peaks", markersize=10, color="red")
    ax.set_xlabel("Column Index (ROI)")
    ax.set_ylabel("Sum of Pixel Values")
    ax.legend()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def filter_transcript(transcript, target_time, window=5):
    snippet = []
    for seg in transcript:
        start = seg.get("start", 0)
        if abs(start - target_time) <= window:
            snippet.append(seg.get("text", ""))
    return " ".join(snippet)

def check_ytdlp_installed():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def download_video_snippet(video_url, start_time, duration=10, output_path="snippet.mp4"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', exist_ok=True)
    try:
        temp_video = "temp_full_video.mp4"
        download_cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", temp_video,
            video_url
        ]
        st.info("Downloading full video for extraction (this might take a moment)...", key="download_full_video")
        subprocess.run(download_cmd, check=True, capture_output=True)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        st.info(f"Extracting {duration}s segment starting at {start_time}s...", key="extract_segment")
        ffmpeg_cmd = [
            ffmpeg_exe,
            "-i", temp_video,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            "-b:a", "128k",
            "-y",
            output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        if os.path.exists(temp_video):
            os.remove(temp_video)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error during video processing: {e.stdout}\n{e.stderr}", key="snippet_error")
        raise Exception(f"Failed to extract video segment: {str(e)}")

# =============================================================================
# 14. UI Pages
# =============================================================================
def show_search_page():
    st.title("Youtube Niche Search")
    # Sidebar: Channel Folder Manager expander
    with st.sidebar.expander("Channel Folder Manager"):
        show_channel_folder_manager()
    # Sidebar: Filters in desired order
    folders = load_channel_folders()
    folder_choice = st.sidebar.selectbox("Select Folder", list(folders.keys()) if folders else ["None"])
    selected_timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["Last 24 hours", "Last 48 hours", "Last 4 days", "Last 7 days", "Last 15 days", "Last 28 days", "3 months", "Lifetime"]
    )
    content_filter = st.sidebar.selectbox("Filter By Content Type", ["Shorts", "Videos", "Both"], index=2)

atexit.register(lambda: logger.info("Application shutting down"))


outlier_score = calculate_outlier_score(video_details['viewCount'], channel_average)
fig = create_performance_chart(benchmark_stats, video_performance, 
video_details['title'][:40] + "..." if len(video_details['title']) > 40 else video_details['title'])
st.plotly_chart(fig, use_container_width=True)
st.subheader("Outlier Analysis")
if outlier_score >= 2.0:
outlier_category = "Significant Positive Outlier"
outlier_class = "outlier-high"
elif outlier_score >= 1.5:
outlier_category = "Positive Outlier"
outlier_class = "outlier-high"
elif outlier_score >= 1.2:
outlier_category = "Slight Positive Outlier"
outlier_class = "outlier-normal"
elif outlier_score >= 0.8:
outlier_category = "Normal Performance"
outlier_class = "outlier-normal"
elif outlier_score >= 0.5:
outlier_category = "Slight Negative Outlier"
outlier_class = "outlier-low"
else:
outlier_category = "Significant Negative Outlier"
outlier_class = "outlier-low"
col1, col2, col3 = st.columns(3)
with col1:
st.markdown(f"""
<div class='metric-card'>
<div>Current Views</div>
<div style='font-size: 24px; font-weight: bold;'>{video_details['viewCount']:,}</div>
</div>
""", unsafe_allow_html=True)
with col2:
st.markdown(f"""
<div class='metric-card'>
<div>Channel Average</div>
<div style='font-size: 24px; font-weight: bold;'>{int(channel_average):,}</div>
</div>
""", unsafe_allow_html=True)
with col3:
st.markdown(f"""
<div class='metric-card'>
<div>Outlier Score</div>
<div style='font-size: 24px; font-weight: bold;' class='{outlier_class}'>{outlier_score:.2f}</div>
<div>{outlier_category}</div>
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class='explanation'>
<p><strong>What this means:</strong></p>
<p>An outlier score of <strong>{outlier_score:.2f}</strong> means this video has <strong>{outlier_score:.2f}x</strong> the views compared to the channel's average at the same age.</p>
<ul>
<li>1.0 = Exactly average performance</li>
<li>&gt;1.0 = Outperforming channel average</li>
<li>&lt;1.0 = Underperforming channel average</li>
</ul>
</div>
""", unsafe_allow_html=True)
st.subheader("Detailed Performance Metrics")
col1, col2 = st.columns(2)
with col1:
if benchmark_median > 0:
vs_median_pct = ((video_details['viewCount'] / benchmark_median) - 1) * 100
st.metric("Compared to Median", f"{vs_median_pct:+.1f}%")
if channel_average > 0:
vs_avg_pct = ((video_details['viewCount'] / channel_average) - 1) * 100
st.metric("Compared to Channel Average", f"{vs_avg_pct:+.1f}%")
with col2:
if benchmark_upper > 0:
vs_upper_pct = ((video_details['viewCount'] / benchmark_upper) - 1) * 100
st.metric("Compared to Upper Band", f"{vs_upper_pct:+.1f}%")
if benchmark_lower > 0:
vs_lower_pct = ((video_details['viewCount'] / benchmark_lower) - 1) * 100
st.metric("Compared to Lower Band", f"{vs_lower_pct:+.1f}%")
