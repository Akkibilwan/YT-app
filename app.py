import os
import time
import json
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
import atexit
import subprocess
import tempfile
import shutil
import pandas as pd
import isodate

# Use Pillow for image processing (instead of OpenCV)
from PIL import Image

# Selenium & related imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import imageio_ffmpeg
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# For Plotly charts (outlier performance chart)
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Set page configuration (only once at the top)
st.set_page_config(page_title="YouTube Video Analysis", page_icon="📊", layout="wide")
# -----------------------------------------------------------------------------

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
# 2. API Keys from Streamlit Secrets
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
# 5. Channel Folders and Related Functions
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
                search_youtube("", [ch["channel_id"]], "3 months", "Both", ttl=7776000, 
                               outlier_video_type="all", outlier_percentile=50, outlier_num_videos=None)

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
                            search_youtube("", [ch_id], "3 months", "Both", ttl=7776000,
                                           outlier_video_type="all", outlier_percentile=50, outlier_num_videos=None)
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
# 6a. Global URL Parsing Helper
# =============================================================================
def extract_video_id(url):
    patterns = [
        r'youtube\.com/watch\?v=([^&\s]+)',
        r'youtu\.be/([^?\s]+)',
        r'youtube\.com/embed/([^?\s]+)',
        r'youtube\.com/v/([^?\s]+)',
        r'youtube\.com/shorts/([^?\s]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.match(r'^[A-Za-z0-9_-]{11}$', url.strip()):
        return url.strip()
    return None

# =============================================================================
# 6b. Transcript & Fallback Functions
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
# 8c. Searching & Calculating Metrics & Integrated Outlier Logic
# =============================================================================
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

    df["formatted_views"] = df["views"].apply(format_number)
    df["comment_to_view_ratio"] = df["cvr_float"].apply(lambda x: f"{(x*100):.2f}%")
    df["comment_to_like_ratio"] = df["clr_float"].apply(lambda x: f"{(x*100):.2f}%")
    
    df["vph_display"] = df["effective_vph"].apply(lambda x: f"{int(round(x,0))} VPH" if x>0 else "0 VPH")
    return df, None

# Modify fetch_all_snippets to store channel_id
def fetch_all_snippets(channel_id, order_param, timeframe, query, published_after):
    all_videos = []
    page_token = None
    base_url = (
        f"https://www.googleapis.com/youtube/v3/search?part=snippet"
        f"&channelId={channel_id}&maxResults=25&type=video&order={order_param}&key={get_youtube_api_key()}"
    )
    if published_after:
        base_url += f"&publishedAfter={published_after}"
    if query:
        base_url += f"&q={query}"
    while True:
        url = base_url
        if page_token:
            url += f"&pageToken={page_token}"
        try:
            logger.info(f"Requesting URL: {url}")
            resp = requests.get(url)
            logger.info(f"Response status code: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during snippet request. URL: {url}")
            break
        items = data.get("items", [])
        for it in items:
            vid_id = it["id"].get("videoId", "")
            if not vid_id:
                continue
            snippet = it["snippet"]
            all_videos.append({
                "video_id": vid_id,
                "title": snippet["title"],
                "channel_name": snippet["channelTitle"],
                "channel_id": snippet.get("channelId", ""),
                "publish_date": format_date(snippet["publishedAt"]),
                "published_at": snippet["publishedAt"],
                "thumbnail": snippet["thumbnails"]["medium"]["url"]
            })
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        if len(all_videos) >= 200:
            break
    return all_videos

def search_youtube(query, channel_ids, timeframe, content_filter, ttl=600,
                   outlier_video_type="all", outlier_percentile=50, outlier_num_videos=None):
    if query.strip() == "" and timeframe == "3 months" and content_filter.lower() == "both":
        ttl = 7776000
    query = query.strip()
    cache_key = build_cache_key(query, channel_ids, timeframe, content_filter, outlier_video_type, outlier_percentile, outlier_num_videos)
    cached = get_cached_result(cache_key, ttl=ttl)
    if cached and isinstance(cached, list) and len(cached) > 0 and "outlier_score" in cached[0]:
        return cached
    order_param = "relevance" if query else "date"
    all_videos = []
    pub_after = None
    if timeframe != "Lifetime":
        now = datetime.now(timezone.utc)
        tmap = {
            "Last 24 hours": now - timedelta(days=1),
            "Last 48 hours": now - timedelta(days=2),
            "Last 4 days": now - timedelta(days=4),
            "Last 7 days": now - timedelta(days=7),
            "Last 15 days": now - timedelta(days=15),
            "Last 28 days": now - timedelta(days=28),
            "3 months": now - timedelta(days=90),
        }
        pub_dt = tmap.get(timeframe)
        if pub_dt and pub_dt < now:
            pub_after = pub_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    for cid in channel_ids:
        snippet_data = fetch_all_snippets(cid, order_param, timeframe, query, pub_after)
        all_videos.extend(snippet_data)
    vid_ids = [x["video_id"] for x in all_videos if x["video_id"]]
    if not vid_ids:
        set_cached_result(cache_key, [])
        return []
    all_stats = {}
    for chunk in chunk_list(vid_ids, 50):
        stats_url = (
            "https://www.googleapis.com/youtube/v3/videos"
            f"?part=statistics,contentDetails&id={','.join(chunk)}&key={get_youtube_api_key()}"
        )
        try:
            resp = requests.get(stats_url)
            resp.raise_for_status()
            dd = resp.json()
            for item in dd.get("items", []):
                vid = item["id"]
                stt = item.get("statistics", {})
                cdt = item.get("contentDetails", {})
                dur_str = cdt.get("duration", "PT0S")
                tot_sec = parse_iso8601_duration(dur_str)
                cat = "Short" if tot_sec < 180 else "Video"
                vc = int(stt.get("viewCount", 0))
                lk = int(stt.get("likeCount", 0))
                cm = int(stt.get("commentCount", 0))
                all_stats[vid] = {
                    "views": vc,
                    "like_count": lk,
                    "comment_count": cm,
                    "duration_seconds": tot_sec,
                    "content_category": cat,
                    "published_at": next((x["published_at"] for x in all_videos if x["video_id"] == vid), None)
                }
        except requests.exceptions.RequestException as e:
            logger.error(f"Stats+contentDetails request failed: {str(e)}")
            set_cached_result(cache_key, [])
            return []
    final_results = []
    for av in all_videos:
        vid = av["video_id"]
        if vid not in all_stats:
            continue
        stat = all_stats[vid]
        combined = {**av, **stat}
        final_results.append(combined)
    df = pd.DataFrame(final_results)
    df, _ = calculate_metrics(df)
    results = df.to_dict("records")
    # --- Integrated Outlier Calculation ---
    channel_benchmarks = {}
    for r in results:
        ch_id = r.get("channel_id")
        if not ch_id:
            r["outlier_score"] = 0
            continue
        try:
            pub_date = datetime.strptime(r["published_at"], "%Y-%m-%dT%H:%M:%SZ").date()
            video_age = (datetime.now().date() - pub_date).days
        except Exception:
            video_age = 0
        # Determine is_short_filter based on outlier_video_type
        if outlier_video_type == "auto":
            is_short_filter = (r.get("content_category", "").lower() == "short")
        elif outlier_video_type == "shorts":
            is_short_filter = True
        elif outlier_video_type == "long_form":
            is_short_filter = False
        else:
            is_short_filter = None
        if ch_id not in channel_benchmarks:
            benchmark = compute_channel_benchmark(ch_id, video_age, is_short_filter, outlier_percentile, outlier_num_videos)
            channel_benchmarks[ch_id] = benchmark
        benchmark_value = channel_benchmarks[ch_id]
        if benchmark_value and benchmark_value > 0:
            r["outlier_score"] = r["views"] / benchmark_value
        else:
            r["outlier_score"] = 0
    # ---------------------------------------
    set_cached_result(cache_key, results)
    if content_filter.lower() == "shorts":
        results = [x for x in results if x.get("content_category") == "Short"]
    elif content_filter.lower() == "videos":
        results = [x for x in results if x.get("content_category") == "Video"]
    return results

# --- Outlier Analysis Helper Functions (from provided code) ---
def parse_duration(duration_str):
    hours = re.search(r'(\d+)H', duration_str)
    minutes = re.search(r'(\d+)M', duration_str)
    seconds = re.search(r'(\d+)S', duration_str)
    total_seconds = 0
    if hours:
        total_seconds += int(hours.group(1)) * 3600
    if minutes:
        total_seconds += int(minutes.group(1)) * 60
    if seconds:
        total_seconds += int(seconds.group(1))
    return total_seconds

def fetch_single_video_outlier(video_id, api_key):
    video_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={api_key}"
    try:
        response = requests.get(video_url).json()
        if 'items' not in response or not response['items']:
            return None
        video_data = response['items'][0]
        duration_str = video_data['contentDetails']['duration']
        duration_seconds = parse_duration(duration_str)
        return {
            'videoId': video_id,
            'title': video_data['snippet']['title'],
            'channelId': video_data['snippet']['channelId'],
            'channelTitle': video_data['snippet']['channelTitle'],
            'publishedAt': video_data['snippet']['publishedAt'],
            'thumbnailUrl': video_data['snippet'].get('thumbnails', {}).get('medium', {}).get('url', ''),
            'viewCount': int(video_data['statistics'].get('viewCount', 0)),
            'likeCount': int(video_data['statistics'].get('likeCount', 0)),
            'commentCount': int(video_data['statistics'].get('commentCount', 0)),
            'duration': duration_seconds,
            'isShort': duration_seconds <= 60
        }
    except Exception as e:
        st.error(f"Error fetching video details: {e}")
        return None

def fetch_channel_videos_outlier(channel_id, max_videos, api_key):
    playlist_url = f"https://www.googleapis.com/youtube/v3/channels?part=contentDetails,snippet,statistics&id={channel_id}&key={api_key}"
    try:
        playlist_res = requests.get(playlist_url).json()
        if 'items' not in playlist_res or not playlist_res['items']:
            st.error("Invalid Channel ID or no uploads found.")
            return None, None, None
        channel_info = playlist_res['items'][0]
        channel_name = channel_info['snippet']['title']
        channel_stats = channel_info['statistics']
        uploads_playlist_id = channel_info['contentDetails']['relatedPlaylists']['uploads']
        videos = []
        next_page_token = ""
        while (max_videos is None or len(videos) < max_videos) and next_page_token is not None:
            playlist_items_url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=contentDetails,snippet&maxResults=50&playlistId={uploads_playlist_id}&key={api_key}"
            if next_page_token:
                playlist_items_url += f"&pageToken={next_page_token}"
            playlist_items_res = requests.get(playlist_items_url).json()
            for item in playlist_items_res.get('items', []):
                video_id = item['contentDetails']['videoId']
                title = item['snippet']['title']
                published_at = item['snippet']['publishedAt']
                videos.append({
                    'videoId': video_id,
                    'title': title,
                    'publishedAt': published_at
                })
                if max_videos is not None and len(videos) >= max_videos:
                    break
            next_page_token = playlist_items_res.get('nextPageToken')
        return videos, channel_name, channel_stats
    except Exception as e:
        st.error(f"Error fetching YouTube data: {e}")
        return None, None, None

def fetch_video_details_outlier(video_ids, api_key):
    if not video_ids:
        return {}
    all_details = {}
    video_chunks = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]
    for chunk in video_chunks:
        video_ids_str = ','.join(chunk)
        details_url = f"https://www.googleapis.com/youtube/v3/videos?part=contentDetails,statistics,snippet&id={video_ids_str}&key={api_key}"
        try:
            details_res = requests.get(details_url).json()
            for item in details_res.get('items', []):
                duration_str = item['contentDetails']['duration']
                duration_seconds = parse_duration(duration_str)
                published_at = item['snippet']['publishedAt']
                all_details[item['id']] = {
                    'duration': duration_seconds,
                    'viewCount': int(item['statistics'].get('viewCount', 0)),
                    'likeCount': int(item['statistics'].get('likeCount', 0)),
                    'commentCount': int(item['statistics'].get('commentCount', 0)),
                    'publishedAt': published_at,
                    'title': item['snippet']['title'],
                    'thumbnailUrl': item['snippet']['thumbnails'].get('medium', {}).get('url', ''),
                    'isShort': duration_seconds <= 60
                }
        except Exception as e:
            st.warning(f"Error fetching details for some videos: {e}")
    return all_details

def generate_historical_data(video_details, max_days, is_short=None):
    today = datetime.now().date()
    all_video_data = []
    for video_id, details in video_details.items():
        if is_short is not None and details['isShort'] != is_short:
            continue
        try:
            publish_date = datetime.fromisoformat(details['publishedAt'].replace('Z', '+00:00')).date()
            video_age_days = (today - publish_date).days
        except:
            continue
        if video_age_days < 3:
            continue
        days_to_generate = video_age_days if max_days > video_age_days else max_days
        total_views = details['viewCount']
        video_data = generate_view_trajectory(video_id, days_to_generate, total_views, details['isShort'])
        all_video_data.extend(video_data)
    if not all_video_data:
        return pd.DataFrame()
    return pd.DataFrame(all_video_data)

def generate_view_trajectory(video_id, days, total_views, is_short):
    data = []
    if is_short:
        trajectory = [total_views * (1 - np.exp(-5 * ((i+1)/days)**1.5)) for i in range(days)]
    else:
        k = 10
        trajectory = [total_views * (1 / (1 + np.exp(-k * ((i+1)/days - 0.35)))) for i in range(days)]
    scaling_factor = total_views / trajectory[-1] if trajectory[-1] > 0 else 1
    trajectory = [v * scaling_factor for v in trajectory]
    noise_factor = 0.05
    for i in range(days):
        noise = np.random.normal(0, noise_factor * total_views)
        if i == 0:
            noisy_value = max(100, trajectory[i] + noise)
        else:
            noisy_value = max(trajectory[i-1] + 10, trajectory[i] + noise)
        trajectory[i] = noisy_value
    daily_views = [trajectory[0]]
    for i in range(1, days):
        daily_views.append(trajectory[i] - trajectory[i-1])
    for day in range(days):
        data.append({
            'videoId': video_id,
            'day': day,
            'daily_views': int(daily_views[day]),
            'cumulative_views': int(trajectory[day])
        })
    return data

def calculate_benchmark(df, band_percentage):
    lower_q = (100 - band_percentage) / 200
    upper_q = 1 - (100 - band_percentage) / 200
    summary = df.groupby('day')['cumulative_views'].agg([
        ('lower_band', lambda x: x.quantile(lower_q)),
        ('upper_band', lambda x: x.quantile(upper_q)),
        ('median', 'median'),
        ('mean', 'mean'),
        ('count', 'count')
    ]).reset_index()
    summary['channel_average'] = (summary['lower_band'] + summary['upper_band']) / 2
    return summary

def calculate_outlier_score(current_views, channel_average):
    if channel_average <= 0:
        return 0
    return current_views / channel_average

def create_performance_chart(benchmark_data, video_data, video_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=benchmark_data['day'], 
        y=benchmark_data['lower_band'],
        name='Typical Performance Range',
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.3)',
        line=dict(width=0),
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_data['day'], 
        y=benchmark_data['channel_average'],
        name='Channel Average',
        line=dict(color='#4285f4', width=2, dash='dash'),
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_data['day'], 
        y=benchmark_data['median'],
        name='Channel Median',
        line=dict(color='#34a853', width=2, dash='dot'),
        mode='lines'
    ))
    actual_data = video_data[video_data['projected'] == False]
    fig.add_trace(go.Scatter(
        x=actual_data['day'], 
        y=actual_data['cumulative_views'],
        name=f'"{video_title}" (Actual)',
        line=dict(color='#ea4335', width=3),
        mode='lines'
    ))
    fig.update_layout(
        title='Video Performance Comparison',
        xaxis_title='Days Since Upload',
        yaxis_title='Cumulative Views',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white'
    )
    return fig

def simulate_video_performance(video_data, benchmark_data):
    try:
        published_at = datetime.fromisoformat(video_data['publishedAt'].replace('Z', '+00:00')).date()
        current_date = datetime.now().date()
        days_since_publish = (current_date - published_at).days
    except:
        days_since_publish = 0
    current_views = video_data['viewCount']
    if days_since_publish < 2:
        days_since_publish = 2
    data = []
    benchmark_day_index = min(days_since_publish, len(benchmark_data) - 1)
    for day in range(days_since_publish + 1):
        if day >= len(benchmark_data):
            break
        if day == days_since_publish:
            cumulative_views = current_views
        else:
            ratio = benchmark_data.loc[day, 'median'] / benchmark_data.loc[benchmark_day_index, 'median'] if benchmark_data.loc[benchmark_day_index, 'median'] > 0 else 0
            cumulative_views = int(current_views * ratio)
        if day == 0:
            daily_views = cumulative_views
        else:
            prev_cumulative = data[-1]['cumulative_views']
            daily_views = max(0, cumulative_views - prev_cumulative)
        data.append({
            'day': day,
            'daily_views': daily_views,
            'cumulative_views': cumulative_views,
            'projected': False
        })
    return pd.DataFrame(data)

def compute_channel_benchmark(channel_id, video_age, is_short_filter, percentile_range, num_videos):
    channel_videos, channel_name, channel_stats = fetch_channel_videos_outlier(channel_id, num_videos, YOUTUBE_API_KEY)
    if not channel_videos:
        return None
    video_ids = [v['videoId'] for v in channel_videos]
    detailed_videos = fetch_video_details_outlier(video_ids, YOUTUBE_API_KEY)
    benchmark_df = generate_historical_data(detailed_videos, video_age, is_short_filter)
    if benchmark_df.empty:
        return None
    benchmark_stats = calculate_benchmark(benchmark_df, percentile_range)
    day_index = min(video_age, len(benchmark_stats) - 1)
    if day_index < 0:
        day_index = 0
    channel_average = benchmark_stats.loc[day_index, 'channel_average']
    return channel_average

# --- Outlier Functions for Outlier Calculation (outlier logic integrated above) ---

# =============================================================================
# 9. Unified Search Page (with Integrated Outlier Calculation)
# =============================================================================
def show_search_page():
    st.title("YouTube Video Search & Outlier Analysis")
    # Sidebar: Folder Manager and Search Parameters
    with st.sidebar.expander("Channel Folder Manager"):
        show_channel_folder_manager()
    folders = load_channel_folders()
    folder_choice = st.sidebar.selectbox("Select Folder", list(folders.keys()) if folders else ["None"])
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["Last 24 hours", "Last 48 hours", "Last 4 days", "Last 7 days", "Last 15 days", "Last 28 days", "3 months", "Lifetime"]
    )
    content_filter = st.sidebar.selectbox("Filter By Content Type", ["Shorts", "Videos", "Both"], index=2)
    keyword = st.sidebar.text_input("Keyword (optional)", "")
    
    # Sidebar: Outlier Calculation Parameters (integrated)
    st.sidebar.markdown("### Outlier Calculation Settings")
    include_all = st.sidebar.checkbox("Include all videos for benchmark", value=True)
    if include_all:
        num_videos_for_benchmark = None
    else:
        num_videos_for_benchmark = st.sidebar.slider("Number of videos for benchmark", min_value=10, max_value=200, value=50, step=10)
    video_type_for_benchmark = st.sidebar.radio("Video Type for Benchmark", options=["all", "long_form", "shorts", "auto"], index=0)
    percentile_range = st.sidebar.slider("Middle Percentage Range for Benchmark", min_value=10, max_value=100, value=50, step=5)
    
    # Determine selected channel IDs from folder
    selected_channel_ids = []
    if folder_choice != "None":
        for ch in folders[folder_choice]:
            selected_channel_ids.append(ch["channel_id"])
    st.write(f"**Selected Folder:** {folder_choice}")
    if folder_choice != "None":
        with st.expander("Channels in this folder", expanded=False):
            if folders[folder_choice]:
                for ch in folders[folder_choice]:
                    st.write(f"- {ch['channel_name']}")
            else:
                st.write("(No channels)")
    
    if st.sidebar.button("Clear Cache (force new)"):
        with sqlite3.connect(DB_PATH) as c:
            c.execute("DELETE FROM youtube_cache")
        st.sidebar.success("Cache cleared. Next search is fresh.")
    
    # If a video has been selected for details, show its details instead of search results.
    if "selected_video_id" in st.session_state:
        show_details_content()
        return

    if st.sidebar.button("Search"):
        if folder_choice == "None" or not selected_channel_ids:
            st.error("No folder or channels selected. Please select a folder with at least one channel.")
        else:
            results = search_youtube(
                keyword, selected_channel_ids, timeframe, content_filter, ttl=600,
                outlier_video_type=video_type_for_benchmark,
                outlier_percentile=percentile_range,
                outlier_num_videos=num_videos_for_benchmark
            )
            st.session_state.search_results = results

    if "search_results" in st.session_state and st.session_state.search_results:
        data = st.session_state.search_results
        sort_options = ["views", "upload_date", "outlier_score", "comment_to_view_ratio", "comment_to_like_ratio", "comment_count"]
        sort_by = st.selectbox("Sort by:", sort_options, index=0)

        def parse_sort_value(item):
            if sort_by == "upload_date":
                try:
                    return datetime.strptime(item["published_at"], "%Y-%m-%dT%H:%M:%SZ")
                except:
                    return datetime.min
            val = item.get(sort_by, 0)
            if sort_by in ("comment_to_view_ratio", "comment_to_like_ratio"):
                return float(val.replace("%", "")) if "%" in val else 0.0
            return float(val) if isinstance(val, (int, float, str)) else 0.0

        sorted_data = sorted(data, key=parse_sort_value, reverse=True)
        st.subheader(f"Found {len(sorted_data)} results (sorted by {sort_by})")

        # Display search results in a grid (3 columns per row)
        for i in range(0, len(sorted_data), 3):
            row_chunk = sorted_data[i:i+3]
            cols = st.columns(3)
            for j in range(3):
                with cols[j]:
                    if j < len(row_chunk):
                        row = row_chunk[j]
                        days_ago = int(round(row.get("days_since_published", 0)))
                        days_ago_text = "today" if days_ago == 0 else f"{days_ago} days ago"
                        outlier_val = f"{row.get('outlier_score', 0):.2f}x"
                        outlier_html = f"""
                        <span style="
                            background-color:#4285f4;
                            color:white;
                            padding:3px 8px;
                            border-radius:12px;
                            font-size:0.95rem;
                            font-weight:bold;">
                            {outlier_val}
                        </span>
                        """
                        watch_url = f"https://www.youtube.com/watch?v={row['video_id']}"
                        container_html = f"""
                        <div style="
                            border: 1px solid #CCC;
                            border-radius: 6px;
                            padding: 12px;
                            height: 400px;
                            overflow: hidden;
                            display: flex;
                            flex-direction: column;
                            justify-content: flex-start;
                            margin-bottom: 1rem;
                        ">
                          <a href="{watch_url}" target="_blank">
                            <img src="{row['thumbnail']}" style="width:100%; border-radius:4px; margin-bottom:0.5rem;" />
                          </a>
                          <div style="font-weight:bold; font-size:1rem; text-align:left; margin-bottom:0.3rem;">
                            {row['title']}
                          </div>
                          <div style="font-size:0.9rem; text-align:left; margin-bottom:0.5rem; color:#555;">
                            {row['channel_name']}
                          </div>
                          <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                            <span style="font-weight:bold; color:#FFA500; font-size:0.95rem;">
                              {row['formatted_views']} views
                            </span>
                            {outlier_html}
                          </div>
                          <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                            <span style="font-size:0.85rem;">
                              {row.get('vph_display', '0 VPH')}
                            </span>
                            <span style="font-size:0.85rem;">
                              Published {days_ago_text}
                            </span>
                          </div>
                        </div>
                        """
                        st.markdown(container_html, unsafe_allow_html=True)
                        if st.button("View Details", key=f"view_{row['video_id']}"):
                            st.session_state.selected_video_id = row["video_id"]
                            st.session_state.selected_video_title = row["title"]
                            st.session_state.selected_video_duration = row["duration_seconds"]
                            st.session_state.selected_video_publish_at = row["published_at"]
                            st.experimental_rerun()
                    else:
                        st.empty()

def show_details_content():
    video_id = st.session_state.get("selected_video_id")
    video_title = st.session_state.get("selected_video_title")
    total_duration = st.session_state.get("selected_video_duration", 0)
    published_at = st.session_state.get("selected_video_publish_at")
    if not video_id or not video_title:
        st.error("No video selected.")
        return

    st.markdown(f"## Details for: {video_title}")
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    st.subheader("Comments")
    comments_key = f"comments_{video_id}"
    if comments_key not in st.session_state:
        # Fetch comments if not already cached
        try:
            comments_url = (
                "https://www.googleapis.com/youtube/v3/commentThreads"
                f"?part=snippet&videoId={video_id}&maxResults=50&order=relevance&key={get_youtube_api_key()}"
            )
            resp = requests.get(comments_url)
            resp.raise_for_status()
            data = resp.json()
            comments = []
            for item in data.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet["textDisplay"]
                like_count = int(snippet.get("likeCount", 0))
                comments.append({"text": text, "likeCount": like_count})
            st.session_state[comments_key] = comments
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            st.session_state[comments_key] = []
    comments = st.session_state[comments_key]
    if comments:
        st.write(f"Total Comments Fetched: {len(comments)}")
        top_5 = sorted(comments, key=lambda c: c["likeCount"], reverse=True)[:5]
        st.markdown("**Top 5 Comments (by Likes):**")
        for c in top_5:
            st.markdown(f"**{c['likeCount']} likes** - {c['text']}", unsafe_allow_html=True)
        with st.spinner("Analyzing comments with GPT..."):
            analysis = "Analysis of comments (placeholder)"  # Replace with actual call if desired
        st.markdown("**Comments Analysis:**")
        st.write(analysis)
    else:
        st.info("No comments available.")

    st.subheader("Script")
    transcript, _ = get_transcript_with_fallback(video_id)
    is_short = ("shorts" in video_url.lower()) or (total_duration < 60)
    if is_short:
        if transcript:
            script_text = " ".join([seg["text"] for seg in transcript])
            st.markdown("**Full Script:**")
            st.write(script_text)
            with st.spinner("Summarizing script with GPT..."):
                short_summary = summarize_script(script_text)
            st.markdown("**Script Summary:**")
            st.write(short_summary)
        else:
            st.info("Transcript unavailable for this video.")
    else:
        with st.spinner("Fetching intro & outro script..."):
            intro_txt, outro_txt = get_intro_outro_transcript(video_id, total_duration)
        st.markdown("**Intro Script (First 1 minute):**")
        st.write(intro_txt if intro_txt else "*No intro script available.*")
        st.markdown("**Outro Script (Last 1 minute):**")
        st.write(outro_txt if (outro_txt and total_duration > 120) else "*No outro script available or video is too short.*")
        with st.spinner("Summarizing intro & outro with GPT..."):
            intro_summary, outro_summary = summarize_intro_outro(intro_txt or "", outro_txt or "")
        st.markdown("**Intro & Outro Summaries:**")
        st.write(intro_summary if intro_summary else "*No summary available.*")

    if st.button("Back to Results"):
        for key in ["selected_video_id", "selected_video_title", "selected_video_duration", "selected_video_publish_at"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

# =============================================================================
# Main – Only one navigation (Search page) is used
# =============================================================================
def main():
    init_db(DB_PATH)
    # Default to search page
    if "page" not in st.session_state:
        st.session_state.page = "search"
    show_search_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error in main UI: {e}")
        st.error("An unexpected error occurred. Please check the logs for details.")
    atexit.register(lambda: logger.info("Application shutting down"))
