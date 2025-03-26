
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
    # Assuming YOUTUBE_API_KEY is the one to use, consistent with app(1).py
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]["key"]
except KeyError:
    try:
        # Fallback to YT_API_KEY if YOUTUBE_API_KEY isn't found
        YOUTUBE_API_KEY = st.secrets["YT_API_KEY"]
    except KeyError:
        st.error("No YouTube API key (YOUTUBE_API_KEY or YT_API_KEY) provided in secrets!")
        YOUTUBE_API_KEY = None
except Exception as e:
    st.error(f"Error loading YouTube API key: {e}")
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
    """Parse ISO 8601 duration format to seconds (used by both files)"""
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
        channel_part = channel_name_or_url.split("youtube.com/channel/")[-1]
        channel_id = channel_part.split("/")[0].split("?")[0]
        if channel_id.startswith("UC"):
            return channel_id

    key = get_youtube_api_key()
    # Try searching directly first
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=id&type=channel&q={channel_name_or_url}&key={key}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        items = response.json().get("items", [])
        if items:
            return items[0]["id"]["channelId"]
    except requests.exceptions.HTTPError as e:
        logger.error(f"Channel search error: {e}")
        # Fallback for handles or custom URLs if search fails might be needed, similar to app.py
        # This simplified version keeps app(1).py's original logic more intact
        if 'response' in locals() and response.status_code == 403:
             st.error(f"YouTube API Error (Quota?): {response.text}")
             return None

    # If simple search fails, try parsing like app.py's logic for handles/users
    # Patterns from app.py
    patterns = [
        r'youtube\.com/c/([^/\s?]+)',
        r'youtube\.com/user/([^/\s?]+)',
        r'youtube\.com/@([^/\s?]+)'
    ]
    identifier = None
    pattern_type = None

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, channel_name_or_url)
        if match:
            identifier = match.group(1)
            if i == 0: pattern_type = 'c'
            elif i == 1: pattern_type = 'user'
            elif i == 2: pattern_type = 'handle'
            break

    if identifier:
        try:
            if pattern_type == 'user':
                 # Try resolving username directly
                 username_url = f"https://www.googleapis.com/youtube/v3/channels?part=id&forUsername={identifier}&key={key}"
                 username_res = requests.get(username_url).json()
                 if 'items' in username_res and username_res['items']:
                     return username_res['items'][0]['id']
                 # If direct lookup fails, fall back to search
                 search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=channel&q={identifier}&key={key}"
            elif pattern_type == 'handle':
                 if identifier.startswith('@'):
                     identifier = identifier[1:]
                 search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=channel&q={identifier}&key={key}"
            else: # 'c' or fallback search
                 search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=channel&q={identifier}&key={key}"

            search_res = requests.get(search_url).json()
            if 'items' in search_res and search_res['items']:
                # Ensure we match the name/handle closely if multiple results
                for item in search_res['items']:
                    # Simple check, might need refinement for exact matching
                    if identifier.lower() in item['snippet']['title'].lower() or \
                       identifier.lower() in item['snippet'].get('customUrl','').lower() or \
                       identifier.lower() in item['snippet'].get('handle','').lower():
                       return item['id']['channelId']
                # If no close match, return the first result as a fallback
                return search_res['items'][0]['id']['channelId']

        except requests.exceptions.RequestException as e:
             logger.error(f"Channel identifier resolution error for '{identifier}': {e}")
        except Exception as e:
             logger.error(f"Unexpected error resolving identifier '{identifier}': {e}")

    logger.warning(f"Could not resolve channel ID for: {channel_name_or_url}")
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
            processed_ids = set() # Avoid duplicates
            with st.spinner("Resolving channel IDs..."):
                 for line in lines:
                     line = line.strip()
                     if not line:
                         continue
                     ch_id = get_channel_id(line)
                     if ch_id and ch_id not in processed_ids:
                         # Attempt to get actual channel name for better display
                         try:
                             ch_details_url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={ch_id}&key={get_youtube_api_key()}"
                             ch_details_res = requests.get(ch_details_url).json()
                             ch_name = ch_details_res['items'][0]['snippet']['title'] if ch_details_res.get('items') else line
                         except Exception:
                             ch_name = line # Fallback to user input
                         channel_list.append({"channel_name": ch_name, "channel_id": ch_id})
                         processed_ids.add(ch_id)
                     elif not ch_id:
                         st.warning(f"Could not find channel ID for '{line}'. Skipping.")

            if not channel_list:
                st.error("At least one valid channel is required.")
                return
            folders[folder_name] = channel_list
            save_channel_folders(folders)
            st.success(f"Folder '{folder_name}' created with {len(channel_list)} channel(s).")
            # Pre-cache data for newly added channels
            with st.spinner("Pre-caching recent data for new channels..."):
                 for ch in channel_list:
                      try:
                           search_youtube("", [ch["channel_id"]], "3 months", "Both", ttl=7776000) # Use long TTL for initial cache
                      except Exception as e:
                           st.warning(f"Could not pre-cache data for {ch['channel_name']}: {e}")

    elif action == "Modify Folder":
        if not folders:
            st.info("No folders available.")
            return
        folder_keys = list(folders.keys())
        selected_folder = st.selectbox("Select Folder to Modify", folder_keys, key="modify_folder_select")

        if selected_folder and selected_folder in folders: # Check existence after selection
            st.write("Channels in this folder:")
            current_channels = folders.get(selected_folder, []) # Use .get for safety
            current_channel_ids = {ch['channel_id'] for ch in current_channels}

            if current_channels:
                # Use DataFrame for better display if many channels
                if len(current_channels) > 10:
                    df_channels = pd.DataFrame(current_channels)
                    st.dataframe(df_channels[['channel_name', 'channel_id']], use_container_width=True)
                else:
                    for ch in current_channels:
                         st.write(f"- {ch['channel_name']} ({ch['channel_id']})")
            else:
                st.write("(No channels yet)")

            modify_action = st.radio("Modify Action", ["Add Channels", "Remove Channel"], key="modify_folder_action")

            if modify_action == "Add Channels":
                st.write("Enter channel name(s) or URL(s) (one per line):")
                new_ch_text = st.text_area("Add Channels", key="add_channels_text")
                if st.button("Add to Folder", key="add_to_folder_btn"):
                    lines = new_ch_text.strip().split("\n")
                    added_channels = []
                    with st.spinner("Resolving and adding channel IDs..."):
                         for line in lines:
                             line = line.strip()
                             if not line:
                                 continue
                             ch_id = get_channel_id(line)
                             if ch_id and ch_id not in current_channel_ids:
                                 try:
                                     ch_details_url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={ch_id}&key={get_youtube_api_key()}"
                                     ch_details_res = requests.get(ch_details_url).json()
                                     ch_name = ch_details_res['items'][0]['snippet']['title'] if ch_details_res.get('items') else line
                                 except Exception:
                                     ch_name = line
                                 new_channel_data = {"channel_name": ch_name, "channel_id": ch_id}
                                 folders[selected_folder].append(new_channel_data)
                                 added_channels.append(new_channel_data)
                                 current_channel_ids.add(ch_id) # Update set immediately
                             elif ch_id and ch_id in current_channel_ids:
                                 st.warning(f"Channel '{line}' (ID: {ch_id}) is already in the folder. Skipping.")
                             elif not ch_id:
                                 st.warning(f"Could not find channel ID for '{line}'. Skipping.")

                    if added_channels:
                         save_channel_folders(folders)
                         st.success(f"Added {len(added_channels)} channel(s) to folder '{selected_folder}'.")
                         # Pre-cache for newly added channels
                         with st.spinner("Pre-caching recent data for new channels..."):
                              for ch_data in added_channels:
                                   try:
                                       search_youtube("", [ch_data["channel_id"]], "3 months", "Both", ttl=7776000)
                                   except Exception as e:
                                       st.warning(f"Could not pre-cache data for {ch_data['channel_name']}: {e}")
                         st.rerun() # Rerun to update the displayed list
                    else:
                         st.info("No new valid channels were added.")


            elif modify_action == "Remove Channel":
                 if current_channels:
                     channel_options = {f"{ch['channel_name']} ({ch['channel_id']})": ch['channel_id'] for ch in current_channels}
                     rem_choice_display = st.selectbox("Select channel to remove", list(channel_options.keys()), key="remove_channel_choice")
                     if st.button("Remove Channel", key="remove_channel_btn"):
                         rem_choice_id = channel_options.get(rem_choice_display)
                         if rem_choice_id:
                             original_count = len(folders[selected_folder])
                             folders[selected_folder] = [c for c in folders[selected_folder] if c["channel_id"] != rem_choice_id]
                             if len(folders[selected_folder]) < original_count:
                                 save_channel_folders(folders)
                                 st.success(f"Channel '{rem_choice_display}' removed from '{selected_folder}'.")
                                 st.rerun() # Rerun to update the list
                             else:
                                 st.error("Failed to remove the selected channel.") # Should not happen if logic is correct
                         else:
                             st.error("Invalid selection.") # Should not happen with selectbox
                 else:
                     st.info("No channels in this folder to remove.")
        # else: Handle case where selected folder might become invalid if deleted concurrently? Low priority.


    elif action == "Delete Folder": # Corrected indentation
        if not folders:
            st.info("No folders available.")
            return
        folder_keys = list(folders.keys())
        selected_folder = st.selectbox("Select Folder to Delete", folder_keys, key="delete_folder_select")
        if selected_folder and selected_folder in folders: # Check existence
            st.warning(f"Are you sure you want to delete the folder '{selected_folder}'? This cannot be undone.")
            if st.button(f"Confirm Delete '{selected_folder}'", key="delete_folder_btn"):
                del folders[selected_folder]
                save_channel_folders(folders)
                st.success(f"Folder '{selected_folder}' deleted.")
                st.rerun() # Rerun to update folder list


# =============================================================================
# 6. Transcript & Fallback
# =============================================================================
def get_transcript(video_id):
    try:
        # Try fetching English first, then allow YouTube to auto-select if English isn't available
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
        except NoTranscriptFound:
             # If English not found, try finding *any* generated transcript
             try:
                 transcript = transcript_list.find_generated_transcript(transcript_list.languages)
             except NoTranscriptFound:
                 # If still no generated transcript, try manual ones (less likely for analysis but possible)
                 try:
                     transcript = transcript_list.find_manually_created_transcript(transcript_list.languages)
                 except NoTranscriptFound:
                     logger.warning(f"No transcript found for {video_id} in any language.")
                     return None
        return transcript.fetch()

    except TranscriptsDisabled:
        logger.warning(f"Transcripts are disabled for video {video_id}")
        return None
    except Exception as e:
        logger.error(f"Transcript error for {video_id}: {e}")
        return None

def download_audio(video_id):
    if not shutil.which("ffmpeg"):
        st.error("ffmpeg not found. Please install ffmpeg and ensure it is in your PATH.")
        logger.error("ffmpeg executable not found in PATH.")
        return None
    if not check_ytdlp_installed():
         st.error("yt-dlp not found. Please install yt-dlp (`pip install yt-dlp`).")
         logger.error("yt-dlp not found.")
         return None

    temp_dir = tempfile.mkdtemp() # Use a unique temp directory
    # Sanitize video_id just in case, although usually safe
    safe_video_id = re.sub(r'[^\w-]', '', video_id)
    output_template = os.path.join(temp_dir, f"{safe_video_id}.%(ext)s")
    audio_path_expected = os.path.join(temp_dir, f"{safe_video_id}.mp3")
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128', # 128kbps is generally enough for speech
        }],
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(), # Explicitly provide ffmpeg path
         # Timeout to prevent hangs on problematic downloads
        'socket_timeout': 30, # seconds
    }

    try:
        import yt_dlp
        logger.info(f"Attempting to download audio for {video_id} to {temp_dir}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if os.path.exists(audio_path_expected):
            logger.info(f"Audio downloaded successfully: {audio_path_expected}")
            # Return path and the directory it's in for later cleanup
            return audio_path_expected, temp_dir
        else:
            # Check if it downloaded with a different extension (e.g., opus, m4a)
            found_audio = None
            for filename in os.listdir(temp_dir):
                 if filename.startswith(safe_video_id) and filename.split('.')[-1] in ['mp3', 'm4a', 'opus', 'ogg', 'wav']:
                     found_audio = os.path.join(temp_dir, filename)
                     logger.warning(f"Audio downloaded with unexpected extension: {filename}. Proceeding.")
                     # If not mp3, potentially convert here if Whisper strictly needs mp3,
                     # but Whisper often handles others. Let's assume it does for now.
                     return found_audio, temp_dir

            logger.error(f"yt-dlp finished but expected audio file not found: {audio_path_expected}")
            shutil.rmtree(temp_dir) # Clean up directory if download failed
            return None, None

    except yt_dlp.utils.DownloadError as e:
         logger.error(f"yt-dlp DownloadError for {video_id}: {e}")
         st.error(f"Error downloading audio: Video might be unavailable or region-locked.")
         shutil.rmtree(temp_dir)
         return None, None
    except Exception as e:
        logger.error(f"Error during audio download process for {video_id}: {e}")
        st.error(f"An unexpected error occurred during audio download: {e}")
        # Clean up the temp directory in case of any error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None, None

def generate_transcript_with_openai(audio_file):
    if not OPENAI_API_KEY:
         st.error("OpenAI API Key not configured. Cannot use Whisper fallback.")
         return None, None

    max_size_bytes = 25 * 1024 * 1024 # OpenAI API limit (slightly under 26MB for safety)
    try:
        actual_size = os.path.getsize(audio_file)
        logger.info(f"Audio file size: {actual_size / (1024*1024):.2f} MB")
    except OSError as e:
        st.error(f"Error accessing audio file {audio_file}: {e}")
        logger.error(f"Could not get size of audio file {audio_file}: {e}")
        return None, None

    file_to_transcribe = audio_file
    temp_snippet_file = None

    if actual_size > max_size_bytes:
        st.warning(f"Audio file is larger than OpenAI's limit ({max_size_bytes / (1024*1024):.1f}MB). Transcribing only the first part.")
        logger.warning(f"Audio file {audio_file} size {actual_size} exceeds limit {max_size_bytes}. Creating snippet.")
        # Create a temporary snippet
        # Calculate max duration based on 128kbps = 16 KB/s approx
        # Max duration ~ max_size_bytes / (16 * 1024) seconds
        max_duration_sec = int(max_size_bytes / (16 * 1024 * 0.9)) # Use 90% margin
        # Limit snippet duration reasonably, e.g., 15-20 mins max?
        snippet_duration = min(max_duration_sec, 15 * 60) # Max 15 mins snippet

        temp_snippet_file = audio_file.replace(".mp3", "_snippet.mp3")
        if temp_snippet_file == audio_file: # Ensure different name if not mp3
             base, ext = os.path.splitext(audio_file)
             temp_snippet_file = f"{base}_snippet{ext}"

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-i", audio_file,
            "-t", str(snippet_duration), # Duration limit
            "-c", "copy", # Try to copy codecs to be fast
            "-y", temp_snippet_file
        ]
        try:
            logger.info(f"Running ffmpeg to create snippet: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            file_to_transcribe = temp_snippet_file
            logger.info(f"Created snippet: {file_to_transcribe}")
            # Verify snippet size
            snippet_size = os.path.getsize(file_to_transcribe)
            if snippet_size > max_size_bytes:
                 logger.error(f"Snippet size {snippet_size} still exceeds limit. Aborting Whisper.")
                 st.error("Failed to create a small enough audio snippet for transcription.")
                 # Cleanup snippet if it exists
                 if temp_snippet_file and os.path.exists(temp_snippet_file):
                     os.remove(temp_snippet_file)
                 return None, None

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg snippet creation failed. Error: {e.stderr.decode()}")
            st.error(f"Failed to create audio snippet for transcription: {e}")
            # Cleanup snippet if it exists
            if temp_snippet_file and os.path.exists(temp_snippet_file):
                os.remove(temp_snippet_file)
            return None, None
        except Exception as e:
             logger.error(f"Unexpected error during snippet creation: {e}")
             st.error(f"Unexpected error creating audio snippet: {e}")
             if temp_snippet_file and os.path.exists(temp_snippet_file):
                 os.remove(temp_snippet_file)
             return None, None

    try:
        logger.info(f"Transcribing {file_to_transcribe} using OpenAI Whisper...")
        with open(file_to_transcribe, "rb") as file:
            # Use the new client if available, otherwise fallback
            try:
                 from openai import OpenAI
                 client = OpenAI(api_key=OPENAI_API_KEY)
                 transcript_response = client.audio.transcriptions.create(
                      model="whisper-1",
                      file=file
                 )
                 text = transcript_response.text
            except ImportError:
                 # Fallback to older openai<1.0 syntax
                 transcript_response = openai.Audio.transcribe(
                      model="whisper-1",
                      file=file,
                      api_key=OPENAI_API_KEY # Pass key explicitly for older versions
                 )
                 text = transcript_response["text"] # type: ignore

        logger.info(f"Whisper transcription successful for {file_to_transcribe}.")

        # Basic segmentation if Whisper doesn't provide timestamps (model 'whisper-1' usually doesn't)
        words = text.split()
        segments = []
        # Estimate words per segment or time per segment
        # This is highly approximate and less useful than real timestamps
        # Let's just return the full text for now, as fake timestamps are misleading
        # For analysis later (intro/outro), the full text might be enough if split is simple
        # We can return a single segment containing the whole text
        total_duration_estimate = actual_size / (16 * 1024) # Rough estimate based on 128kbps

        segments.append({
             "start": 0.0,
             "duration": total_duration_estimate, # Placeholder duration
             "text": text
        })

        # Clean up the snippet file if it was created
        if temp_snippet_file and os.path.exists(temp_snippet_file):
            os.remove(temp_snippet_file)
            logger.info(f"Removed temporary snippet file: {temp_snippet_file}")

        return segments, "openai_whisper" # Indicate source

    except openai.APIError as e:
         logger.error(f"OpenAI API error during Whisper transcription: {e}")
         st.error(f"OpenAI API error: {e}. Check API key, usage limits, or file format.")
         return None, None
    except Exception as e:
        logger.error(f"Error during OpenAI Whisper transcription: {e}")
        st.error(f"Error during Whisper transcription: {e}")
        # Cleanup snippet if it exists and something went wrong during transcription
        if temp_snippet_file and os.path.exists(temp_snippet_file):
             try:
                 os.remove(temp_snippet_file)
             except OSError:
                 pass # Ignore cleanup error
        return None, None

def get_transcript_with_fallback(video_id):
    # Use session state to cache per video_id during a single run
    cache_key_data = f"transcript_data_{video_id}"
    cache_key_source = f"transcript_source_{video_id}"

    if cache_key_data in st.session_state:
        logger.info(f"Using cached transcript for {video_id} from session state.")
        return st.session_state[cache_key_data], st.session_state.get(cache_key_source)

    logger.info(f"Attempting to fetch YouTube transcript for {video_id}")
    direct_transcript = get_transcript(video_id)

    if direct_transcript:
        logger.info(f"Successfully fetched YouTube transcript for {video_id}")
        st.session_state[cache_key_data] = direct_transcript
        st.session_state[cache_key_source] = "youtube"
        return direct_transcript, "youtube"

    # If YouTube transcript fails, try fallback
    logger.warning(f"YouTube transcript failed for {video_id}. Attempting Whisper fallback.")
    st.warning("YouTube transcript not available, attempting audio download & Whisper transcription (this may take a while)...")

    audio_file_path, audio_temp_dir = download_audio(video_id)

    if audio_file_path and audio_temp_dir:
        logger.info(f"Audio downloaded for {video_id}. Proceeding with Whisper.")
        whisper_transcript, whisper_source = generate_transcript_with_openai(audio_file_path)

        # Clean up the audio file and its directory *after* transcription attempt
        try:
            logger.info(f"Cleaning up audio temp directory: {audio_temp_dir}")
            shutil.rmtree(audio_temp_dir)
        except Exception as e:
            logger.error(f"Failed to clean up audio temp directory {audio_temp_dir}: {e}")

        if whisper_transcript:
            logger.info(f"Whisper transcription successful for {video_id}.")
            st.success("Whisper transcription successful.")
            st.session_state[cache_key_data] = whisper_transcript
            st.session_state[cache_key_source] = whisper_source
            return whisper_transcript, whisper_source
        else:
            logger.error(f"Whisper transcription failed for {video_id}.")
            st.error("Whisper transcription fallback failed.")
            st.session_state[cache_key_data] = None
            st.session_state[cache_key_source] = None
            return None, None
    else:
        logger.error(f"Audio download failed for {video_id}. Cannot use Whisper fallback.")
        st.error("Audio download failed. Cannot perform Whisper transcription.")
        st.session_state[cache_key_data] = None
        st.session_state[cache_key_source] = None
        return None, None


def get_intro_outro_transcript(video_id, total_duration):
    # This function might need adjustment depending on whether Whisper provides usable timestamps
    # or just one large text block.

    transcript, source = get_transcript_with_fallback(video_id)
    if not transcript:
        return (None, None)

    end_intro_sec = min(60, total_duration * 0.2 if total_duration else 60) # First 60s or 20%
    start_outro_sec = max(total_duration - 60, total_duration * 0.8 if total_duration else 0) # Last 60s or from 80%

    intro_lines = []
    outro_lines = []

    if source == "youtube": # YouTube provides good timestamps
        for item in transcript:
            start_sec = float(item["start"])
            # Use end time if available, otherwise estimate from duration
            end_sec = start_sec + float(item.get("duration", 2.0)) # Assume 2s duration if missing
            # Check for overlap with intro/outro periods
            if max(start_sec, 0) < end_intro_sec:
                 intro_lines.append(item["text"])
            if start_sec >= start_outro_sec and start_sec < total_duration:
                 outro_lines.append(item["text"])

    elif source == "openai_whisper":
        # Whisper might return one large block or segmented text.
        # If one block, we have to crudely split. If segmented, might be better.
        full_text = " ".join(seg["text"] for seg in transcript)
        words = full_text.split()
        num_words = len(words)

        if num_words == 0 or total_duration == 0:
            return (None, None)

        words_per_second = num_words / total_duration if total_duration > 0 else 3 # Estimate WPM/WPS

        intro_word_count = int(end_intro_sec * words_per_second)
        outro_start_word_index = int(start_outro_sec * words_per_second)

        intro_text = " ".join(words[:intro_word_count]) if intro_word_count > 0 else None
        outro_text = " ".join(words[outro_start_word_index:]) if outro_start_word_index < num_words else None
        # Filter out very short/empty strings
        intro_text = intro_text if intro_text and len(intro_text.split()) > 5 else None
        outro_text = outro_text if outro_text and len(outro_text.split()) > 5 else None
        return (intro_text, outro_text)

    else: # Unknown source or format
         logger.warning(f"Unknown transcript source '{source}' for {video_id}. Cannot reliably extract intro/outro.")
         # Fallback: try to use first/last few segments if structure looks like YouTube's
         if isinstance(transcript, list) and transcript and 'text' in transcript[0]:
             intro_text = " ".join(seg["text"] for seg in transcript[:10]) # Guess first 10 segments
             outro_text = " ".join(seg["text"] for seg in transcript[-10:]) # Guess last 10 segments
             return (intro_text, outro_text)
         else:
             return (None, None)

    intro_full_text = " ".join(intro_lines) if intro_lines else None
    outro_full_text = " ".join(outro_lines) if outro_lines else None

    return (intro_full_text, outro_full_text)


def summarize_intro_outro(intro_text, outro_text):
    if not intro_text and not outro_text:
        return (None, None)

    if not OPENAI_API_KEY:
         st.error("OpenAI API key not configured. Cannot summarize.")
         return (None, None)

    # Generate a hash based on the combined text for caching
    combined_text = (intro_text or "") + "||" + (outro_text or "")
    cache_key = f"intro_outro_summary_{hashlib.sha256(combined_text.encode()).hexdigest()}"

    # Check session state cache
    if cache_key in st.session_state:
        logger.info(f"Using cached intro/outro summary for hash {cache_key[-8:]}")
        # Assuming cached value is the summary text directly
        return (st.session_state[cache_key], st.session_state[cache_key]) # Return tuple format

    logger.info("Generating new intro/outro summary using OpenAI...")
    prompt_parts = []
    if intro_text:
        prompt_parts.append(f"Intro snippet:\n{intro_text[:2000]}\n") # Limit length
    if outro_text:
        prompt_parts.append(f"Outro snippet:\n{outro_text[:2000]}\n") # Limit length

    prompt_parts.append(
        "Based *only* on the text provided above, produce two concise bullet-point summaries (3-5 points each):\n"
        "1. Key points or hooks mentioned in the Intro snippet.\n"
        "2. Key takeaways or calls to action mentioned in the Outro snippet.\n"
        "If a snippet is missing or too short to analyze, state that clearly for that section. "
        "Format the output clearly, starting with 'Intro Summary:' and 'Outro Summary:'."
    )
    prompt_str = "\n".join(prompt_parts)

    try:
        # Use updated client syntax if possible
         from openai import OpenAI
         client = OpenAI(api_key=OPENAI_API_KEY)
         response = client.chat.completions.create(
             model="gpt-3.5-turbo", # Use a cheaper/faster model for summarization
             messages=[{"role": "user", "content": prompt_str}],
             max_tokens=300,
             temperature=0.5,
         )
         result_txt = response.choices[0].message.content

    except ImportError:
         # Fallback to older openai<1.0 syntax
         response = openai.ChatCompletion.create(
             model="gpt-3.5-turbo",
             messages=[{"role": "user", "content": prompt_str}],
             max_tokens=300,
             temperature=0.5,
             api_key=OPENAI_API_KEY
         )
         result_txt = response.choices[0].message['content'] # type: ignore


    # Store the raw result in session cache
    st.session_state[cache_key] = result_txt
    logger.info(f"Stored new intro/outro summary in cache for hash {cache_key[-8:]}")

    # Return in the expected tuple format (even though both elements are the same now)
    return (result_txt, result_txt)

    # Error handling for API calls
    except openai.APIError as e:
         logger.error(f"OpenAI API error during intro/outro summarization: {e}")
         st.error(f"OpenAI API error during summarization: {e}")
         return ("*Summary failed due to API error.*", "*Summary failed due to API error.*")
    except Exception as e:
        logger.error(f"Unexpected error during intro/outro summarization: {e}")
        st.error(f"Unexpected error during summarization: {e}")
        return ("*Summary failed due to an unexpected error.*", "*Summary failed due to an unexpected error.*")


def summarize_script(script_text):
    if not script_text or not script_text.strip():
        return "No script text to summarize."

    if not OPENAI_API_KEY:
         st.error("OpenAI API key not configured. Cannot summarize script.")
         return "Summary failed: OpenAI key missing."

    # Use a simpler hash for caching, focusing on content
    hashed = hashlib.sha256(script_text.strip().encode("utf-8")).hexdigest()

    if "script_summary_cache" not in st.session_state:
        st.session_state["script_summary_cache"] = {}
    if hashed in st.session_state["script_summary_cache"]:
        logger.info(f"Using cached script summary for hash {hashed[-8:]}")
        return st.session_state["script_summary_cache"][hashed]

    logger.info("Generating new script summary using OpenAI...")
    # Truncate long scripts to avoid excessive token usage/cost
    max_chars = 10000 # Approx 2500 tokens
    truncated_script = script_text[:max_chars]
    if len(script_text) > max_chars:
         logger.warning(f"Script text truncated to {max_chars} characters for summarization.")

    prompt_content = (
        f"Please provide a concise, neutral summary (around 100-150 words) "
        f"of the key topics discussed in the following video script:\n\n"
        f"'''\n{truncated_script}\n'''\n\n"
        f"Focus on the main subject matter and information presented."
    )

    try:
         # Use updated client syntax if possible
         from openai import OpenAI
         client = OpenAI(api_key=OPENAI_API_KEY)
         response = client.chat.completions.create(
             model="gpt-3.5-turbo", # Cheaper model is fine for summary
             messages=[{"role": "user", "content": prompt_content}],
             max_tokens=250, # Limit output size
             temperature=0.5,
         )
         summary = response.choices[0].message.content

    except ImportError:
         # Fallback to older openai<1.0 syntax
         response = openai.ChatCompletion.create(
             model="gpt-3.5-turbo",
             messages=[{"role": "user", "content": prompt_content}],
             max_tokens=250,
             temperature=0.5,
             api_key=OPENAI_API_KEY
         )
         summary = response.choices[0].message['content'] # type: ignore

    # Cache the result
    st.session_state["script_summary_cache"][hashed] = summary
    logger.info(f"Stored new script summary in cache for hash {hashed[-8:]}")
    return summary

    # Error handling
    except openai.APIError as e:
        logger.error(f"OpenAI API error during script summarization: {e}")
        st.error(f"OpenAI API error during script summary: {e}")
        return "Script summary failed due to API error."
    except Exception as e:
        logger.error(f"Unexpected error during script summarization: {e}")
        st.error(f"Unexpected error during script summary: {e}")
        return "Script summary failed due to an unexpected error."


# =============================================================================
# 8. Searching & Calculating Outliers (<<<< INTEGRATION POINT >>>>)
# =============================================================================
def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Function from app.py required for its outlier logic (if we were fully implementing it)
# def parse_duration(duration_str): # Already defined above as parse_iso8601_duration
#    pass # Use parse_iso8601_duration

# Helper for app.py's logic - Simulating view trajectory (Simplified Placeholder)
# NOTE: Implementing the full simulation accurately within this structure is complex
# and likely violates the "don't change anything else" rule.
# This placeholder shows where it *would* go but won't be used in the minimal change approach.
def generate_view_trajectory_placeholder(video_id, days, total_views, is_short):
    """Placeholder for the view simulation logic from app.py."""
    # In a full implementation, this would generate daily view data
    # based on the logic in app.py (exponential for shorts, sigmoid for long-form).
    # For the minimal change, this function is NOT used.
    data = []
    # Simplified: linear growth just for structure example
    daily_avg = total_views / days if days > 0 else 0
    for day in range(days):
        cumulative = int(daily_avg * (day + 1))
        daily = int(daily_avg) if day > 0 else cumulative
        data.append({
            'videoId': video_id,
            'day': day,
            'daily_views': daily,
            'cumulative_views': cumulative
        })
    return data

# Helper for app.py's logic - Calculating benchmark (Simplified Placeholder)
# NOTE: As above, this is part of the app.py logic not used in the minimal change.
def calculate_benchmark_placeholder(df_history, band_percentage=50):
    """Placeholder for benchmark calculation from app.py."""
    # In a full implementation, this calculates quantiles (bands), median, mean
    # from the simulated historical data (`df_history`).
    # For the minimal change, this function is NOT used.
    if df_history.empty:
        return pd.DataFrame(columns=['day', 'lower_band', 'upper_band', 'median', 'mean', 'channel_average'])

    # Simplified: just calculate mean per day as a placeholder 'average'
    summary = df_history.groupby('day')['cumulative_views'].agg([
        ('mean', 'mean')
    ]).reset_index()
    summary['lower_band'] = summary['mean'] * 0.5 # Placeholder band
    summary['upper_band'] = summary['mean'] * 1.5 # Placeholder band
    summary['median'] = summary['mean'] # Placeholder median
    summary['channel_average'] = summary['mean'] # Placeholder average
    summary['count'] = df_history.groupby('day').size().values # Keep count for info
    return summary

# --- The MODIFIED calculate_metrics function ---
def calculate_metrics(df):
    """
    Calculates performance metrics for videos in the DataFrame.
    INTEGRATION: Replaces the original outlier_score calculation
                 with a simplified version inspired by app.py, primarily using vph_ratio.
                 It does NOT implement the full view trajectory simulation from app.py
                 due to structural incompatibility and the 'keep rest same' constraint.
    """
    if df.empty:
        return df, None # Return empty dataframe if input is empty

    now_utc = datetime.now(timezone.utc) # Use timezone-aware now

    # Convert published_at to datetime objects, handling potential errors
    df['published_at_dt'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)

    # Drop rows where published_at couldn't be parsed
    df.dropna(subset=['published_at_dt'], inplace=True)
    if df.empty:
        logger.warning("DataFrame empty after dropping rows with invalid publish dates.")
        return df, None

    # Calculate age consistently
    df['hours_since_published'] = ((now_utc - df['published_at_dt']).dt.total_seconds() / 3600).round(1)
    # Ensure hours_since_published is not zero or negative to avoid division errors
    df['hours_since_published'] = df['hours_since_published'].apply(lambda x: max(x, 0.1)) # Min 0.1 hours
    df['days_since_published'] = (df['hours_since_published'] / 24).round(2)

    # --- Standard Metrics (Mostly from original app(1).py) ---
    # Ensure numeric types for calculation columns, coercing errors to NaN or 0
    for col in ['views', 'like_count', 'comment_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Raw VPH
    df['raw_vph'] = df['views'] / df['hours_since_published']

    # Peak VPH (using first 30 days = 720 hours)
    # Apply minimum hours condition correctly
    df['peak_hours'] = df['hours_since_published'].apply(lambda x: min(x, 720.0))
    df['peak_vph'] = df['views'] / df['peak_hours']

    # Effective VPH (Use raw for recent, peak for older, same as original)
    df['effective_vph'] = df.apply(lambda row: row['raw_vph'] if row['days_since_published'] < 90 else row['peak_vph'], axis=1)

    # Engagement Rate (weighted comments)
    df['engagement_metric'] = df['like_count'] + 5 * df['comment_count']
    df['engagement_rate'] = df['engagement_metric'] / df['views'].apply(lambda x: max(x, 1)) # Avoid division by zero views

    # Comment/Like to View Ratios (Floats for calculation)
    df['cvr_float'] = df['comment_count'] / df['views'].apply(lambda x: max(x, 1))
    df['clr_float'] = df['like_count'] / df['views'].apply(lambda x: max(x, 1))

    # --- Channel Averages (Crucial for Relative Metrics) ---
    # Calculate averages *per channel* if multiple channels are present
    channel_metrics = {}
    results_list = []

    for channel_id, group in df.groupby('channelId'):
        group = group.copy() # Avoid SettingWithCopyWarning

        # Filter for recent videos (last 30 days) within this channel's group
        # Use 'days_since_published' which is already calculated
        recent_videos = group[group['days_since_published'] <= 30].copy()

        # If no videos in the last 30 days, use up to 90 days for averaging
        if recent_videos.empty:
            recent_videos = group[group['days_since_published'] <= 90].copy()
            # If still empty, use all videos in the group as fallback
            if recent_videos.empty:
                 logger.warning(f"No videos within 90 days for channel {channel_id}. Using all {len(group)} videos for average.")
                 recent_videos = group.copy()

        # --- Calculate Channel Average Metrics ---
        # Use 'effective_vph' for averaging VPH (more stable than raw VPH)
        channel_avg_vph = recent_videos['effective_vph'].mean()

        # Calculate average engagement rate
        channel_avg_engagement = recent_videos['engagement_rate'].mean()

        # Calculate average CVR and CLR
        channel_avg_cvr = recent_videos['cvr_float'].mean()
        channel_avg_clr = recent_videos['clr_float'].mean()

        # Store averages for this channel
        channel_metrics[channel_id] = {
            'avg_vph': channel_avg_vph,
            'avg_engagement': channel_avg_engagement,
            'avg_cvr': channel_avg_cvr,
            'avg_clr': channel_avg_clr,
            'num_videos_for_avg': len(recent_videos)
        }
        logger.info(f"Channel {channel_id} averages (based on {len(recent_videos)} videos): VPH={channel_avg_vph:.2f}, Eng={channel_avg_engagement:.4f}, CVR={channel_avg_cvr:.4f}, CLR={channel_avg_clr:.4f}")

        # --- Calculate Ratios Relative to Channel Average ---
        # Use stored averages, with safe division (minimum denominator)
        group['channel_avg_vph'] = channel_avg_vph
        group['channel_avg_engagement'] = channel_avg_engagement
        group['channel_avg_cvr'] = channel_avg_cvr
        group['channel_avg_clr'] = channel_avg_clr

        group['vph_ratio'] = group['effective_vph'] / max(channel_avg_vph, 0.1) # Min avg VPH of 0.1
        group['engagement_ratio'] = group['engagement_rate'] / max(channel_avg_engagement, 0.0001) # Min avg engagement 0.01%
        group['outlier_cvr'] = group['cvr_float'] / max(channel_avg_cvr, 0.0001) # Min avg CVR 0.01%
        group['outlier_clr'] = group['clr_float'] / max(channel_avg_clr, 0.0001) # Min avg CLR 0.01%

        # --- **** NEW OUTLIER SCORE CALCULATION **** ---
        # Inspired by app.py's concept (relative performance) but using VPH ratio as the primary driver,
        # as the full simulation isn't feasible here.
        # Removed the log transform and recency factor from the original app(1).py formula.
        # Weighting remains similar to original app(1).py's combined_performance idea.

        # Calculate a combined performance score (weighted VPH and engagement)
        # Adjust weights if needed: 90% VPH, 10% Engagement
        group['combined_performance'] = (0.9 * group['vph_ratio']) + (0.1 * group['engagement_ratio'])

        # The 'outlier_score' is now this combined performance ratio.
        # A score of 1.0 means perfectly average performance based on this weighted metric.
        # A score of 2.0 means twice the average performance.
        group['outlier_score'] = group['combined_performance']

        # Keep breakout_score linked to outlier_score for consistency with original structure
        group['breakout_score'] = group['outlier_score']

        # --- Formatting for Display ---
        group['formatted_views'] = group['views'].apply(format_number)
        # Use CVR/CLR floats for percentage formatting
        group['comment_to_view_ratio'] = group['cvr_float'].apply(lambda x: f"{x*100:.2f}%")
        group['comment_to_like_ratio'] = group['clr_float'].apply(lambda x: f"{x*100:.2f}%") # Renamed from app(1), CLR is Like/View
        group['like_to_view_ratio'] = group['clr_float'].apply(lambda x: f"{x*100:.2f}%") # Added for clarity

        # Display VPH
        group['vph_display'] = group['effective_vph'].apply(lambda x: f"{int(round(x,0))} VPH" if x>0 else "0 VPH")

        results_list.append(group)

    # Concatenate results from all channel groups
    if not results_list:
         logger.warning("No results generated after processing groups.")
         return pd.DataFrame(), None # Return empty if something went wrong

    final_df = pd.concat(results_list)

    # --- Final Cleanup & Return ---
    # Drop intermediate calculation columns if desired (optional)
    # final_df = final_df.drop(columns=['published_at_dt', 'peak_hours', 'engagement_metric'])

    # Sort by outlier score by default before returning? Or leave sorting to UI.
    # final_df = final_df.sort_values(by='outlier_score', ascending=False)

    # Return the dataframe with all calculated metrics and None (second element was unused)
    return final_df, None
# --- END OF MODIFIED calculate_metrics ---


def fetch_all_snippets(channel_id, order_param, timeframe, query, published_after):
    """Fetches basic video snippet data (ID, title, publish time) for a channel."""
    all_videos = []
    page_token = None
    try:
        key = get_youtube_api_key() # Get key once
    except Exception as e:
        st.error(f"Cannot fetch snippets: {e}")
        return []

    base_url = (
        f"https://www.googleapis.com/youtube/v3/search?part=snippet"
        f"&channelId={channel_id}&maxResults=50&type=video&order={order_param}&key={key}" # Max 50 per page
    )
    if published_after:
        base_url += f"&publishedAfter={published_after}"
    if query:
        # Ensure query is URL-encoded (requests usually handles this, but good practice)
        from urllib.parse import quote_plus
        base_url += f"&q={quote_plus(query)}"

    max_results_limit = 200 # Limit total results to prevent excessive API usage
    fetched_count = 0

    while fetched_count < max_results_limit:
        url = base_url
        if page_token:
            url += f"&pageToken={page_token}"

        try:
            # logger.debug(f"Requesting Snippets URL: {url}") # Use debug level
            resp = requests.get(url, timeout=20) # Add timeout
            # logger.debug(f"Response status code: {resp.status_code}")

            # Handle API errors gracefully
            if resp.status_code == 403:
                 logger.error(f"API Key Error (Forbidden): {resp.text}")
                 st.error(f"YouTube API error (Forbidden/Quota Exceeded?). Please check your API key and quota.")
                 break # Stop fetching for this channel
            elif resp.status_code == 400:
                 logger.error(f"API Bad Request: {resp.text}")
                 st.error(f"YouTube API error (Bad Request): {resp.text}. Check parameters.")
                 break # Stop fetching
            resp.raise_for_status() # Raise other HTTP errors

            data = resp.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error during snippet request. URL: {url.split('&key=')[0]}... Error: {str(e)}")
            # Log response text if possible
            try:
                 logger.error(f"Response text: {resp.text}")
            except NameError: # If resp doesn't exist
                 logger.error("No response object available.")
            except Exception as E:
                 logger.error(f"Could not log response text: {E}")

            st.error(f"Network error fetching video list for channel {channel_id}. Please try again later.")
            break # Stop fetching for this channel

        items = data.get("items", [])
        if not items and fetched_count == 0:
             logger.warning(f"No video snippets found for channel {channel_id} with current filters.")
             # st.info(f"No videos found for channel {channel_id} matching criteria.") # Maybe too noisy
             break # No videos found at all

        for it in items:
            vid_id = it.get("id", {}).get("videoId") # Safer access
            if not vid_id:
                logger.warning(f"Found item without videoId: {it}")
                continue

            snippet = it.get("snippet")
            if not snippet:
                 logger.warning(f"Found item without snippet: {it}")
                 continue

            # Check if publishedAt exists and is valid format
            published_at_raw = snippet.get("publishedAt")
            if not published_at_raw or not isinstance(published_at_raw, str):
                 logger.warning(f"Missing or invalid publishedAt for video {vid_id}: {published_at_raw}")
                 # Decide: skip video or use a placeholder? Skipping is safer.
                 continue

            all_videos.append({
                "video_id": vid_id,
                "title": snippet.get("title", "No Title"),
                "channel_name": snippet.get("channelTitle", "Unknown Channel"),
                "channelId": snippet.get("channelId"), # Store channelId from snippet
                "publish_date": format_date(published_at_raw), # Original formatted date
                "published_at": published_at_raw, # ISO format for precise calculations
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", "")
            })
            fetched_count += 1
            if fetched_count >= max_results_limit:
                break # Exit inner loop

        # Check if we need to continue pagination
        page_token = data.get("nextPageToken")
        if not page_token:
            break # No more pages

    logger.info(f"Fetched {len(all_videos)} snippets for channel {channel_id}.")
    return all_videos


def search_youtube(query, channel_ids, timeframe, content_filter, ttl=600):
    """
    Searches YouTube channels, fetches video stats, calculates metrics including the new outlier score.
    """
    query = query.strip()
    # Adjust TTL based on search parameters (long TTL for broad, recent channel scans)
    is_broad_scan = not query and timeframe == "3 months" and content_filter.lower() == "both"
    effective_ttl = 7776000 if is_broad_scan else ttl # 90 days TTL for broad scans

    # Generate cache key including all relevant parameters
    cache_key_params = [query, sorted(channel_ids), timeframe, content_filter]
    cache_key = build_cache_key(*cache_key_params)

    # Try fetching from SQLite cache first
    cached_data = get_cached_result(cache_key, ttl=effective_ttl)
    if cached_data is not None: # Check explicitly for None, as empty list is a valid cache result
        logger.info(f"Cache hit for key {cache_key[-8:]} (ttl={effective_ttl})")
        # Ensure cached data is in the expected DataFrame format or convert it
        if isinstance(cached_data, list):
             # Check if it looks like the old list-of-dicts format
             if cached_data and isinstance(cached_data[0], dict):
                  try:
                       # Try converting list of dicts to DataFrame
                       cached_df = pd.DataFrame(cached_data)
                       # Verify essential columns exist from the new calculate_metrics
                       required_cols = ['video_id', 'outlier_score', 'vph_ratio', 'effective_vph']
                       if all(col in cached_df.columns for col in required_cols):
                            logger.info("Loaded cached data (list format) and converted to DataFrame.")
                            # Apply content filter *after* loading from cache
                            if content_filter.lower() == "shorts":
                                cached_df = cached_df[cached_df.get("content_category") == "Short"]
                            elif content_filter.lower() == "videos":
                                cached_df = cached_df[cached_df.get("content_category") == "Video"]
                            return cached_df.to_dict("records") # Return as list of dicts for consistency downstream
                       else:
                            logger.warning("Cached list data missing required columns. Refetching.")
                            delete_cache_key(cache_key) # Invalidate cache
                  except Exception as e:
                       logger.warning(f"Error processing cached list data: {e}. Refetching.")
                       delete_cache_key(cache_key) # Invalidate cache

             elif not cached_data: # Empty list is valid cache
                  logger.info("Returning cached empty list.")
                  return []
             else:
                  logger.warning("Cached data is a list but not of dicts. Refetching.")
                  delete_cache_key(cache_key)

        elif isinstance(cached_data, dict) and '_is_dataframe' in cached_data:
             # Assuming cached data is a dict representing a DataFrame (e.g., to_dict('split'))
             try:
                  cached_df = pd.DataFrame(**cached_data['data']) # Reconstruct DataFrame
                  logger.info("Loaded cached data (dict format) and reconstructed DataFrame.")
                  # Apply content filter *after* loading from cache
                  if content_filter.lower() == "shorts":
                      cached_df = cached_df[cached_df.get("content_category") == "Short"]
                  elif content_filter.lower() == "videos":
                      cached_df = cached_df[cached_df.get("content_category") == "Video"]
                  return cached_df.to_dict("records") # Return as list of dicts
             except Exception as e:
                  logger.warning(f"Error reconstructing DataFrame from cached dict: {e}. Refetching.")
                  delete_cache_key(cache_key) # Invalidate cache
        else:
             logger.warning(f"Unknown cached data format ({type(cached_data)}). Refetching.")
             delete_cache_key(cache_key) # Invalidate bad cache entry

    # If cache miss or invalid cache, proceed with API fetching
    logger.info(f"Cache miss or invalid cache for key {cache_key[-8:]}. Fetching fresh data.")
    st.info("Fetching fresh data from YouTube API...") # Notify user

    order_param = "relevance" if query else "date" # Sort by relevance if keyword search, else by date
    all_snippets = []

    # Calculate publishedAfter date based on timeframe
    pub_after_iso = None
    if timeframe != "Lifetime":
        now_utc = datetime.now(timezone.utc)
        delta_map = {
            "Last 24 hours": timedelta(days=1),
            "Last 48 hours": timedelta(days=2),
            "Last 4 days": timedelta(days=4),
            "Last 7 days": timedelta(days=7),
            "Last 15 days": timedelta(days=15),
            "Last 28 days": timedelta(days=28),
            "3 months": timedelta(days=90), # Approx 3 months
        }
        delta = delta_map.get(timeframe)
        if delta:
            pub_after_dt = now_utc - delta
            pub_after_iso = pub_after_dt.strftime('%Y-%m-%dT%H:%M:%SZ') # ISO 8601 format

    # Fetch snippets for all selected channels
    with st.spinner(f"Fetching video list for {len(channel_ids)} channel(s)..."):
        all_snippets = []
        for cid in channel_ids:
            logger.info(f"Fetching snippets for channel: {cid}")
            channel_snippets = fetch_all_snippets(cid, order_param, timeframe, query, pub_after_iso)
            all_snippets.extend(channel_snippets)
            time.sleep(0.1) # Small delay between channel fetches


    if not all_snippets:
        logger.warning("No snippets found for any selected channel with the given criteria.")
        st.warning("No videos found matching your criteria.")
        set_cached_result(cache_key, []) # Cache the empty result
        return []

    # Deduplicate snippets based on video_id (in case a video appears in multiple searches/channels?)
    unique_snippets = {s['video_id']: s for s in all_snippets}.values()
    vid_ids = [s["video_id"] for s in unique_snippets]
    logger.info(f"Found {len(vid_ids)} unique video IDs to fetch details for.")

    # Fetch statistics and content details in chunks
    all_stats_details = {}
    max_ids_per_request = 50
    video_id_chunks = list(chunk_list(vid_ids, max_ids_per_request))

    with st.spinner(f"Fetching details for {len(vid_ids)} videos..."):
         try:
             key = get_youtube_api_key() # Get key once
             for i, chunk in enumerate(video_id_chunks):
                 logger.info(f"Fetching details chunk {i+1}/{len(video_id_chunks)} ({len(chunk)} IDs)")
                 ids_str = ','.join(chunk)
                 stats_url = (
                     "https://www.googleapis.com/youtube/v3/videos"
                     f"?part=statistics,contentDetails,snippet&id={ids_str}&key={key}" # Added snippet to get duration reliably
                 )
                 try:
                      resp = requests.get(stats_url, timeout=20)
                      if resp.status_code == 403:
                          logger.error(f"API Key Error (Forbidden) fetching details: {resp.text}")
                          st.error("YouTube API error (Forbidden/Quota Exceeded?) fetching video details.")
                          raise requests.exceptions.RequestException("API Forbidden")
                      elif resp.status_code == 400:
                           logger.error(f"API Bad Request fetching details: {resp.text}")
                           st.error(f"YouTube API error (Bad Request) fetching details: {resp.text}")
                           raise requests.exceptions.RequestException("API Bad Request")
                      resp.raise_for_status() # Raise other HTTP errors

                      details_data = resp.json()
                      for item in details_data.get("items", []):
                          vid = item["id"]
                          stats = item.get("statistics", {})
                          content = item.get("contentDetails", {})
                          snippet = item.get("snippet", {}) # Get snippet for reliable publish time

                          duration_str = content.get("duration", "PT0S")
                          duration_sec = parse_iso8601_duration(duration_str)

                          # Determine category (Short/Video) based on duration
                          # Use <= 60 seconds for Shorts for consistency with app.py
                          # Official definition might vary slightly (e.g., vertical format), but duration is a good proxy.
                          category = "Short" if duration_sec <= 60 else "Video"

                          # Get stats, default to 0 if missing
                          views = int(stats.get("viewCount", 0))
                          likes = int(stats.get("likeCount", 0))
                          comments = int(stats.get("commentCount", 0))

                          # Get publish time from snippet for accuracy
                          published_at_iso = snippet.get("publishedAt")

                          all_stats_details[vid] = {
                              "views": views,
                              "like_count": likes,
                              "comment_count": comments,
                              "duration_seconds": duration_sec,
                              "content_category": category,
                              "published_at": published_at_iso # Use reliable timestamp from details API
                          }
                 except requests.exceptions.RequestException as e:
                      logger.error(f"Details request failed for chunk {i+1}: {str(e)}")
                      st.error(f"Network error fetching video details (chunk {i+1}). Some data might be missing.")
                      # Continue to next chunk if possible, some data is better than none
                 time.sleep(0.1) # Small delay between detail fetches

         except Exception as e: # Catch error getting API key
              st.error(f"Cannot fetch video details: {e}")
              set_cached_result(cache_key, []) # Cache empty on critical failure
              return []


    # Combine snippet data with stats/details
    combined_results = []
    for snippet_data in unique_snippets:
        vid = snippet_data["video_id"]
        if vid in all_stats_details:
            stats_info = all_stats_details[vid]
            # Update published_at in snippet_data if the details API provided a more reliable one
            if stats_info.get("published_at"):
                 snippet_data["published_at"] = stats_info["published_at"]
                 # Also update the formatted date for consistency
                 snippet_data["publish_date"] = format_date(stats_info["published_at"])

            # Merge dictionaries, prioritizing details API data for stats/duration
            combined = {**snippet_data, **stats_info}
            combined_results.append(combined)
        else:
            logger.warning(f"Missing stats/details for video ID: {vid}. Skipping this video.")


    if not combined_results:
         logger.warning("No results after combining snippets and details.")
         st.warning("Could not fetch details for any found videos.")
         set_cached_result(cache_key, []) # Cache empty result
         return []

    # Create DataFrame and calculate metrics
    logger.info(f"Calculating metrics for {len(combined_results)} videos.")
    with st.spinner("Calculating performance metrics..."):
         df_results = pd.DataFrame(combined_results)
         # Ensure channelId is present from snippets
         if 'channelId' not in df_results.columns:
              logger.error("Channel ID missing in combined data. Cannot calculate channel averages.")
              st.error("Internal error: Channel ID missing. Cannot calculate metrics accurately.")
              # Attempt to fallback using channel_name? Risky. Best to return empty.
              set_cached_result(cache_key, [])
              return []

         # Drop rows where essential data might still be missing (e.g., failed publish date parse)
         df_results.dropna(subset=['published_at', 'channelId'], inplace=True)

         if df_results.empty:
              logger.warning("DataFrame empty after final cleaning before metrics calculation.")
              set_cached_result(cache_key, [])
              return []

         # Call the modified calculate_metrics function
         df_with_metrics, _ = calculate_metrics(df_results) # Second return value is unused

    if df_with_metrics.empty:
         logger.warning("Metrics calculation returned an empty DataFrame.")
         st.warning("Failed to calculate performance metrics for the videos.")
         set_cached_result(cache_key, []) # Cache empty result
         return []

    # Cache the results (as DataFrame representation) before filtering
    try:
        # Convert DataFrame to a serializable format (e.g., dict) for caching
        # Using 'split' format is often robust
        cacheable_data = {'_is_dataframe': True, 'data': df_with_metrics.to_dict('split')}
        set_cached_result(cache_key, cacheable_data)
        logger.info(f"Successfully cached results for key {cache_key[-8:]}")
    except Exception as e:
        logger.error(f"Failed to cache results: {e}")
        st.warning(f"Could not cache results due to: {e}") # Non-critical


    # Apply content filter *after* calculation and caching
    if content_filter.lower() == "shorts":
        final_df = df_with_metrics[df_with_metrics["content_category"] == "Short"]
    elif content_filter.lower() == "videos":
        final_df = df_with_metrics[df_with_metrics["content_category"] == "Video"]
    else: # "Both"
        final_df = df_with_metrics

    logger.info(f"Returning {len(final_df)} videos after content filtering ('{content_filter}').")
    # Return results as list of dictionaries for the UI
    return final_df.to_dict("records")


# =============================================================================
# 9. Comments & Analysis
# =============================================================================
def analyze_comments(comments):
    if not comments:
        return "No comments provided for analysis."

    if not OPENAI_API_KEY:
         st.error("OpenAI API key not configured. Cannot analyze comments.")
         return "Analysis failed: OpenAI key missing."

    # Create a representative string/hash from comments for caching
    # Use a subset of comments if too many, and hash the text content
    sample_size = min(len(comments), 100) # Analyze up to 100 comments
    comments_text = "\n".join([c.get("text", "") for c in comments[:sample_size]])
    hashed = hashlib.sha256(comments_text.encode("utf-8")).hexdigest()

    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}
    if hashed in st.session_state["analysis_cache"]:
        logger.info(f"Using cached comment analysis for hash {hashed[-8:]}")
        return st.session_state["analysis_cache"][hashed]

    logger.info(f"Generating new comment analysis for {len(comments)} comments using OpenAI...")

    # Prepare the prompt for GPT
    prompt_content = (
        f"Analyze the following YouTube comments (up to {sample_size} shown) and provide a concise summary in three sections:\n"
        "1.  **Positive Sentiment:** Briefly summarize the main positive feedback or appreciation (2-3 bullet points).\n"
        "2.  **Negative/Critical Sentiment:** Briefly summarize the main criticisms, complaints, or negative points (2-3 bullet points).\n"
        "3.  **Topic Suggestions/Questions:** List the top 3-5 recurring questions or suggested topics mentioned in the comments.\n\n"
        "Keep the summaries brief and neutral. If sentiment is overwhelmingly one-sided, state that.\n\n"
        "--- Comments ---\n"
        f"{comments_text}"
        "\n--- End Comments ---"
    )

    try:
        # Use updated client syntax if possible
         from openai import OpenAI
         client = OpenAI(api_key=OPENAI_API_KEY)
         response = client.chat.completions.create(
             model="gpt-3.5-turbo", # Sufficient for this task
             messages=[{"role": "user", "content": prompt_content}],
             max_tokens=400, # Allow slightly more space for the structured output
             temperature=0.6,
         )
         analysis_text = response.choices[0].message.content

    except ImportError:
         # Fallback to older openai<1.0 syntax
         response = openai.ChatCompletion.create(
             model="gpt-3.5-turbo",
             messages=[{"role": "user", "content": prompt_content}],
             max_tokens=400,
             temperature=0.6,
             api_key=OPENAI_API_KEY
         )
         analysis_text = response.choices[0].message['content'] # type: ignore

    # Cache the result
    st.session_state["analysis_cache"][hashed] = analysis_text
    logger.info(f"Stored new comment analysis in cache for hash {hashed[-8:]}")
    return analysis_text

    # Error Handling
    except openai.APIError as e:
        logger.error(f"OpenAI API error during comment analysis: {e}")
        st.error(f"OpenAI API error during comment analysis: {e}")
        return "Comment analysis failed due to API error."
    except Exception as e:
        logger.error(f"Unexpected error during comment analysis: {e}")
        st.error(f"Unexpected error during comment analysis: {e}")
        return "Comment analysis failed due to an unexpected error."

def get_video_comments(video_id, max_comments=100):
    """Fetches top or relevant comments for a video."""
    comments = []
    page_token = None
    try:
        key = get_youtube_api_key()
    except Exception as e:
        st.error(f"Cannot fetch comments: {e}")
        return []

    # Fetch by relevance first, as it often surfaces interesting comments
    # Alternative: 'time' for newest comments
    order = "relevance"
    logger.info(f"Fetching up to {max_comments} comments for video {video_id} (order: {order})")

    while len(comments) < max_comments:
        try:
            url = (
                "https://www.googleapis.com/youtube/v3/commentThreads"
                f"?part=snippet&videoId={video_id}&maxResults={min(50, max_comments - len(comments))}" # Fetch up to 50 or remaining needed
                f"&order={order}&textFormat=plainText&key={key}" # Use plain text
            )
            if page_token:
                url += f"&pageToken={page_token}"

            resp = requests.get(url, timeout=15)

            if resp.status_code == 403:
                 # Handle comments disabled specifically
                 try:
                      error_details = resp.json()
                      if any(err.get('reason') == 'commentsDisabled' for err in error_details.get('error',{}).get('errors',[])):
                           logger.warning(f"Comments are disabled for video {video_id}.")
                           st.info("Comments appear to be disabled for this video.")
                           return [] # Return empty list, not an error
                 except Exception:
                      pass # Ignore parsing errors, fall through to general error
                 logger.error(f"API Key Error (Forbidden) fetching comments: {resp.text}")
                 st.error("YouTube API error (Forbidden/Quota?) fetching comments.")
                 break # Stop trying
            resp.raise_for_status() # Raise other HTTP errors

            data = resp.json()

            items = data.get("items", [])
            if not items and len(comments) == 0:
                 logger.info(f"No comments found for video {video_id}.")
                 # Check if zero comments is the actual state
                 try:
                      stats_url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={video_id}&key={key}"
                      stats_resp = requests.get(stats_url).json()
                      comment_count = int(stats_resp.get('items', [{}])[0].get('statistics', {}).get('commentCount', -1))
                      if comment_count == 0:
                           logger.info(f"Confirmed zero comments for video {video_id} via stats API.")
                           return [] # No comments exist
                 except Exception as e:
                      logger.warning(f"Could not verify comment count via stats API: {e}")
                 # If count check fails or > 0, maybe API issue?
                 st.warning("Could not retrieve comments (API issue or none exist).")
                 break


            for item in items:
                try:
                    # Get top-level comment snippet
                    top_comment = item.get("snippet", {}).get("topLevelComment", {})
                    snippet = top_comment.get("snippet")
                    if not snippet:
                        logger.warning(f"Comment item missing snippet: {item}")
                        continue

                    text = snippet.get("textDisplay", "") # Use textDisplay which might have basic formatting
                    # Clean potential HTML? textFormat=plainText should prevent this, but sanitize anyway
                    text_cleaned = re.sub(r'<.*?>', '', text) # Basic HTML tag removal

                    like_count = int(snippet.get("likeCount", 0))
                    author = snippet.get("authorDisplayName", "Unknown Author")
                    published_at = snippet.get("publishedAt") # ISO timestamp

                    comments.append({
                        "text": text_cleaned,
                        "likeCount": like_count,
                        "author": author,
                        "published_at": published_at,
                        "comment_id": top_comment.get("id") # Include comment ID
                        })
                except Exception as e:
                     logger.error(f"Error processing comment item: {item} - Error: {e}")
                     continue # Skip malformed comment

            # Check for next page
            page_token = data.get("nextPageToken")
            if not page_token:
                break # No more pages

        except requests.exceptions.Timeout:
             logger.error(f"Timeout fetching comments for {video_id}.")
             st.error("Timeout fetching comments. Please try again.")
             break
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching comments for {video_id}: {e}")
            # Log response if possible
            try: logger.error(f"Response: {resp.text}")
            except Exception: pass
            st.error(f"Error retrieving comments: {e}")
            break # Stop trying

    logger.info(f"Fetched {len(comments)} comments successfully for {video_id}.")
    return comments


# =============================================================================
# 10. Retention Analysis (Keep as is from app(1).py)
# =============================================================================
# NOTE: Selenium-based retention requires a specific environment setup (browser, driver)
# which might not be available in all deployment scenarios (like basic Streamlit Cloud).
# Consider adding checks or warnings about environment compatibility.

# Check for necessary executables early
CHROMIUM_PATH = "/usr/bin/chromium" # Path for Linux environments like Streamlit Cloud
# Check if running locally vs deployed for binary path?
# if os.path.exists(CHROMIUM_PATH):
#      logger.info(f"Using Chromium binary at: {CHROMIUM_PATH}")
# else:
#      logger.warning(f"Chromium binary not found at {CHROMIUM_PATH}. Retention analysis might fail.")
#      # Consider trying default webdriver-manager path if local?

def check_selenium_setup():
     """Checks if required components for Selenium are likely present."""
     # 1. Check for Chrome/Chromium (simple check, might need refinement)
     browser_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium-browser") or shutil.which("chromium")
     if not browser_path and not os.path.exists(CHROMIUM_PATH):
          st.warning("⚠️ Chrome/Chromium browser not found. Retention analysis requires it.")
          logger.warning("Chrome/Chromium executable not found.")
          return False
     if not browser_path: browser_path = CHROMIUM_PATH # Assume Linux path if others fail

     # 2. Check if webdriver-manager can find/install a driver (doesn't guarantee it runs)
     try:
          # Explicitly specify a version if needed, otherwise let it manage
          # driver_path = ChromeDriverManager().install()
          # Forcing version based on original code:
          driver_version = "120.0.6099.224" # Hardcoded in original, might need update
          logger.info(f"Checking/Installing ChromeDriver version: {driver_version}")
          # Suppress webdriver-manager logs if possible? It's noisy.
          os.environ['WDM_LOG_LEVEL'] = '0' # Try to silence WDM logs
          driver_path = ChromeDriverManager(driver_version=driver_version).install()
          if not driver_path or not os.path.exists(driver_path):
               st.warning("⚠️ ChromeDriver could not be installed by webdriver-manager.")
               logger.warning("webdriver-manager failed to install driver.")
               return False
          logger.info(f"ChromeDriver seems available at: {driver_path}")
     except Exception as e:
          st.warning(f"⚠️ Error setting up ChromeDriver: {e}. Retention analysis might fail.")
          logger.error(f"Error installing/checking ChromeDriver via webdriver-manager: {e}")
          return False

     # 3. Check ffmpeg (already done elsewhere, but good to have here too)
     if not shutil.which("ffmpeg"):
          st.warning("⚠️ ffmpeg not found in PATH. Required for some retention features (like snippets).")
          # No need to return False, as basic analysis might still work
     # 4. Check yt-dlp
     if not check_ytdlp_installed():
          st.warning("⚠️ yt-dlp not found. Required for downloading video snippets in retention analysis.")

     logger.info("Selenium setup appears potentially viable.")
     return True # Setup seems okay


def load_cookies(driver, cookie_file="youtube_cookies.json"):
    if not os.path.exists(cookie_file):
        logger.warning(f"Cookie file '{cookie_file}' not found. Proceeding without cookies.")
        st.info("YouTube cookie file not found. Retention graph might be less accurate or unavailable if login is required.")
        return False # Indicate cookies were not loaded

    logger.info(f"Loading cookies from {cookie_file}")
    try:
        with open(cookie_file, "r", encoding="utf-8") as f:
            cookies = json.load(f)
    except Exception as e:
        logger.error(f"Error reading cookie file {cookie_file}: {e}")
        st.error(f"Error reading cookie file: {e}")
        return False

    # Navigate to the domain *before* adding cookies
    driver.get("https://www.youtube.com/")
    time.sleep(2) # Allow page to start loading

    added_count = 0
    skipped_count = 0
    for cookie in cookies:
        # Clean cookie attributes that cause issues with Selenium
        # Handle 'sameSite' mapping if needed (Strict, Lax, None)
        if 'sameSite' in cookie:
             if cookie['sameSite'] not in ['Strict', 'Lax', 'None']:
                  # If invalid value, remove it or default? Removing is safer.
                  # logger.debug(f"Removing invalid sameSite value: {cookie['sameSite']}")
                  del cookie['sameSite']

        # Rename 'expiry' to 'expires' if present (common in exported cookies)
        if "expiry" in cookie:
            cookie["expires"] = cookie.pop("expiry")

        # Remove domain if it's not specific enough (e.g., '.youtube.com') - optional refinement
        # if 'domain' in cookie and cookie['domain'].startswith('.'):
             # try adding without domain? or keep it? Keep for now.

        try:
            driver.add_cookie(cookie)
            added_count += 1
        except Exception as e:
            skipped_count += 1
            # Log only if excessive errors occur?
            if skipped_count < 5 or skipped_count % 50 == 0:
                 logger.warning(f"Skipping cookie, possibly invalid: {cookie.get('name', 'N/A')}. Error: {e}")

    logger.info(f"Added {added_count} cookies, skipped {skipped_count}.")
    if added_count > 0:
         # Refresh page *after* adding cookies
         logger.info("Refreshing page after loading cookies.")
         driver.refresh()
         time.sleep(3) # Allow refresh to complete
         return True
    else:
         logger.warning("No valid cookies were added.")
         return False


def capture_player_screenshot_with_hover(video_url, timestamp, output_path="player_retention.png", use_cookies=True):
    # Check setup before starting browser
    if not check_selenium_setup():
         raise EnvironmentError("Selenium/Browser setup is not valid. Cannot capture retention.")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080") # Standard desktop size
    options.add_argument("--no-sandbox") # Required for running as root/in containers
    options.add_argument("--disable-dev-shm-usage") # Overcome limited resource problems
    options.add_argument("--disable-gpu") # Sometimes necessary in headless
    options.add_argument("--lang=en-US,en") # Set language to English
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36") # Common user agent

    # Specify Chromium path if needed (e.g., Streamlit Cloud)
    if os.path.exists(CHROMIUM_PATH):
        options.binary_location = CHROMIUM_PATH
        logger.info(f"Using Chromium binary location: {CHROMIUM_PATH}")

    driver = None # Initialize driver to None for finally block
    try:
        # Use webdriver-manager to get the driver path
        # driver_version = "120.0.6099.224" # Version from original code - may need updates!
        # Use specific version or let manager decide? Let manager decide for robustness unless specific version is critical.
        logger.info("Initializing Chrome driver...")
        service = Service(ChromeDriverManager().install()) # Let manager find appropriate version
        # service = Service(ChromeDriverManager(driver_version=driver_version).install()) # Force version

        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(45) # Timeout for page loads
        logger.info("Driver initialized.")

        # Load cookies if requested
        cookies_loaded = False
        if use_cookies:
             try:
                  cookies_loaded = load_cookies(driver, "youtube_cookies.json")
             except Exception as e:
                  logger.error(f"Error loading cookies: {e}")
                  st.warning(f"Error occurred while loading cookies: {e}. Proceeding without.")

        # Navigate to the video URL
        logger.info(f"Navigating to video URL: {video_url}")
        driver.get(video_url)
        logger.info("Waiting for page elements (up to 15s)...")
        # Wait for video player element to be present
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        try:
             WebDriverWait(driver, 15).until(
                  EC.presence_of_element_located((By.TAG_NAME, "video"))
             )
             logger.info("Video element found.")
             # Add extra sleep to allow player UI to potentially load fully
             time.sleep(5)
        except TimeoutException:
             logger.error("Timeout waiting for video element on page.")
             st.error("Failed to load video player interface in time.")
             # Capture screenshot anyway for debugging?
             try: driver.save_screenshot("debug_timeout_screenshot.png")
             except: pass
             raise TimeoutError("Video player did not load.")

        # Try to dismiss consent forms / popups if they exist
        consent_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Accept') or contains(., 'Agree') or contains(@aria-label, 'Accept')]")
        if consent_buttons:
            try:
                consent_buttons[0].click()
                logger.info("Clicked a potential consent button.")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Could not click consent button: {e}")


        # --- Interaction Logic ---
        video_element_selector = "video.html5-main-video" # More specific selector
        try:
             # 1. Get initial duration (best effort)
             initial_duration = driver.execute_script(f"return document.querySelector('{video_element_selector}')?.duration || 0;")
             logger.info(f"Initial reported duration (JS): {initial_duration:.2f} seconds")
             if initial_duration <= 0:
                  logger.warning("Could not get valid video duration initially.")
                  # Try again after a delay?
                  time.sleep(3)
                  initial_duration = driver.execute_script(f"return document.querySelector('{video_element_selector}')?.duration || 0;")
                  logger.info(f"Second attempt duration (JS): {initial_duration:.2f} seconds")


             # 2. Pause the video if it's playing
             driver.execute_script(f"document.querySelector('{video_element_selector}')?.pause();")
             time.sleep(0.5)
             logger.info("Paused video.")

             # 3. Seek to the target timestamp
             # Use a timestamp near the end, but not exactly the end, to ensure graph is visible
             # If initial_duration is unreliable, use the provided 'timestamp' cautiously
             seek_target = initial_duration - 5 if initial_duration > 10 else initial_duration * 0.9
             if seek_target <= 0: # Fallback if duration is still unknown
                  seek_target = timestamp if timestamp > 0 else 30 # Use input or default 30s
             logger.info(f"Seeking video to: {seek_target:.2f} seconds")
             driver.execute_script(f"document.querySelector('{video_element_selector}').currentTime = {seek_target};")
             # Wait for seek to potentially complete and UI to update
             time.sleep(3)

             # 4. Find the player and progress bar elements
             # Use robust selectors that are less likely to change
             player_element = WebDriverWait(driver, 10).until(
                 EC.presence_of_element_located((By.CSS_SELECTOR, "#movie_player")) # Common container ID
             )
             # Find progress bar padding or container - might need adjustment based on YT UI changes
             # XPATH from original: "//div[@class='ytp-progress-bar-padding']" - check if still valid
             # Alternative CSS: ".ytp-progress-bar-container" or similar
             progress_bar_element_xpath = "//div[contains(@class, 'ytp-progress-bar-container')]" # More robust class name part
             progress_bar = WebDriverWait(player_element, 10).until(
                 EC.presence_of_element_located((By.XPATH, progress_bar_element_xpath))
             )
             logger.info("Player and progress bar elements located.")

             # 5. Hover over the progress bar to reveal the retention graph
             logger.info("Hovering over progress bar...")
             ActionChains(driver).move_to_element(progress_bar).perform()
             # Wait for the graph animation/popup to appear
             time.sleep(2.5) # Increased wait time for graph stability

             # 6. Capture screenshot of the player element
             logger.info(f"Capturing player screenshot to: {output_path}")
             player_element.screenshot(output_path)
             logger.info("Screenshot captured successfully.")

             # 7. Get duration again (might be more reliable now)
             final_duration = driver.execute_script(f"return document.querySelector('{video_element_selector}')?.duration || 0;")
             logger.info(f"Final reported duration (JS): {final_duration:.2f} seconds")
             effective_duration = final_duration if final_duration > 0 else initial_duration
             if effective_duration <= 0:
                  logger.warning("Could not determine video duration.")
                  # Fallback: return the input timestamp? Or raise error?
                  # Let's return 0 and handle it in the calling function
                  return 0.0


        except (NoSuchElementException, TimeoutException) as e:
             logger.error(f"Selenium interaction failed: Element not found or timed out. Error: {e}")
             st.error(f"Error interacting with YouTube player interface: {e}. Retention graph might be missing.")
             # Capture full page screenshot for debugging
             try: driver.save_screenshot("debug_interaction_error.png")
             except: pass
             raise RuntimeError("Failed to interact with YouTube player for retention.") from e
        except Exception as e:
             logger.error(f"Unexpected error during Selenium interaction: {e}")
             st.error(f"Unexpected error during retention analysis: {e}")
             try: driver.save_screenshot("debug_unexpected_error.png")
             except: pass
             raise # Re-raise the exception


    finally:
        # Ensure the driver is quit even if errors occur
        if driver:
            logger.info("Quitting Selenium driver.")
            driver.quit()

    # Return the most reliable duration found
    return effective_duration if effective_duration > 0 else 0.0


def detect_retention_peaks(image_path, crop_ratio=0.15, height_threshold=150, distance=30, top_n=5):
    """Analyzes the retention graph screenshot to find peaks."""
    if not os.path.exists(image_path):
        logger.error(f"Retention screenshot file not found at: {image_path}")
        raise FileNotFoundError(f"File {image_path} not found.")

    logger.info(f"Analyzing retention peaks in: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image from {image_path} using OpenCV.")
        raise ValueError(f"Failed to read image from {image_path}.")

    height, width, _ = img.shape
    if height == 0 or width == 0:
         logger.error(f"Invalid image dimensions: {width}x{height}")
         raise ValueError("Image has invalid dimensions (zero width or height).")

    # --- Define Region of Interest (ROI) ---
    # Crop from the bottom, but slightly less aggressively than 20%? Try 15%.
    # The graph is usually near the bottom but above the absolute edge.
    roi_start_y = int(height * (1 - crop_ratio))
    roi_end_y = height
    roi = img[roi_start_y:roi_end_y, 0:width]
    logger.info(f"ROI defined: Y from {roi_start_y} to {roi_end_y} (Height: {roi_end_y - roi_start_y}), Width: {width}")

    if roi.shape[0] == 0 or roi.shape[1] == 0:
         logger.error(f"ROI has invalid dimensions after cropping: {roi.shape}. Original: {img.shape}, Crop Ratio: {crop_ratio}")
         st.warning("Could not define a valid region for analyzing the retention graph. Check screenshot.")
         # Return empty results gracefully
         return np.array([]), None, None, width, np.array([])


    # --- Image Processing ---
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding might be better than fixed if lighting varies?
    # _, binary_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY) # Original fixed threshold
    # Try adaptive thresholding:
    binary_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2) # Block size 11, C=2
    # The retention graph is usually white/light grey on dark background. We want the graph line.
    # Let's invert if adaptive threshold makes the graph black
    # Check average pixel value, if mostly white (high value), it might be inverted
    if np.mean(binary_roi) > 128:
         binary_roi = cv2.bitwise_not(binary_roi) # Invert to make graph white
         logger.info("Inverted binary ROI as it seemed mostly white.")


    # --- Peak Detection ---
    # Sum pixel values vertically to get a 1D profile of the graph's brightness/presence
    col_sums = np.sum(binary_roi, axis=0)

    # Adjust peak finding parameters:
    # `height_threshold`: Minimum sum of white pixels to be considered part of the graph.
    # Depends on ROI height and graph thickness. If ROI height is ~162 (1080*0.15), a thin line might be < 100?
    # Let's lower it slightly from 200.
    # `distance`: Minimum horizontal distance (pixels) between peaks. Prevents finding multiple peaks on one bump.
    # 30 pixels seems reasonable on a 1920 width image.

    effective_height_threshold = height_threshold # Use the parameter passed in
    logger.info(f"Finding peaks with height >= {effective_height_threshold} and distance >= {distance}")

    peaks, properties = find_peaks(col_sums, height=effective_height_threshold, distance=distance)

    if peaks.size == 0:
        logger.warning(f"No peaks found with current settings (height={effective_height_threshold}, distance={distance}). Trying lower threshold.")
        # Retry with a lower threshold if no peaks found initially
        effective_height_threshold = max(50, height_threshold // 2) # Lower threshold, min 50
        peaks, properties = find_peaks(col_sums, height=effective_height_threshold, distance=distance)
        if peaks.size > 0:
             logger.info(f"Found {len(peaks)} peaks with reduced threshold {effective_height_threshold}.")
        else:
             logger.warning("Still no peaks found even with reduced threshold.")
             st.warning("Could not detect significant retention peaks in the graph image.")


    # Select top N peaks based on their height (brightness sum) if more than N are found
    if len(peaks) > top_n:
        logger.info(f"Found {len(peaks)} peaks, selecting top {top_n} based on height.")
        peak_heights = properties["peak_heights"]
        top_indices = np.argsort(peak_heights)[-top_n:] # Indices of the top N peaks
        top_peaks = np.sort(peaks[top_indices]) # Get the peak locations and sort them by time (x-axis)
    else:
        top_peaks = peaks # Already sorted by find_peaks

    logger.info(f"Detected {len(top_peaks)} final peaks at x-coordinates: {top_peaks}")

    # Return peaks, ROI images (optional), width for scaling, and the brightness profile
    return top_peaks, roi, binary_roi, width, col_sums


def capture_frame_at_time(video_url, target_time, output_path="frame_retention.png", use_cookies=True):
    """Captures a single frame screenshot at a specific video timestamp using Selenium."""
     # Check setup before starting browser
    if not check_selenium_setup():
         raise EnvironmentError("Selenium/Browser setup is not valid. Cannot capture frame.")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en-US,en")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    if os.path.exists(CHROMIUM_PATH):
        options.binary_location = CHROMIUM_PATH
        logger.info(f"Using Chromium binary location: {CHROMIUM_PATH}")

    driver = None
    try:
        logger.info("Initializing Chrome driver for frame capture...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(45)
        logger.info("Driver initialized.")

        cookies_loaded = False
        if use_cookies:
             try:
                  cookies_loaded = load_cookies(driver, "youtube_cookies.json")
             except Exception as e:
                  logger.error(f"Error loading cookies: {e}")


        logger.info(f"Navigating to video URL: {video_url}")
        driver.get(video_url)
        logger.info("Waiting for video element (up to 15s)...")
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        try:
             WebDriverWait(driver, 15).until(
                  EC.presence_of_element_located((By.TAG_NAME, "video"))
             )
             logger.info("Video element found.")
             time.sleep(5) # Allow player UI to potentially load fully
        except TimeoutException:
             logger.error("Timeout waiting for video element on page.")
             st.error("Failed to load video player interface in time for frame capture.")
             try: driver.save_screenshot("debug_frame_timeout.png")
             except: pass
             raise TimeoutError("Video player did not load for frame capture.")

        # Try to dismiss consent forms
        consent_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Accept') or contains(., 'Agree') or contains(@aria-label, 'Accept')]")
        if consent_buttons:
            try:
                consent_buttons[0].click()
                logger.info("Clicked a potential consent button.")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Could not click consent button: {e}")


        # --- Frame Capture Logic ---
        video_element_selector = "video.html5-main-video"
        try:
             # 1. Pause the video first
             driver.execute_script(f"document.querySelector('{video_element_selector}')?.pause();")
             time.sleep(0.5)

             # 2. Seek to the target time
             logger.info(f"Seeking video to target time: {target_time:.2f} seconds")
             driver.execute_script(f"document.querySelector('{video_element_selector}').currentTime = {target_time};")
             # Wait for seek to complete and frame to potentially update
             # This might require a slight play/pause cycle if seeking alone doesn't refresh frame accurately in headless
             time.sleep(2)

             # Optional: Short play/pause cycle to force frame refresh
             # logger.info("Performing short play/pause cycle to ensure frame update...")
             # driver.execute_script(f"document.querySelector('{video_element_selector}')?.play();")
             # time.sleep(0.2) # Play for a very short duration
             # driver.execute_script(f"document.querySelector('{video_element_selector}')?.pause();")
             # time.sleep(0.5) # Wait for pause to take effect

             # 3. Find the player element
             player_element = WebDriverWait(driver, 10).until(
                 EC.presence_of_element_located((By.CSS_SELECTOR, "#movie_player"))
             )

             # 4. Capture the screenshot
             logger.info(f"Capturing frame screenshot to: {output_path}")
             player_element.screenshot(output_path)
             logger.info("Frame screenshot captured successfully.")

             # Verify file exists and is not empty
             if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                  logger.error(f"Screenshot file was not created or is empty: {output_path}")
                  raise RuntimeError("Screenshot failed - file empty or not created.")


        except (NoSuchElementException, TimeoutException) as e:
             logger.error(f"Selenium frame capture failed: Element not found or timed out. Error: {e}")
             st.error(f"Error interacting with YouTube player for frame capture: {e}.")
             try: driver.save_screenshot("debug_frame_interaction_error.png")
             except: pass
             raise RuntimeError("Failed to interact with YouTube player for frame capture.") from e
        except Exception as e:
             logger.error(f"Unexpected error during Selenium frame capture: {e}")
             st.error(f"Unexpected error during frame capture: {e}")
             try: driver.save_screenshot("debug_frame_unexpected_error.png")
             except: pass
             raise

    finally:
        if driver:
            logger.info("Quitting Selenium driver (frame capture).")
            driver.quit()

    return output_path # Return the path where the frame was saved


def plot_brightness_profile(col_sums, peaks):
    """Generates a plot of the column brightness sums and detected peaks."""
    if col_sums is None or col_sums.size == 0:
         logger.warning("Cannot plot brightness profile: column sums data is empty.")
         # Return None or an empty buffer? Returning None is clearer.
         return None

    buf = BytesIO()
    try:
        fig, ax = plt.subplots(figsize=(10, 3)) # Wider figure
        ax.plot(col_sums, label="Retention Graph Profile", color="#4285f4")
        # Plot peaks if they exist
        if peaks is not None and peaks.size > 0:
            # Ensure peaks are within bounds of col_sums
            valid_peaks = peaks[peaks < len(col_sums)]
            if valid_peaks.size > 0:
                 ax.plot(valid_peaks, col_sums[valid_peaks], "x", label="Detected Peaks", markersize=8, color="#ea4335", markeredgewidth=2)

        ax.set_xlabel("Horizontal Position (Pixels in ROI)")
        ax.set_ylabel("Summed Brightness")
        ax.set_title("Detected Retention Profile")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100) # Save with decent resolution
        plt.close(fig) # Close the plot to free memory
        buf.seek(0)
        logger.info("Successfully generated brightness profile plot.")
        return buf
    except Exception as e:
        logger.error(f"Error generating brightness profile plot: {e}")
        plt.close(fig) # Ensure figure is closed even on error
        return None # Return None on error


def filter_transcript(transcript, target_time, window=5):
    """Extracts transcript text around a specific target time."""
    if not transcript:
        return ""

    snippet_texts = []

    # Handle different transcript formats
    if isinstance(transcript, list) and transcript and 'start' in transcript[0] and 'text' in transcript[0]:
         # Likely YouTube format list of dicts
         for seg in transcript:
             try:
                 start = float(seg.get("start", 0))
                 duration = float(seg.get("duration", 2.0)) # Estimate duration if missing
                 end = start + duration
                 # Check if the segment [start, end] overlaps with [target-window, target+window]
                 if max(start, target_time - window) < min(end, target_time + window):
                      snippet_texts.append(seg.get("text", "").strip())
             except (ValueError, TypeError):
                 # Ignore segments with invalid time data
                 continue
    elif isinstance(transcript, list) and transcript and 'text' in transcript[0] and 'start' not in transcript[0]:
         # Likely Whisper format with potentially one segment
         full_text = " ".join(seg.get("text","") for seg in transcript)
         # Cannot accurately filter by time without timestamps
         # Return a generic message or try a very rough split?
         logger.warning("Cannot filter transcript by time: Timestamps missing (likely Whisper output).")
         return "(Transcript snippet unavailable due to missing timestamps)"

    elif isinstance(transcript, str): # If transcript is just a single string
         # Cannot filter by time
         logger.warning("Cannot filter transcript by time: Transcript is a single string.")
         return "(Transcript snippet unavailable: Format lacks timestamps)"

    else:
         logger.warning(f"Unrecognized transcript format for filtering: {type(transcript)}")
         return "(Transcript snippet unavailable: Unrecognized format)"


    # Join the collected snippet texts
    full_snippet = " ".join(snippet_texts).strip()

    # Limit length if excessively long?
    max_len = 500
    if len(full_snippet) > max_len:
         return full_snippet[:max_len] + "..."
    elif not full_snippet:
         return "(No transcript text found near this time)"
    else:
         return full_snippet


def check_ytdlp_installed():
    """Checks if yt-dlp command is available in the system PATH."""
    try:
        # Run yt-dlp --version, capture output, check return code
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, check=True, timeout=5)
        logger.info(f"yt-dlp found, version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.warning("yt-dlp command not found in PATH.")
        return False
    except subprocess.TimeoutExpired:
         logger.warning("yt-dlp --version command timed out.")
         return False # Assume not working if it hangs
    except subprocess.CalledProcessError as e:
        logger.warning(f"yt-dlp --version command failed: {e}")
        return False
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error checking yt-dlp: {e}")
        return False


def download_video_snippet(video_url, start_time, duration=10, output_path="snippet.mp4"):
    """Downloads a short video snippet using yt-dlp and ffmpeg."""
    if not check_ytdlp_installed():
        st.error("yt-dlp is required to download video snippets.")
        raise EnvironmentError("yt-dlp not found.")
    if not shutil.which("ffmpeg"):
         st.error("ffmpeg is required to extract video snippets.")
         raise EnvironmentError("ffmpeg not found.")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory for snippet: {output_dir}")

    # Format timespec for yt-dlp download archive
    # Format: HH:MM:SS.ms or seconds
    start_str = str(timedelta(seconds=start_time))
    end_str = str(timedelta(seconds=start_time + duration))
    timespec = f"{start_str}-{end_str}"

    # Use yt-dlp's --download-sections feature combined with ffmpeg remuxing
    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]', # Download decent quality video and audio separately
        'outtmpl': output_path.replace('.mp4', '.%(ext)s'), # Temporary output template
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
        'socket_timeout': 60, # Longer timeout for video download
        # Specify sections to download *using ffmpeg*: This tells yt-dlp to use ffmpeg for extraction
        'download_ranges': lambda info_dict, ydl: ydl.download_range_func(info_dict, [(start_time, start_time + duration)]),
        # Force keyframes? Might improve seeking accuracy but slower. Generally not needed with download_ranges.
        # 'force_keyframes_at_cuts': True,
        # Remux into MP4 container after downloading
        'postprocessors': [{
             'key': 'FFmpegVideoConvertor',
             'preferedformat': 'mp4', # Ensure output is mp4
        }],
        # Important: Keep fragments after download for FFmpeg processing
        'keep_fragments': True, # May be needed depending on format
    }


    try:
        import yt_dlp
        logger.info(f"Attempting to download snippet for {video_url} ({timespec}) using yt-dlp sections...")
        st_info_placeholder = st.info(f"Downloading snippet ({duration}s starting at {start_time:.1f}s)...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Check if the final output file exists
        # yt-dlp should rename the final remuxed file to output_path
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st_info_placeholder.success(f"Snippet downloaded successfully: {output_path}")
            logger.info(f"Snippet downloaded and remuxed successfully to {output_path}")
            return output_path
        else:
            # Check if intermediate files exist but final didn't get created
            base_output = output_path.replace('.mp4', '')
            found_files = [f for f in os.listdir(output_dir or '.') if f.startswith(os.path.basename(base_output))]
            logger.error(f"yt-dlp finished but expected output file '{output_path}' not found or empty. Intermediate files: {found_files}")
            st.error("Snippet download failed: Final file not created.")
            # Clean up intermediate files if possible
            for f in found_files:
                 try: os.remove(os.path.join(output_dir or '.', f))
                 except OSError: pass
            raise RuntimeError("yt-dlp failed to create the final snippet file.")


    except yt_dlp.utils.DownloadError as e:
         logger.error(f"yt-dlp DownloadError during snippet download: {e}")
         st.error(f"Error downloading snippet: Video might be unavailable, region-locked, or section download failed.")
         raise RuntimeError(f"yt-dlp download error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during snippet download for {video_url}: {e}")
        st.error(f"An unexpected error occurred during snippet download: {e}")
        # Attempt cleanup of partial files
        try:
             if os.path.exists(output_path): os.remove(output_path)
             base_output = output_path.replace('.mp4', '')
             found_files = [f for f in os.listdir(output_dir or '.') if f.startswith(os.path.basename(base_output))]
             for f in found_files:
                  try: os.remove(os.path.join(output_dir or '.', f))
                  except OSError: pass
        except Exception as cleanup_e:
             logger.error(f"Error during snippet cleanup: {cleanup_e}")
        raise # Re-raise the original exception


# =============================================================================
# 14. UI Pages (Keep structure, integrate results from new calculate_metrics)
# =============================================================================
def show_search_page():
    st.title("Youtube Niche Search")

    # Sidebar: Filters and Controls
    with st.sidebar:
        st.header("Search Filters")

        # Folder Selection
        folders = load_channel_folders()
        available_folders = list(folders.keys())
        if not available_folders:
             st.warning("No channel folders created yet.")
             st.info("Use the 'Channel Folder Manager' below to add channels.")
             folder_choice = None
        else:
             folder_choice = st.selectbox("Select Channel Folder", available_folders, index=0, key="folder_choice_sb")

        # Timeframe
        timeframe_options = ["Last 24 hours", "Last 48 hours", "Last 4 days", "Last 7 days", "Last 15 days", "Last 28 days", "3 months", "Lifetime"]
        selected_timeframe = st.selectbox("Timeframe", timeframe_options, index=timeframe_options.index("3 months"), key="timeframe_sb") # Default 3 months

        # Content Type Filter
        content_filter_options = ["Both", "Videos", "Shorts"]
        content_filter = st.selectbox("Filter Content Type", content_filter_options, index=0, key="content_filter_sb")

        # Minimum Outlier Score Filter
        # Scale might change with new formula, default to 0 but explain it's relative
        min_outlier_score = st.number_input("Minimum Outlier Score (1.0 = Avg)", value=0.0, min_value=0.0, step=0.1, format="%.2f", key="min_outlier_sb",
                                            help="Filters results by minimum performance relative to channel average (1.0 = average, >1.0 = above average).")

        # Keyword Search
        search_query = st.text_input("Keyword Search (optional)", key="query_sb")

        # Search Button
        search_button_pressed = st.button("Search", key="search_button_sb", type="primary")

        st.divider()

        # Cache Clearing
        if st.button("Clear Cache & Reload Data"):
             try:
                  with sqlite3.connect(DB_PATH) as conn:
                      conn.execute("DELETE FROM youtube_cache")
                  st.session_state.clear() # Clear session state as well
                  st.success("Cache cleared. Search again to fetch fresh data.")
                  logger.info("SQLite cache cleared by user.")
             except Exception as e:
                  st.error(f"Failed to clear cache: {e}")
                  logger.error(f"Failed to clear cache: {e}")

        st.divider()
        # Channel Folder Manager
        with st.expander("Channel Folder Manager"):
             show_channel_folder_manager()


    # Main Page Content Area
    if folder_choice:
        st.write(f"**Selected Folder**: {folder_choice}")
        selected_channel_ids = [ch["channel_id"] for ch in folders[folder_choice]]
        if not selected_channel_ids:
             st.warning(f"Folder '{folder_choice}' contains no channels.")
        else:
             # Optionally show channels in the selected folder
             with st.expander("Channels in this folder", expanded=False):
                  if folders[folder_choice]:
                       # Display as a neat list or table
                       ch_names = [ch['channel_name'] for ch in folders[folder_choice]]
                       if len(ch_names) < 15:
                            st.write(", ".join(ch_names))
                       else:
                            st.dataframe({'Channel Name': ch_names}, use_container_width=True, height=150)
                  else:
                       st.write("(No channels)")
    else:
        st.info("Select a channel folder from the sidebar to begin.")
        selected_channel_ids = []


    # Execute Search when button is pressed and folder/channels are selected
    if search_button_pressed:
        if not folder_choice or not selected_channel_ids:
            st.error("Please select a folder containing channels before searching.")
        else:
            # Store search parameters in session state to persist results
            st.session_state.search_params = {
                "query": search_query,
                "channel_ids": selected_channel_ids,
                "timeframe": selected_timeframe,
                "content_filter": content_filter,
                "min_outlier_score": min_outlier_score,
                "folder_choice": folder_choice # Remember which folder was searched
            }
            # Clear previous results before fetching new ones
            if 'search_results' in st.session_state:
                 del st.session_state['search_results']
            st.session_state.page = "search" # Ensure we stay on search page
            # Trigger rerun to fetch data (fetch happens below based on session state)
            st.rerun()


    # Display Results if they exist in session state
    # Check if search was triggered and results should be fetched/displayed
    if 'search_params' in st.session_state and st.session_state.get("page") == "search":
        search_params = st.session_state.search_params

        # Check if results are already fetched for these params
        if 'search_results' not in st.session_state or \
           st.session_state.get('_search_params_for_results') != search_params:

            # Fetch results
            try:
                 results = search_youtube(
                     search_params["query"],
                     search_params["channel_ids"],
                     search_params["timeframe"],
                     search_params["content_filter"] # Pass the raw filter value
                     # TTL handled inside search_youtube
                 )
                 st.session_state.search_results = results
                 st.session_state._search_params_for_results = search_params # Mark results as current
            except Exception as e:
                 st.error(f"An error occurred during search: {e}")
                 logger.error(f"Error during search_youtube call: {e}", exc_info=True)
                 st.session_state.search_results = [] # Set empty results on error

        # Now display the results (if any) from session state
        results_to_display = st.session_state.get('search_results', [])

        # Apply the outlier score filter (client-side filtering after fetch)
        min_score_filter = search_params["min_outlier_score"]
        if min_score_filter > 0:
            original_count = len(results_to_display)
            # Ensure 'outlier_score' exists and is numeric, default to 0 if missing/invalid
            results_to_display = [
                r for r in results_to_display
                if pd.to_numeric(r.get("outlier_score"), errors='coerce', downcast=None) is not None and
                   pd.to_numeric(r.get("outlier_score"), errors='coerce', downcast=None) >= min_score_filter
            ]
            filtered_count = len(results_to_display)
            if original_count > 0 and filtered_count < original_count:
                 st.info(f"Filtered {original_count - filtered_count} results below minimum outlier score ({min_score_filter:.2f}).")


        if not results_to_display:
            st.info("No videos found matching all criteria.")
        else:
            # --- Sorting Options ---
            # Use updated metric names, ensure they exist in the data
            sort_options_map = {
                "Outlier Score": "outlier_score",
                "Views": "views",
                "Upload Date": "published_at", # Use ISO string for sorting
                "VPH Ratio": "vph_ratio",
                "Comment/View Ratio": "cvr_float", # Sort by float value
                "Like/View Ratio": "clr_float", # Sort by float value
                "Comment Count": "comment_count",
                "Like Count": "like_count",
                "Effective VPH": "effective_vph"
            }
            # Default sort by outlier score
            sort_label = st.selectbox("Sort results by:", list(sort_options_map.keys()), index=0, key="sort_by_sb")
            sort_key = sort_options_map[sort_label]

            # --- Sorting Logic ---
            try:
                # Handle sorting for different types
                if sort_key == "published_at":
                    # Sort by date string (ISO format ensures correct order)
                    sorted_data = sorted(results_to_display, key=lambda x: x.get(sort_key, "1970-01-01T00:00:00Z"), reverse=True)
                else:
                    # Sort numerically, handling potential missing values (treat as 0)
                    sorted_data = sorted(results_to_display, key=lambda x: pd.to_numeric(x.get(sort_key), errors='coerce') or 0, reverse=True)
            except Exception as e:
                 st.error(f"Error sorting results by {sort_label}: {e}")
                 logger.error(f"Sorting error: {e}", exc_info=True)
                 sorted_data = results_to_display # Fallback to unsorted


            st.subheader(f"Found {len(sorted_data)} results (sorted by {sort_label})")

            # --- Display Results in Cards ---
            num_columns = 3 # Fixed number of columns
            num_rows = (len(sorted_data) + num_columns - 1) // num_columns

            for i in range(num_rows):
                 cols = st.columns(num_columns)
                 row_items = sorted_data[i * num_columns : (i + 1) * num_columns]

                 for j, item in enumerate(row_items):
                      with cols[j]:
                           # Calculate display values safely
                           days_ago = int(round(item.get("days_since_published", 0)))
                           days_ago_text = "today" if days_ago == 0 else (f"yesterday" if days_ago == 1 else f"{days_ago} days ago")
                           outlier_val_num = pd.to_numeric(item.get('outlier_score'), errors='coerce')
                           outlier_val_str = f"{outlier_val_num:.2f}x" if outlier_val_num is not None else "N/A"
                           # Determine color based on outlier score
                           if outlier_val_num is None: outlier_color = "#888" # Grey for N/A
                           elif outlier_val_num >= 1.5: outlier_color = "#1e8e3e" # Green for high
                           elif outlier_val_num >= 0.8: outlier_color = "#188038" # Light Green for normal/slight high
                           else: outlier_color = "#c53929" # Red for low

                           outlier_html = f"""
                           <span style="
                               background-color:{outlier_color};
                               color:white;
                               padding:3px 8px;
                               border-radius:12px;
                               font-size:0.9em;
                               font-weight:bold;
                               display: inline-block;
                           ">
                               {outlier_val_str}
                           </span>
                           """
                           watch_url = f"https://www.youtube.com/watch?v={item['video_id']}"
                           thumbnail_url = item.get('thumbnail', '') or 'https://via.placeholder.com/320x180.png?text=No+Thumbnail'

                           # Card HTML structure
                           card_html = f"""
                           <div style="
                               border: 1px solid #e0e0e0;
                               border-radius: 8px;
                               padding: 10px;
                               margin-bottom: 15px;
                               height: 420px; /* Fixed height */
                               display: flex;
                               flex-direction: column;
                               background-color: #ffffff;
                               box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                           ">
                             <a href="{watch_url}" target="_blank" style="text-decoration: none; color: inherit;">
                               <img src="{thumbnail_url}" alt="Video Thumbnail" style="width:100%; border-radius:4px; margin-bottom:8px; object-fit: cover; aspect-ratio: 16/9;" />
                               <div style="font-weight: 600; font-size: 0.95rem; line-height: 1.3; height: 40px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; margin-bottom: 4px;">
                                 {item.get('title', 'No Title')}
                               </div>
                             </a>
                             <div style="font-size: 0.85rem; color: #5f6368; margin-bottom: 8px; height: 18px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                               {item.get('channel_name', 'Unknown Channel')}
                             </div>
                             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                               <span style="font-weight: 500; color: #202124; font-size: 0.9rem;">
                                 {item.get('formatted_views', 'N/A')} views
                               </span>
                               {outlier_html}
                             </div>
                             <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #5f6368;">
                               <span>
                                 {item.get('vph_display', 'N/A')}
                               </span>
                               <span>
                                 {days_ago_text}
                               </span>
                             </div>
                             <div style="margin-top: auto; padding-top: 5px;"> {/* Push button to bottom */}
                                 {/* Button added via Streamlit below */}
                             </div>
                           </div>
                           """
                           st.markdown(card_html, unsafe_allow_html=True)

                           # Add "View Details" button within the column
                           button_key = f"view_details_{item['video_id']}"
                           if st.button("View Details", key=button_key, use_container_width=True):
                               st.session_state.selected_video_id = item["video_id"]
                               st.session_state.selected_video_title = item["title"]
                               st.session_state.selected_video_data = item # Store all data
                               st.session_state.page = "details"
                               st.rerun() # Go to details page


def show_details_page():
    # Retrieve selected video data from session state
    video_id = st.session_state.get("selected_video_id")
    video_title = st.session_state.get("selected_video_title")
    video_data = st.session_state.get("selected_video_data") # Full data dictionary

    # Back button always available
    if st.button("⬅️ Back to Search Results", key="details_back_button_top"):
        # Clear selected video state? Optional, keeps context if they come back
        # if "selected_video_id" in st.session_state: del st.session_state["selected_video_id"]
        # if "selected_video_data" in st.session_state: del st.session_state["selected_video_data"]
        st.session_state.page = "search"
        st.rerun()

    if not video_id or not video_title or not video_data:
        st.error("No video selected or data missing. Please go back to Search.")
        # st.stop() # Removed stop, back button handles navigation
        return

    st.title(f"Video Details: {video_title}")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    st.markdown(f"[Watch on YouTube]({video_url})", unsafe_allow_html=True)

    # --- Display Basic Info & Metrics ---
    col1, col2 = st.columns([1, 2])
    with col1:
         thumbnail_url = video_data.get('thumbnail', '') or 'https://via.placeholder.com/320x180.png?text=No+Thumbnail'
         st.image(thumbnail_url, use_column_width=True)

    with col2:
        st.subheader("Performance Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Views", video_data.get('formatted_views', 'N/A'))
        m_col2.metric("Likes", format_number(video_data.get('like_count', 0)))
        m_col3.metric("Comments", format_number(video_data.get('comment_count', 0)))

        outlier_score_val = pd.to_numeric(video_data.get('outlier_score'), errors='coerce')
        outlier_score_disp = f"{outlier_score_val:.2f}x" if outlier_score_val is not None else "N/A"
        m_col1.metric("Outlier Score", outlier_score_disp, help="Performance relative to channel avg (1.0 = avg)")

        vph_ratio_val = pd.to_numeric(video_data.get('vph_ratio'), errors='coerce')
        vph_ratio_disp = f"{vph_ratio_val:.2f}x" if vph_ratio_val is not None else "N/A"
        m_col2.metric("VPH Ratio", vph_ratio_disp, help="Views per hour relative to channel avg")

        m_col3.metric("Effective VPH", video_data.get('vph_display', 'N/A'))

        cvr_disp = video_data.get('comment_to_view_ratio', 'N/A')
        outlier_cvr_val = pd.to_numeric(video_data.get('outlier_cvr'), errors='coerce')
        outlier_cvr_disp = f"({outlier_cvr_val:.2f}x avg)" if outlier_cvr_val is not None else ""
        st.markdown(f"**Comment/View Ratio:** {cvr_disp} {outlier_cvr_disp}")

        clr_disp = video_data.get('like_to_view_ratio', 'N/A') # Use the correct field name
        outlier_clr_val = pd.to_numeric(video_data.get('outlier_clr'), errors='coerce')
        outlier_clr_disp = f"({outlier_clr_val:.2f}x avg)" if outlier_clr_val is not None else ""
        st.markdown(f"**Like/View Ratio:** {clr_disp} {outlier_clr_disp}")

        duration_sec = video_data.get('duration_seconds', 0)
        duration_min, duration_s = divmod(duration_sec, 60)
        st.markdown(f"**Duration:** {int(duration_min)}m {int(duration_s)}s ({video_data.get('content_category', 'N/A')})")
        st.markdown(f"**Published:** {video_data.get('publish_date', 'N/A')} ({int(round(video_data.get('days_since_published', 0)))} days ago)")
        st.markdown(f"**Channel:** {video_data.get('channel_name', 'N/A')}")

    st.divider()

    # --- Comments Analysis ---
    st.subheader("💬 Comments Analysis")
    comments_key = f"comments_{video_id}"
    analysis_key = f"analysis_{video_id}"

    # Fetch comments if not already in session state for this video
    if comments_key not in st.session_state:
        with st.spinner(f"Fetching comments for {video_id}..."):
            st.session_state[comments_key] = get_video_comments(video_id, max_comments=100) # Fetch up to 100

    comments = st.session_state[comments_key]

    if not comments:
        st.info("No comments available or comments are disabled for this video.")
    else:
        st.write(f"Analyzed based on ~{len(comments)} comments fetched (ordered by relevance).")
        # Display top 5 comments by likes
        with st.expander("Show Top 5 Comments (by Likes)", expanded=False):
             top_5_comments = sorted(comments, key=lambda c: c.get("likeCount", 0), reverse=True)[:5]
             if not top_5_comments:
                  st.write("No comments with likes found.")
             else:
                  for i, c in enumerate(top_5_comments):
                       st.markdown(f"**{i+1}. {c.get('likeCount', 0)} likes** - _{c.get('author', 'Anon')}_", unsafe_allow_html=False) # Use markdown, disable HTML
                       st.caption(f'"{c.get("text", "")[:200]}..."') # Show preview in caption
                       st.markdown("---")


        # Perform and display analysis if not already done
        if analysis_key not in st.session_state:
            with st.spinner("Analyzing comments with AI..."):
                st.session_state[analysis_key] = analyze_comments(comments)

        analysis_result = st.session_state[analysis_key]
        st.markdown("**AI Comments Summary:**")
        st.markdown(analysis_result) # Display analysis text

    st.divider()

    # --- Script Analysis ---
    st.subheader("📜 Script Analysis")
    total_duration = video_data.get('duration_seconds', 0)
    is_short = video_data.get('content_category') == "Short" # Use pre-calculated category

    # Get transcript (uses fallback logic including Whisper)
    transcript_key = f"transcript_data_{video_id}"
    transcript_source_key = f"transcript_source_{video_id}"
    if transcript_key not in st.session_state:
         with st.spinner("Fetching transcript (may use Whisper fallback)..."):
              # get_transcript_with_fallback updates session state internally
              get_transcript_with_fallback(video_id)

    transcript = st.session_state.get(transcript_key)
    source = st.session_state.get(transcript_source_key)

    if not transcript:
        st.warning("Transcript is unavailable for this video.")
    else:
        st.caption(f"Transcript source: {source or 'Unknown'}")
        full_script_text = " ".join([seg.get("text", "") for seg in transcript])

        if is_short:
            st.markdown("**Short Video Script Summary:**")
            with st.spinner("Summarizing short's script with AI..."):
                short_summary = summarize_script(full_script_text)
            st.write(short_summary)
            with st.expander("Show Full Script Text (Short)", expanded=False):
                 st.text_area("Full Script", full_script_text, height=200, key="short_script_text")
        else: # Long-form video
            st.markdown("**Intro & Outro Analysis:**")
            intro_outro_key = f"intro_outro_summary_{video_id}"
            if intro_outro_key not in st.session_state:
                 with st.spinner("Extracting & summarizing intro/outro script snippets..."):
                      intro_txt, outro_txt = get_intro_outro_transcript(video_id, total_duration)
                      # Summarize using AI (uses internal caching)
                      summary_text, _ = summarize_intro_outro(intro_txt, outro_txt) # We expect one combined summary now
                      st.session_state[intro_outro_key] = {
                           "intro_raw": intro_txt,
                           "outro_raw": outro_txt,
                           "summary": summary_text
                      }

            intro_outro_data = st.session_state[intro_outro_key]
            st.markdown("**AI Summary of Intro/Outro:**")
            st.write(intro_outro_data.get("summary", "*Summary unavailable.*"))

            with st.expander("Show Raw Intro/Outro Snippets", expanded=False):
                 st.markdown("**Intro Snippet (approx first 60s):**")
                 st.caption(intro_outro_data.get("intro_raw") or "*Not available or extracted.*")
                 st.markdown("**Outro Snippet (approx last 60s):**")
                 st.caption(intro_outro_data.get("outro_raw") or "*Not available or extracted.*")

            # Option to summarize the *entire* long script
            if st.button("Summarize Full Long-form Script (AI)", key="summarize_full_long"):
                 with st.spinner("Summarizing full script..."):
                      full_summary = summarize_script(full_script_text)
                 st.session_state[f"full_script_summary_{video_id}"] = full_summary

            if f"full_script_summary_{video_id}" in st.session_state:
                 st.markdown("**AI Summary of Full Script:**")
                 st.write(st.session_state[f"full_script_summary_{video_id}"])

            with st.expander("Show Full Script Text (Long-form)", expanded=False):
                 st.text_area("Full Script", full_script_text, height=300, key="long_script_text")


    st.divider()

    # --- Retention Analysis ---
    st.subheader("📈 Retention Analysis (Experimental)")
    retention_state_key_prefix = f"retention_{video_id}_"

    # Check eligibility for retention analysis
    published_at_iso = video_data.get("published_at")
    can_run_retention = False
    if is_short:
        st.info("Retention analysis is typically less informative for short videos.")
    elif not published_at_iso:
         st.warning("Cannot determine video age for retention analysis (publish date missing).")
    else:
        try:
            published_dt = datetime.fromisoformat(published_at_iso.replace('Z', '+00:00')) # Make timezone-aware (UTC)
            age_delta = datetime.now(timezone.utc) - published_dt
            if age_delta < timedelta(days=2): # Need at least ~2 days for data to populate
                st.info(f"Retention data may be incomplete. Video is too recent (published {age_delta.days} days ago). Needs ~2+ days.")
            elif not os.path.exists("youtube_cookies.json"):
                 st.warning("`youtube_cookies.json` not found. Retention analysis requires YouTube login cookies.")
                 st.markdown("See [instructions](https://github.com/demberto/youtube-studio-api/blob/main/youtube_studio_api/auth.py#L13) on how to get cookies using a browser extension.", unsafe_allow_html=True) # Link to example
            elif not check_selenium_setup():
                 st.error("Browser/Driver setup failed. Cannot run retention analysis.")
            else:
                 can_run_retention = True # Eligible to run

        except Exception as e:
            logger.error(f"Error checking retention eligibility: {e}")
            st.warning(f"Could not determine retention eligibility: {e}")


    if can_run_retention:
        # Button to trigger analysis
        if st.button("Run Retention Analysis", key="run_retention_button"):
             st.session_state[retention_state_key_prefix + 'triggered'] = True
             # Clear previous results for this video if re-running
             for key in list(st.session_state.keys()):
                  if key.startswith(retention_state_key_prefix) and key != retention_state_key_prefix + 'triggered':
                       del st.session_state[key]
             st.rerun() # Rerun to execute the analysis block

        # Execute analysis if triggered
        if st.session_state.get(retention_state_key_prefix + 'triggered'):
             try:
                 # 1. Capture Screenshot with Hover
                 screenshot_key = retention_state_key_prefix + 'screenshot_path'
                 duration_key = retention_state_key_prefix + 'duration'
                 if screenshot_key not in st.session_state:
                      with st.spinner("Capturing retention graph screenshot via Selenium..."):
                           # Use a unique path per video in temp dir
                           temp_retention_dir = tempfile.mkdtemp(prefix=f"retention_{video_id}_")
                           screenshot_path = os.path.join(temp_retention_dir, "player_retention.png")
                           logger.info(f"Attempting retention capture for {video_id}")
                           # Pass video duration if known, else use 0 as placeholder
                           base_timestamp = total_duration if total_duration > 0 else 120
                           vid_duration = capture_player_screenshot_with_hover(video_url, timestamp=base_timestamp, output_path=screenshot_path, use_cookies=True)

                           if os.path.exists(screenshot_path):
                                st.session_state[screenshot_key] = screenshot_path
                                st.session_state[duration_key] = vid_duration # Store the duration returned by capture
                                logger.info(f"Retention screenshot saved to {screenshot_path}, duration: {vid_duration}")
                           else:
                                st.error("Retention screenshot capture failed.")
                                raise RuntimeError("Screenshot file not found after capture.")

                 # Display screenshot if successful
                 if screenshot_key in st.session_state:
                     st.image(st.session_state[screenshot_key], caption="Player Screenshot with Retention Graph")
                     retention_vid_duration = st.session_state.get(duration_key, 0)
                     if retention_vid_duration <= 0:
                          st.warning("Could not reliably determine video duration during capture.")
                          # Use duration from API as fallback if > 0
                          retention_vid_duration = total_duration if total_duration > 0 else 1 # Avoid division by zero

                     # 2. Detect Peaks
                     peaks_key = retention_state_key_prefix + 'peaks'
                     if peaks_key not in st.session_state:
                          with st.spinner("Analyzing retention peaks from screenshot..."):
                               screenshot_file = st.session_state[screenshot_key]
                               peaks, roi, binary_roi, roi_width, col_sums = detect_retention_peaks(
                                   screenshot_file,
                                   crop_ratio=0.15, # Adjusted crop
                                   height_threshold=100, # Lowered threshold
                                   distance=25, # Adjusted distance
                                   top_n=5
                               )
                               st.session_state[peaks_key] = peaks
                               st.session_state[retention_state_key_prefix + 'roi_width'] = roi_width
                               st.session_state[retention_state_key_prefix + 'col_sums'] = col_sums # Store for plotting
                               logger.info(f"Detected {len(peaks)} peaks: {peaks}")

                     # Display peaks and plot
                     peaks_data = st.session_state.get(peaks_key)
                     col_sums_data = st.session_state.get(retention_state_key_prefix + 'col_sums')
                     roi_width_data = st.session_state.get(retention_state_key_prefix + 'roi_width')

                     if peaks_data is not None:
                          st.write(f"Detected {len(peaks_data)} potential retention peak(s).")
                          # Plot brightness profile
                          profile_plot = plot_brightness_profile(col_sums_data, peaks_data)
                          if profile_plot:
                               st.image(profile_plot, caption="Retention Graph Brightness Profile")
                          else:
                               st.warning("Could not generate brightness profile plot.")

                          if len(peaks_data) > 0 and roi_width_data > 0:
                               # 3. Analyze Each Peak
                               st.markdown("**Peak Analysis:**")
                               snippet_duration_sec = st.slider("Snippet Duration around Peaks (sec):", min_value=4, max_value=20, value=8, step=2, key="snippet_dur_slider")

                               for idx, peak_x in enumerate(peaks_data):
                                    # Calculate timestamp corresponding to peak's x-coordinate
                                    peak_time_sec = (peak_x / roi_width_data) * retention_vid_duration
                                    st.markdown(f"--- \n**Peak {idx+1} at ~ {peak_time_sec:.1f} seconds**")

                                    peak_frame_key = f"{retention_state_key_prefix}peak_{idx+1}_frame"
                                    if peak_frame_key not in st.session_state:
                                         with st.spinner(f"Capturing frame for Peak {idx+1}..."):
                                              frame_output_dir = os.path.dirname(st.session_state[screenshot_key]) # Use same temp dir
                                              frame_output_path = os.path.join(frame_output_dir, f"peak_{idx+1}_frame.png")
                                              capture_frame_at_time(video_url, target_time=peak_time_sec, output_path=frame_output_path, use_cookies=True)
                                              if os.path.exists(frame_output_path):
                                                   st.session_state[peak_frame_key] = frame_output_path
                                              else:
                                                   st.error(f"Failed to capture frame for Peak {idx+1}")
                                                   st.session_state[peak_frame_key] = None # Mark as failed

                                    # Display frame
                                    if st.session_state.get(peak_frame_key):
                                         st.image(st.session_state[peak_frame_key], caption=f"Frame at {peak_time_sec:.1f} sec")

                                    # Transcript snippet
                                    if transcript:
                                         snippet_text = filter_transcript(transcript, target_time=peak_time_sec, window=snippet_duration_sec / 2)
                                         st.markdown(f"**Transcript around {peak_time_sec:.1f}s (±{snippet_duration_sec / 2}s):**")
                                         st.caption(snippet_text or "*No text found.*")

                                    # Video snippet (if yt-dlp available)
                                    peak_snippet_key = f"{retention_state_key_prefix}peak_{idx+1}_snippet"
                                    if check_ytdlp_installed():
                                         if peak_snippet_key not in st.session_state:
                                              with st.spinner(f"Downloading video snippet for Peak {idx+1}..."):
                                                   snippet_output_dir = os.path.dirname(st.session_state[screenshot_key])
                                                   snippet_output_path = os.path.join(snippet_output_dir, f"peak_{idx+1}_snippet.mp4")
                                                   adjusted_start_time = max(0, peak_time_sec - snippet_duration_sec / 2)
                                                   try:
                                                        download_video_snippet(video_url, start_time=adjusted_start_time, duration=snippet_duration_sec, output_path=snippet_output_path)
                                                        if os.path.exists(snippet_output_path) and os.path.getsize(snippet_output_path) > 0:
                                                             st.session_state[peak_snippet_key] = snippet_output_path
                                                        else:
                                                             st.error(f"Video snippet download failed for Peak {idx+1}.")
                                                             st.session_state[peak_snippet_key] = None
                                                   except Exception as snip_e:
                                                        st.error(f"Error downloading snippet for Peak {idx+1}: {snip_e}")
                                                        logger.error(f"Snippet download error: {snip_e}", exc_info=True)
                                                        st.session_state[peak_snippet_key] = None

                                         # Display video snippet
                                         if st.session_state.get(peak_snippet_key):
                                              st.write(f"**Video Snippet (±{snippet_duration_sec / 2:.1f}s around peak):**")
                                              st.video(st.session_state[peak_snippet_key])
                                         else:
                                              # Only show error if download was attempted and failed
                                              if peak_snippet_key in st.session_state:
                                                   st.warning("Could not display video snippet for this peak.")

                                    else: # yt-dlp not installed
                                         st.info("Install yt-dlp to enable video snippet downloads for peaks.")


                          else: # No peaks detected or invalid width
                               if peaks_data is not None and len(peaks_data) == 0:
                                    st.info("No significant retention peaks were detected in the analysis.")
                               elif roi_width_data <= 0:
                                    st.error("Could not determine the width of the retention graph region.")

                     else: # Peak detection failed earlier
                          st.error("Retention peak detection failed.")

             except EnvironmentError as env_e: # Catch specific setup errors
                  st.error(f"Retention Analysis Setup Error: {env_e}")
                  logger.error(f"Retention Env Error: {env_e}")
             except (RuntimeError, TimeoutError, FileNotFoundError) as run_e: # Catch errors from Selenium/analysis steps
                  st.error(f"Retention Analysis Runtime Error: {run_e}")
                  logger.error(f"Retention Runtime Error: {run_e}", exc_info=True)
             except Exception as e:
                  st.error(f"An unexpected error occurred during retention analysis: {e}")
                  logger.error(f"Unexpected Retention Error: {e}", exc_info=True)
             finally:
                  # Reset trigger after execution (success or failure)
                  st.session_state[retention_state_key_prefix + 'triggered'] = False
                  # Consider cleaning up temp files associated with retention analysis here?
                  # Need to track the temp directory created.
                  # temp_dir_to_clean = os.path.dirname(st.session_state.get(screenshot_key,""))
                  # if temp_dir_to_clean and os.path.isdir(temp_dir_to_clean) and temp_dir_to_clean.startswith(tempfile.gettempdir()):
                  #      logger.info(f"Attempting to clean up retention temp dir: {temp_dir_to_clean}")
                  #      # Be cautious with rmtree
                  #      # shutil.rmtree(temp_dir_to_clean, ignore_errors=True)


def main():
    # Initialize database on first run
    if 'db_initialized' not in st.session_state:
         init_db(DB_PATH)
         st.session_state.db_initialized = True

    # Set page config (only once)
    st.set_page_config(page_title="YouTube Niche Search & Analysis", layout="wide", page_icon="📊")

    # Simple navigation based on session state
    if "page" not in st.session_state:
        st.session_state.page = "search"

    page = st.session_state.get("page")

    if page == "search":
        show_search_page()
    elif page == "details":
        show_details_page()
    else: # Default to search page if state is invalid
        st.session_state.page = "search"
        show_search_page()

# Cleanup function to log shutdown
def app_shutdown():
     logger.info("Application shutting down.")
     # Clean up any global resources if needed

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch unexpected errors in main execution flow
        logger.error(f"FATAL: Unhandled exception in main UI loop: {e}", exc_info=True)
        st.error(f"An critical error occurred: {e}. Please check the logs or restart the application.")
    finally:
         # Register cleanup function
         atexit.register(app_shutdown)


# --- END OF FILE ---
```
