"""
app.py - A self-contained Streamlit application for YouTube Niche Search.
It integrates:
  • A Channel Folder Manager (with support for adding channel URLs)
  • Real YouTube search using the YouTube Data API (API key stored in st.secrets)
  • Retention analysis and outlier scoring.
"""

import os
import time
import json
import re
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.signal import find_peaks
import sqlite3
import subprocess
import pandas as pd
import isodate
import logging
from logging.handlers import RotatingFileHandler
import atexit
from datetime import datetime, timedelta, timezone
import requests

# =============================================================================
# Shared Helper Functions (Data Processing & Outlier Calculation)
# =============================================================================
def calculate_outlier_score(current_views, channel_average):
    """Calculate outlier score as the ratio of current views to the channel's average."""
    try:
        return current_views / channel_average if channel_average != 0 else 0
    except Exception:
        return 0

def get_outlier_category(outlier_score):
    """Determine outlier category and CSS class based on the outlier score."""
    if outlier_score >= 2.0:
        return "Significant Positive Outlier", "outlier-high"
    elif outlier_score >= 1.5:
        return "Positive Outlier", "outlier-high"
    elif outlier_score >= 1.2:
        return "Slight Positive Outlier", "outlier-normal"
    elif outlier_score >= 0.8:
        return "Normal Performance", "outlier-normal"
    elif outlier_score >= 0.5:
        return "Slight Negative Outlier", "outlier-low"
    else:
        return "Significant Negative Outlier", "outlier-low"

def process_video_data(data):
    """
    Process raw video data and calculate performance metrics including outlier score.
    Expected data columns: views, likes, comments, view_count, channel_average.
    """
    df = pd.DataFrame(data)
    df["combined_performance"] = df["views"] + df["likes"] * 2 + df["comments"] * 3
    df["log_performance"] = np.log1p(df["combined_performance"])
    df["outlier_score"] = df["view_count"] / df["channel_average"]
    df["breakout_score"] = df["outlier_score"]
    return df

# =============================================================================
# Logging Setup
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("app.log", maxBytes=1000000, backupCount=3)
logger.addHandler(handler)

# =============================================================================
# Constants and Paths
# =============================================================================
DB_PATH = "youtube_data.db"
CHANNEL_FOLDERS_FILE = "channel_folders.json"

# =============================================================================
# Database and Cache Functions
# =============================================================================
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS youtube_cache (
            id TEXT PRIMARY KEY,
            data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def cache_result(key, data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO youtube_cache (id, data) VALUES (?, ?)", (key, json.dumps(data)))
    conn.commit()
    conn.close()

def load_cached_result(key):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT data FROM youtube_cache WHERE id = ?", (key,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

# =============================================================================
# Channel Folder Persistence Functions
# =============================================================================
def load_channel_folders():
    """Load folder data from JSON file."""
    if os.path.exists(CHANNEL_FOLDERS_FILE):
        with open(CHANNEL_FOLDERS_FILE, "r") as f:
            folders = json.load(f)
    else:
        folders = {}
    return folders

def save_channel_folders(folders):
    """Save folder data to JSON file."""
    with open(CHANNEL_FOLDERS_FILE, "w") as f:
        json.dump(folders, f, indent=4)

# =============================================================================
# Utility: Extract Channel ID from URL or input
# =============================================================================
def extract_channel_id(input_str):
    """
    If the input string is a YouTube channel URL, extract the channel ID.
    Otherwise, return the stripped input.
    """
    input_str = input_str.strip()
    # Check for URL with "/channel/"
    m = re.search(r"youtube\.com/channel/([a-zA-Z0-9_-]+)", input_str)
    if m:
        return m.group(1)
    # Check for URL with "/user/" (for simplicity, return the username; ideally, you should convert it)
    m = re.search(r"youtube\.com/user/([a-zA-Z0-9_-]+)", input_str)
    if m:
        return m.group(1)
    # Otherwise, assume input_str is already a channel ID.
    return input_str

# =============================================================================
# Advanced Channel Folder Manager UI
# =============================================================================
def show_channel_folder_manager():
    """
    Manage channel folders using a single form with an Action dropdown:
      - Create New Folder
      - Add Channels to Existing Folder
      - Delete Folder
    """
    st.subheader("Manage Channel Folders")
    folders = load_channel_folders()

    if folders:
        st.write("**Existing Folders:**")
        for folder_name, channels in folders.items():
            st.write(f"- {folder_name} ({len(channels)} channels)")
    else:
        st.write("No folders available yet.")

    action = st.selectbox("Action", ["Create New Folder", "Add Channels to Existing Folder", "Delete Folder"])
    folder_name = ""
    folder_choice = ""
    channels_input = ""

    if action == "Create New Folder":
        folder_name = st.text_input("Folder Name")
        channels_input = st.text_area("Enter at least one channel URL or ID (one per line):", height=100)
    elif action == "Add Channels to Existing Folder":
        if not folders:
            st.info("No folders available. Please create a folder first.")
            return
        folder_choice = st.selectbox("Select Folder", list(folders.keys()))
        channels_input = st.text_area("Enter at least one channel URL or ID (one per line):", height=100)
    elif action == "Delete Folder":
        if not folders:
            st.info("No folders available to delete.")
            return
        folder_choice = st.selectbox("Select Folder to Delete", list(folders.keys()))

    button_label = {"Create New Folder": "Create Folder",
                    "Add Channels to Existing Folder": "Add Channels",
                    "Delete Folder": "Delete Folder"}[action]

    if st.button(button_label):
        if action == "Create New Folder":
            name = folder_name.strip()
            if not name:
                st.error("Folder name cannot be empty.")
                return
            if name in folders:
                st.error("Folder already exists. Choose a different name or use 'Add Channels'.")
                return
            lines = [line.strip() for line in channels_input.splitlines() if line.strip()]
            if not lines:
                st.error("Please enter at least one channel.")
                return
            channel_list = [{"channel_name": line, "channel_id": extract_channel_id(line)} for line in lines]
            folders[name] = channel_list
            save_channel_folders(folders)
            st.success(f"Folder '{name}' created with {len(channel_list)} channel(s).")
        elif action == "Add Channels to Existing Folder":
            if not folder_choice:
                st.error("No folder selected.")
                return
            lines = [line.strip() for line in channels_input.splitlines() if line.strip()]
            if not lines:
                st.error("Please enter at least one channel.")
                return
            for line in lines:
                folders[folder_choice].append({"channel_name": line, "channel_id": extract_channel_id(line)})
            save_channel_folders(folders)
            st.success(f"Added {len(lines)} channel(s) to folder '{folder_choice}'.")
        elif action == "Delete Folder":
            if not folder_choice:
                st.error("No folder selected.")
                return
            if folder_choice in folders:
                del folders[folder_choice]
                save_channel_folders(folders)
                st.success(f"Folder '{folder_choice}' deleted.")
            else:
                st.error("Folder not found. Please refresh and try again.")

# =============================================================================
# YouTube Search Using the YouTube Data API
# =============================================================================
def fetch_youtube_results(keyword, channel_ids, timeframe, content_filter):
    """
    Fetch search results from the YouTube Data API.
    API key is read from st.secrets["youtube"]["api_key"].
    """
    results = []
    api_key = st.secrets["youtube"]["api_key"]
    base_search_url = "https://www.googleapis.com/youtube/v3/search"

    now = datetime.utcnow()
    if timeframe == "Last 24 hours":
        published_after = now - timedelta(days=1)
    elif timeframe == "Last 48 hours":
        published_after = now - timedelta(days=2)
    elif timeframe == "Last 4 days":
        published_after = now - timedelta(days=4)
    elif timeframe == "Last 7 days":
        published_after = now - timedelta(days=7)
    elif timeframe == "Last 15 days":
        published_after = now - timedelta(days=15)
    elif timeframe == "Last 28 days":
        published_after = now - timedelta(days=28)
    elif timeframe == "3 months":
        published_after = now - timedelta(days=90)
    elif timeframe == "Lifetime":
        published_after = None
    else:
        published_after = None

    published_after_str = published_after.isoformat("T") + "Z" if published_after else None

    for channel_id in channel_ids:
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": 10,
            "type": "video",
            "order": "date",
            "key": api_key
        }
        if keyword:
            params["q"] = keyword
        if published_after_str:
            params["publishedAfter"] = published_after_str

        response = requests.get(base_search_url, params=params)
        if response.status_code == 200:
            data = response.json()
            results.extend(data.get("items", []))
        else:
            st.error(f"Error fetching YouTube results: {response.text}")
    return results

def search_youtube(keyword, channel_ids, timeframe, content_filter, ttl=600):
    """
    Wrapper for fetching YouTube results.
    """
    return fetch_youtube_results(keyword, channel_ids, timeframe, content_filter)

# =============================================================================
# Other Video Analysis Functions (Retention, etc.)
# =============================================================================
def analyze_comments(comments):
    return "Positive sentiment with suggestions for improvement."

def summarize_script(script_text):
    return "This short video quickly covers the main topic."

def get_transcript_with_fallback(video_id):
    transcript = [{"start": 0, "text": "Welcome to the video."}]
    return transcript, "dummy"

def get_intro_outro_transcript(video_id, total_duration):
    return "This is the intro of the video.", "This is the outro of the video."

def summarize_intro_outro(intro, outro):
    return "Intro summary.", "Outro summary."

def capture_player_screenshot_with_hover(video_url, timestamp, output_path, use_cookies):
    time.sleep(2)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(output_path, dummy_image)
    return 10

def detect_retention_peaks(screenshot_path, crop_ratio, height_threshold, distance, top_n):
    roi_width = 640
    col_sums = np.random.randint(0, 255, size=(roi_width,))
    peaks, _ = find_peaks(col_sums, height=height_threshold, distance=distance)
    return peaks[:top_n], None, None, roi_width, col_sums

def plot_brightness_profile(col_sums, peaks):
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(col_sums, label="Brightness Sum")
    ax.plot(peaks, col_sums[peaks], "x", label="Detected Peaks", markersize=10)
    ax.set_xlabel("Column Index (ROI)")
    ax.set_ylabel("Sum of Pixel Values")
    ax.legend()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def capture_frame_at_time(video_url, target_time, output_path, use_cookies):
    time.sleep(1)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(output_path, dummy_frame)
    return output_path

def filter_transcript(transcript, target_time, window=5):
    snippet = []
    for seg in transcript:
        if abs(seg.get("start", 0) - target_time) <= window:
            snippet.append(seg.get("text", ""))
    return " ".join(snippet)

def check_ytdlp_installed():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def download_video_snippet(video_url, start_time, duration=10, output_path="snippet.mp4"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    try:
        temp_video = "temp_full_video.mp4"
        download_cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", temp_video,
            video_url
        ]
        st.info("Downloading full video for extraction...", key="download_full_video")
        subprocess.run(download_cmd, check=True, capture_output=True)
        import imageio_ffmpeg
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
        st.error(f"Error during video processing: {e.stdout}\\n{e.stderr}", key="snippet_error")
        raise Exception(f"Failed to extract video segment: {str(e)}")

# =============================================================================
# UI Pages
# =============================================================================
def show_search_page():
    st.title("YouTube Niche Search")
    
    # Sidebar: Advanced Channel Folder Manager
    with st.sidebar.expander("Channel Folder Manager"):
        show_channel_folder_manager()

    folders = load_channel_folders()
    folder_choice = st.sidebar.selectbox("Select Folder", list(folders.keys()) if folders else ["None"])
    selected_timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["Last 24 hours", "Last 48 hours", "Last 4 days", "Last 7 days", "Last 15 days", "Last 28 days", "3 months", "Lifetime"]
    )
    content_filter = st.sidebar.selectbox("Filter By Content Type", ["Shorts", "Videos", "Both"], index=2)
    min_outlier_score = st.sidebar.number_input("Minimum Outlier Score", value=0.0, step=0.1, format="%.2f")
    search_query = st.sidebar.text_input("Keyword (optional)", "")

    selected_channel_ids = []
    if folder_choice != "None":
        for ch in folders.get(folder_choice, []):
            selected_channel_ids.append(ch["channel_id"])

    st.write(f"**Selected Folder**: {folder_choice}")
    if folder_choice != "None":
        with st.expander("Channels in this folder", expanded=False):
            if folders.get(folder_choice):
                for ch in folders[folder_choice]:
                    st.write(f"- {ch['channel_name']}")
            else:
                st.write("(No channels)")

    if st.sidebar.button("Clear Cache (force new)"):
        with sqlite3.connect(DB_PATH) as c:
            c.execute("DELETE FROM youtube_cache")
        st.sidebar.success("Cache cleared. Next search is fresh.")

    if st.sidebar.button("Search"):
        if folder_choice == "None" or not selected_channel_ids:
            st.error("No folder or channels selected. Please select a folder with at least one channel.")
        else:
            results = search_youtube(search_query, selected_channel_ids, selected_timeframe, content_filter, ttl=600)
            if min_outlier_score > 0:
                results = [r for r in results if float(r.get("statistics", {}).get("viewCount", 0)) >= min_outlier_score]
            st.session_state.search_results = results
            st.session_state.page = "search"

    if "search_results" in st.session_state and st.session_state.search_results:
        data = st.session_state.search_results
        sort_options = [
            "views",
            "upload_date",
            "outlier_score",
            "outlier_cvr",
            "outlier_clr",
            "comment_to_view_ratio",
            "comment_to_like_ratio",
            "comment_count"
        ]
        sort_by = st.selectbox("Sort by:", sort_options, index=0)

        def parse_sort_value(item):
            if sort_by == "upload_date":
                try:
                    return datetime.strptime(item["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                except:
                    return datetime.min
            val = item.get(sort_by, 0)
            try:
                return float(val)
            except:
                return 0.0

        sorted_data = sorted(data, key=parse_sort_value, reverse=True)
        st.subheader(f"Found {len(sorted_data)} results (sorted by {sort_by})")

        for i in range(0, len(sorted_data), 3):
            row_chunk = sorted_data[i:i+3]
            cols = st.columns(3)
            for j in range(3):
                with cols[j]:
                    if j < len(row_chunk):
                        row = row_chunk[j]
                        published = row["snippet"].get("publishedAt", "")
                        try:
                            pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                            days_ago = (datetime.utcnow() - pub_date).days
                        except:
                            days_ago = 0
                        days_ago_text = "today" if days_ago == 0 else f"{days_ago} days ago"
                        # For display, we use a dummy outlier value
                        outlier_val = "N/A"
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
                        video_id = row.get("id", {}).get("videoId", "unknown")
                        watch_url = f"https://www.youtube.com/watch?v={video_id}"
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
                            <img src="{row['snippet']['thumbnails']['medium']['url']}" style="width:100%; border-radius:4px; margin-bottom:0.5rem;" />
                          </a>
                          <div style="font-weight:bold; font-size:1rem; text-align:left; margin-bottom:0.3rem;">
                            {row['snippet']['title']}
                          </div>
                          <div style="font-size:0.9rem; text-align:left; margin-bottom:0.5rem; color:#555;">
                            {row['snippet']['channelTitle']}
                          </div>
                          <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                            <span style="font-size:0.85rem;">
                              Published {days_ago_text}
                            </span>
                            {outlier_html}
                          </div>
                        </div>
                        """
                        st.markdown(container_html, unsafe_allow_html=True)
                        if st.checkbox("View more analytics", key=f"toggle_{video_id}"):
                            st.write(f"**Channel:** {row['snippet']['channelTitle']}")
                            if st.button("View Details", key=f"view_{video_id}"):
                                st.session_state.selected_video_id = video_id
                                st.session_state.selected_video_title = row["snippet"]["title"]
                                st.session_state.selected_video_publish_at = row["snippet"]["publishedAt"]
                                st.session_state.page = "details"
                                st.stop()
                    else:
                        st.empty()

def show_details_page():
    video_id = st.session_state.get("selected_video_id")
    video_title = st.session_state.get("selected_video_title")
    published_at = st.session_state.get("selected_video_publish_at")

    if not video_id or not video_title:
        st.write("No video selected. Please go back to Search.")
        if st.button("Back to Search", key="details_back"):
            st.session_state.page = "search"
            st.stop()
        return

    st.title(f"Details for: {video_title}")
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    st.subheader("Comments")
    # Replace with actual comment fetching if needed.
    comments = []
    if comments:
        st.write(f"Total Comments Fetched: {len(comments)}")
    else:
        st.write("No comments available for this video.")

    st.subheader("Script")
    transcript, _ = get_transcript_with_fallback(video_id)
    if transcript:
        script_text = " ".join([seg["text"] for seg in transcript])
        st.markdown("**Full Script:**")
        st.write(script_text)
        with st.spinner("Summarizing script..."):
            summary = summarize_script(script_text)
        st.subheader("Script Summary")
        st.write(summary)
    else:
        st.info("Transcript unavailable.")

    st.subheader("Retention Analysis")
    st.info("Retention analysis functionality not fully implemented in this demo.")

    if st.button("Back to Search", key="details_back_button"):
        st.session_state.page = "search"
        st.stop()

def main():
    init_db(DB_PATH)
    if "page" not in st.session_state:
        st.session_state.page = "search"
    page = st.session_state.get("page")
    if page == "search":
        show_search_page()
    elif page == "details":
        show_details_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred. Please check the logs for details.")
        logger.error(f"Unexpected error in main UI: {e}")
    atexit.register(lambda: logger.info("Application shutting down"))
