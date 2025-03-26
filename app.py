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

# Import shared functions from shared.py
from shared import process_video_data

# =============================================================================
# Setup Logging
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
# Database and Utility Functions
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
# Channel Folder Persistence
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

    # Display existing folders summary.
    if folders:
        st.write("**Existing Folders:**")
        for folder_name, channels in folders.items():
            st.write(f"- {folder_name} ({len(channels)} channels)")
    else:
        st.write("No folders available yet.")

    # Action selectbox.
    action = st.selectbox("Action", ["Create New Folder", "Add Channels to Existing Folder", "Delete Folder"])

    # Prepare form fields.
    folder_name = ""
    folder_choice = ""
    channels_input = ""

    if action == "Create New Folder":
        folder_name = st.text_input("Folder Name")
        channels_input = st.text_area("Enter at least one channel name or URL (one per line):", height=100)

    elif action == "Add Channels to Existing Folder":
        if not folders:
            st.info("No folders available. Please create a folder first.")
            return
        folder_choice = st.selectbox("Select Folder", list(folders.keys()))
        channels_input = st.text_area("Enter at least one channel name or URL (one per line):", height=100)

    elif action == "Delete Folder":
        if not folders:
            st.info("No folders available to delete.")
            return
        folder_choice = st.selectbox("Select Folder to Delete", list(folders.keys()))

    button_label = {
        "Create New Folder": "Create Folder",
        "Add Channels to Existing Folder": "Add Channels",
        "Delete Folder": "Delete Folder"
    }[action]

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
            channel_list = []
            for line in lines:
                channel_entry = {
                    "channel_name": line,
                    "channel_id": line
                }
                channel_list.append(channel_entry)
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
                channel_entry = {
                    "channel_name": line,
                    "channel_id": line
                }
                folders[folder_choice].append(channel_entry)
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
# Video Search & Analysis Functions
# =============================================================================
def search_youtube(keyword, channel_ids, timeframe, content_filter, ttl=600):
    """
    Dummy implementation for search.
    Replace with actual YouTube API calls or DB queries.
    """
    dummy_data = [
        {
            "video_id": "abc123",
            "title": "Sample Video 1",
            "thumbnail": "https://img.youtube.com/vi/abc123/0.jpg",
            "channel_name": "Channel One",
            "views": 15000,
            "likes": 300,
            "comments": 25,
            "view_count": 15000,
            "channel_average": 10000,
            "recency_factor": 1.0,
            "combined_performance": 15000 + 300 * 2 + 25 * 3,
            "formatted_views": "15K",
            "vph_display": "100 VPH",
            "published_at": "2023-03-15T12:00:00Z",
            "days_since_published": 10,
            "outlier_score": 1.5,
            "outlier_cvr": 1.2,
            "outlier_clr": 0.8,
            "comment_to_view_ratio": "0.17%",
            "comment_to_like_ratio": "8.33%"
        },
    ]
    return dummy_data

def analyze_comments(comments):
    return "Positive sentiment with suggestions for improvement."

def summarize_script(script_text):
    return "This short video quickly covers the main topic."

def get_transcript_with_fallback(video_id):
    transcript = [{"start": 0, "text": "Welcome to the video."}]
    source = "dummy"
    return transcript, source

def get_intro_outro_transcript(video_id, total_duration):
    intro_txt = "This is the intro of the video."
    outro_txt = "This is the outro of the video."
    return intro_txt, outro_txt

def summarize_intro_outro(intro, outro):
    return "Intro summary.", "Outro summary."

# =============================================================================
# Retention Analysis Functions
# =============================================================================
def capture_player_screenshot_with_hover(video_url, timestamp, output_path, use_cookies):
    time.sleep(2)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(output_path, dummy_image)
    return 10

def detect_retention_peaks(screenshot_path, crop_ratio, height_threshold, distance, top_n):
    roi_width = 640
    col_sums = np.random.randint(0, 255, size=(roi_width,))
    peaks, _ = find_peaks(col_sums, height=height_threshold, distance=distance)
    peaks = peaks[:top_n]
    return peaks, None, None, roi_width, col_sums

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
    st.title("Youtube Niche Search")
    # Sidebar: Channel Folder Manager
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
                results = [r for r in results if r.get("outlier_score", 0) >= min_outlier_score]
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
                    return datetime.strptime(item["published_at"], "%Y-%m-%dT%H:%M:%SZ")
                except:
                    return datetime.min
            val = item.get(sort_by, 0)
            if sort_by in ("comment_to_view_ratio", "comment_to_like_ratio"):
                return float(val.replace("%", "")) if "%" in val else 0.0
            return float(val) if isinstance(val, (int, float, str)) else 0.0

        sorted_data = sorted(data, key=parse_sort_value, reverse=True)
        st.subheader(f"Found {len(sorted_data)} results (sorted by {sort_by})")

        # Display results in 3 columns per row.
        for i in range(0, len(sorted_data), 3):
            row_chunk = sorted_data[i:i+3]
            cols = st.columns(3)
            for j in range(3):
                with cols[j]:
                    if j < len(row_chunk):
                        row = row_chunk[j]
                        days_ago = int(round(row.get("days_since_published", 0)))
                        days_ago_text = "today" if days_ago == 0 else f"{days_ago} days ago"
                        outlier_val = f"{row['outlier_score']:.2f}x"
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
                        if st.checkbox("View more analytics", key=f"toggle_{row['video_id']}"):
                            st.write(f"**Channel**: {row['channel_name']}")
                            st.write(f"**Category**: {row.get('content_category', 'N/A')}")
                            st.write(f"**Comments**: {row['comment_count']}")
                            st.markdown(f"**C/V Ratio**: {row['comment_to_view_ratio']} (Outlier: {row['outlier_cvr']:.2f})")
                            st.markdown(f"**C/L Ratio**: {row['comment_to_like_ratio']} (Outlier: {row['outlier_clr']:.2f})")
                            if st.button("View Details", key=f"view_{row['video_id']}"):
                                st.session_state.selected_video_id = row["video_id"]
                                st.session_state.selected_video_title = row["title"]
                                st.session_state.selected_video_duration = row.get("duration_seconds", 0)
                                st.session_state.selected_video_publish_at = row["published_at"]
                                st.session_state.page = "details"
                                st.stop()
                    else:
                        st.empty()

def show_details_page():
    video_id = st.session_state.get("selected_video_id")
    video_title = st.session_state.get("selected_video_title")
    total_duration = st.session_state.get("selected_video_duration", 0)
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
    comments_key = f"comments_{video_id}"
    if comments_key not in st.session_state:
        st.session_state[comments_key] = []  # Replace with actual comment fetching.
    comments = st.session_state[comments_key]
    if comments:
        st.write(f"Total Comments Fetched: {len(comments)}")
        top_5 = sorted(comments, key=lambda c: c["likeCount"], reverse=True)[:5]
        st.write("**Top 5 Comments (by Likes):**")
        for c in top_5:
            st.markdown(f"**{c['likeCount']} likes** - {c['text']}", unsafe_allow_html=True)
        with st.spinner("Analyzing comments with GPT..."):
            analysis = analyze_comments(comments)
        st.write("**Comments Analysis (Positive, Negative, Suggested Topics):**")
        st.write(analysis)
    else:
        st.write("No comments available for this video.")

    st.subheader("Script")
    transcript, source = get_transcript_with_fallback(video_id)
    is_short = ("shorts" in video_url.lower()) or (total_duration < 60)
    if is_short:
        if transcript:
            script_text = " ".join([seg["text"] for seg in transcript])
            st.markdown("**Full Script:**")
            st.write(script_text)
            with st.spinner("Summarizing short's script with GPT..."):
                short_summary = summarize_script(script_text)
            st.subheader("Script Summary")
            st.write(short_summary)
        else:
            st.info("Transcript unavailable for this short.")
    else:
        with st.spinner("Fetching intro & outro script..."):
            intro_txt, outro_txt = get_intro_outro_transcript(video_id, total_duration)
        st.markdown("**Intro Script (First 1 minute):**")
        st.write(intro_txt if intro_txt else "*No intro script available.*")
        st.markdown("**Outro Script (Last 1 minute):**")
        st.write(outro_txt if (outro_txt and total_duration > 120) else "*No outro script available or video is too short.*")
        with st.spinner("Summarizing intro & outro script with GPT..."):
            intro_summary, outro_summary = summarize_intro_outro(intro_txt or "", outro_txt or "")
        st.subheader("Intro & Outro Summaries")
        st.write(intro_summary if intro_summary else "*No summary available.*")

    st.subheader("Retention Analysis")
    if is_short:
        st.info("Retention analysis is available only for full-length videos. This appears to be a short video.")
    else:
        if published_at:
            try:
                published_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - published_dt) < timedelta(days=3):
                    st.info("Retention analysis is available only for videos posted at least 3 days ago.")
                else:
                    if st.button("Run Retention Analysis"):
                        try:
                            base_timestamp = total_duration if total_duration and total_duration > 0 else 120
                            st.info(f"Using base timestamp: {base_timestamp:.2f} sec (from video duration)")
                            snippet_duration = st.number_input("Snippet Duration for Peaks (sec):", value=5, min_value=3, max_value=30, step=1)
                            with st.spinner("Capturing retention screenshot..."):
                                screenshot_path = "player_retention.png"
                                duration = capture_player_screenshot_with_hover(video_url, timestamp=base_timestamp, output_path=screenshot_path, use_cookies=True)
                            if os.path.exists(screenshot_path):
                                st.image(screenshot_path, caption="Player Screenshot for Retention Analysis")
                            else:
                                st.error("Retention screenshot not found.")
                                return
                            with st.spinner("Analyzing retention peaks..."):
                                peaks, roi, binary_roi, roi_width, col_sums = detect_retention_peaks(
                                    screenshot_path, crop_ratio=0.2, height_threshold=200, distance=20, top_n=5
                                )
                            st.write(f"Detected {len(peaks)} retention peak(s): {peaks}")
                            buf = plot_brightness_profile(col_sums, peaks)
                            st.image(buf, caption="Brightness Profile with Detected Peaks")
                            for idx, peak_x in enumerate(peaks):
                                peak_time = (peak_x / roi_width) * duration
                                st.markdown(f"**Retention Peak {idx+1} at {peak_time:.2f} sec**")
                                peak_frame_path = f"peak_frame_retention_{idx+1}.png"
                                with st.spinner(f"Capturing frame for retention peak {idx+1}..."):
                                    capture_frame_at_time(video_url, target_time=peak_time, output_path=peak_frame_path, use_cookies=True)
                                if os.path.exists(peak_frame_path):
                                    st.image(peak_frame_path, caption=f"Frame at {peak_time:.2f} sec")
                                else:
                                    st.error(f"Failed to capture frame for retention peak {idx+1}")
                                if check_ytdlp_installed():
                                    adjusted_start_time = max(0, peak_time - snippet_duration / 2)
                                    snippet_path = f"peak_snippet_{idx+1}.mp4"
                                    with st.spinner(f"Downloading video snippet for retention peak {idx+1}..."):
                                        download_video_snippet(video_url, start_time=adjusted_start_time, duration=snippet_duration, output_path=snippet_path)
                                    if os.path.exists(snippet_path) and os.path.getsize(snippet_path) > 0:
                                        st.write(f"Video Snippet for Peak {idx+1} (±{snippet_duration/2} sec)")
                                        st.video(snippet_path)
                                    else:
                                        st.error(f"Video snippet download failed for retention peak {idx+1}.")
                                else:
                                    st.error("yt-dlp is not installed. Cannot download video snippet.")
                                if transcript:
                                    snippet_text = filter_transcript(transcript, target_time=peak_time, window=5)
                                    if snippet_text:
                                        st.markdown(f"**Transcript Snippet:** {snippet_text}")
                                    else:
                                        st.write("No transcript snippet found for this peak.")
                        except Exception as e:
                            st.error(f"Error during retention analysis: {e}")
            except Exception as e:
                logger.error(f"Error parsing published_at: {e}")
        else:
            st.info("No published date available; cannot determine retention analysis eligibility.")
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
        logger.error(f"Unexpected error in main UI: {e}")
        st.error("An unexpected error occurred. Please check the logs for details.")
    atexit.register(lambda: logger.info("Application shutting down"))
