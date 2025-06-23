import streamlit as st
import sys
import pdf
import video
from pathlib import Path

# Add pdf and video directories to sys.path so we can import their modules
dirs = [str(Path(__file__).parent / "pdf"), str(Path(__file__).parent / "video")]
for d in dirs:
    if d not in sys.path:
        sys.path.insert(0, d)

import pdf
import video

st.set_page_config(page_title="Unified Workspace", layout="wide")

st.sidebar.title("Workspace")
page = st.sidebar.radio("Go to", ("PDF Reader", "Video to Notes"))

if page == "PDF Reader":
    pdf.render_pdf_reader()
elif page == "Video to Notes":
    video.render_video_to_notes()
