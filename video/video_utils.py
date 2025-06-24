import streamlit as st
import yt_dlp
import whisper
import os
from openai import OpenAI

# --- UI ---
st.title("YouTube Video Note-Taker")
st.write("Enter a YouTube video link, and get clean, structured notes.")
st.write("After reviewing the notes, you can generate a quiz to test your understanding of the video.")
st.markdown("""
**If you encounter a 'Sign in to confirm you’re not a bot' error, please upload your YouTube cookies.txt file exported from your browser using the [Get cookies.txt](https://chrome.google.com/webstore/detail/get-cookiestxt/) extension.**
""")
cookies_file = st.file_uploader("Upload cookies.txt (optional, for protected/blocked videos)", type=["txt"])

youtube_url = st.text_input("Enter YouTube URL:")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if youtube_url and api_key:
    if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        st.error("Please enter a valid YouTube URL (must start with http:// or https://)")
    else:
        client = OpenAI(api_key=api_key)
        model = whisper.load_model("base")
        transcript = None
        notes = None
        audio_path = "audio.mp4"
        try:
            # Download audio using yt-dlp
            with st.spinner("Downloading audio with yt-dlp..."):
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': audio_path,
                    'noplaylist': True,
                    'quiet': True,
                }
                # If user uploaded a cookies.txt file, save it and pass to yt-dlp
                if cookies_file is not None:
                    cookies_path = "cookies.txt"
                    with open(cookies_path, "wb") as f:
                        f.write(cookies_file.read())
                    ydl_opts['cookiefile'] = cookies_path
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(youtube_url, download=True)
                if not os.path.exists(audio_path):
                    raise Exception("Audio download failed.")

            # Transcribe with Whisper
            with st.spinner("Transcribing with Whisper..."):
                result = model.transcribe(audio_path)
                transcript = result["text"]

            st.subheader("Raw Transcript")
            st.text_area("Transcript:", transcript, height=300)

            # Generate notes using OpenAI GPT
            with st.spinner("Generating notes with GPT..."):
                prompt = f"""
You are a helpful assistant that summarizes YouTube videos.
Create structured notes from the following transcript.
Include:
- Section headings
- Bullet points of key ideas
- Highlight important facts

Transcript:
{transcript}
"""
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                notes = response.choices[0].message.content

            st.subheader("Notes")
            st.markdown(notes)
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Clean up the audio file if it exists
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
