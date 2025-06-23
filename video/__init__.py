import streamlit as st
import yt_dlp
import whisper
import os
from openai import OpenAI

def render_video_to_notes():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Lato:wght@400;700&display=swap');
        """ + open("video/style.css").read() + """
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("YouTube Video Note-Taker")
    st.write("Enter a YouTube video link, and get clean, structured notes.")
    st.markdown("""
    **If you encounter a 'Sign in to confirm you’re not a bot' error, please upload your YouTube cookies.txt file exported from your browser using the [Get cookies.txt](https://chrome.google.com/webstore/detail/get-cookiestxt/) extension.**
    """)
    cookies_file = st.file_uploader("Upload cookies.txt (optional, for protected/blocked videos)", type=["txt"], key="video_cookies")
    youtube_url = st.text_input("Enter YouTube URL:", key="video_url")
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="video_api_key")
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

                # --- Quiz Feature ---
                import json, re
                st.header("Test your Understanding")
                if st.button("Generate Quiz Questions", key="video_generate_quiz"):
                    with st.spinner("Generating quiz questions using AI..."):
                        quiz_prompt = (
                            "Create a quiz of 5 multiple-choice questions based on the following transcript. "
                            "For each question, provide 4 options (A, B, C, D), indicate the correct answer, and provide a brief explanation for the correct answer. "
                            "Return ONLY valid JSON, no explanation, no markdown, no preamble. "
                            "Format: [ { 'question': '...', 'options': ['A ...', 'B ...', 'C ...', 'D ...'], 'answer': 'A', 'explanation': '...'}, ... ]\n\nTranscript:"
                            + transcript
                        )
                        quiz_response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": quiz_prompt}],
                            temperature=0.7
                        )
                        quiz_text = quiz_response.choices[0].message.content
                        def extract_json(text):
                            match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
                            if match:
                                return match.group(1).strip()
                            match = re.search(r'(\[.*\])', text, re.DOTALL)
                            if match:
                                return match.group(1).strip()
                            return text.strip()
                        quiz_text_clean = extract_json(quiz_text)
                        try:
                            quiz_data = json.loads(quiz_text_clean)
                            st.session_state['video_quiz_data'] = quiz_data
                            st.session_state['video_quiz_submitted'] = False
                        except Exception:
                            st.error("Could not parse quiz questions. Here is the raw output. Please try again:")
                            st.markdown(f"<pre>{quiz_text}</pre>", unsafe_allow_html=True)
                            st.session_state['video_quiz_data'] = None
                            st.session_state['video_quiz_submitted'] = False
                quiz_data = st.session_state.get('video_quiz_data')
                quiz_submitted = st.session_state.get('video_quiz_submitted', False)
                if quiz_data and not quiz_submitted:
                    with st.form("video_quiz_form"):
                        user_answers = []
                        for idx, q in enumerate(quiz_data):
                            st.write(f"**Q{idx+1}: {q['question']}**")
                            options = q['options']
                            user_choice = st.radio("Select an answer:", options, key=f"video_quiz_{idx}")
                            user_answers.append(user_choice)
                        submitted = st.form_submit_button("Submit Quiz")
                        if submitted:
                            st.session_state['video_quiz_user_answers'] = user_answers
                            st.session_state['video_quiz_submitted'] = True
                            st.experimental_rerun()
                # Show feedback after submit
                if quiz_data and st.session_state.get('video_quiz_submitted') and st.session_state.get('video_quiz_user_answers'):
                    score = 0
                    user_answers = st.session_state['video_quiz_user_answers']
                    for idx, q in enumerate(quiz_data):
                        correct_letter = q.get('answer')
                        correct_option = [opt for opt in q['options'] if opt.startswith(correct_letter)][0] if correct_letter else None
                        explanation = q.get('explanation', 'No explanation provided.')
                        user_option = user_answers[idx]
                        if user_option.startswith(correct_letter):
                            st.success(f"Q{idx+1}: Correct!")
                            score += 1
                        else:
                            st.error(f"Q{idx+1}: Incorrect. Correct answer: {correct_option}")
                        st.info(f"Explanation: {explanation}")
                    st.info(f"Your score: {score} out of {len(quiz_data)}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Clean up the audio file if it exists
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass
