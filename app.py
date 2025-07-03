import os
import uuid
import asyncio
from flask import Flask, render_template, request, send_from_directory
import whisper
from googletrans import Translator
from pydub import AudioSegment
import edge_tts

# Ensure FFmpeg is in PATH
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin"

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Your language codes for translation
LANGUAGES = {
    'Hindi': 'hi',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Bengali': 'bn',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Punjabi': 'pa',
    'Urdu': 'ur',
}

# Mapping for Edge TTS voices.
# For some languages, there might not be a dedicated neural voice.
# Here we map to a similar voice (or fallback to Hindi) where needed.
EDGE_TTS_VOICES = {
    'hi': "hi-IN-SwaraNeural",
    'mr': "hi-IN-SwaraNeural",  # No native Marathi voice; fallback to Hindi
    'ta': "ta-IN-SaranyaNeural",
    'te': "te-IN-SaiNeural",
    'gu': "gu-IN-PrabhatNeural",
    'bn': "bn-IN-NabanitaNeural",
    'kn': "hi-IN-SwaraNeural",  # Fallback
    'ml': "hi-IN-SwaraNeural",  # Fallback
    'pa': "hi-IN-SwaraNeural",  # Fallback
    'ur': "hi-IN-SwaraNeural",  # Fallback
}

def synthesize_text(text, output_file, voice):
    """Synthesize text to speech using edge-tts and save to output_file."""
    # This wrapper runs the async function in a synchronous way.
    asyncio.run(_synthesize_text(text, output_file, voice))

async def _synthesize_text(text, output_file, voice):
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(output_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        language = request.form['language']
        lang_code = LANGUAGES[language]
        voice = EDGE_TTS_VOICES.get(lang_code, "hi-IN-SwaraNeural")
        filename = str(uuid.uuid4()) + ".mp4"
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(video_path)

        # Extract audio from video using FFmpeg
        audio_path = video_path.replace('.mp4', '.wav')
        os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y')

        # Transcribe with Whisper and get segments with timestamps
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, verbose=True)
        segments = result.get("segments", [])
        if not segments:
            segments = [{"start": 0.0, "end": result.get("duration", 0), "text": result.get("text", "")}]

        # Initialize translator
        translator = Translator()

        # Prepare to collect synthesized audio segments
        final_audio = AudioSegment.empty()

        # Process each segment individually
        for i, seg in enumerate(segments):
            seg_text = seg["text"].strip()
            if not seg_text:
                continue

            # Translate the segment text
            seg_translated = translator.translate(seg_text, dest=lang_code).text

            # Synthesize this segment's translated text using Edge TTS
            seg_audio_file = os.path.join(OUTPUT_FOLDER, f"segment_{i}.mp3")
            synthesize_text(seg_translated, seg_audio_file, voice)

            # Load the synthesized audio
            seg_audio = AudioSegment.from_file(seg_audio_file)
            # Calculate expected duration for this segment (in milliseconds)
            expected_duration = int((seg["end"] - seg["start"]) * 1000)

            # Adjust segment audio duration:
            # Always add a pause after the segment
            pause_duration = 300  # 300ms pause between segments
            pause = AudioSegment.silent(duration=pause_duration)
            seg_audio = seg_audio + pause

            # Append this segment to the final audio
            final_audio += seg_audio

        # Save the final combined audio
        final_audio_file = os.path.join(OUTPUT_FOLDER, f"final_{filename.replace('.mp4', '.mp3')}")
        final_audio.export(final_audio_file, format="mp3")

        # Replace the video's audio with the synthesized, timed audio using FFmpeg
        dubbed_path = os.path.join(OUTPUT_FOLDER, f'dubbed_{filename}')
        os.system(f'ffmpeg -i "{video_path}" -i "{final_audio_file}" -map 0:v -map 1:a -c:v copy -shortest "{dubbed_path}" -y')

        # (Optionally, you can combine all the translated segment texts to show in the result page.)
        combined_translation = "\n".join(
            translator.translate(seg["text"].strip(), dest=lang_code).text for seg in segments if seg["text"].strip()
        )

        return render_template('result.html', translated=combined_translation,
                               output_file=os.path.basename(dubbed_path))

    return render_template('index.html', languages=LANGUAGES.keys())

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
