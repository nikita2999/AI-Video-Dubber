import os
import tempfile
import shutil
import uuid
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
import whisper
import edge_tts
import asyncio
from googletrans import Translator
import subprocess
import json
import re
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

translator = Translator()
model = whisper.load_model("base")  # Change to "small" or "medium" for better accuracy

# Gender mapping for major languages - using more natural-sounding voices
GENDER_VOICE_MAP = {
    "en": {"Male": "en-US-DavisNeural", "Female": "en-US-JennyNeural"},
    "hi": {"Male": "hi-IN-MadhurNeural", "Female": "hi-IN-NeerjaNeural"},
    "fr": {"Male": "fr-FR-HenriNeural", "Female": "fr-FR-DeniseNeural"},
    "es": {"Male": "es-ES-AlvaroNeural", "Female": "es-ES-ElviraNeural"},
    "de": {"Male": "de-DE-ConradNeural", "Female": "de-DE-KatjaNeural"},
    "bn": {"Male": "bn-BD-PradeepNeural", "Female": "bn-BD-NabanitaNeural"},
    "ta": {"Male": "ta-IN-ValluvarNeural", "Female": "ta-IN-PallaviNeural"},
    "te": {"Male": "te-IN-MohanNeural", "Female": "te-IN-ShrutiNeural"},
    "ml": {"Male": "ml-IN-MidhunNeural", "Female": "ml-IN-SobhanaNeural"},
    "gu": {"Male": "gu-IN-NiranjanNeural", "Female": "gu-IN-DhwaniNeural"}
}

# List of available Edge TTS voices for each language
EDGE_TTS_VOICES = {
    "en": [
        ("en-US-DavisNeural", "English (US) - Davis (Male)"),
        ("en-US-JennyNeural", "English (US) - Jenny (Female)"),
        ("en-US-SaraNeural", "English (US) - Sara (Female)"),
        ("en-US-AriaNeural", "English (US) - Aria (Female)"),
        ("en-US-GuyNeural", "English (US) - Guy (Male)"),
        ("en-US-AnaNeural", "English (US) - Ana (Female)"),
        ("en-US-AmberNeural", "English (US) - Amber (Female)"),
        ("en-US-AshleyNeural", "English (US) - Ashley (Female)")
    ],
    "hi": [
        ("hi-IN-MadhurNeural", "Hindi - Madhur (Male)"),
        ("hi-IN-SwaraNeural", "Hindi - Swara (Female)"),
        ("hi-IN-NeerjaNeural", "Hindi - Neerja (Female)")
    ],
    "fr": [
        ("fr-FR-HenriNeural", "French - Henri (Male)"),
        ("fr-FR-DeniseNeural", "French - Denise (Female)")
    ],
    "es": [
        ("es-ES-AlvaroNeural", "Spanish - Alvaro (Male)"),
        ("es-ES-ElviraNeural", "Spanish - Elvira (Female)")
    ],
    "de": [
        ("de-DE-ConradNeural", "German - Conrad (Male)"),
        ("de-DE-KatjaNeural", "German - Katja (Female)")
    ],
    "bn": [
        ("bn-BD-PradeepNeural", "Bengali - Pradeep (Male)"),
        ("bn-BD-NabanitaNeural", "Bengali - Nabanita (Female)")
    ],
    "ta": [
        ("ta-IN-ValluvarNeural", "Tamil - Valluvar (Male)"),
        ("ta-IN-PallaviNeural", "Tamil - Pallavi (Female)")
    ],
    "te": [
        ("te-IN-MohanNeural", "Telugu - Mohan (Male)"),
        ("te-IN-ShrutiNeural", "Telugu - Shruti (Female)")
    ],
    "ml": [
        ("ml-IN-MidhunNeural", "Malayalam - Midhun (Male)"),
        ("ml-IN-SobhanaNeural", "Malayalam - Sobhana (Female)")
    ],
    "gu": [
        ("gu-IN-NiranjanNeural", "Gujarati - Niranjan (Male)"),
        ("gu-IN-DhwaniNeural", "Gujarati - Dhwani (Female)")
    ]
}


def detect_speakers_and_genders(audio_path, segments):
    """
    Detect different speakers and their genders using audio analysis.
    This is a simplified approach - for production, use pyannote.audio or similar.
    """
    speakers = []
    current_speaker = 0
    
    for i, segment in enumerate(segments):
        # Simple speaker change detection based on audio characteristics
        # In a real implementation, you'd use a proper speaker diarization model
        
        if i == 0:
            # First speaker
            gender = detect_speaker_gender_from_segment(audio_path, segment['start'], segment['end'])
            speakers.append({
                'speaker_id': current_speaker,
                'gender': gender,
                'segments': [i]
            })
        else:
            # Check if this might be a different speaker
            # Simple heuristic: if there's a significant gap or different audio characteristics
            prev_segment = segments[i-1]
            gap = segment['start'] - prev_segment['end']
            
            # If there's a significant gap (>1 second), it might be a different speaker
            if gap > 1.0:
                current_speaker += 1
                gender = detect_speaker_gender_from_segment(audio_path, segment['start'], segment['end'])
                speakers.append({
                    'speaker_id': current_speaker,
                    'gender': gender,
                    'segments': [i]
                })
            else:
                # Same speaker, add to existing speaker's segments
                speakers[-1]['segments'].append(i)
    
    return speakers

def detect_speaker_gender_from_segment(audio_path, start_time, end_time):
    """
    Detect gender from a specific audio segment using more sophisticated analysis.
    """
    try:
        # Load the audio segment
        audio = AudioSegment.from_wav(audio_path)
        segment_audio = audio[start_time*1000:end_time*1000]
        
        if len(segment_audio) == 0:
            return "Female"  # Default
        
        # Calculate various audio features
        rms = segment_audio.rms
        db = segment_audio.dBFS
        
        # Basic frequency analysis
        samples = segment_audio.get_array_of_samples()
        if len(samples) > 0:
            # Calculate zero crossing rate (indicator of pitch)
            zero_crossings = sum(1 for i in range(1, len(samples)) if samples[i-1] * samples[i] < 0)
            zcr = zero_crossings / len(samples)
            
            # Higher ZCR often indicates higher pitch (female)
            # Higher RMS often indicates male voice
            if rms > 1500 and zcr < 0.1:
                return "Male"
            elif rms < 800 and zcr > 0.15:
                return "Female"
            else:
                # Use a combination of features
                if rms > 1200:
                    return "Male"
                else:
                    return "Female"
        
        return "Female"  # Default
    except Exception as e:
        print(f"Error in gender detection: {e}")
        return "Female"  # Default fallback

def preserve_timing_and_voices(audio_path, translated_segments, target_lang, selected_voice=None):
    """
    Generate dubbed audio with natural voices and minimal timing adjustments.
    """
    # Load original audio for timing reference
    original_audio = AudioSegment.from_wav(audio_path)
    
    # Detect speakers and their genders
    speakers = detect_speakers_and_genders(audio_path, translated_segments)
    
    # Create a list to store audio segments
    dubbed_segments = []
    
    for i, segment in enumerate(translated_segments):
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        # Find which speaker this segment belongs to
        speaker_gender = "Female"  # Default
        for speaker in speakers:
            if i in speaker['segments']:
                speaker_gender = speaker['gender']
                break
        
        # Select appropriate voice with fallback
        voice_map = GENDER_VOICE_MAP.get(target_lang, GENDER_VOICE_MAP["en"])
        voice = voice_map.get(speaker_gender, list(voice_map.values())[0])
        
        # Debug voice selection
        print(f"Segment {i}: target_lang={target_lang}, speaker_gender={speaker_gender}, selected_voice={voice}")
        
        # Generate TTS for this segment with natural pacing
        segment_audio_path = os.path.join(UPLOAD_FOLDER, f"segment_{i}.mp3")
        
        # Add retry logic with fallback voice
        max_retries = 3
        for attempt in range(max_retries):
            try:
                asyncio.run(generate_tts(text, voice, segment_audio_path))
                break  # Success, exit retry loop
            except Exception as e:
                print(f"TTS attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    # Try with a fallback voice
                    fallback_voice = "en-US-JennyNeural"  # Universal fallback
                    print(f"Trying fallback voice: {fallback_voice}")
                    try:
                        asyncio.run(generate_tts(text, fallback_voice, segment_audio_path))
                    except Exception as fallback_error:
                        print(f"Fallback voice also failed: {fallback_error}")
                        raise fallback_error
                else:
                    # Try a different voice from the same language
                    available_voices = EDGE_TTS_VOICES.get(target_lang, EDGE_TTS_VOICES["en"])
                    if available_voices:
                        voice = available_voices[0][0]  # Use first available voice
                        print(f"Trying alternative voice: {voice}")
        
        # Load the generated audio
        dubbed_audio = AudioSegment.from_mp3(segment_audio_path)
        
        # Calculate target duration (preserve original timing)
        target_duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Natural timing adjustment - only trim if significantly longer, don't speed up
        if len(dubbed_audio) > 0:
            # If the dubbed audio is much longer than the original segment, trim it
            if len(dubbed_audio) > target_duration * 1.5:  # 50% longer
                dubbed_audio = dubbed_audio[:target_duration]
            # If it's shorter, add silence at the end to match timing
            elif len(dubbed_audio) < target_duration:
                silence_duration = target_duration - len(dubbed_audio)
                silence = AudioSegment.silent(duration=silence_duration)
                dubbed_audio = dubbed_audio + silence
            # If it's within reasonable range, keep it as is (no speed changes)
        
        dubbed_segments.append({
            'audio': dubbed_audio,
            'start': start_time,
            'end': end_time
        })
        
        # Clean up temporary file
        os.remove(segment_audio_path)
    
    # Combine all segments in order
    final_audio = AudioSegment.silent(duration=len(original_audio))
    
    for segment in dubbed_segments:
        start_ms = segment['start'] * 1000
        audio = segment['audio']
        
        # Overlay the dubbed audio at the correct position
        final_audio = final_audio.overlay(audio, position=start_ms)
    
    return final_audio

@app.route("/", methods=["GET", "POST"])
def index():
    languages = {
        "Hindi": "hi",
        "Bengali": "bn",
        "Tamil": "ta",
        "Telugu": "te",
        "Gujarati": "gu",
        "Malayalam": "ml",
        "English": "en",
        "French": "fr",
        "German": "de",
        "Spanish": "es"
    }
    
    if request.method == "POST":
        file = request.files["video"]
        lang_code = request.form["language"]
        selected_voice = request.form.get("voice")

        if not file or not lang_code:
            return "Missing file or language."

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Extract original audio
        video = VideoFileClip(file_path)
        audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_original.wav")
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)

        # Transcribe with timestamps
        result = model.transcribe(audio_path, word_timestamps=True)
        
        # Extract segments with timing information
        segments = []
        for segment in result["segments"]:
            if segment["text"].strip():  # Only include non-empty segments
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
        
        # Translate all segments
        translated_segments = []
        for segment in segments:
            translated = translator.translate(segment['text'], dest=lang_code)
            translated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': translated.text
            })
        
        # Generate dubbed audio with timing preservation and voice switching
        dubbed_audio = preserve_timing_and_voices(audio_path, translated_segments, lang_code, selected_voice)
        
        # Save the dubbed audio
        dubbed_audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_dubbed.wav")
        dubbed_audio.export(dubbed_audio_path, format="wav")
        
        # Merge dubbed audio with original video
        output_video_path = os.path.join(UPLOAD_FOLDER, f"dubbed_{uuid.uuid4()}.mp4")
        subprocess.call([
            "ffmpeg", "-y", "-i", file_path, "-i", dubbed_audio_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_video_path
        ])

        # Return JSON response with download link
        return jsonify({
            "success": True,
            "message": "Video dubbing completed successfully!",
            "download_url": f"/download/{os.path.basename(output_video_path)}",
            "filename": os.path.basename(output_video_path)
        })

    return render_template("index.html", languages=languages, edge_voices=EDGE_TTS_VOICES)

@app.route("/download/<filename>")
def download_file(filename):
    """Download a dubbed video file"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

async def generate_tts(text, voice, output_path):
    # Add debugging information
    print(f"Generating TTS for text: '{text[:50]}...' with voice: {voice}")
    
    # Validate text and voice
    if not text or not text.strip():
        print("Warning: Empty text provided to TTS")
        return
    
    if not voice:
        print("Warning: No voice provided to TTS")
        return
    
    try:
        # Use more natural speech settings
        communicate = edge_tts.Communicate(
            text, 
            voice,
            rate="+0%",  # Normal speed
            volume="+0%",  # Normal volume
            pitch="+0Hz"   # Normal pitch (must be in Hz format)
        )
        await communicate.save(output_path)
        print(f"TTS generated successfully: {output_path}")
    except Exception as e:
        print(f"TTS generation failed: {e}")
        print(f"Text: '{text}'")
        print(f"Voice: {voice}")
        
        # Try with a simpler configuration
        try:
            print("Trying with simplified TTS configuration...")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            print(f"TTS generated successfully with simplified config: {output_path}")
        except Exception as simple_error:
            print(f"Simplified TTS also failed: {simple_error}")
            raise e  # Re-raise the original error

@app.route("/languages")
def get_languages():
    return {
        "languages": {
            "Hindi": "hi",
            "Bengali": "bn",
            "Tamil": "ta",
            "Telugu": "te",
            "Gujarati": "gu",
            "Malayalam": "ml",
            "English": "en",
            "French": "fr",
            "German": "de",
            "Spanish": "es"
        }
    }

if __name__ == "__main__":
    app.run(debug=True)
