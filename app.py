import os
import uuid
import time
import pandas as pd
from difflib import get_close_matches
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import speech_recognition as sr
from gtts import gTTS
import requests
import json
import subprocess
import threading
import logging
import tempfile
import io
from werkzeug.exceptions import RequestEntityTooLarge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Set maximum file size (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 🔐 Set Gemini API
API_KEY = "AIzaSyAqzYkdcMBQqtVE7JKAYXi7iBBk5R0aW58"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Language configuration mapping frontend language codes to speech recognition and TTS
LANGUAGE_CONFIG = {
    'en': {
        'speech_lang': 'en-US',
        'tts_lang': 'en',
        'display_name': 'English',
        'exit_words': ['exit', 'stop', 'bye', 'goodbye']
    },
    'hi': {
        'speech_lang': 'hi-IN', 
        'tts_lang': 'hi',
        'display_name': 'Hindi',
        'exit_words': ['बंद करो', 'रोको', 'खत्म', 'exit']
    },
    'gu': {
        'speech_lang': 'gu-IN',
        'tts_lang': 'gu', 
        'display_name': 'Gujarati',
        'exit_words': ['બંધ કરો', 'રોકો', 'બસ', 'exit']
    }
}

# Global variable to store custom Q&A data for different languages
CUSTOM_QA = {
    'en': {},
    'hi': {},
    'gu': {}
}

# Language-specific CSV file mapping
CSV_FILES = {
    'en': 'coco.csv',
    'hi': 'coco_hindi.csv',
    'gu': 'coco_gujarati.csv'
}

# Load custom Q&A dataset for a specific language
def load_custom_qa(csv_path, language_code='en'):
    """Load custom Q&A dataset from CSV file for specific language"""
    global CUSTOM_QA
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"Custom Q&A file not found: {csv_path}")
            return {}
            
        df = pd.read_csv(csv_path)
        qa_dict = {}
        
        # Check if required columns exist
        if 'Prompt' not in df.columns or 'Responses' not in df.columns:
            logger.error(f"CSV file {csv_path} must contain 'Prompt' and 'Responses' columns")
            return {}
        
        for idx, row in df.iterrows():
            try:
                question = str(row['Prompt']).strip().lower()
                answer = str(row['Responses']).strip()
                
                # Skip empty entries
                if question and answer and question != 'nan' and answer != 'nan':
                    qa_dict[question] = answer
            except Exception as e:
                logger.warning(f"Error processing row {idx} in {csv_path}: {e}")
                continue
                
        logger.info(f"Loaded {len(qa_dict)} custom Q&A entries from {csv_path} for {LANGUAGE_CONFIG[language_code]['display_name']}")
        CUSTOM_QA[language_code] = qa_dict
        return qa_dict
        
    except Exception as e:
        logger.error(f"Error loading custom Q&A file {csv_path}: {e}")
        return {}

# Load all language-specific Q&A datasets
def load_all_custom_qa():
    """Load Q&A datasets for all supported languages"""
    total_loaded = 0
    for lang_code, csv_file in CSV_FILES.items():
        qa_dict = load_custom_qa(csv_file, lang_code)
        total_loaded += len(qa_dict)
    
    logger.info(f"Total custom Q&A entries loaded across all languages: {total_loaded}")
    return total_loaded

# Function to find a matching custom answer for specific language
def get_custom_answer(user_input, language_code='en'):
    """Find matching answer from custom Q&A dataset for specific language"""
    if language_code not in CUSTOM_QA or not CUSTOM_QA[language_code]:
        return None
        
    try:
        user_input_lower = user_input.strip().lower()
        lang_qa = CUSTOM_QA[language_code]
        
        # First try exact match
        if user_input_lower in lang_qa:
            logger.info(f"Found exact match for: {user_input} in {LANGUAGE_CONFIG[language_code]['display_name']}")
            return lang_qa[user_input_lower]
        
        # Then try fuzzy matching
        matches = get_close_matches(user_input_lower, lang_qa.keys(), n=1, cutoff=0.75)
        if matches:
            matched_question = matches[0]
            logger.info(f"Found fuzzy match in {LANGUAGE_CONFIG[language_code]['display_name']}: '{user_input}' -> '{matched_question}'")
            return lang_qa[matched_question]
            
        return None
        
    except Exception as e:
        logger.error(f"Error in custom answer matching for {language_code}: {e}")
        return None

# Function to reload custom Q&A data for specific language
def reload_custom_qa(language_code=None):
    """Reload custom Q&A data from CSV files"""
    if language_code and language_code in CSV_FILES:
        csv_path = CSV_FILES[language_code]
        return load_custom_qa(csv_path, language_code)
    else:
        # Reload all languages
        return load_all_custom_qa()

@app.route("/", methods=["GET"])
def home():
    """Serve the main HTML page"""
    return render_template("index.html")

@app.route("/health", methods=["GET"])  
def health_check():
    """Health check endpoint"""
    qa_counts = {lang: len(CUSTOM_QA[lang]) for lang in CUSTOM_QA}
    return jsonify({
        "status": "healthy",
        "supported_languages": ["english", "hindi", "gujarati"],
        "custom_qa_entries": qa_counts,
        "total_custom_qa_entries": sum(qa_counts.values()),
        "timestamp": time.time()
    })

@app.route("/reload-qa", methods=["POST"])
def reload_qa():
    """Endpoint to reload custom Q&A data"""
    try:
        language_code = request.json.get('language') if request.json else None
        
        if language_code and language_code in CSV_FILES:
            # Reload specific language
            qa_dict = reload_custom_qa(language_code)
            count = len(qa_dict)
            lang_name = LANGUAGE_CONFIG[language_code]['display_name']
            return jsonify({
                "status": "success",
                "message": f"Reloaded {count} custom Q&A entries for {lang_name}",
                "language": language_code,
                "count": count
            })
        else:
            # Reload all languages
            total_count = reload_custom_qa()
            return jsonify({
                "status": "success",
                "message": f"Reloaded {total_count} custom Q&A entries across all languages",
                "total_count": total_count,
                "counts_by_language": {lang: len(CUSTOM_QA[lang]) for lang in CUSTOM_QA}
            })
    except Exception as e:
        logger.error(f"Error reloading Q&A data: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def speech_to_text(audio_file_path, language_code):
    """Convert speech to text using Google Speech Recognition"""
    recognizer = sr.Recognizer()
    
    # Configuration for better recognition
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        
        # Get language configuration
        lang_config = LANGUAGE_CONFIG.get(language_code, LANGUAGE_CONFIG['en'])
        speech_lang = lang_config['speech_lang']
        
        # Recognize speech in the specified language
        text = recognizer.recognize_google(audio, language=speech_lang)
        logger.info(f"Speech recognized in {lang_config['display_name']}: {text}")
        return text
        
    except sr.UnknownValueError:
        logger.warning("Speech not understood")
        raise Exception("Could not understand audio. Please speak clearly.")
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        raise Exception("Speech recognition service error")
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        raise

def generate_response(user_text, language_code):
    """Generate response using custom Q&A first, then fallback to Gemini API"""
    
    # First check custom Q&A dataset for the specific language
    custom_answer = get_custom_answer(user_text, language_code)
    if custom_answer:
        lang_name = LANGUAGE_CONFIG[language_code]['display_name']
        logger.info(f"Using custom answer from {lang_name} dataset for: {user_text}")
        return custom_answer, True  # Return answer and flag indicating it's from custom dataset
    
    # If no custom answer found, use Gemini API
    logger.info(f"No custom answer found in {LANGUAGE_CONFIG[language_code]['display_name']} dataset, using Gemini API for: {user_text}")
    gemini_response = generate_gemini_response(user_text, language_code)
    return gemini_response, False  # Return answer and flag indicating it's from Gemini

def generate_gemini_response(user_text, language_code):
    """Generate response using Gemini API in specified language"""
    lang_config = LANGUAGE_CONFIG.get(language_code, LANGUAGE_CONFIG['en'])
    language_name = lang_config['display_name']
    
    # Create language-specific prompt
    if language_code == 'hi':
        system_prompt = """You are a helpful AI assistant. You MUST respond only in Hindi using Devanagari script. 
        Do not mix English words. Keep responses conversational, helpful, and under 100 words.
        Avoid emojis and special characters."""
    elif language_code == 'gu':
        system_prompt = """You are a helpful AI assistant. You MUST respond only in Gujarati using Gujarati script.
        Do not mix English words. Keep responses conversational, helpful, and under 100 words.
        Avoid emojis and special characters."""
    else:  # English
        system_prompt = """You are a helpful AI assistant. You MUST respond only in English.
        Keep responses conversational, helpful, and under 100 words.
        Avoid emojis and special characters."""
    
    prompt = f"{system_prompt}\nUser said: {user_text}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        reply = data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        logger.info(f"Gemini response in {language_name}: {reply[:100]}...")
        return reply
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        
        # Fallback responses in respective languages
        fallback_responses = {
            'hi': "मुझे तकनीकी समस्या हो रही है। कृपया फिर से कोशिश करें।",
            'gu': "મને તકનીકી સમસ્યા આવી રહી છે. કૃપા કરીને ફરીથી પ્રયાસ કરો.",
            'en': "I'm experiencing technical difficulties. Please try again."
        }
        return fallback_responses.get(language_code, fallback_responses['en'])

def text_to_speech(text, language_code, output_path):
    """Convert text to speech using gTTS"""
    try:
        lang_config = LANGUAGE_CONFIG.get(language_code, LANGUAGE_CONFIG['en'])
        tts_lang = lang_config['tts_lang']
        
        # Clean text for better TTS
        cleaned_text = text.replace('*', '').replace('#', '').replace('`', '').strip()
        
        if not cleaned_text:
            logger.error("Empty text for TTS")
            return False
        
        # Limit text length
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:500] + "..."
        
        logger.info(f"Generating TTS in {lang_config['display_name']}: {cleaned_text[:50]}...")
        
        # Create gTTS object
        tts = gTTS(text=cleaned_text, lang=tts_lang, slow=False)
        
        # Save to temporary MP3 file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_path = temp_file.name
        
        tts.save(temp_path)
        
        # Convert MP3 to WAV using ffmpeg
        if convert_mp3_to_wav(temp_path, output_path):
            os.remove(temp_path)  # Clean up temp file
            logger.info(f"TTS generated successfully: {output_path}")
            return True
        else:
            logger.error("Failed to convert MP3 to WAV")
            return False
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False

def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', mp3_path,
            '-ar', '22050',  # Sample rate
            '-ac', '1',      # Mono
            '-c:a', 'pcm_s16le',
            wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            return True
        return False
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return False

def convert_audio_format(input_path, output_path):
    """Convert input audio to WAV format for speech recognition"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000',  # 16kHz for speech recognition
            '-ac', '1',      # Mono
            '-c:a', 'pcm_s16le',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Audio conversion successful: {input_path} -> {output_path}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Audio format conversion error: {e}")
        return False

def cleanup_old_files():
    """Remove old recording files"""
    recordings_dir = 'recordings'
    if not os.path.exists(recordings_dir):
        return
        
    current_time = time.time()
    cleaned_count = 0
    
    try:
        for filename in os.listdir(recordings_dir):
            file_path = os.path.join(recordings_dir, filename)
            try:
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
                        cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not clean up file {file_path}: {e}")
                
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old files")
            
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

@app.route("/process", methods=["POST"])
def process_audio():
    """Main endpoint to process voice input and return AI response"""
    start_time = time.time()
    
    # Validate request
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio_data']
    language_code = request.form.get('language', 'en')
    
    if audio_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Validate language code
    if language_code not in LANGUAGE_CONFIG:
        language_code = 'en'  # Default to English
        
    # Create directories
    os.makedirs('recordings', exist_ok=True)
    cleanup_old_files()
    
    unique_id = str(uuid.uuid4())[:8]
    webm_path = os.path.join('recordings', f"input_{unique_id}.webm")
    wav_input_path = os.path.join('recordings', f"input_{unique_id}.wav")
    wav_output_path = os.path.join('recordings', f"output_{unique_id}.wav")

    try:
        # Save uploaded file
        audio_file.save(webm_path)
        logger.info(f"Processing audio in {LANGUAGE_CONFIG[language_code]['display_name']}")
        
        # Convert audio format for speech recognition
        if not convert_audio_format(webm_path, wav_input_path):
            return jsonify({"error": "Failed to convert audio format"}), 500
        
        # Clean up original file
        try:
            os.remove(webm_path)
        except:
            pass

        # Convert speech to text
        try:
            user_text = speech_to_text(wav_input_path, language_code)
            logger.info(f"User said: '{user_text}'")
            
            if not user_text.strip():
                return jsonify({"error": "No speech detected"}), 400
                
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return jsonify({"error": str(e)}), 400

        # Generate AI response (custom Q&A first, then Gemini)
        ai_response, is_custom = generate_response(user_text, language_code)
        response_source = f"Custom Q&A ({LANGUAGE_CONFIG[language_code]['display_name']})" if is_custom else "Gemini API"
        logger.info(f"AI Response from {response_source}: '{ai_response[:100]}...'")

        # Convert response to speech
        if text_to_speech(ai_response, language_code, wav_output_path):
            # Clean up input file
            try:
                os.remove(wav_input_path)
            except:
                pass
            
            processing_time = time.time() - start_time
            logger.info(f"Request processed in {processing_time:.2f} seconds using {response_source}")
            
            # Return audio file with additional headers for debugging
            response = send_file(
                wav_output_path,
                mimetype='audio/wav',
                as_attachment=False,
                download_name=f'response_{unique_id}.wav'
            )
            response.headers['X-Response-Source'] = response_source
            response.headers['X-Processing-Time'] = f"{processing_time:.2f}"
            response.headers['X-Language'] = language_code
            return response
        else:
            return jsonify({"error": "Failed to generate audio response"}), 500

    except RequestEntityTooLarge:
        return jsonify({"error": "Audio file too large"}), 413
    except Exception as e:
        logger.error(f"Processing error: {e}")
        
        # Cleanup on error
        for file_path in [webm_path, wav_input_path, wav_output_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
                
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Background cleanup task
def start_cleanup_scheduler():
    """Start background cleanup task"""
    def cleanup_scheduler():
        while True:
            time.sleep(1800)  # Run every 30 minutes
            try:
                cleanup_old_files()
            except Exception as e:
                logger.error(f"Scheduled cleanup error: {e}")
    
    cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
    cleanup_thread.start()
    logger.info("Background cleanup scheduler started")

if __name__ == "__main__":
    # Ensure recordings directory exists
    os.makedirs('recordings', exist_ok=True)
    
    # Load custom Q&A data for all languages at startup
    total_loaded = load_all_custom_qa()
    
    # Start background cleanup
    start_cleanup_scheduler()
    
    # Log startup information
    logger.info("=== 3D AI Avatar Flask App with Multilingual Custom Q&A Support Starting ===")
    logger.info("Supported Languages: English, Hindi, Gujarati")
    logger.info(f"Total custom Q&A entries loaded: {total_loaded}")
    
    # Log per-language statistics
    for lang_code, qa_dict in CUSTOM_QA.items():
        lang_name = LANGUAGE_CONFIG[lang_code]['display_name']
        csv_file = CSV_FILES[lang_code]
        logger.info(f"  {lang_name} ({csv_file}): {len(qa_dict)} entries")
    
    # Run the Flask app with HTTPS
    cert_file = os.path.abspath("cert.pem")
    key_file = os.path.abspath("key.pem")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=True,
            threaded=True,
            ssl_context=(cert_file, key_file)
        )
    else:
        logger.warning("SSL certificates not found. Running without HTTPS.")
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=True,
            threaded=True
        )