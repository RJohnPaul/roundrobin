meeting_transcript='''So, just to kick things off, I wanted to mention that the UI overhaul is finally complete on staging. We’re now just waiting for feedback from the design team before pushing it to production.

Right, and we also need to double-check how the new color scheme looks on smaller screens. Someone mentioned it was a bit off on mobile last time.

Yeah, I noticed that too—especially on iPhone SE and similar compact viewports. We might need to tweak the padding a bit on those cards.

That reminds me, what's the timeline for the analytics dashboard? Weren’t we aiming to get it done by next Friday?

We were, but there’s been a slight delay due to the API integration. The data’s not coming through as expected from the CRM endpoint.

We could mock the data for now, just to unblock the frontend work. Then plug in the actual feed once it's fixed.

That’s a good idea. Also, about the feedback we got on the onboarding flow—should we start implementing those suggestions now or wait for the next sprint?

Honestly, some of them are pretty quick fixes. Like changing the tooltip text and swapping a few icons. I think we can slide them into this sprint without too much disruption.

Agreed. The bigger requests—like the multi-step walkthrough—can be scoped for the next cycle. But let’s patch what we can this week.

Before we wrap, one more thing: the export-to-PDF feature. Who’s taking that on?

I think it was unassigned. But it makes sense to pair it with the report summary task—there’s overlap.

Perfect. Let’s make sure we add it to the board. Alright, anything else before we close?

Nope, that’s all from my side.

Sounds good—thanks, everyone.'''
from flask import Flask, render_template, request, jsonify
import re
import numpy as np
from numpy.linalg import norm
from numpy import dot
from groq import Groq
import os
import tempfile
import threading
from werkzeug.utils import secure_filename
import faster_whisper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Load API key from environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Global progress tracking
_transcription_progress = {}
_progress_lock = threading.Lock()

def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def clean_latex_output(text):
    """
    Clean LaTeX output to remove explanatory text and extract only the LaTeX code
    """
    # Remove common explanatory phrases
    explanatory_phrases = [
        r"Here is the LaTeX code.*?:",
        r"Here's the LaTeX.*?:",
        r"Below is the LaTeX.*?:",
        r"The LaTeX code is.*?:",
        r"Note that I used.*",
        r"I used.*",
        r"This.*uses.*",
        r"```latex",
        r"```"
    ]
    
    for phrase in explanatory_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Find the LaTeX document start and end
    lines = text.split('\n')
    latex_lines = []
    in_latex = False
    
    for line in lines:
        line = line.strip()
        
        # Start capturing when we find \documentclass
        if '\\documentclass' in line:
            in_latex = True
            latex_lines.append(line)
        elif in_latex:
            latex_lines.append(line)
            # Stop capturing after \end{document}
            if '\\end{document}' in line:
                break
    
    result = '\n'.join(latex_lines).strip()
    
    # If we couldn't extract properly, return the cleaned text
    if not result or '\\documentclass' not in result:
        # Remove leading/trailing non-LaTeX text
        text = text.strip()
        # Remove everything before \documentclass
        if '\\documentclass' in text:
            start_idx = text.find('\\documentclass')
            text = text[start_idx:]
        # Remove everything after \end{document}
        if '\\end{document}' in text:
            end_idx = text.find('\\end{document}') + len('\\end{document}')
            text = text[:end_idx]
        result = text
    
    return result

def chunk_text(text, max_words=150):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def cluster_meeting_transcript(transcript):
    sentences = re.split(r'(?<=[.!?]) +', transcript)
    chunk_size = 5
    step_size = 2
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), step_size)]

    embeddings = [np.random.rand(384) for _ in chunks]
    similarities = [
        dot(embeddings[i], embeddings[i + 1]) / (norm(embeddings[i]) * norm(embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    smoothed = []
    window = 3
    for i in range(len(similarities)):
        start, end = max(0, i - window // 2), min(len(similarities), i + window // 2 + 1)
        smoothed.append(np.mean(similarities[start:end]))

    avg, std = np.mean(smoothed), np.std(smoothed)
    threshold = avg - 1.2 * std
    boundaries = [0] + [i + 1 for i, s in enumerate(smoothed) if s < threshold]

    clusters = []
    for i in range(1, len(boundaries)):
        clusters.append(' '.join(chunks[boundaries[i - 1]:boundaries[i]]))
    if boundaries[-1] < len(chunks):
        clusters.append(' '.join(chunks[boundaries[-1]:]))

    return clusters

def generate_summary(text, mode="detailed"):
    """
    Generate summary with API fallback to basic text processing
    """
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    
    if groq_api_key:
        if mode == "detailed":
            prompt = f"""
Please summarize the following segment of a meeting transcript in a clear and detailed way.
Capture all relevant points and insights from the text only:\n\n{text}
"""
        else:
            prompt = f"""
Summarize the following meeting transcript in a concise, high-level way.
Extract only the key topics and insights based on the content:\n\n{text}
"""
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192"
            )
            return remove_think_tags(chat_completion.choices[0].message.content)
        except Exception as e:
            print(f"API error in summary generation: {e}, using fallback")
    
    # Fallback: return the original text with basic formatting
    # In a production system, you could implement more sophisticated text processing here
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if line and len(line) > 3:  # Filter out very short lines
            filtered_lines.append(line)
    
    if mode == "detailed":
        return '\n'.join(filtered_lines)
    else:
        # For high-level mode, take key sentences
        key_lines = []
        for line in filtered_lines[:10]:  # Limit to first 10 meaningful lines
            if any(keyword in line.lower() for keyword in ['discussed', 'decided', 'agreed', 'action', 'next', 'follow']):
                key_lines.append(line)
        
        return '\n'.join(key_lines) if key_lines else '\n'.join(filtered_lines[:5])

def generate_latex_mom(summary_text):
    """
    Enhanced LaTeX generation with fallback to offline generation
    """
    # First try with API if key is available
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    
    if groq_api_key:
        prompt = f"""
You are a LaTeX document generator. Your ONLY task is to output pure LaTeX code for a professional "Minutes of Meeting" document.

CRITICAL REQUIREMENTS:
- Output ONLY LaTeX code - no explanations, no comments, no additional text
- Start with \\documentclass and end with \\end{{document}}
- Do NOT include phrases like "Here is the LaTeX code" or "Note that I used"
- Extract key information from timestamps and content
- Use professional MoM formatting

CONTENT TO PROCESS:
{summary_text}

OUTPUT ONLY THE LATEX CODE:"""
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192"
            )
            result = remove_think_tags(chat_completion.choices[0].message.content).strip()
            
            # Clean the result to extract only LaTeX code
            result = clean_latex_output(result)
            
            # Verify the result looks like LaTeX
            if '\\documentclass' in result and '\\begin{document}' in result:
                return result
        except Exception as e:
            print(f"API error: {e}, falling back to offline generation")
    
    # Fallback to offline generation
    return generate_latex_mom_offline(summary_text)

def generate_latex_mom_offline(summary_text):
    """
    Generate LaTeX MoM without API dependency - offline fallback
    """
    
    # Parse the input text to extract meaningful content
    lines = summary_text.split('\n')
    
    # Clean timestamp patterns and extract content
    content_lines = []
    speakers = set()
    timestamps = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Extract timestamp patterns [X.XXs -> Y.YYs]
        timestamp_match = re.search(r'\[([\d.]+)s\s*->\s*([\d.]+)s\]', line)
        if timestamp_match:
            start_time = float(timestamp_match.group(1))
            end_time = float(timestamp_match.group(2))
            timestamps.append((start_time, end_time))
            # Remove timestamp and get the content
            content = re.sub(r'\[[\d.]+s\s*->\s*[\d.]+s\]', '', line).strip()
            if content:
                content_lines.append(content)
        else:
            content_lines.append(line)
        
        # Extract speaker names (simple pattern: "Name:")
        speaker_match = re.search(r'^([A-Za-z\s]+):', line)
        if speaker_match:
            speakers.add(speaker_match.group(1).strip())
    
    # Determine meeting metadata
    meeting_duration = ""
    if timestamps:
        total_duration = max([end for _, end in timestamps])
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        meeting_duration = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    # Generate LaTeX document
    latex_content = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{fancyhdr}}
\\usepackage{{enumitem}}
\\usepackage{{xcolor}}
\\usepackage{{titlesec}}
\\usepackage{{amsmath}}

% Header and footer setup
\\pagestyle{{fancy}}
\\fancyhf{{}}
\\fancyhead[L]{{\\textbf{{Minutes of Meeting}}}}
\\fancyhead[R]{{\\today}}
\\fancyfoot[C]{{\\thepage}}

% Section formatting
\\titleformat{{\\section}}{{\\large\\bfseries\\color{{blue!70!black}}}}{{\\thesection}}{{1em}}{{}}
\\titleformat{{\\subsection}}{{\\normalsize\\bfseries}}{{\\thesubsection}}{{1em}}{{}}

\\title{{\\textbf{{Minutes of Meeting}}}}
\\author{{Generated by AudioMoM}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Meeting Information}}
\\begin{{itemize}}
    \\item \\textbf{{Date:}} \\today
    \\item \\textbf{{Duration:}} {meeting_duration if meeting_duration else "Not specified"}
    \\item \\textbf{{Meeting Type:}} Recorded Session
\\end{{itemize}}

"""

    # Add attendees if identified
    if speakers:
        latex_content += """\\section{Attendees}
\\begin{itemize}
"""
        for speaker in sorted(speakers):
            clean_speaker = speaker.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$')
            latex_content += f"    \\item {clean_speaker}\n"
        latex_content += "\\end{itemize}\n\n"

    # Add meeting content
    latex_content += """\\section{Meeting Content}
"""
    
    # Group content into meaningful sections
    current_section = []
    sections = []
    
    for content in content_lines:
        if not content:
            continue
            
        # Clean content for LaTeX
        clean_content = content.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#')
        
        # Check if this looks like a new topic/section
        if any(keyword in content.lower() for keyword in ['agenda', 'first', 'next', 'second', 'third', 'finally', 'before we', "let's"]):
            if current_section:
                sections.append(current_section)
                current_section = []
        
        current_section.append(clean_content)
    
    if current_section:
        sections.append(current_section)
    
    # Output sections or all content
    if len(sections) > 1:
        for i, section in enumerate(sections, 1):
            latex_content += f"\\subsection{{Discussion Point {i}}}\n"
            for content in section:
                latex_content += f"{content}\n\n"
    else:
        # Single section with all content
        latex_content += "\\begin{itemize}\n"
        for content in content_lines:
            if content.strip():
                clean_content = content.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#')
                latex_content += f"    \\item {clean_content}\n"
        latex_content += "\\end{itemize}\n\n"

    # Add standard closing sections
    latex_content += """\\section{Summary}
This meeting covered the key topics outlined in the discussion points above."""

    if timestamps:
        latex_content += f" The session lasted approximately {meeting_duration}."

    latex_content += """

\\section{Next Steps}
\\begin{itemize}
    \\item Review and approve these minutes
    \\item Follow up on any action items discussed
    \\item Schedule next meeting if required
\\end{itemize}

\\end{document}"""

    return latex_content

ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

# Global model cache to avoid reloading
_model_cache = {}
_model_lock = threading.Lock()

def get_cached_model(model_size="tiny", device="cpu", compute_type="float32"):
    """
    Get or create a cached model instance to avoid repeated loading
    Enhanced for maximum performance with tiny model as default
    """
    cache_key = f"{model_size}_{device}_{compute_type}"
    
    with _model_lock:
        if cache_key not in _model_cache:
            print(f"Loading Whisper model: {model_size} (first time)")
            _model_cache[cache_key] = faster_whisper.WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=None,
                cpu_threads=2,  # Reduced for faster startup
                num_workers=1   # Single worker for stability
            )
            print("Model loaded and cached successfully")
        else:
            print(f"Using cached Whisper model: {model_size}")
        
        return _model_cache[cache_key]

def transcribe_audio_file(file_path, model_size="tiny", language=None, include_timestamps=True, speaker_identification=False, session_id=None):
    """
    Optimized transcribe audio file using faster-whisper with caching and performance improvements
    Enhanced with real-time progress tracking
    """
    import threading
    import time
    result_container = {'result': None, 'exception': None}
    
    def update_progress(progress, status):
        """Update progress for real-time tracking"""
        if session_id:
            with _progress_lock:
                if session_id in _transcription_progress:
                    _transcription_progress[session_id]['progress'] = progress
                    _transcription_progress[session_id]['status'] = status
    
    def transcribe_with_timeout():
        try:
            print(f"Starting optimized transcription with model: {model_size}")
            update_progress(5, f"Loading {model_size} model...")
            
            # Use cached model to avoid reloading
            model = get_cached_model(model_size, "cpu", "float32")
            update_progress(15, "Model loaded, preparing audio...")
            
            # Set language to None for auto-detection if 'auto' is specified
            transcribe_language = None if language == 'auto' else language
            
            print(f"Starting transcription of file: {file_path}")
            print(f"Language: {transcribe_language}, Timestamps: {include_timestamps}")
            update_progress(25, "Starting transcription...")
            
            # Optimized transcription parameters for maximum speed
            segments, info = model.transcribe(
                file_path, 
                beam_size=1,  # Minimal beam size for speed
                language=transcribe_language,
                word_timestamps=False,  # Disable word timestamps for speed
                vad_filter=True,  # Enable VAD filtering
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Larger chunks for speed
                    speech_pad_ms=50  # Minimal padding
                ),
                condition_on_previous_text=False,  # Disable for speed
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.8,  # Higher threshold for speed
                temperature=0,  # Deterministic, fastest
                patience=1.0,  # Minimal patience
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                prompt_reset_on_temperature=0.5,
                without_timestamps=not include_timestamps,  # Skip timestamps if not needed
                initial_prompt=None  # No initial prompt for speed
            )
            
            print(f"Transcription completed. Detected language: {info.language}")
            update_progress(70, "Processing transcription segments...")
            
            # Fast formatting without excessive logging
            transcription_lines = []
            segment_count = 0
            
            for segment in segments:
                segment_count += 1
                if include_timestamps:
                    transcription_lines.append(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                else:
                    transcription_lines.append(segment.text)
                
                # Update progress periodically
                if segment_count % 10 == 0:
                    progress = min(70 + (segment_count * 20 / max(segment_count, 100)), 90)
                    update_progress(progress, f"Processed {segment_count} segments...")
            
            transcription_text = "\n".join(transcription_lines)
            
            print(f"Transcription formatting completed. Total segments: {segment_count}")
            update_progress(95, "Finalizing transcription...")
            
            result_container['result'] = {
                'success': True,
                'transcription': transcription_text,
                'language': info.language,
                'language_probability': info.language_probability
            }
            
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            update_progress(0, f"Error: {str(e)}")
            result_container['exception'] = e
    
    try:
        # Start transcription in a separate thread with timeout
        thread = threading.Thread(target=transcribe_with_timeout)
        thread.daemon = True
        thread.start()
        
        # Wait for completion with timeout (5 minutes)
        timeout_seconds = 300
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print("Transcription timed out")
            return {
                'success': False,
                'error': f'Transcription timed out after {timeout_seconds} seconds. Try using a smaller model or shorter audio file.'
            }
        
        if result_container['exception']:
            raise result_container['exception']
        
        if result_container['result']:
            return result_container['result']
        else:
            return {
                'success': False,
                'error': 'Transcription completed but no result was returned'
            }
        
    except Exception as e:
        print(f"Transcription failed: {str(e)}")
        return {
            'success': False,
            'error': f'Transcription failed: {str(e)}'
        }

@app.route("/", methods=["GET", "POST"])
def index():
    latex_code = ""
    transcript = ""

    if request.method == "POST":
        transcript = request.form.get("transcript", "")
        if transcript.strip():
            clusters = cluster_meeting_transcript(transcript)
            cluster_summaries = []

            for cluster in clusters:
                if len(cluster.split()) > 300:
                    chunks = chunk_text(cluster)
                    summaries = [generate_summary(c) for c in chunks]
                    cluster_summaries.append(" ".join(summaries))
                else:
                    cluster_summaries.append(generate_summary(cluster))

            combined_summary = " ".join(cluster_summaries)
            final_summary = generate_summary(combined_summary, mode="highlevel")
            latex_code = generate_latex_mom(final_summary)

    return render_template("index.html", latex_code=latex_code, transcript=transcript)

@app.route("/generate-mom", methods=["POST"])
def generate_mom():
    try:
        transcript = request.form.get("transcript", "")
        if not transcript.strip():
            return jsonify({'error': 'No transcript provided'}), 400
            
        clusters = cluster_meeting_transcript(transcript)
        cluster_summaries = []

        for cluster in clusters:
            if len(cluster.split()) > 300:
                chunks = chunk_text(cluster)
                summaries = [generate_summary(c) for c in chunks]
                cluster_summaries.append(" ".join(summaries))
            else:
                cluster_summaries.append(generate_summary(cluster))

        combined_summary = " ".join(cluster_summaries)
        final_summary = generate_summary(combined_summary, mode="highlevel")
        latex_code = generate_latex_mom(final_summary)
        
        return jsonify({'latex_code': latex_code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio():
    try:
        print(f"Received transcribe-audio request")
        print(f"Files in request: {list(request.files.keys())}")
        print(f"Form data: {dict(request.form)}")
          # Check if file was uploaded
        if 'audio_file' not in request.files:
            print("Error: No audio file provided")
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio_file']
        print(f"File received: {file.filename}")
        print(f"File content type: {file.content_type}")
        print(f"File stream position: {file.stream.tell()}")
        
        # Reset stream position to beginning
        file.stream.seek(0)
        
        if file.filename == '':
            print("Error: No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print(f"Error: Invalid file type for {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload MP3, WAV, M4A, OGG, FLAC, or AAC files.'}), 400
          
        # Get settings from form
        language = request.form.get('language', 'auto')
        include_timestamps = request.form.get('include_timestamps', 'true').lower() == 'true'
        speaker_identification = request.form.get('speaker_identification', 'false').lower() == 'true'
        model_size = request.form.get('model', 'base')
        
        print(f"Settings: language={language}, timestamps={include_timestamps}, speaker_id={speaker_identification}, model={model_size}")
          
        # Disable speaker identification for now due to complexity and hanging issues
        if speaker_identification:
            print("Warning: Speaker identification is currently disabled due to stability issues")
            speaker_identification = False
            
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        print(f"Saving file to: {temp_path}")
        file.save(temp_path)
        
        # Verify file was saved correctly
        if not os.path.exists(temp_path):
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        file_size = os.path.getsize(temp_path)
        print(f"File saved successfully, size: {file_size} bytes")
        
        if file_size == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        try:            
            print("Starting transcription...")
            
            # Generate a unique session ID for progress tracking
            import uuid
            session_id = str(uuid.uuid4())
            
            # Initialize progress
            with _progress_lock:
                _transcription_progress[session_id] = {
                    'progress': 0,
                    'status': 'Initializing transcription...',
                    'completed': False,
                    'error': None
                }
            
            # Transcribe the audio with progress tracking
            result = transcribe_audio_file(
                temp_path, 
                model_size=model_size,
                language=language,
                include_timestamps=include_timestamps,
                speaker_identification=speaker_identification,
                session_id=session_id
            )
            
            print(f"Transcription result: success={result.get('success')}")
            if result.get('success'):
                transcription_length = len(result.get('transcription', ''))
                print(f"Transcription length: {transcription_length} characters")
                
                # Mark as completed
                with _progress_lock:
                    _transcription_progress[session_id]['completed'] = True
                    _transcription_progress[session_id]['progress'] = 100
                    _transcription_progress[session_id]['status'] = 'Transcription completed'
                
                return jsonify({
                    'transcription': result['transcription'],
                    'language': result['language'],
                    'language_probability': result['language_probability'],
                    'session_id': session_id
                })
            else:
                print(f"Transcription failed: {result.get('error')}")
                
                # Mark as failed
                with _progress_lock:
                    _transcription_progress[session_id]['error'] = result.get('error')
                    _transcription_progress[session_id]['status'] = 'Transcription failed'
                
                return jsonify({'error': result['error'], 'session_id': session_id}), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
                
    except Exception as e:
        print(f"Exception in transcribe_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route("/transcription-progress/<session_id>")
def get_transcription_progress(session_id):
    """Get real-time transcription progress"""
    with _progress_lock:
        progress_data = _transcription_progress.get(session_id, {
            'progress': 0,
            'status': 'Session not found',
            'completed': False,
            'error': 'Invalid session ID'
        })
    
    return jsonify(progress_data)

@app.route("/audio-transcription")
def audio_transcription():
    """Route specifically for audio transcription page"""
    return render_template("index.html")

@app.route("/test-upload")
def test_upload():
    """Test upload page"""
    return app.send_static_file('../test_upload.html')

@app.route("/debug-upload")
def debug_upload():
    """Debug upload page"""
    return app.send_static_file('../debug_upload.html')

@app.route('/debug')
def debug_test():
    with open('debug_test.html', 'r') as f:
        return f.read()
    
@app.route('/minimal')
def minimal_test():
    with open('minimal_test.html', 'r') as f:
        return f.read()

@app.route('/simple')
def simple_test():
    with open('simple_test.html', 'r') as f:
        return f.read()

if __name__ == "__main__":
    app.run(debug=True, port=8080, use_reloader=False)
