"""
Flask Web Interface for Multimodal DeepFake Detection System
Updated with thumbnail generation and cleanup
"""

import os
import shutil
import atexit
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tempfile
from predict import predict_video_with_audio_extraction

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['THUMBNAIL_FOLDER'] = 'thumbnails'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['THUMBNAIL_FOLDER'], exist_ok=True)

def cleanup_on_exit():
    """Clean up all temporary files when server stops"""
    try:
        # Clean up uploads folder
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Clean up thumbnails folder
        if os.path.exists(app.config['THUMBNAIL_FOLDER']):
            for file in os.listdir(app.config['THUMBNAIL_FOLDER']):
                file_path = os.path.join(app.config['THUMBNAIL_FOLDER'], file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        print("\n✅ Cleaned up all temporary files")
    except Exception as e:
        print(f"\n⚠️ Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_thumbnail(video_path, thumbnail_path, time_offset=1.0):
    """
    Generate a thumbnail from video at specified time offset
    
    Args:
        video_path: Path to video file
        thumbnail_path: Path to save thumbnail
        time_offset: Time in seconds to capture frame (default: 1.0)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame number
        frame_number = int(fps * time_offset)
        
        # If requested frame exceeds video length, use middle frame
        if frame_number >= total_frames:
            frame_number = total_frames // 2
        
        # Set video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print(f"❌ Could not read frame from video")
            return False
        
        # Resize frame to reasonable thumbnail size (max width: 640px)
        height, width = frame.shape[:2]
        max_width = 640
        
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save thumbnail
        success = cv2.imwrite(thumbnail_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        if success:
            print(f"✅ Thumbnail generated: {thumbnail_path}")
        else:
            print(f"❌ Failed to save thumbnail")
        
        return success
    
    except Exception as e:
        print(f"❌ Error generating thumbnail: {e}")
        return False

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images"""
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and analysis"""
    
    # Check if file was uploaded
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
    video_path = None
    thumbnail_path = None
    audio_path = None
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(os.times()[4] * 1000))
        unique_filename = f"{timestamp}_{filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(video_path)
        
        # Generate thumbnail
        thumbnail_filename = f"{timestamp}_thumbnail.jpg"
        thumbnail_path = os.path.join(app.config['THUMBNAIL_FOLDER'], thumbnail_filename)
        thumbnail_generated = generate_thumbnail(video_path, thumbnail_path)
        
        thumbnail_url = f"/thumbnails/{thumbnail_filename}" if thumbnail_generated else None
        
        # Process video and get prediction
        result = predict_video_with_audio_extraction(
            video_path,
            video_model_path='best_video_model.pth',
            audio_model_path='best_audio_model.h5',
            scaler_path='video_scaler.pkl'
        )
        
        # Add thumbnail URL to result
        if result:
            result['thumbnail_url'] = thumbnail_url
        
        # Clean up uploaded file and extracted audio
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"🗑️ Deleted video: {video_path}")
            
            # Check for extracted audio
            audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"🗑️ Deleted audio: {audio_path}")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
        
        if result is None:
            return jsonify({'error': 'Could not process video. Please ensure video contains faces.'}), 400
        
        return jsonify(result), 200
    
    except Exception as e:
        # Clean up on error
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            if thumbnail_path and os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/delete-thumbnail', methods=['POST'])
def delete_thumbnail():
    """Delete a specific thumbnail"""
    try:
        data = request.get_json()
        thumbnail_url = data.get('thumbnail_url', '')
        
        if thumbnail_url:
            # Extract filename from URL
            thumbnail_filename = thumbnail_url.split('/')[-1]
            thumbnail_path = os.path.join(app.config['THUMBNAIL_FOLDER'], thumbnail_filename)
            
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                print(f"🗑️ Deleted thumbnail: {thumbnail_path}")
                return jsonify({'success': True}), 200
        
        return jsonify({'success': False, 'error': 'Thumbnail not found'}), 404
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Check if models are loaded"""
    video_model_exists = os.path.exists('best_video_model.pth')
    audio_model_exists = os.path.exists('best_audio_model.h5')
    scaler_exists = os.path.exists('video_scaler.pkl')
    
    return jsonify({
        'status': 'ready' if (video_model_exists and audio_model_exists) else 'not_ready',
        'video_model': video_model_exists,
        'audio_model': audio_model_exists,
        'scaler': scaler_exists
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DEEPFAKE DETECTOR WEB INTERFACE")
    print("="*60)
    print("\nStarting server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        print("\nShutting down server...")
        cleanup_on_exit()