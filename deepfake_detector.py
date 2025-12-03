"""
Multimodal DeepFake Detection System
Based on "A Multimodal Framework for DeepFake Detection" paper

This implementation includes:
1. Feature extraction for videos (9 facial features)
2. Mel-spectrogram generation for audio
3. ANN model for video classification (PyTorch)
4. VGG19 model for audio classification (TensorFlow)
5. Combined multimodal prediction
"""

import os
# suppress oneDNN informational message and lower TF log level (0=ALL,1=ERROR,2=WARNING,3=FATAL)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import pandas as pd
import librosa
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Using alternative face detection.")

from scipy.spatial import distance
from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array

# ==================== VIDEO FEATURE EXTRACTION ====================

class VideoFeatureExtractor:
    """Extract 9 facial features from videos as described in the paper"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Try to load dlib predictor if available
        self.dlib_available = DLIB_AVAILABLE
        if self.dlib_available:
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                # Download shape predictor if needed
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                if not os.path.exists(predictor_path):
                    print(f"Downloading dlib shape predictor...")
                    import urllib.request
                    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                    print(f"Please download from: {url}")
                    print(f"Extract and place in current directory as: {predictor_path}")
                    self.dlib_available = False
                else:
                    self.predictor = dlib.shape_predictor(predictor_path)
            except Exception as e:
                print(f"Could not load dlib: {e}")
                self.dlib_available = False
        
    def extract_features_from_video(self, video_path, max_frames=30):
        """Extract all 9 features from a video file"""
        cap = cv2.VideoCapture(video_path)
        features_list = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            features = self.extract_frame_features(frame)
            if features is not None:
                features_list.append(features)
            frame_count += 1
        
        cap.release()
        
        if len(features_list) == 0:
            return None
        
        # Average features across frames
        return np.mean(features_list, axis=0)
    
    def extract_frame_features(self, frame):
        """Extract features from a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        
        x, y, w, h = faces[0]
        face_roi = rgb_frame[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]
        
        # Get facial landmarks
        if self.dlib_available:
            landmarks = self.get_dlib_landmarks(gray_face, face_roi.shape)
        else:
            landmarks = self.get_simple_landmarks(gray_face, face_roi.shape)
        
        if landmarks is None:
            return None
        
        # Extract all 9 features (simplified when using basic detection)
        features = []
        
        # 1-2. Nose and lip size (estimated from face proportions)
        nose_size = h * 0.25  # Approximate nose size
        lip_size = w * 0.3    # Approximate lip size
        features.extend([nose_size, lip_size])
        
        # 3-4. Contrast and Correlation (GLCM features)
        contrast, correlation = self.calculate_texture_features(gray_face)
        features.extend([contrast, correlation])
        
        # 5. Eye aspect ratio (from detected eyes)
        ear = self.calculate_eye_aspect_ratio_simple(gray_face, face_roi.shape)
        features.append(ear)
        
        # 6. Inter-pupil distance (estimated)
        ipd = w * 0.4  # Typical IPD is ~40% of face width
        features.append(ipd)
        
        # 7. Cheekbone height (from face geometry)
        cheekbone_height = h * 0.35
        features.append(cheekbone_height)
        
        # 8-10. Head pose (simplified estimation)
        head_pose = self.estimate_head_pose_simple(gray_face)
        features.extend(head_pose)
        
        # 11-13. Skin tone (L, C1, C2 in oRGB color space)
        skin_tone = self.calculate_skin_tone(face_roi)
        features.extend(skin_tone)
        
        return np.array(features)
    
    def get_dlib_landmarks(self, gray_face, shape):
        """Get facial landmarks using dlib"""
        try:
            rect = dlib.rectangle(0, 0, shape[1], shape[0])
            landmarks = self.predictor(gray_face, rect)
            return landmarks
        except:
            return None
    
    def get_simple_landmarks(self, gray_face, shape):
        """Simple landmark estimation without dlib/mediapipe"""
        # Return a simple structure for basic feature extraction
        return {'simple': True}
    
    def calculate_eye_aspect_ratio_simple(self, gray_face, shape):
        """Simplified eye aspect ratio using eye cascade"""
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        
        if len(eyes) < 2:
            return 0.3  # Default EAR value
        
        # Calculate simple eye aspect ratio from detected eye regions
        eye_heights = [h for (x, y, w, h) in eyes]
        eye_widths = [w for (x, y, w, h) in eyes]
        
        avg_height = np.mean(eye_heights)
        avg_width = np.mean(eye_widths)
        
        ear = avg_height / (avg_width + 1e-6)
        return ear
    
    def estimate_head_pose_simple(self, gray_face):
        """Simplified head pose estimation"""
        h, w = gray_face.shape
        
        # Calculate image moments for orientation
        moments = cv2.moments(gray_face)
        
        # Simple orientation indicators
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            
            # Normalize to [-1, 1] range
            pitch = (cy / h - 0.5) * 2
            yaw = (cx / w - 0.5) * 2
            
            # Calculate roll from edge detection
            edges = cv2.Canny(gray_face, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            roll = 0
            if lines is not None and len(lines) > 0:
                angles = [line[0][1] for line in lines[:5]]
                roll = np.mean(angles) - np.pi/2
        else:
            pitch, yaw, roll = 0, 0, 0
        
        return [pitch * 45, yaw * 45, roll * 45]  # Scale to degrees
    
    def calculate_nose_size(self, landmarks, shape):
        """Calculate nose size using landmarks"""
        if not self.dlib_available:
            h, w = shape[:2]
            return h * 0.25  # Approximate
        
        # Dlib landmarks: nose tip = 30, nose bridge = 27
        nose_tip = landmarks.part(30)
        nose_bridge = landmarks.part(27)
        
        tip = np.array([nose_tip.x, nose_tip.y])
        bridge = np.array([nose_bridge.x, nose_bridge.y])
        
        return distance.euclidean(tip, bridge)
    
    def calculate_lip_size(self, landmarks, shape):
        """Calculate lip size"""
        if not self.dlib_available:
            h, w = shape[:2]
            return w * 0.3  # Approximate
        
        # Dlib landmarks: mouth corners = 48, 54
        left_corner = landmarks.part(48)
        right_corner = landmarks.part(54)
        
        left = np.array([left_corner.x, left_corner.y])
        right = np.array([right_corner.x, right_corner.y])
        
        return distance.euclidean(left, right)
    
    def calculate_texture_features(self, gray_roi):
        """Calculate GLCM contrast and correlation"""
        # Resize for efficiency
        gray_small = cv2.resize(gray_roi, (64, 64))
        
        # Calculate GLCM
        glcm = graycomatrix(gray_small, [1], [0], 256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        return contrast, correlation
    
    def calculate_eye_aspect_ratio(self, landmarks, shape):
        """Calculate eye aspect ratio using dlib landmarks"""
        if not self.dlib_available:
            return self.calculate_eye_aspect_ratio_simple(None, shape)
        
        # Left eye landmarks for dlib: 36-41
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        left_eye_pts = np.array([[p.x, p.y] for p in left_eye])
        
        # Calculate EAR for left eye
        A = distance.euclidean(left_eye_pts[1], left_eye_pts[5])
        B = distance.euclidean(left_eye_pts[2], left_eye_pts[4])
        C = distance.euclidean(left_eye_pts[0], left_eye_pts[3])
        
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def calculate_interpupilary_distance(self, landmarks, shape):
        """Calculate inter-pupillary distance"""
        if not self.dlib_available:
            h, w = shape[:2]
            return w * 0.4  # Approximate
        
        # Dlib: left eye center ≈ 39, right eye center ≈ 42
        left_eye = landmarks.part(39)
        right_eye = landmarks.part(42)
        
        left = np.array([left_eye.x, left_eye.y])
        right = np.array([right_eye.x, right_eye.y])
        
        return distance.euclidean(left, right)
    
    def calculate_cheekbone_height(self, landmarks, shape):
        """Calculate cheekbone height"""
        if not self.dlib_available:
            h, w = shape[:2]
            return h * 0.35  # Approximate
        
        # Dlib landmarks: nose tip = 30, chin = 8, cheek = 2
        nose_tip = landmarks.part(30)
        chin = landmarks.part(8)
        cheek = landmarks.part(2)
        
        nose_pt = np.array([nose_tip.x, nose_tip.y])
        chin_pt = np.array([chin.x, chin.y])
        cheek_pt = np.array([cheek.x, cheek.y])
        
        # Height difference between cheekbone and chin
        height = abs(cheek_pt[1] - chin_pt[1])
        
        return height
    
    def estimate_head_pose(self, landmarks, shape):
        """Estimate head pose angles using dlib landmarks"""
        if not self.dlib_available:
            return self.estimate_head_pose_simple(None)
        
        h, w = shape[:2]
        
        # Key 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye
            (225.0, 170.0, -135.0),      # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth
            (150.0, -150.0, -125.0)      # Right mouth
        ])
        
        # 2D image points from dlib landmarks
        image_points = np.array([
            [landmarks.part(30).x, landmarks.part(30).y],   # Nose
            [landmarks.part(8).x, landmarks.part(8).y],     # Chin
            [landmarks.part(36).x, landmarks.part(36).y],   # Left eye
            [landmarks.part(45).x, landmarks.part(45).y],   # Right eye
            [landmarks.part(48).x, landmarks.part(48).y],   # Left mouth
            [landmarks.part(54).x, landmarks.part(54).y]    # Right mouth
        ], dtype="double")
        
        # Camera matrix
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return [0, 0, 0]
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Get Euler angles
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = euler_angles.flatten()[:3]
        
        return [pitch, yaw, roll]
    
    def calculate_skin_tone(self, face_roi):
        """Calculate skin tone in oRGB color space"""
        # Convert to float
        rgb = face_roi.astype(float) / 255.0
        
        # Transformation matrix for oRGB
        transform = np.array([
            [0.299, 0.587, 0.114],
            [0.500, 0.500, -1.000],
            [0.866, -0.866, 0.000]
        ])
        
        # Reshape for matrix multiplication
        pixels = rgb.reshape(-1, 3)
        orgb = np.dot(pixels, transform.T)
        
        # Calculate mean values
        L = np.mean(orgb[:, 0])
        C1 = np.mean(orgb[:, 1])
        C2 = np.mean(orgb[:, 2])
        
        return [L, C1, C2]


# ==================== AUDIO FEATURE EXTRACTION ====================

class AudioFeatureExtractor:
    """Extract mel-spectrograms from audio files"""
    
    def __init__(self, n_mels=128, target_shape=(224, 224)):
        self.n_mels = n_mels
        self.target_shape = target_shape
    
    def extract_mel_spectrogram(self, audio_path, duration=4.0):
        """Extract mel-spectrogram from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=duration)
            
            # Generate mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=self.n_mels
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to target shape
            mel_spec_resized = cv2.resize(
                mel_spec_db, 
                self.target_shape, 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize to [0, 1]
            mel_spec_norm = (mel_spec_resized - mel_spec_resized.min()) / \
                           (mel_spec_resized.max() - mel_spec_resized.min() + 1e-8)
            
            # Convert to 3-channel image for VGG19
            mel_spec_3ch = np.stack([mel_spec_norm] * 3, axis=-1)
            
            return mel_spec_3ch
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None


# ==================== ANN MODEL FOR VIDEO (PyTorch) ====================

class DeepFakeVideoANN(nn.Module):
    """Artificial Neural Network for video deepfake detection"""
    
    def __init__(self, input_size=13):
        super(DeepFakeVideoANN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class VideoDataset(Dataset):
    """PyTorch Dataset for video features"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_video_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the ANN model for video classification"""
    
    # Create datasets
    train_dataset = VideoDataset(X_train, y_train)
    val_dataset = VideoDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepFakeVideoANN(input_size=X_train.shape[1]).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_video_model.pth')
    
    return model, history


# ==================== VGG19 MODEL FOR AUDIO (TensorFlow) ====================

def create_audio_model(input_shape=(224, 224, 3)):
    """Create VGG19 model for audio deepfake detection"""
    
    # Load pre-trained VGG19
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze all layers except last 4
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_audio_model(X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """Train VGG19 model for audio classification"""
    
    model = create_audio_model()
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save model
    model.save('best_audio_model.h5')
    
    return model, history


# ==================== MULTIMODAL DETECTION SYSTEM ====================

class MultimodalDeepFakeDetector:
    """Combined video and audio deepfake detection system"""
    
    def __init__(self, video_model_path='best_video_model.pth', 
                 audio_model_path='best_audio_model.h5'):
        
        # Initialize feature extractors
        self.video_extractor = VideoFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        
        # Load models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Video model (PyTorch)
        self.video_model = DeepFakeVideoANN(input_size=13).to(self.device)
        if os.path.exists(video_model_path):
            self.video_model.load_state_dict(torch.load(video_model_path))
        self.video_model.eval()
        
        # Audio model (TensorFlow)
        if os.path.exists(audio_model_path):
            self.audio_model = tf.keras.models.load_model(audio_model_path)
        else:
            self.audio_model = None
        
        # Scaler for video features
        self.scaler = StandardScaler()
    
    def predict_video(self, video_path):
        """Predict if video is deepfake"""
        features = self.video_extractor.extract_features_from_video(video_path)
        
        if features is None:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            prediction = self.video_model(features_tensor).item()
        
        return prediction
    
    def predict_audio(self, audio_path):
        """Predict if audio is deepfake"""
        if self.audio_model is None:
            return None
        
        mel_spec = self.audio_extractor.extract_mel_spectrogram(audio_path)
        
        if mel_spec is None:
            return None
        
        # Predict
        mel_spec_batch = np.expand_dims(mel_spec, axis=0)
        prediction = self.audio_model.predict(mel_spec_batch, verbose=0)[0][0]
        
        return prediction
    
    def predict_multimodal(self, video_path, audio_path, threshold=0.5):
        """
        Multimodal prediction: classify as deepfake if either component is fake
        
        Returns:
            dict: {
                'video_prediction': float,
                'audio_prediction': float,
                'final_prediction': str,
                'confidence': float
            }
        """
        video_pred = self.predict_video(video_path)
        audio_pred = self.predict_audio(audio_path)
        
        # Classify as deepfake if either is above threshold
        is_deepfake = (video_pred and video_pred > threshold) or \
                     (audio_pred and audio_pred > threshold)
        
        confidence = max(video_pred if video_pred else 0, 
                        audio_pred if audio_pred else 0)
        
        result = {
            'video_prediction': video_pred,
            'audio_prediction': audio_pred,
            'final_prediction': 'DEEPFAKE' if is_deepfake else 'REAL',
            'confidence': confidence,
            'video_label': 'DEEPFAKE' if video_pred and video_pred > threshold else 'REAL',
            'audio_label': 'DEEPFAKE' if audio_pred and audio_pred > threshold else 'REAL'
        }
        
        return result


# ==================== MAIN TRAINING PIPELINE ====================

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Multimodal DeepFake Detection System")
    print("=" * 60)
    
    # Set up data directories
    data_dir = "data"
    video_real_dir = os.path.join(data_dir, "videos", "real")
    video_fake_dir = os.path.join(data_dir, "videos", "fake")
    audio_real_dir = os.path.join(data_dir, "audio", "real")
    audio_fake_dir = os.path.join(data_dir, "audio", "fake")
    
    # Create directories if they don't exist
    os.makedirs(video_real_dir, exist_ok=True)
    os.makedirs(video_fake_dir, exist_ok=True)
    os.makedirs(audio_real_dir, exist_ok=True)
    os.makedirs(audio_fake_dir, exist_ok=True)
    
    print("\nExpected data structure:")
    print("data/")
    print("  ├── videos/")
    print("  │   ├── real/  (real videos)")
    print("  │   └── fake/  (deepfake videos)")
    print("  └── audio/")
    print("      ├── real/  (real audio)")
    print("      └── fake/  (deepfake audio)")
    
    # Check if data exists
    video_real_files = [f for f in os.listdir(video_real_dir) if f.endswith(('.mp4', '.avi'))]
    video_fake_files = [f for f in os.listdir(video_fake_dir) if f.endswith(('.mp4', '.avi'))]
    audio_real_files = [f for f in os.listdir(audio_real_dir) if f.endswith(('.wav', '.mp3'))]
    audio_fake_files = [f for f in os.listdir(audio_fake_dir) if f.endswith(('.wav', '.mp3'))]
    
    print(f"\nFound:")
    print(f"  - {len(video_real_files)} real videos")
    print(f"  - {len(video_fake_files)} fake videos")
    print(f"  - {len(audio_real_files)} real audio files")
    print(f"  - {len(audio_fake_files)} fake audio files")
    
    if len(video_real_files) == 0 or len(video_fake_files) == 0:
        print("\n⚠️  No video data found! Please add video files to train the model.")
        print("Place videos in data/videos/real/ and data/videos/fake/")
    
    if len(audio_real_files) == 0 or len(audio_fake_files) == 0:
        print("\n⚠️  No audio data found! Please add audio files to train the model.")
        print("Place audio files in data/audio/real/ and data/audio/fake/")
    
    print("\n" + "=" * 60)
    print("Training pipeline ready!")
    print("Add your data files and run the training functions:")
    print("  - train_video_model() for video classification")
    print("  - train_audio_model() for audio classification")
    print("=" * 60)


if __name__ == "__main__":
    main()