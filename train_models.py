"""
Complete training script for Multimodal DeepFake Detection System
"""

import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import from your main file
from deepfake_detector import (
    VideoFeatureExtractor,
    AudioFeatureExtractor,
    train_video_model,
    train_audio_model
)

def extract_video_features(data_dir):
    """Extract features from all videos in dataset"""
    print("\n" + "="*60)
    print("EXTRACTING VIDEO FEATURES")
    print("="*60)
    
    video_extractor = VideoFeatureExtractor()
    features = []
    labels = []
    
    # Real videos
    real_dir = os.path.join(data_dir, "videos", "real")
    print(f"\nProcessing real videos from: {real_dir}")
    for i, video_file in enumerate(os.listdir(real_dir)):
        if video_file.endswith(('.mp4', '.avi')):
            video_path = os.path.join(real_dir, video_file)
            print(f"  [{i+1}] Processing: {video_file}")
            feature = video_extractor.extract_features_from_video(video_path)
            if feature is not None:
                features.append(feature)
                labels.append(0)  # Real
    
    # Fake videos
    fake_dir = os.path.join(data_dir, "videos", "fake")
    print(f"\nProcessing fake videos from: {fake_dir}")
    for i, video_file in enumerate(os.listdir(fake_dir)):
        if video_file.endswith(('.mp4', '.avi')):
            video_path = os.path.join(fake_dir, video_file)
            print(f"  [{i+1}] Processing: {video_file}")
            feature = video_extractor.extract_features_from_video(video_path)
            if feature is not None:
                features.append(feature)
                labels.append(1)  # Fake
    
    X = np.array(features)
    y = np.array(labels)
    print(f"\nTotal videos processed: {len(X)}")
    print(f"  - Real: {np.sum(y == 0)}")
    print(f"  - Fake: {np.sum(y == 1)}")
    
    return X, y


def extract_audio_features(data_dir):
    """Extract mel-spectrograms from all audio files"""
    print("\n" + "="*60)
    print("EXTRACTING AUDIO FEATURES")
    print("="*60)
    
    audio_extractor = AudioFeatureExtractor()
    features = []
    labels = []
    
    # Real audio
    real_dir = os.path.join(data_dir, "audio", "real")
    print(f"\nProcessing real audio from: {real_dir}")
    for i, audio_file in enumerate(os.listdir(real_dir)):
        if audio_file.endswith(('.wav', '.mp3')):
            audio_path = os.path.join(real_dir, audio_file)
            print(f"  [{i+1}] Processing: {audio_file}")
            mel_spec = audio_extractor.extract_mel_spectrogram(audio_path)
            if mel_spec is not None:
                features.append(mel_spec)
                labels.append(0)  # Real
    
    # Fake audio
    fake_dir = os.path.join(data_dir, "audio", "fake")
    print(f"\nProcessing fake audio from: {fake_dir}")
    for i, audio_file in enumerate(os.listdir(fake_dir)):
        if audio_file.endswith(('.wav', '.mp3')):
            audio_path = os.path.join(fake_dir, audio_file)
            print(f"  [{i+1}] Processing: {audio_file}")
            mel_spec = audio_extractor.extract_mel_spectrogram(audio_path)
            if mel_spec is not None:
                features.append(mel_spec)
                labels.append(1)  # Fake
    
    X = np.array(features)
    y = np.array(labels)
    print(f"\nTotal audio files processed: {len(X)}")
    print(f"  - Real: {np.sum(y == 0)}")
    print(f"  - Fake: {np.sum(y == 1)}")
    
    return X, y


def train_video_pipeline(data_dir="data"):
    """Complete training pipeline for video model"""
    print("\n" + "="*60)
    print("VIDEO MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Extract features
    X, y = extract_video_features(data_dir)
    
    if len(X) < 10:
        print("\n⚠️  Not enough data to train! Need at least 10 videos.")
        return None, None
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    with open('video_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler to: video_scaler.pkl")
    
    # Train model
    print("\nTraining video model...")
    model, history = train_video_model(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=50,
        batch_size=32
    )
    
    print("\n✓ Video model training complete!")
    print("Model saved to: best_video_model.pth")
    
    return model, history


def train_audio_pipeline(data_dir="data"):
    """Complete training pipeline for audio model"""
    print("\n" + "="*60)
    print("AUDIO MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Extract features
    X, y = extract_audio_features(data_dir)
    
    if len(X) < 10:
        print("\n⚠️  Not enough data to train! Need at least 10 audio files.")
        return None, None
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples")
    
    # Train model
    print("\nTraining audio model...")
    model, history = train_audio_model(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=32
    )
    
    print("\n✓ Audio model training complete!")
    print("Model saved to: best_audio_model.h5")
    
    return model, history


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("MULTIMODAL DEEPFAKE DETECTION - TRAINING")
    print("="*60)
    
    data_dir = "data"
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"\n⚠️  Data directory not found: {data_dir}")
        print("Please create the following structure:")
        print("data/")
        print("  ├── videos/")
        print("  │   ├── real/")
        print("  │   └── fake/")
        print("  └── audio/")
        print("      ├── real/")
        print("      └── fake/")
        return
    
    # Train video model
    video_model, video_history = train_video_pipeline(data_dir)
    
    # Train audio model
    audio_model, audio_history = train_audio_pipeline(data_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - best_video_model.pth (Video model)")
    print("  - best_audio_model.h5 (Audio model)")
    print("  - video_scaler.pkl (Feature scaler)")
    print("\nYou can now use these models for prediction!")
    print("="*60)


if __name__ == "__main__":
    main()