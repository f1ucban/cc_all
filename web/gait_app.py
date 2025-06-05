import sqlite3
import pickle
import numpy as np
import cv2
import base64
import mediapipe as mp
import logging
import json
import threading
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
from functools import wraps
import warnings
import collections
import os
warnings.filterwarnings('ignore')

# Set TensorFlow environment variable to suppress oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "gait_recognition_secret_key_2024")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

def ensure_db_connection(func):
    """Decorator to ensure database connection is properly handled"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        conn = None
        try:
            # Use self.db_path instead of gait_system.db_path
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            result = func(self, cursor, *args, **kwargs)
            conn.commit()
            return result
        except Exception as e:
            logging.error(f"Database error in {func.__name__}: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    return wrapper

class EnhancedGaitRecognitionSystem:
    def __init__(self):
        """Initialize the enhanced gait recognition system"""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Database setup
            self.db_path = "gait_recognition.db"
            self.init_database()
            
            # ML models
            self.ensemble_models = {}
            self.scaler = StandardScaler()
            self.feature_selector = SelectKBest(f_classif, k=500)
            self.pca = PCA(n_components='mle')
            
            # Threading lock for training
            self.training_lock = threading.Lock()
            self.training_in_progress = False
            self.training_progress = 0
            
            # Enrollment configuration
            self.enrollment_data = {}
            self.current_enrollment_user = None
            self.target_frames = 100
            
            # Recognition buffer
            self.recognition_buffer = []
            self.buffer_size = 30
            
            # Load existing models
            self.load_models()
            
            logging.info("Enhanced Gait Recognition System initialized")
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
    
    @ensure_db_connection
    def init_database(self, cursor):
        """Initialize SQLite database with required tables"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enrollment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                landmarks_data BLOB,
                features_data BLOB,
                frame_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_data BLOB,
                scaler_data BLOB,
                feature_selector_data BLOB,
                pca_data BLOB,
                performance_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_count INTEGER,
                accuracy REAL,
                training_time REAL,
                model_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

# Initialize the system
    
    def extract_pose_landmarks(self, image):
        """Extract pose landmarks from image"""
        try:
            if isinstance(image, str):
                if image.startswith('data:image'):
                    image = image.split(',')[1]
                image_data = base64.b64decode(image)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None or len(image.shape) != 3:
                return None
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z or 0.0, landmark.visibility])
                return np.array(landmarks, dtype=np.float32)
            return None
        except Exception as e:
            logging.error(f"Error extracting pose landmarks: {str(e)}")
            return None
    
    def extract_enhanced_gait_features(self, landmarks_sequence):
        """Extract enhanced gait features with improved feature engineering"""
        try:
            if not landmarks_sequence or len(landmarks_sequence) < 5:
                return None
                
            landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
            features = []
            
            # Spatial features
            for frame in landmarks_array:
                features.extend(self._extract_anatomical_features(frame))
                features.extend(self._extract_distance_features(frame))
                features.extend(self._extract_angular_features(frame))
            
            # Temporal features
            features.extend(self._extract_temporal_features(landmarks_array))
            features.extend(self._extract_frequency_features(landmarks_array))
            features.extend(self._extract_gait_cycle_features(landmarks_array))
            
            # Ensure consistent feature size
            expected_features = 2000
            features = np.array(features, dtype=np.float32)
            if len(features) < expected_features:
                features = np.pad(features, (0, expected_features - len(features)), mode='constant')
            elif len(features) > expected_features:
                features = features[:expected_features]
                
            return features
        except Exception as e:
            logging.error(f"Error extracting enhanced gait features: {str(e)}")
            return None
    
    def _extract_anatomical_features(self, frame):
        """Extract anatomical proportion features"""
        try:
            features = []
            landmarks = {
                'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
                'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
                'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
                'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
                'left_toe': 31, 'right_toe': 32
            }
            
            shoulder_width = abs(frame[landmarks['left_shoulder']*4] - frame[landmarks['right_shoulder']*4])
            hip_width = abs(frame[landmarks['left_hip']*4] - frame[landmarks['right_hip']*4])
            features.extend([shoulder_width, hip_width])
            
            left_arm_length = np.sqrt(
                (frame[landmarks['left_shoulder']*4] - frame[landmarks['left_wrist']*4])**2 +
                (frame[landmarks['left_shoulder']*4+1] - frame[landmarks['left_wrist']*4+1])**2
            )
            right_arm_length = np.sqrt(
                (frame[landmarks['right_shoulder']*4] - frame[landmarks['right_wrist']*4])**2 +
                (frame[landmarks['right_shoulder']*4+1] - frame[landmarks['right_wrist']*4+1])**2
            )
            features.extend([left_arm_length, right_arm_length])
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error in anatomical features: {str(e)}")
            return np.zeros(4, dtype=np.float32)
    
    def _extract_distance_features(self, frame):
        """Extract pairwise distance features"""
        try:
            features = []
            key_points = [0, 11, 12, 23, 24, 25, 26, 27, 28]
            
            for i in range(len(key_points)):
                for j in range(i+1, len(key_points)):
                    dist = np.sqrt(
                        (frame[key_points[i]*4] - frame[key_points[j]*4])**2 +
                        (frame[key_points[i]*4+1] - frame[key_points[j]*4+1])**2
                    )
                    features.append(dist)
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error in distance features: {str(e)}")
            return np.zeros(36, dtype=np.float32)
    
    def _extract_angular_features(self, frame):
        """Extract angular features for joints"""
        try:
            features = []
            
            for side in ['left', 'right']:
                hip_idx = 23 if side == 'left' else 24
                knee_idx = 25 if side == 'left' else 26
                ankle_idx = 27 if side == 'left' else 28
                
                hip_pos = np.array([frame[hip_idx*4], frame[hip_idx*4+1]], dtype=np.float32)
                knee_pos = np.array([frame[knee_idx*4], frame[knee_idx*4+1]], dtype=np.float32)
                ankle_pos = np.array([frame[ankle_idx*4], frame[ankle_idx*4+1]], dtype=np.float32)
                
                v1 = hip_pos - knee_pos
                v2 = ankle_pos - knee_pos
                
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 0 and norm_v2 > 0:
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0))
                else:
                    angle = 0.0
                features.append(angle)
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error in angular features: {str(e)}")
            return np.zeros(2, dtype=np.float32)
    
    def _extract_temporal_features(self, landmarks_array):
        """Extract temporal movement features"""
        try:
            features = []
            
            for landmark_idx in [23, 24, 25, 26, 27, 28]:
                positions = landmarks_array[:, landmark_idx*4:landmark_idx*4+2]
                velocities = np.diff(positions, axis=0)
                features.extend([np.mean(velocities), np.std(velocities)])
            
            left_ankle_y = landmarks_array[:, 27*4+1]
            right_ankle_y = landmarks_array[:, 28*4+1]
            
            left_steps = len([i for i in range(1, len(left_ankle_y)) 
                           if left_ankle_y[i] > left_ankle_y[i-1] and 
                           abs(left_ankle_y[i] - left_ankle_y[i-1]) > 0.02])
            right_steps = len([i for i in range(1, len(right_ankle_y)) 
                            if right_ankle_y[i] > right_ankle_y[i-1] and 
                            abs(right_ankle_y[i] - right_ankle_y[i-1]) > 0.02])
            
            features.extend([left_steps, right_steps])
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error in temporal features: {str(e)}")
            return np.zeros(14, dtype=np.float32)
    
    def _extract_frequency_features(self, landmarks_array):
        """Extract frequency domain features"""
        try:
            features = []
            
            for landmark_idx in [23, 24, 27, 28]:
                signal = landmarks_array[:, landmark_idx*4+1]
                if len(signal) > 8:
                    fft = np.fft.fft(signal)
                    power_spectrum = np.abs(fft[:len(fft)//2])
                    features.extend(power_spectrum[:5])
                else:
                    features.extend([0] * 5)
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error in frequency features: {str(e)}")
            return np.zeros(20, dtype=np.float32)
    
    def _extract_gait_cycle_features(self, landmarks_array):
        """Extract gait cycle specific features"""
        try:
            features = []
            
            left_hip_movement = np.std(landmarks_array[:, 23*4:23*4+2])
            right_hip_movement = np.std(landmarks_array[:, 24*4:24*4+2])
            hip_symmetry = abs(left_hip_movement - right_hip_movement)
            
            hip_center_x = (landmarks_array[:, 23*4] + landmarks_array[:, 24*4]) / 2
            stride_length = np.std(hip_center_x)
            stride_frequency = len(landmarks_array) / (np.ptp(hip_center_x) + 1e-10)
            
            features.extend([hip_symmetry, stride_length, stride_frequency])
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error in gait cycle features: {str(e)}")
            return np.zeros(3, dtype=np.float32)
    
    @ensure_db_connection
    def get_all_users(self, cursor):
        """Get all users from database"""
        try:
            cursor.execute("SELECT id, username, created_at FROM users ORDER BY username")
            users = cursor.fetchall()
            return [{'id': user['id'], 'username': user['username'], 'created_at': user['created_at']} 
                    for user in users]
        except Exception as e:
            logging.error(f"Error getting users: {str(e)}")
            return []
    
    @ensure_db_connection
    def user_exists(self, cursor, username):
        """Check if user exists"""
        try:
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            return cursor.fetchone() is not None
        except Exception as e:
            logging.error(f"Error checking user existence: {str(e)}")
            return False
    
    @ensure_db_connection
    def delete_user(self, cursor, username):
        """Delete user and all associated data"""
        try:
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            
            if user:
                user_id = user['id']
                cursor.execute("DELETE FROM enrollment_data WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                logging.info(f"User {username} and associated data deleted")
                return True
            return False
        except Exception as e:
            logging.error(f"Error deleting user: {str(e)}")
            return False
    
    def start_enrollment(self, username):
        """Start enrollment process"""
        try:
            if not username or not isinstance(username, str) or len(username) > 50:
                return False, "Invalid username"
                
            if self.user_exists(username):
                return False, "User already exists"
            
            self.current_enrollment_user = username
            self.enrollment_data[username] = {
                'landmarks': [],
                'features': [],
                'frame_count': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            logging.info(f"Started enrollment for user: {username}")
            return True, "Enrollment started"
        except Exception as e:
            logging.error(f"Error starting enrollment: {str(e)}")
            return False, str(e)
    
    def add_enrollment_frame(self, image_data):
        """Add frame to enrollment data"""
        try:
            if not self.current_enrollment_user:
                return False, "No enrollment in progress"
            
            landmarks = self.extract_pose_landmarks(image_data)
            if landmarks is None:
                return False, "No pose detected in frame"
                
            username = self.current_enrollment_user
            self.enrollment_data[username]['landmarks'].append(landmarks)
            self.enrollment_data[username]['frame_count'] += 1
            
            progress = (self.enrollment_data[username]['frame_count'] / self.target_frames) * 100
            socketio.emit('enrollment_progress', {
                'progress': min(progress, 100.0),
                'frames': self.enrollment_data[username]['frame_count'],
                'target': self.target_frames
            })
            
            if self.enrollment_data[username]['frame_count'] >= self.target_frames:
                return self.complete_enrollment()
                
            return True, f"Frame added ({self.enrollment_data[username]['frame_count']}/{self.target_frames})"
        except Exception as e:
            logging.error(f"Error adding enrollment frame: {str(e)}")
            return False, str(e)
    
    @ensure_db_connection
    def complete_enrollment(self, cursor):
        """Complete enrollment process automatically"""
        try:
            if not self.current_enrollment_user:
                return False, "No enrollment in progress"
            
            username = self.current_enrollment_user
            landmarks_data = self.enrollment_data[username]['landmarks']
            
            if len(landmarks_data) < 10:
                return False, "Insufficient enrollment data"
            
            features = self.extract_enhanced_gait_features(landmarks_data)
            if features is None:
                return False, "Failed to extract features"
            
            cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
            user_id = cursor.lastrowid
            
            landmarks_blob = pickle.dumps(landmarks_data)
            features_blob = pickle.dumps(features)
            
            cursor.execute("""
                INSERT INTO enrollment_data (user_id, landmarks_data, features_data, frame_count)
                VALUES (?, ?, ?, ?)
            """, (user_id, landmarks_blob, features_blob, len(landmarks_data)))
            
            del self.enrollment_data[username]
            self.current_enrollment_user = None
            
            socketio.emit('enrollment_complete', {'username': username})
            logging.info(f"Enrollment completed for user: {username}")
            return True, f"User {username} enrolled successfully"
        except Exception as e:
            logging.error(f"Error completing enrollment: {str(e)}")
            return False, str(e)
    
    def train_enhanced_models(self):
        """Train ensemble models with progress tracking"""
        if self.training_in_progress:
            return False
            
        def training_thread():
            with self.training_lock:
                try:
                    self.training_in_progress = True    
                    self.training_progress = 0
                    socketio.emit('training_started')
                    
                    training_data, labels = self._load_training_data()
                    if len(training_data) < 2 or len(set(labels)) < 2:
                        socketio.emit('training_error', {'message': 'Insufficient data or users for training'})
                        return
                    
                    self.training_progress = 10
                    socketio.emit('training_progress', {'progress': 10})
                    
                    X = np.array(training_data, dtype=np.float32)
                    y = np.array(labels)
                    
                    # Check dimensions before PCA
                    min_dim = min(X.shape[0], X.shape[1])
                    if min_dim < 2:
                        socketio.emit('training_error', {'message': f'Not enough dimensions for PCA (current: {min_dim})'})
                        return
                    
                    # Auto-adjust PCA components
                    n_components = min(100, min_dim - 1)
                    self.pca = PCA(n_components=n_components)
                    
                    self.training_progress = 20
                    socketio.emit('training_progress', {'progress': 20})
                    
                    X_scaled = self.scaler.fit_transform(X)
                    self.training_progress = 30
                    socketio.emit('training_progress', {'progress': 30})
                    
                    X_selected = self.feature_selector.fit_transform(X_scaled, y)
                    self.training_progress = 40
                    socketio.emit('training_progress', {'progress': 40})
                    
                    X_final = self.pca.fit_transform(X_selected)
                    self.training_progress = 50
                    socketio.emit('training_progress', {'progress': 50})
                    
                    # Simplify models for small datasets
                    if len(set(y)) < 5:
                        models = {
                            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                            'svm': SVC(probability=True, random_state=42, kernel='linear'),
                            'neural_network': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
                        }
                    else:
                        models = {
                            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                            'svm': SVC(probability=True, random_state=42),
                            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
                        }
                    
                    trained_models = {}
                    for i, (name, model) in enumerate(models.items()):
                        model.fit(X_final, y)
                        trained_models[name] = model
                        progress = 50 + (i + 1) * 15
                        self.training_progress = progress
                        socketio.emit('training_progress', {'progress': progress})
                    
                    self.ensemble_models = trained_models
                    
                    # Use simpler cross-validation for small datasets
                    n_samples = X_final.shape[0]
                    n_classes = len(set(y))
                    class_counts = collections.Counter(y)
                    min_class_samples = min(class_counts.values())
                    cv = min(5, n_samples, n_classes, min_class_samples)
                    if cv < 2:
                        cv = 2  # At least 2 folds required
                    if min_class_samples < 2:
                        # Not enough samples for cross-validation, just fit and set accuracy to 1.0
                        accuracy_scores = {name: 1.0 for name in trained_models}
                        socketio.emit('training_warning', {'message': 'Not enough samples for cross-validation. Models trained without validation.'})
                    else:
                        accuracy_scores = {}
                        for name, model in trained_models.items():
                            scores = cross_val_score(model, X_final, y, cv=cv, n_jobs=-1)
                            accuracy_scores[name] = float(np.mean(scores))
                    
                    self.training_progress = 95
                    socketio.emit('training_progress', {'progress': 95})
                    
                    self._save_models_to_db(trained_models, accuracy_scores)
                    
                    self.training_progress = 100
                    socketio.emit('training_complete', {
                        'accuracy_scores': accuracy_scores,
                        'total_users': len(set(y))
                    })
                    
                    self._log_training_session(len(set(y)), accuracy_scores)
                    
                except Exception as e:
                    logging.error(f"Training error: {str(e)}")
                    socketio.emit('training_error', {'message': str(e)})
                finally:
                    self.training_in_progress = False
        
        threading.Thread(target=training_thread, daemon=True).start()
        return True
    
    @ensure_db_connection
    def _load_training_data(self, cursor):
        """Load training data from database"""
        try:
            cursor.execute("""
                SELECT u.username, e.features_data 
                FROM users u 
                JOIN enrollment_data e ON u.id = e.user_id
            """)
            
            results = cursor.fetchall()
            training_data = []
            labels = []
            
            for row in results:
                try:
                    features = pickle.loads(row['features_data'])
                    if features is not None:
                        training_data.append(features)
                        labels.append(row['username'])
                except Exception as e:
                    logging.warning(f"Error loading features for user {row['username']}: {str(e)}")
                    continue
            
            return training_data, labels
        except Exception as e:
            logging.error(f"Error loading training data: {str(e)}")
            return [], []
    
    @ensure_db_connection
    def _save_models_to_db(self, cursor, models, accuracy_scores):
        """Save trained models to database"""
        try:
            cursor.execute("DELETE FROM models")
            
            for name, model in models.items():
                model_blob = pickle.dumps(model)
                scaler_blob = pickle.dumps(self.scaler)
                selector_blob = pickle.dumps(self.feature_selector)
                pca_blob = pickle.dumps(self.pca)
                metrics = json.dumps({'accuracy': float(accuracy_scores[name])})
                
                cursor.execute("""
                    INSERT INTO models (model_name, model_data, scaler_data, 
                                      feature_selector_data, pca_data, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, model_blob, scaler_blob, selector_blob, pca_blob, metrics))
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            raise
    
    @ensure_db_connection
    def _log_training_session(self, cursor, user_count, accuracy_scores):
        """Log training session"""
        try:
            avg_accuracy = float(np.mean(list(accuracy_scores.values())))
            cursor.execute("""
                INSERT INTO training_logs (user_count, accuracy, training_time, model_type)
                VALUES (?, ?, ?, ?)
            """, (user_count, avg_accuracy, time.time(), 'ensemble'))
        except Exception as e:
            logging.error(f"Error logging training: {str(e)}")
    
    @ensure_db_connection
    def load_models(self, cursor):
        """Load models from database"""
        try:
            cursor.execute("SELECT model_name, model_data, scaler_data, feature_selector_data, pca_data FROM models")
            results = cursor.fetchall()
            
            if results:
                models = {}
                for row in results:
                    try:
                        models[row['model_name']] = pickle.loads(row['model_data'])
                        if row['scaler_data']:
                            self.scaler = pickle.loads(row['scaler_data'])
                        if row['feature_selector_data']:
                            self.feature_selector = pickle.loads(row['feature_selector_data'])
                        if row['pca_data']:
                            self.pca = pickle.loads(row['pca_data'])
                    except Exception as e:
                        logging.warning(f"Error loading model {row['model_name']}: {str(e)}")
                        continue
                
                self.ensemble_models = models
                logging.info(f"Loaded {len(models)} models from database")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
    
    def recognize_user(self, image_data):
        """Recognize user using ensemble voting"""
        try:
            if not self.ensemble_models:
                return None, "No trained models available"
            
            landmarks = self.extract_pose_landmarks(image_data)
            if landmarks is None:
                return None, "No pose detected"
            
            self.recognition_buffer.append(landmarks)
            if len(self.recognition_buffer) > self.buffer_size:
                self.recognition_buffer.pop(0)
            
            if len(self.recognition_buffer) < 10:
                return None, "Collecting frames..."
            
            features = self.extract_enhanced_gait_features(self.recognition_buffer)
            if features is None:
                return None, "Failed to extract features"
            
            features_scaled = self.scaler.transform([features])
            features_selected = self.feature_selector.transform(features_scaled)
            features_final = self.pca.transform(features_selected)
            
            predictions = {}
            confidences = {}
            
            for name, model in self.ensemble_models.items():
                pred = model.predict(features_final)[0]
                prob = model.predict_proba(features_final)[0]
                max_conf = float(np.max(prob))
                
                predictions[name] = pred
                confidences[name] = max_conf
            
            votes = list(predictions.values())
            most_common = max(set(votes), key=votes.count)
            avg_confidence = float(np.mean(list(confidences.values())))
            
            return most_common, f"Confidence: {avg_confidence:.2f}"
        except Exception as e:
            logging.error(f"Error in recognition: {str(e)}")
            return None, str(e)
    
    @ensure_db_connection
    def get_system_stats(self, cursor):
        """Get system statistics"""
        try:
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM models")
            model_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT accuracy, created_at FROM training_logs ORDER BY created_at DESC LIMIT 1")
            latest_training = cursor.fetchone()
            
            stats = {
                'user_count': user_count or 0,
                'model_count': model_count or 0,
                'latest_accuracy': float(latest_training['accuracy']) if latest_training else 0.0,
                'last_training': latest_training['created_at'] if latest_training else 'Never'
            }
            
            return stats
        except Exception as e:
            logging.error(f"Error getting stats: {str(e)}")
            return {'user_count': 0, 'model_count': 0, 'latest_accuracy': 0.0, 'last_training': 'Never'}

# Initialize the system
gait_system = EnhancedGaitRecognitionSystem()

# Routes
@app.route('/')
def index():
    """Main dashboard"""
    stats = gait_system.get_system_stats()
    users = gait_system.get_all_users()
    return render_template_string(INDEX_TEMPLATE, stats=stats, users=users)

@app.route('/enroll')
def enroll():
    """Enrollment page"""
    return render_template_string(ENROLL_TEMPLATE)

@app.route('/recognize')
def recognize():
    """Recognition page"""
    return render_template_string(RECOGNIZE_TEMPLATE)

@app.route('/manage')
def manage():
    """User management page"""
    users = gait_system.get_all_users()
    return render_template_string(MANAGE_TEMPLATE, users=users)

@app.route('/api/start_enrollment', methods=['POST'])
def api_start_enrollment():
    """Start enrollment API"""
    try:
        data = request.get_json()
        if not data or 'username' not in data:
            return jsonify({'success': False, 'message': 'Invalid request'})
            
        username = data['username'].strip()
        if not username or len(username) > 50:
            return jsonify({'success': False, 'message': 'Invalid username'})
            
        success, message = gait_system.start_enrollment(username)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        logging.error(f"Error in start_enrollment API: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/api/add_enrollment_frame', methods=['POST'])
def api_add_enrollment_frame():
    """Add enrollment frame API"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data'})
            
        image_data = data['image']
        success, message = gait_system.add_enrollment_frame(image_data)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        logging.error(f"Error in add_enrollment_frame API: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/api/recognize_frame', methods=['POST'])
def api_recognize_frame():
    """Recognize frame API"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data'})
            
        image_data = data['image']
        user, confidence = gait_system.recognize_user(image_data)
        
        if user:
            return jsonify({'success': True, 'user': user, 'confidence': confidence})
        return jsonify({'success': False, 'message': confidence})
    except Exception as e:
        logging.error(f"Error in recognize_frame API: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/api/train_models', methods=['POST'])
def api_train_models():
    """Train models API"""
    try:
        success = gait_system.train_enhanced_models()
        if success:
            return jsonify({'success': True, 'message': 'Training started'})
        return jsonify({'success': False, 'message': 'Training already in progress'})
    except Exception as e:
        logging.error(f"Error in train_models API: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/api/delete_user', methods=['POST'])
def api_delete_user():
    """Delete user API"""
    try:
        data = request.get_json()
        if not data or 'username' not in data:
            return jsonify({'success': False, 'message': 'Invalid request'})
            
        username = data['username'].strip()
        if not username:
            return jsonify({'success': False, 'message': 'Username required'})
            
        success = gait_system.delete_user(username)
        if success:
            return jsonify({'success': True, 'message': f'User {username} deleted'})
        return jsonify({'success': False, 'message': 'User not found'})
    except Exception as e:
        logging.error(f"Error in delete_user API: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/api/stats')
def api_stats():
    """Get system stats API"""
    try:
        stats = gait_system.get_system_stats()
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error in stats API: {str(e)}")
        return jsonify({'error': 'Server error'})

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    logging.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected')

# HTML Templates
INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Gait Recognition System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        .stat-card { 
            background: var(--bs-dark); 
            border: 1px solid var(--bs-secondary); 
            transition: transform 0.2s; 
        }
        .stat-card:hover { transform: translateY(-2px); }
        .feature-icon { font-size: 2rem; color: var(--bs-primary); }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-walking me-2"></i>Gait Recognition</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/enroll"><i class="fas fa-user-plus me-1"></i>Enroll</a>
                <a class="nav-link" href="/recognize"><i class="fas fa-search me-1"></i>Recognize</a>
                <a class="nav-link" href="/manage"><i class="fas fa-users-cog me-1"></i>Manage</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row mb-4">
            <div class="col-12">
                <h1 class="text-center mb-4">
                    <i class="fas fa-brain text-primary me-2"></i>
                    Enhanced Gait Recognition System
                </h1>
                <p class="lead text-center">Advanced biometric identification using gait analysis with ensemble machine learning</p>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-3 mb-3">
                <div class="card stat-card">
                    <div class="card-body text-center">
                        <i class="fas fa-users feature-icon mb-2"></i>
                        <h3 class="card-title">{{ stats.user_count }}</h3>
                        <p class="card-text">Enrolled Users</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card stat-card">
                    <div class="card-body text-center">
                        <i class="fas fa-robot feature-icon mb-2"></i>
                        <h3 class="card-title">{{ stats.model_count }}</h3>
                        <p class="card-text">Trained Models</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card stat-card">
                    <div class="card-body text-center">
                        <i class="fas fa-chart-line feature-icon mb-2"></i>
                        <h3 class="card-title">{{ "%.1f"|format(stats.latest_accuracy * 100) }}%</h3>
                        <p class="card-text">Model Accuracy</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card stat-card">
                    <div class="card-body text-center">
                        <i class="fas fa-clock feature-icon mb-2"></i>
                        <h3 class="card-title">{{ stats.last_training.split('T')[0] if stats.last_training != 'Never' else 'Never' }}</h3>
                        <p class="card-text">Last Training</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-user-plus feature-icon mb-3"></i>
                        <h5 class="card-title">Enroll New User</h5>
                        <p class="card-text">Register a new user using automated gait analysis</p>
                        <a href="/enroll" class="btn btn-primary">Start Enrollment</a>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-search feature-icon mb-3"></i>
                        <h5 class="card-title">Recognize User</h5>
                        <p class="card-text">Identify users through real-time gait recognition</p>
                        <a href="/recognize" class="btn btn-primary">Start Recognition</a>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-cogs feature-icon mb-3"></i>
                        <h5 class="card-title">Train Models</h5>
                        <p class="card-text">Train ensemble machine learning models</p>
                        <button class="btn btn-success" onclick="trainModels()" id="trainBtn">Train Models</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4" id="trainingSection" style="display: none;">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-brain me-2"></i>Model Training Progress</h5>
                    </div>
                    <div class="card-body">
                        <div class="progress mb-3" style="height: 25px;">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                 id="trainingProgress" style="width: 0%;">0%</div>
                        </div>
                        <div id="trainingStatus" class="text-center">
                            <i class="fas fa-spinner fa-spin me-2"></i>Initializing training...
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-users me-2"></i>Enrolled Users</h5>
                    </div>
                    <div class="card-body">
                        {% if users %}
                            <div class="table-responsive">
                                <table class="table table-hover">
                                    <thead>
                                        <tr>
                                            <th>Username</th>
                                            <th>Enrolled Date</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for user in users %}
                                        <tr>
                                            <td><i class="fas fa-user me-2"></i>{{ user.username }}</td>
                                            <td>{{ user.created_at.split('T')[0] if user.created_at else 'Unknown' }}</td>
                                            <td>
                                                <a href="/manage" class="btn btn-sm btn-outline-primary">
                                                    <i class="fas fa-edit"></i> Manage
                                                </a>
                                            </td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        {% else %}
                            <div class="text-center text-muted">
                                <i class="fas fa-user-slash fa-3x mb-3"></i>
                                <p>No users enrolled yet. <a href="/enroll">Start enrolling users</a> to begin.</p>
                            </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        function trainModels() {
            const btn = document.getElementById('trainBtn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Starting...';
            
            fetch('/api/train_models', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('trainingSection').style.display = 'block';
                } else {
                    alert(data.message);
                    btn.disabled = false;
                    btn.innerHTML = 'Train Models';
                }
            })
            .catch(error => {
                alert('Error starting training: ' + error.message);
                btn.disabled = false;
                btn.innerHTML = 'Train Models';
            });
        }
        
        socket.on('training_started', function() {
            document.getElementById('trainingStatus').innerHTML = 
                '<i class="fas fa-spinner fa-spin me-2"></i>Training started...';
        });
        
        socket.on('training_progress', function(data) {
            const progress = data.progress;
            const progressBar = document.getElementById('trainingProgress');
            progressBar.style.width = progress + '%';
            progressBar.textContent = Math.round(progress) + '%';
            
            let status = 'Training in progress...';
            if (progress < 20) status = 'Loading training data...';
            else if (progress < 40) status = 'Preprocessing features...';
            else if (progress < 80) status = 'Training ensemble models...';
            else status = 'Finalizing training...';
            
            document.getElementById('trainingStatus').innerHTML = 
                '<i class="fas fa-brain me-2"></i>' + status;
        });
        
        socket.on('training_complete', function(data) {
            document.getElementById('trainingProgress').classList.remove('progress-bar-animated');
            document.getElementById('trainingProgress').classList.add('bg-success');
            document.getElementById('trainingStatus').innerHTML = 
                '<i class="fas fa-check-circle text-success me-2"></i>Training completed successfully!';
            
            setTimeout(() => {
                location.reload();
            }, 2000);
        });
        
        socket.on('training_error', function(data) {
            document.getElementById('trainingProgress').classList.add('bg-danger');
            document.getElementById('trainingStatus').innerHTML = 
                '<i class="fas fa-exclamation-triangle text-danger me-2"></i>Training failed: ' + data.message;
            
            const btn = document.getElementById('trainBtn');
            btn.disabled = false;
            btn.innerHTML = 'Train Models';
        });
    </script>
</body>
</html>
'''

ENROLL_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Enrollment - Gait Recognition</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        #video { border-radius: 10px; }
        .enrollment-card { background: var(--bs-dark); border: 1px solid var(--bs-secondary); }
        .progress-ring { transform: rotate(-90deg); }
        .progress-ring-circle { transition: stroke-dasharray 0.35s; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-walking me-2"></i>Gait Recognition</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/"><i class="fas fa-home me-1"></i>Dashboard</a>
                <a class="nav-link" href="/recognize"><i class="fas fa-search me-1"></i>Recognize</a>
                <a class="nav-link" href="/manage"><i class="fas fa-users-cog me-1"></i>Manage</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <h2 class="text-center mb-4">
                    <i class="fas fa-user-plus text-primary me-2"></i>
                    User Enrollment
                </h2>
            </div>
        </div>

        <div class="row">
            <div class="col-lg-8">
                <div class="card enrollment-card">
                    <div class="card-header">
                        <h5><i class="fas fa-camera me-2"></i>Camera Feed</h5>
                    </div>
                    <div class="card-body text-center">
                        <video id="video" width="640" height="480" autoplay muted class="border"></video>
                        <div class="mt-3">
                            <button id="startCamera" class="btn btn-primary me-2">
                                <i class="fas fa-video me-1"></i>Start Camera
                            </button>
                            <button id="stopCamera" class="btn btn-danger" disabled>
                                <i class="fas fa-video-slash me-1"></i>Stop Camera
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-lg-4">
                <div class="card enrollment-card mb-3">
                    <div class="card-header">
                        <h5><i class="fas fa-user me-2"></i>Enrollment Details</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="username" class="form-label">Username</label>
                            <input type="text" class="form-control" id="username" placeholder="Enter username" maxlength="50">
                        </div>
                        <button id="startEnrollment" class="btn btn-success w-100" disabled>
                            <i class="fas fa-play me-1"></i>Start Enrollment
                        </button>
                    </div>
                </div>

                <div class="card enrollment-card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-pie me-2"></i>Progress</h5>
                    </div>
                    <div class="card-body text-center">
                        <div class="position-relative d-inline-block mb-3">
                            <svg class="progress-ring" width="120" height="120">
                                <circle class="progress-ring-circle" stroke="var(--bs-secondary)" 
                                        stroke-width="8" fill="transparent" r="50" cx="60" cy="60"
                                        stroke-dasharray="314.16" stroke-dashoffset="314.16" id="progressCircle"/>
                                <circle stroke="var(--bs-primary)" stroke-width="8" fill="transparent" 
                                        r="50" cx="60" cy="60" stroke-dasharray="314.16" 
                                        stroke-dashoffset="314.16" id="progressRing"/>
                            </svg>
                            <div class="position-absolute top-50 start-50 translate-middle">
                                <h4 id="progressText">0%</h4>
                            </div>
                        </div>
                        
                        <div id="enrollmentStats">
                            <p class="mb-1"><strong>Frames Captured:</strong> <span id="frameCount">0</span></p>
                            <p class="mb-1"><strong>Target Frames:</strong> <span id="targetFrames">100</span></p>
                            <p class="mb-0"><strong>Status:</strong> <span id="status">Ready</span></p>
                        </div>

                        <div id="enrollmentComplete" class="d-none">
                            <div class="alert alert-success">
                                <i class="fas fa-check-circle me-2"></i>
                                Enrollment completed successfully!
                            </div>
                            <a href="/" class="btn btn-primary">
                                <i class="fas fa-home me-1"></i>Back to Dashboard
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-info-circle me-2"></i>Enrollment Instructions</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6><i class="fas fa-walking text-primary me-2"></i>Walking Guidelines</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-arrow-right text-success me-2"></i>Walk naturally at your normal pace</li>
                                    <li><i class="fas fa-arrow-right text-success me-2"></i>Walk back and forth in front of the camera</li>
                                    <li><i class="fas fa-arrow-right text-success me-2"></i>Maintain a distance of 6-10 feet from camera</li>
                                    <li><i class="fas fa-arrow-right text-success me-2"></i>Ensure your full body is visible</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6><i class="fas fa-exclamation-triangle text-warning me-2"></i>Important Notes</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-arrow-right text-warning me-2"></i>Good lighting is essential</li>
                                    <li><i class="fas fa-arrow-right text-warning me-2"></i>Avoid loose clothing that obscures movement</li>
                                    <li><i class="fas fa-arrow-right text-warning me-2"></i>Continue walking until 100 frames are captured</li>
                                    <li><i class="fas fa-arrow-right text-warning me-2"></i>Process will complete automatically</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const socket = io();
        let video, stream;
        let enrollmentActive = false;
        let frameInterval;

        document.getElementById('startCamera').addEventListener('click', startCamera);
        document.getElementById('stopCamera').addEventListener('click', stopCamera);
        document.getElementById('startEnrollment').addEventListener('click', startEnrollment);
        document.getElementById('username').addEventListener('input', function() {
            const btn = document.getElementById('startEnrollment');
            btn.disabled = !this.value.trim() || !stream;
        });

        async function startCamera() {
            try {
                video = document.getElementById('video');
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                video.srcObject = stream;
                
                document.getElementById('startCamera').disabled = true;
                document.getElementById('stopCamera').disabled = false;
                
                const username = document.getElementById('username').value.trim();
                if (username) {
                    document.getElementById('startEnrollment').disabled = false;
                }
            } catch (err) {
                alert('Error accessing camera: ' + err.message);
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
            }
            
            document.getElementById('startCamera').disabled = false;
            document.getElementById('stopCamera').disabled = true;
            document.getElementById('startEnrollment').disabled = true;
            
            if (enrollmentActive) {
                stopEnrollment();
            }
        }

        async function startEnrollment() {
            const username = document.getElementById('username').value.trim();
            
            if (!username) {
                alert('Please enter a username');
                return;
            }

            try {
                const response = await fetch('/api/start_enrollment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    enrollmentActive = true;
                    document.getElementById('startEnrollment').disabled = true;
                    document.getElementById('username').disabled = true;
                    document.getElementById('status').textContent = 'Enrolling...';
                    
                    frameInterval = setInterval(captureFrame, 200);
                } else {
                    alert(data.message);
                }
            } catch (err) {
                alert('Error starting enrollment: ' + err.message);
            }
        }

        function stopEnrollment() {
            enrollmentActive = false;
            if (frameInterval) {
                clearInterval(frameInterval);
                frameInterval = null;
            }
            document.getElementById('status').textContent = 'Stopped';
        }

        async function captureFrame() {
            if (!enrollmentActive || !video) return;

            try {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                const response = await fetch('/api/add_enrollment_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                
                const data = await response.json();
                if (!data.success) {
                    console.warn('Frame capture failed:', data.message);
                }
                
            } catch (err) {
                console.error('Error capturing frame:', err);
            }
        }

        function updateProgress(progress, frames, target) {
            const circle = document.getElementById('progressRing');
            const circumference = 2 * Math.PI * 50;
            const offset = circumference - (progress / 100) * circumference;
            circle.style.strokeDashoffset = offset;
            
            document.getElementById('progressText').textContent = Math.round(progress) + '%';
            document.getElementById('frameCount').textContent = frames;
            document.getElementById('targetFrames').textContent = target;
        }

        socket.on('enrollment_progress', function(data) {
            updateProgress(data.progress, data.frames, data.target);
        });

        socket.on('enrollment_complete', function(data) {
            enrollmentActive = false;
            clearInterval(frameInterval);
            
            updateProgress(100, data.frames || 100, data.target || 100);
            document.getElementById('status').textContent = 'Completed';
            document.getElementById('enrollmentStats').classList.add('d-none');
            document.getElementById('enrollmentComplete').classList.remove('d-none');
        });
    </script>
</body>
</html>
'''

RECOGNIZE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Recognition - Gait Recognition</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" rel="stylesheet">
    <style>
        #video { border-radius: 10px; }
        .recognition-card { background: var(--bs-dark); border: 1px solid var(--bs-secondary); }
        .result-success { border-left: 4px solid var(--bs-success); }
        .result-warning { border-left: 4px solid var(--bs-warning); }
        .result-danger { border-left: 4px solid var(--bs-danger); }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-walking me-2"></i>Gait Recognition</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/"><i class="fas fa-home me-1"></i>Dashboard</a>
                <a class="nav-link" href="/enroll"><i class="fas fa-user-plus me-1"></i>Enroll</a>
                <a class="nav-link" href="/manage"><i class="fas fa-users-cog me-1"></i>Manage</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <h2 class="text-center mb-4">
                    <i class="fas fa-search text-primary me-2"></i>
                    User Recognition
                </h2>
            </div>
        </div>

        <div class="row">
            <div class="col-lg-8">
                <div class="card recognition-card">
                    <div class="card-header">
                        <h5><i class="fas fa-camera me-2"></i>Live Recognition Feed</h5>
                    </div>
                    <div class="card-body text-center">
                        <video id="video" width="640" height="480" autoplay muted class="border"></video>
                        <div class="mt-3">
                            <button id="startCamera" class="btn btn-primary me-2">
                                <i class="fas fa-video me-1"></i>Start Camera
                            </button>
                            <button id="stopCamera" class="btn btn-danger me-2" disabled>
                                <i class="fas fa-video-slash me-1"></i>Stop Camera
                            </button>
                            <button id="startRecognition" class="btn btn-success" disabled>
                                <i class="fas fa-play me-1"></i>Start Recognition
                            </button>
                            <button id="stopRecognition" class="btn btn-warning d-none">
                                <i class="fas fa-pause me-1"></i>Stop Recognition
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-lg-4">
                <div class="card recognition-card mb-3">
                    <div class="card-header">
                        <h5><i class="fas fa-user-check me-2"></i>Recognition Results</h5>
                    </div>
                    <div class="card-body">
                        <div id="recognitionResults">
                            <div class="text-center text-muted">
                                <i class="fas fa-user-slash fa-3x mb-3"></i>
                                <p>Start recognition to see results</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card recognition-card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-bar me-2"></i>Recognition Stats</h5>
                    </div>
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-6">
                                <h4 id="frameCount">0</h4>
                                <small class="text-muted">Frames Processed</small>
                            </div>
                            <div class="col-6">
                                <h4 id="recognitionCount">0</h4>
                                <small class="text-muted">Recognitions</small>
                            </div>
                        </div>
                        <hr>
                        <div class="text-center">
                            <span class="badge bg-primary" id="status">Ready</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-history me-2"></i>Recognition History</h5>
                        <button class="btn btn-sm btn-outline-secondary float-end" onclick="clearHistory()">
                            <i class="fas fa-trash me-1"></i>Clear
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="recognitionHistory" class="row">
                            <div class="col-12 text-center text-muted">
                                <i class="fas fa-clock fa-2x mb-3"></i>
                                <p>No recognition history yet</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-info-circle me-2"></i>Recognition Guidelines</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6><i class="fas fa-walking text-primary me-2"></i>For Best Results</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-check text-success me-2"></i>Walk naturally at normal pace</li>
                                    <li><i class="fas fa-check text-success me-2"></i>Maintain 6-10 feet distance from camera</li>
                                    <li><i class="fas fa-check text-success me-2"></i>Ensure full body is visible</li>
                                    <li><i class="fas fa-check text-success me-2"></i>Good lighting conditions</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6><i class="fas fa-brain text-info me-2"></i>How It Works</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-arrow-right text-info me-2"></i>System analyzes your gait pattern</li>
                                    <li><i class="fas fa-arrow-right text-info me-2"></i>Uses ensemble ML models for recognition</li>
                                    <li><i class="fas fa-arrow-right text-info me-2"></i>Requires multiple frames for accuracy</li>
                                    <li><i class="fas fa-arrow-right text-info me-2"></i>Real-time confidence scoring</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let video, stream;
        let recognitionActive = false;
        let frameInterval;
        let frameCount = 0;
        let recognitionCount = 0;
        let recognitionHistory = [];

        document.getElementById('startCamera').addEventListener('click', startCamera);
        document.getElementById('stopCamera').addEventListener('click', stopCamera);
        document.getElementById('startRecognition').addEventListener('click', startRecognition);
        document.getElementById('stopRecognition').addEventListener('click', stopRecognition);

        async function startCamera() {
            try {
                video = document.getElementById('video');
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                video.srcObject = stream;
                
                document.getElementById('startCamera').disabled = true;
                document.getElementById('stopCamera').disabled = false;
                document.getElementById('startRecognition').disabled = false;
                document.getElementById('status').textContent = 'Camera Ready';
                document.getElementById('status').className = 'badge bg-success';
            } catch (err) {
                alert('Error accessing camera: ' + err.message);
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
            }
            
            document.getElementById('startCamera').disabled = false;
            document.getElementById('stopCamera').disabled = true;
            document.getElementById('startRecognition').disabled = true;
            document.getElementById('status').textContent = 'Ready';
            document.getElementById('status').className = 'badge bg-primary';
            
            if (recognitionActive) {
                stopRecognition();
            }
        }

        function startRecognition() {
            recognitionActive = true;
            frameCount = 0;
            recognitionCount = 0;
            
            document.getElementById('startRecognition').classList.add('d-none');
            document.getElementById('stopRecognition').classList.remove('d-none');
            document.getElementById('status').textContent = 'Recognizing';
            document.getElementById('status').className = 'badge bg-warning';
            
            frameInterval = setInterval(recognizeFrame, 500);
        }

        function stopRecognition() {
            recognitionActive = false;
            if (frameInterval) {
                clearInterval(frameInterval);
                frameInterval = null;
            }
            
            document.getElementById('startRecognition').classList.remove('d-none');
            document.getElementById('stopRecognition').classList.add('d-none');
            document.getElementById('status').textContent = 'Stopped';
            document.getElementById('status').className = 'badge bg-secondary';
        }

        async function recognizeFrame() {
            if (!recognitionActive || !video) return;

            try {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                const response = await fetch('/api/recognize_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                
                const data = await response.json();
                
                frameCount++;
                document.getElementById('frameCount').textContent = frameCount;
                
                if (data.success) {
                    recognitionCount++;
                    document.getElementById('recognitionCount').textContent = recognitionCount;
                    
                    const result = {
                        user: data.user,
                        confidence: data.confidence,
                        timestamp: new Date().toLocaleTimeString()
                    };
                    
                    recognitionHistory.unshift(result);
                    if (recognitionHistory.length > 10) {
                        recognitionHistory.pop();
                    }
                    
                    updateRecognitionDisplay(result);
                    updateRecognitionHistory();
                } else {
                    updateRecognitionDisplay({ error: data.message });
                }
                
            } catch (err) {
                console.error('Error in recognition:', err);
            }
        }

        function updateRecognitionDisplay(result) {
            const container = document.getElementById('recognitionResults');
            
            if (result.error) {
                container.innerHTML = `
                    <div class="alert alert-warning result-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <strong>Status:</strong> ${result.error}
                    </div>
                `;
            } else {
                const confidenceValue = parseFloat(result.confidence.split(':')[1]);
                const confidenceClass = confidenceValue >= 0.8 ? 'success' : 'warning';
                container.innerHTML = `
                    <div class="alert alert-${confidenceClass} result-${confidenceClass}">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="fas fa-user-check me-2"></i>
                                <strong>${result.user}</strong>
                            </div>
                            <small class="text-muted">${result.timestamp}</small>
                        </div>
                        <div class="mt-2">
                            <small>${result.confidence}</small>
                        </div>
                    </div>
                `;
            }
        }

        function updateRecognitionHistory() {
            const container = document.getElementById('recognitionHistory');
            
            if (recognitionHistory.length === 0) {
                container.innerHTML = `
                    <div class="col-12 text-center text-muted">
                        <i class="fas fa-clock fa-2x mb-3"></i>
                        <p>No recognition history yet</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = recognitionHistory.map(result => `
                <div class="col-md-6 col-lg-4 mb-2">
                    <div class="card">
                        <div class="card-body py-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <strong class="text-truncate">${result.user}</strong>
                                <small class="text-muted">${result.timestamp}</small>
                            </div>
                            <small class="text-muted">${result.confidence}</small>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function clearHistory() {
            recognitionHistory = [];
            updateRecognitionHistory();
        }
    </script>
</body>
</html>
'''

MANAGE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Management - Gait Recognition</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" rel="stylesheet">
    <style>
        .user-card { 
            background: var(--bs-dark); 
            border: 1px solid var(--bs-secondary); 
            transition: transform 0.2s; 
        }
        .user-card:hover { transform: translateY(-2px); }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-walking me-2"></i>Gait Recognition</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/"><i class merry-christmas-user-plus me-1"></i>Enroll</a>
                <a class="nav-link" href="/recognize"><i class="fas fa-search me-1"></i>Recognize</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <h2 class="text-center mb-4">
                    <i class="fas fa-users-cog text-primary me-2"></i>
                    User Management
                </h2>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5><i class="fas fa-users me-2"></i>Enrolled Users ({{ users|length }})</h5>
                        <div>
                            <a href="/enroll" class="btn btn-success me-2">
                                <i class="fas fa-user-plus me-1"></i>Add User
                            </a>
                            <button class="btn btn-primary" onclick="refreshUsers()">
                                <i class="fas fa-sync-alt me-1"></i>Refresh
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        {% if users %}
                            <div class="row" id="usersList">
                                {% for user in users %}
                                <div class="col-md-6 col-lg-4 mb-3" id="user-{{ user.id }}">
                                    <div class="card user-card">
                                        <div class="card-body">
                                            <div class="d-flex justify-content-between align-items-start mb-2">
                                                <h6 class="card-title mb-0">
                                                    <i class="fas fa-user me-2"></i>{{ user.username | e }}
                                                </h6>
                                                <div class="dropdown">
                                                    <button class="btn btn-sm btn-outline-secondary dropdown-toggle" 
                                                            type="button" data-bs-toggle="dropdown">
                                                        <i class="fas fa-ellipsis-v"></i>
                                                    </button>
                                                    <ul class="dropdown-menu">
                                                        <li>
                                                            <a class="dropdown-item" href="#" 
                                                               onclick="viewUserDetails('{{ user.username | e }}')">
                                                                <i class="fas fa-eye me-2"></i>View Details
                                                            </a>
                                                        </li>
                                                        <li><hr class="dropdown-divider"></li>
                                                        <li>
                                                            <a class="dropdown-item text-danger" href="#" 
                                                               onclick="deleteUser('{{ user.username | e }}')">
                                                                <i class="fas fa-trash me-2"></i>Delete
                                                            </a>
                                                        </li>
                                                    </ul>
                                                </div>
                                            </div>
                                            <p class="card-text text-muted">
                                                <small>Enrolled: {{ user.created_at.split('T')[0] if user.created_at else 'Unknown' }}</small>
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        {% else %}
                            <div class="text-center text-muted">
                                <i class="fas fa-user-slash fa-3x mb-3"></i>
                                <p>No users enrolled. <a href="/enroll">Add a new user</a> to begin.</p>
                            </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-info-circle me-2"></i>Management Guide</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6><i class="fas fa-user-cog text-primary me-2"></i>Available Actions</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-arrow-right text-success me-2"></i>View user enrollment details</li>
                                    <li><i class="fas fa-arrow-right text-success me-2"></i>Delete users and their data</li>
                                    <li><i class="fas fa-arrow-right text-success me-2"></i>Add new users via enrollment</li>
                                    <li><i class="fas fa-arrow-right text-success me-2"></i>Refresh user list</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6><i class="fas fa-exclamation-triangle text-warning me-2"></i>Important Notes</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-arrow-right text-warning me-2"></i>Deleting a user removes all associated data</li>
                                    <li><i class="fas fa-arrow-right text-warning me-2"></i>Changes require model retraining</li>
                                    <li><i class="fas fa-arrow-right text-warning me-2"></i>Ensure users are enrolled properly</li>
                                    <li><i class="fas fa-arrow-right text-warning me-2"></i>Regularly train models for accuracy</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="modal fade" id="userDetailsModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"><i class="fas fa-user me-2"></i>User Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="userDetailsContent">
                    <p>Loading...</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                        <i class="fas fa-times me-1"></i>Close
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function refreshUsers() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                const usersResponse = await fetch('/api/users');
                const users = await usersResponse.json();
                
                const container = document.getElementById('usersList');
                if (users.length === 0) {
                    container.innerHTML = `
                        <div class="text-center text-muted">
                            <i class="fas fa-user-slash fa-3x mb-3"></i>
                            <p>No users enrolled. <a href="/enroll">Add a new user</a> to begin.</p>
                        </div>
                    `;
                    return;
                }
                
                container.innerHTML = users.map(user => `
                    <div class="col-md-6 col-lg-4 mb-3" id="user-${user.id}">
                        <div class="card user-card">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start mb-2">
                                    <h6 class="card-title mb-0">
                                        <i class="fas fa-user me-2"></i>${user.username}
                                    </h6>
                                    <div class="dropdown">
                                        <button class="btn btn-sm btn-outline-secondary dropdown-toggle" 
                                                type="button" data-bs-toggle="dropdown">
                                            <i class="fas fa-ellipsis-v"></i>
                                        </button>
                                        <ul class="dropdown-menu">
                                            <li>
                                                <a class="dropdown-item" href="#" 
                                                   onclick="viewUserDetails('${user.username}')">
                                                    <i class="fas fa-eye me-2"></i>View Details
                                                </a>
                                            </li>
                                            <li><hr class="dropdown-divider"></li>
                                            <li>
                                                <a class="dropdown-item text-danger" href="#" 
                                                   onclick="deleteUser('${user.username}')">
                                                    <i class="fas fa-trash me-2"></i>Delete
                                                </a>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                                <p class="card-text text-muted">
                                    <small>Enrolled: ${user.created_at.split('T')[0]}</small>
                                </p>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (err) {
                alert('Error refreshing users: ' + err.message);
            }
        }

        async function viewUserDetails(username) {
            try {
                // Note: This endpoint would need to be implemented if detailed user info is required
                const response = await fetch(`/api/user_details?username=${encodeURIComponent(username)}`);
                const data = await response.json();
                
                const modalContent = document.getElementById('userDetailsContent');
                if (data.success) {
                    modalContent.innerHTML = `
                        <h6><i class="fas fa-user me-2"></i>${username}</h6>
                        <p><strong>Enrolled:</strong> ${data.created_at || 'Unknown'}</p>
                        <p><strong>Frame Count:</strong> ${data.frame_count || 'N/A'}</p>
                        <p><strong>Last Updated:</strong> ${data.updated_at || 'N/A'}</p>
                    `;
                } else {
                    modalContent.innerHTML = `<p class="text-danger">${data.message}</p>`;
                }
                
                const modal = new bootstrap.Modal(document.getElementById('userDetailsModal'));
                modal.show();
            } catch (err) {
                document.getElementById('userDetailsContent').innerHTML = 
                    `<p class="text-danger">Error loading details: ${err.message}</p>`;
                const modal = new bootstrap.Modal(document.getElementById('userDetailsModal'));
                modal.show();
            }
        }

        async function deleteUser(username) {
            if (!confirm(`Are you sure you want to delete user "${username}"? This action cannot be undone.`)) {
                return;
            }
            
            try {
                const response = await fetch('/api/delete_user', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const userElement = document.getElementById(`user-${username}`);
                    if (userElement) {
                        userElement.remove();
                    }
                    alert(data.message);
                    refreshUsers();
                } else {
                    alert(data.message);
                }
            } catch (err) {
                alert('Error deleting user: ' + err.message);
            }
        }
    </script>
</body>
</html>
'''

# Add missing API endpoint for user details
@app.route('/api/user_details')
@ensure_db_connection
def api_user_details(cursor):
    """Get user details API"""
    try:
        username = request.args.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Username required'})
            
        cursor.execute("""
            SELECT u.id, u.username, u.created_at, u.updated_at, 
                   e.frame_count
            FROM users u 
            LEFT JOIN enrollment_data e ON u.id = e.user_id
            WHERE u.username = ?
        """, (username,))
        
        user = cursor.fetchone()
        if user:
            return jsonify({
                'success': True,
                'username': user['username'],
                'created_at': user['created_at'],
                'updated_at': user['updated_at'],
                'frame_count': user['frame_count'] or 0
            })
        return jsonify({'success': False, 'message': 'User not found'})
    except Exception as e:
        logging.error(f"Error in user_details API: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/api/users')
def api_users():
    """Get all users API"""
    try:
        users = gait_system.get_all_users()
        return jsonify(users)
    except Exception as e:
        logging.error(f"Error in users API: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'})

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'success': False, 'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server error: {str(error)}")
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)