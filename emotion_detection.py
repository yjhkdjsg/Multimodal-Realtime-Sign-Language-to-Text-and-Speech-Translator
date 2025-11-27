import threading
import queue
import math
from collections import deque
from typing import Dict
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

class AsyncEmotionDetector:
    
    def __init__(
        self,
        buffer_size: int = 15,
        skip_rate: int = 10,
        min_confidence: float = 0.3,
        conf_threshold: float = 35.0,
        model_name: str = 'FER', 
        use_histogram_eq: bool = True,
        use_weighted_voting: bool = True
    ):
        self.skip_rate = skip_rate
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.use_histogram_eq = use_histogram_eq
        self.use_weighted_voting = use_weighted_voting
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0, 
            min_detection_confidence=min_confidence
        )
        print(f"[AsyncEmotion] Warming up DeepFace with {model_name} model...")
        try:
            dummy = np.zeros((48, 48, 3), dtype=np.uint8)
            if model_name == 'ensemble':
                DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, 
                               detector_backend='skip', silent=True)
            else:
                DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, 
                               detector_backend='skip', silent=True)
        except:
            pass

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()
        
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
        self.latest_result = {
            "emotion": "neutral", 
            "confidence": 0.0,
            "is_active": False,
            "all_emotions": {}
        }
        self.frame_count = 0

    def start(self):
        if self.running: return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def process_frame(self, frame: np.ndarray):
        """Non-blocking push to worker thread."""
        self.frame_count += 1
        if self.frame_count % self.skip_rate != 0:
            return
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_emotion_data(self) -> Dict:
        """Get smoothed emotion data."""
        with self.lock:
            return self.latest_result.copy()

    def _worker_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                aligned_face = self._extract_aligned_face(frame)
                if aligned_face is not None:
                    if self.model_name == 'ensemble':
                        emotion_data = self._analyze_ensemble(aligned_face)
                    else:
                        emotion_data = self._analyze_single_model(aligned_face)
                    
                    if emotion_data:
                        dominant = emotion_data['dominant_emotion']
                        conf = emotion_data['confidence']
                        all_emotions = emotion_data['all_emotions']

                        with self.lock:
                            if conf > self.conf_threshold:
                                self.emotion_buffer.append(dominant)
                                self.confidence_buffer.append(conf)
                            
                            if self.emotion_buffer:
                                if self.use_weighted_voting:
                                    smoothed = self._get_smoothed_emotion_weighted()
                                else:
                                    smoothed = max(set(self.emotion_buffer), 
                                                 key=self.emotion_buffer.count)
                            else:
                                smoothed = dominant
                            
                            avg_conf = np.mean(self.confidence_buffer) if self.confidence_buffer else conf
                                
                            self.latest_result = {
                                "emotion": smoothed,
                                "confidence": float(avg_conf),
                                "is_active": True,
                                "all_emotions": all_emotions
                            }
                else:
                    with self.lock:
                        self.latest_result["is_active"] = False
            except Exception as e:
                pass

    def _analyze_single_model(self, face_img):
        """Analyze using a single DeepFace model."""
        try:
            results = DeepFace.analyze(
                img_path=face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )
            
            res = results[0] if isinstance(results, list) else results
            dominant = res['dominant_emotion']
            all_emotions = res['emotion']
            conf = all_emotions[dominant]
            
            return {
                'dominant_emotion': dominant,
                'confidence': conf,
                'all_emotions': all_emotions
            }
        except:
            return None

    def _analyze_ensemble(self, face_img):
        try:
            results_list = []
            
            res1 = DeepFace.analyze(face_img, actions=['emotion'], 
                                   enforce_detection=False, detector_backend='skip', silent=True)
            results_list.append(res1[0] if isinstance(res1, list) else res1)
            
            brightened = cv2.convertScaleAbs(face_img, alpha=1.1, beta=10)
            res2 = DeepFace.analyze(brightened, actions=['emotion'], 
                                   enforce_detection=False, detector_backend='skip', silent=True)
            results_list.append(res2[0] if isinstance(res2, list) else res2)
            
            combined_emotions = {}
            for res in results_list:
                for emotion, score in res['emotion'].items():
                    combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score
            
            for emotion in combined_emotions:
                combined_emotions[emotion] /= len(results_list)
            
            dominant = max(combined_emotions, key=combined_emotions.get)
            
            return {
                'dominant_emotion': dominant,
                'confidence': combined_emotions[dominant],
                'all_emotions': combined_emotions
            }
        except:
            return None

    def _get_smoothed_emotion_weighted(self):
        """Weighted voting with recency bias - recent emotions matter more."""
        if not self.emotion_buffer:
            return None
        
        weights = np.linspace(0.5, 1.0, len(self.emotion_buffer))
        emotion_scores = {}
        
        for emotion, weight in zip(self.emotion_buffer, weights):
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + weight
        
        return max(emotion_scores, key=emotion_scores.get)

    def _extract_aligned_face(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None
        
        det = max(results.detections, 
                 key=lambda d: d.location_data.relative_bounding_box.width)
        kp = det.location_data.relative_keypoints
        
        right_eye = np.array([kp[0].x * w, kp[0].y * h])
        left_eye = np.array([kp[1].x * w, kp[1].y * h])
        
        eye_center = ((right_eye[0] + left_eye[0]) / 2, 
                     (right_eye[1] + left_eye[1]) / 2)
        dy = left_eye[1] - right_eye[1]
        dx = left_eye[0] - right_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        current_eye_dist = np.linalg.norm(left_eye - right_eye)
        desired_eye_dist = 16.0 
        scale = desired_eye_dist / current_eye_dist if current_eye_dist > 0 else 1.0
        
        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        
        M[0, 2] += (24 - eye_center[0])
        M[1, 2] += (18 - eye_center[1]) 
        
        aligned = cv2.warpAffine(frame, M, (48, 48))
        
        if self.use_histogram_eq:
            aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            aligned_gray = cv2.equalizeHist(aligned_gray)
            aligned = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)
        
        return aligned

if __name__ == "__main__":
    detector_fast = AsyncEmotionDetector(
        model_name='Emotion',
        skip_rate=10,
        buffer_size=12
    )
    
    detector_accurate = AsyncEmotionDetector(
        model_name='FER',
        skip_rate=8,
        buffer_size=15,
        conf_threshold=40.0
    )
    
    detector_best = AsyncEmotionDetector(
        model_name='ensemble',
        skip_rate=12,  
        buffer_size=20,
        conf_threshold=45.0,
        use_histogram_eq=True,
        use_weighted_voting=True
    )
    
    detector_accurate.start()