"""
Face Recognition Model
Junior Developer Implementation
Uses Haar Cascades for face detection
"""

import cv2
import numpy as np
import os

class FaceRecognitionModel:
    """
    This class handles face detection in images
    It uses OpenCV's Haar Cascade classifier
    """
    
    def __init__(self):
        """Initialize the face detector"""
        # Path to the Haar cascade file (provided by OpenCV)
        cascade_path = 'haarcascade_frontalface_default.xml'
        
        # Try to load the cascade file
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("Face detector loaded successfully! - face_model.py:25")
        else:
            print(f"Warning: {cascade_path} not found! - face_model.py:27")
            print("Please download it from OpenCV or place it in the project folder - face_model.py:28")
            self.face_cascade = None
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image
        Args:
            image_path: Path to the image file
        Returns:
            List of dictionaries containing face information
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path} - face_model.py:42")
            return []
        
        # Convert to grayscale (Haar cascades work better on grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # If no cascade file, return empty list
        if self.face_cascade is None:
            return []
        
        # Detect faces
        # Parameters:
        # - scaleFactor: How much image size is reduced each scale (1.1 = 10% reduction)
        # - minNeighbors: How many neighbors each rectangle should have (higher = fewer detections)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Process each detected face
        face_images = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract the face region
            face_img = gray[y:y+h, x:x+w]
            
            # Store face information
            face_info = {
                'index': i,
                'bbox': (x, y, w, h),  # Bounding box coordinates
                'size': w * h,  # Area of the face
                'aspect_ratio': w / h if h > 0 else 0  # Width to height ratio
            }
            
            # Simple gender guess based on aspect ratio (just for fun, not accurate!)
            # In real apps, you'd use a trained model for this
            if face_info['aspect_ratio'] > 0.9:
                face_info['estimated_gender'] = 'male'
                face_info['confidence'] = 0.55  # Low confidence, just for demo
            else:
                face_info['estimated_gender'] = 'female'
                face_info['confidence'] = 0.55
            
            face_images.append(face_info)
        
        return face_images
    
    def analyze_face(self, image_path):
        """
        Comprehensive face analysis
        Args:
            image_path: Path to the image file
        Returns:
            Dictionary with analysis results
        """
        # Detect faces
        faces = self.detect_faces(image_path)
        
        # If no faces detected
        if not faces:
            return {
                'faces_detected': 0,
                'message': 'No faces detected in the image. Try a clearer photo with good lighting.',
                'faces': []
            }
        
        # Build the result
        result = {
            'faces_detected': len(faces),
            'message': f'Found {len(faces)} face(s) in the image',
            'faces': faces
        }
        
        # Add a tip if multiple faces
        if len(faces) > 1:
            result['tip'] = 'Multiple faces detected! The analysis shows all faces found.'
        
        return result