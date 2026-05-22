"""
Models package for Face Recognition App
This file tells Python that this folder is a package
"""

from .face_model import FaceRecognitionModel
from .image_analyzer import ImageAnalyzer

# What gets imported when we do "from models import *"
__all__ = ['FaceRecognitionModel', 'ImageAnalyzer']