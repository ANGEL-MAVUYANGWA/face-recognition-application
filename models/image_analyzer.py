"""
Image Analyzer
Junior Developer Implementation
Analyzes image quality, colors, and composition
"""

import cv2
import numpy as np
from PIL import Image

class ImageAnalyzer:
    """
    This class analyzes various aspects of an image:
    - Quality (blur, brightness, contrast)
    - Colors (colorfulness, dominant colors)
    - Composition (aspect ratio, symmetry)
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        # Try to load AI model for image captioning (optional)
        self.use_ai = False
        try:
            # This is optional - if the libraries aren't installed, skip
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.use_ai = True
            print("AI image captioning model loaded! - image_analyzer.py:32")
        except Exception as e:
            print(f"AI model not available: {e} - image_analyzer.py:34")
            print("Image captioning will use simple descriptions instead - image_analyzer.py:35")
    
    def analyze_image(self, image_path):
        """
        Analyze an image and return all metrics
        Args:
            image_path: Path to the image file
        Returns:
            Dictionary with all analysis results
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not load image'}
        
        # Run all analyses
        result = {
            'image_quality': self._analyze_quality(img),
            'color_analysis': self._analyze_colors(img),
            'composition': self._analyze_composition(img),
            'description': self._generate_description(image_path)
        }
        
        return result
    
    def _analyze_quality(self, img):
        """
        Analyze image quality metrics
        Args:
            img: OpenCV image (BGR format)
        Returns:
            Dictionary with quality metrics
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Calculate blur score using Laplacian variance
        # Higher variance = sharper image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        # 2. Calculate average brightness (0-255)
        brightness = np.mean(gray)
        
        # 3. Calculate contrast (standard deviation of pixel values)
        contrast = np.std(gray)
        
        # 4. Convert metrics to quality scores (0-100)
        # For blur: assume 500 is "good" sharpness
        quality_score = min(100, max(0, (blur_score / 500) * 100))
        
        # Determine status messages
        is_blurry = blur_score < 100
        is_dark = brightness < 80
        is_bright = brightness > 200
        is_low_contrast = contrast < 40
        
        return {
            'blur_score': float(blur_score),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'quality_score': float(quality_score),
            'is_blurry': is_blurry,
            'is_dark': is_dark,
            'is_bright': is_bright,
            'is_low_contrast': is_low_contrast,
            'status': self._get_quality_status(is_blurry, is_dark, is_low_contrast)
        }
    
    def _get_quality_status(self, is_blurry, is_dark, is_low_contrast):
        """
        Get a human-readable quality status
        """
        if is_blurry:
            return "Image is blurry - try using a steadier hand"
        if is_dark:
            return "Image is too dark - try better lighting"
        if is_low_contrast:
            return "Image has low contrast - subjects may not stand out"
        return "Image quality looks good!"
    
    def _analyze_colors(self, img):
        """
        Analyze color characteristics of the image
        Args:
            img: OpenCV image
        Returns:
            Dictionary with color analysis
        """
        # Convert BGR to RGB for analysis
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate colorfulness metric
        # Formula from: "Measuring colourfulness in natural images" (Hasler and Süsstrunk)
        R = rgb_img[:, :, 0].astype(float)
        G = rgb_img[:, :, 1].astype(float)
        B = rgb_img[:, :, 2].astype(float)
        
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        
        colorfulness = np.mean(rg) + np.mean(yb)
        
        # Calculate average saturation (simplified)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv_img[:, :, 1])
        
        # Get dominant colors (simplified - sample pixels)
        # Reshape to list of pixels and find most common colors
        pixels = rgb_img.reshape(-1, 3)
        
        # Sample only every 100th pixel for performance
        sample_size = min(1000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[indices]
        
        # Get unique colors and their counts (rounded to multiples of 20 for grouping)
        rounded = (sample_pixels // 20) * 20
        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        
        # Get top 3 colors
        top_indices = np.argsort(counts)[-3:][::-1]
        dominant_colors = unique[top_indices].tolist() if len(top_indices) > 0 else []
        
        return {
            'colorfulness': float(colorfulness),
            'avg_saturation': float(avg_saturation),
            'dominant_colors': dominant_colors,
            'is_grayscale': colorfulness < 20,
            'is_colorful': colorfulness > 50,
            'description': self._get_color_description(colorfulness, avg_saturation)
        }
    
    def _get_color_description(self, colorfulness, saturation):
        """
        Get a human-readable color description
        """
        if colorfulness < 20:
            return "This image is mostly grayscale or has very muted colors"
        elif colorfulness < 50:
            return "This image has moderate colors"
        else:
            return "This image is very colorful!"
    
    def _analyze_composition(self, img):
        """
        Analyze image composition (layout, symmetry, etc.)
        Args:
            img: OpenCV image
        Returns:
            Dictionary with composition metrics
        """
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        # Calculate symmetry score
        # Split image into left and right halves and compare
        mid = w // 2
        left_half = img[:, :mid]
        right_half = cv2.flip(img[:, mid:], 1)
        
        # Make halves the same size for comparison
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate similarity (1 = identical, 0 = completely different)
        if left_half.shape == right_half.shape and left_half.size > 0:
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            similarity = 1 - (np.mean(diff) / 255)
        else:
            similarity = 0.5
        
        # Determine orientation
        if aspect_ratio > 1.2:
            orientation = "landscape"
            orientation_desc = "wider than tall"
        elif aspect_ratio < 0.8:
            orientation = "portrait"
            orientation_desc = "taller than wide"
        else:
            orientation = "square"
            orientation_desc = "approximately square"
        
        return {
            'width': w,
            'height': h,
            'aspect_ratio': round(aspect_ratio, 2),
            'symmetry_score': float(similarity),
            'orientation': orientation,
            'orientation_desc': orientation_desc,
            'is_symmetric': similarity > 0.7,
            'composition_tip': self._get_composition_tip(aspect_ratio, similarity)
        }
    
    def _get_composition_tip(self, aspect_ratio, symmetry):
        """
        Provide a simple composition tip
        """
        if symmetry > 0.7:
            return "This image is quite symmetric - good for formal compositions"
        elif aspect_ratio > 1.2:
            return "Landscape orientation works well for scenery and group photos"
        elif aspect_ratio < 0.8:
            return "Portrait orientation is great for individual portraits"
        else:
            return "Consider the rule of thirds for better composition"
    
    def _generate_description(self, image_path):
        """
        Generate a text description of the image
        Uses AI if available, otherwise provides simple description
        """
        if self.use_ai:
            try:
                # Use BLIP model for AI captioning
                image = Image.open(image_path).convert('RGB')
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                out = self.model.generate(**inputs, max_length=50)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                return f"AI Analysis: {caption}"
            except Exception as e:
                return f"AI description unavailable: {str(e)}"
        else:
            # Simple description based on analysis
            return "Upload an image to see AI-powered analysis. For better results, install torch and transformers."