import numpy as np
import cv2
from ex2_utils import get_patch
from ex3_utils import create_cosine_window, create_gauss_peak
from pathlib import Path
import sys

# Add toolkit directory to Python path
sys.path.append(str(Path("/toolkit-dir").resolve()))
from utils.tracker import Tracker

class MosseTracker(Tracker):
    
    def __init__(self):
        # Tracking parameters
        self.learning_rate = 0.15       # How quickly the filter adapts
        self.gaussian_std = 2           # Width of target response peak
        self.regularization = 0.1       # Prevents division by zero
        self.search_area_scale = 1      # Enlargement factor for search region
        
        # Tracking state
        self.target_center = None       # (x,y) coordinates of target center
        self.search_window_size = None  # Size of search window (w,h)
        self.target_size = None         # Original target size (w,h)
        self.cosine_window = None      # Window to reduce boundary effects
        self.correlation_filter = None  # Learned filter in frequency domain
        self.ideal_response_fft = None # FFT of desired Gaussian response

    def name(self):
        return "mosse_tracker"

    def preprocess_patch(self, image_patch):

        # Convert to grayscale if needed
        if len(image_patch.shape) == 3:
            image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast and normalize
        image_patch = np.log1p(image_patch)  # Log transform
        image_patch -= np.mean(image_patch)  # Zero-mean
        image_patch /= np.linalg.norm(image_patch)  # Unit variance
        
        # Apply window function and convert to frequency domain
        return np.fft.fft2(image_patch * self.cosine_window)

    def compute_filter(self, patch_fft):

        numerator = self.ideal_response_fft * np.conj(patch_fft)
        denominator = (patch_fft * np.conj(patch_fft)) + self.regularization
        return numerator / denominator

    def initialize(self, image, bounding_box):

        # Calculate target center position
        box_x, box_y, box_w, box_h = bounding_box
        self.target_center = (box_x + box_w/2, box_y + box_h/2)
        
        # Set sizes ensuring odd dimensions for FFT efficiency
        self.target_size = (round(box_w) | 1, round(box_h) | 1)
        self.search_window_size = (
            round(box_w * self.search_area_scale) | 1, 
            round(box_h * self.search_area_scale) | 1
        )

        # Create preprocessing window and desired response
        self.cosine_window = create_cosine_window(self.search_window_size)
        gaussian_response = create_gauss_peak(self.search_window_size, self.gaussian_std)
        self.ideal_response_fft = np.fft.fft2(gaussian_response)

        # Initialize correlation filter
        initial_patch = get_patch(image, self.target_center, self.search_window_size)[0]
        initial_patch_fft = self.preprocess_patch(initial_patch)
        self.correlation_filter = self.compute_filter(initial_patch_fft)

    def track(self, image):

        # Extract and preprocess search region
        search_patch = get_patch(image, self.target_center, self.search_window_size)[0]
        patch_fft = self.preprocess_patch(search_patch)
        
        # Compute correlation response
        response = np.fft.ifft2(self.correlation_filter * patch_fft)
        
        # Find displacement from response peak
        peak_row, peak_col = np.unravel_index(response.argmax(), response.shape)
        if peak_col > self.search_window_size[0] // 2:
            peak_col -= self.search_window_size[0]
        if peak_row > self.search_window_size[1] // 2:
            peak_row -= self.search_window_size[1]

        # Update target position
        new_x = self.target_center[0] + peak_col
        new_y = self.target_center[1] + peak_row
        self.target_center = (new_x, new_y)

        # Update correlation filter
        updated_patch = get_patch(image, self.target_center, self.search_window_size)[0]
        updated_filter = self.compute_filter(self.preprocess_patch(updated_patch))
        self.correlation_filter = (
            (1 - self.learning_rate) * self.correlation_filter + 
            self.learning_rate * updated_filter
        )

        # Return bounding box [x, y, width, height]
        return [
            self.target_center[0] - self.target_size[0] // 2,
            self.target_center[1] - self.target_size[1] // 2,
            self.target_size[0],
            self.target_size[1]
        ]
    