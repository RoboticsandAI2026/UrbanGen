import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import traceback
from shapely.geometry import Polygon, LineString
from scipy.spatial import distance

class BoundaryProcessor:
    """
    Specialized processor for handling hand-drawn boundary inputs.
    This processor cleans, denoises, and extracts proper boundaries from sketch inputs.
    """
    
    def __init__(self):
        # Parameters for boundary detection
        self.edge_threshold1 = 30
        self.edge_threshold2 = 150
        self.hough_threshold = 25
        self.min_line_length = 30
        self.max_line_gap = 20
        self.intersection_distance_threshold = 20
        self.line_extension_factor = 0.2  # How much to extend lines for better intersection
        
    def is_sketch_input(self, image):
        """
        Determine if an image is likely a hand-drawn sketch rather than a photograph
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate the histogram of the image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / (gray.shape[0] * gray.shape[1])
        
        # Get histogram statistics
        non_zero_bins = np.count_nonzero(hist > 0.001)
        hist_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
        
        # Sketches typically have:
        # - Low histogram entropy (few distinct colors)
        # - Few populated histogram bins
        # - Higher edge density
        is_sketch = (hist_entropy < 4.5 and non_zero_bins < 50) or edge_density > 0.15
        
        return is_sketch
    
    def preprocess_sketch(self, image):
        """
        Preprocess a sketch input to enhance boundaries
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply thresholding to enhance contrast
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Clean up salt & pepper noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert if necessary - want dark lines on light background
        if np.mean(binary) > 127:
            binary = 255 - binary
            
        return binary
    
    def extract_lines_from_sketch(self, preprocessed_image):
        """
        Extract line segments from a preprocessed sketch
        """
        # Apply edge detection
        edges = cv2.Canny(preprocessed_image, self.edge_threshold1, self.edge_threshold2)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
            
        return [line[0] for line in lines]
    
    def find_line_intersections(self, lines):
        """
        Find the intersection points of the detected lines
        """
        intersections = []
        
        # Convert lines to shapely LineString objects with extensions
        shapely_lines = []
        for x1, y1, x2, y2 in lines:
            # Calculate line extension
            dx = x2 - x1
            dy = y2 - y1
            
            # Extend the line in both directions
            ext_x1 = x1 - dx * self.line_extension_factor
            ext_y1 = y1 - dy * self.line_extension_factor
            ext_x2 = x2 + dx * self.line_extension_factor
            ext_y2 = y2 + dy * self.line_extension_factor
            
            line = LineString([(ext_x1, ext_y1), (ext_x2, ext_y2)])
            shapely_lines.append(line)
        
        # Find all intersections
        for i, line1 in enumerate(shapely_lines):
            for j, line2 in enumerate(shapely_lines):
                if i >= j:  # Avoid duplicate comparisons and self-intersection
                    continue
                    
                if line1.intersects(line2):
                    point = line1.intersection(line2)
                    intersections.append((point.x, point.y))
        
        return intersections
    
    def filter_corner_points(self, intersections, image_shape):
        """
        Filter intersection points to find the corners of the boundary
        """
        if not intersections:
            return []
            
        # Convert to numpy array
        points = np.array(intersections)
        
        # Remove duplicates by clustering nearby points
        filtered_points = []
        used_indices = set()
        
        for i, point in enumerate(points):
            if i in used_indices:
                continue
                
            nearby_points = []
            nearby_indices = []
            
            for j, other_point in enumerate(points):
                if j in used_indices:
                    continue
                    
                dist = distance.euclidean(point, other_point)
                if dist < self.intersection_distance_threshold:
                    nearby_points.append(other_point)
                    nearby_indices.append(j)
            
            if nearby_points:
                cluster_center = np.mean(nearby_points, axis=0)
                filtered_points.append(cluster_center)
                used_indices.update(nearby_indices)
        
        # Ensure points are inside the image boundaries
        h, w = image_shape[:2]
        filtered_points = [p for p in filtered_points if 0 <= p[0] < w and 0 <= p[1] < h]
        
        return filtered_points
    
    def order_boundary_points(self, points):
        """
        Order points to form a proper boundary polygon
        """
        if len(points) < 3:
            return np.array([])
            
        # Convert to numpy array if not already
        points = np.array(points)
        
        # Find approximate center of points
        center = np.mean(points, axis=0)
        
        # Sort points by angle around center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        ordered_points = points[sorted_indices]
        
        # Close the polygon by adding the first point at the end
        # ordered_boundary = np.vstack([ordered_points, ordered_points[0]])
        
        return ordered_points
    
    def extract_boundary_from_sketch(self, image):
        """
        Main function to extract a clean boundary from a hand-drawn sketch
        """
        try:
            # Check if it's a sketch
            if not self.is_sketch_input(image):
                print("Input doesn't appear to be a sketch. Using standard boundary extraction.")
                return None
                
            # Preprocess the sketch
            preprocessed = self.preprocess_sketch(image)
            
            # Extract lines from the sketch
            lines = self.extract_lines_from_sketch(preprocessed)
            if not lines:
                print("No lines detected in the sketch.")
                return None
                
            # Find line intersections
            intersections = self.find_line_intersections(lines)
            if not intersections:
                print("No intersections found between lines.")
                return None
            
            # Filter intersection points to find corners
            filtered_corners = self.filter_corner_points(intersections, image.shape)
            if len(filtered_corners) < 3:
                print(f"Not enough corners found: {len(filtered_corners)}")
                return None
                
            # Order boundary points to form a proper polygon
            boundary = self.order_boundary_points(filtered_corners)
            
            return {
                "boundary": boundary,
                "preprocessed_image": preprocessed,
                "detected_lines": lines,
                "corners": filtered_corners
            }
            
        except Exception as e:
            print(f"Error extracting boundary from sketch: {e}")
            traceback.print_exc()
            return None
    
    def visualize_boundary_extraction(self, image, extraction_result, output_path=None):
        """
        Visualize the boundary extraction process
        """
        if extraction_result is None:
            print("No extraction results to visualize.")
            return None
            
        # Create RGB visualization
        if len(image.shape) == 2:
            vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_img = image.copy()
            
        # Create 2x2 subplot figure
        plt.figure(figsize=(12, 10))
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap='gray')
        axs[0, 0].set_title("Original Sketch")
        axs[0, 0].axis("off")
        
        # Preprocessed image
        axs[0, 1].imshow(extraction_result["preprocessed_image"], cmap='gray')
        axs[0, 1].set_title("Preprocessed Sketch")
        axs[0, 1].axis("off")
        
        # Detected lines
        line_img = np.zeros_like(vis_img)
        for x1, y1, x2, y2 in extraction_result["detected_lines"]:
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axs[1, 0].imshow(line_img)
        axs[1, 0].set_title("Detected Lines")
        axs[1, 0].axis("off")
        
        # Extracted boundary
        boundary_img = np.zeros_like(vis_img)
        corners = np.array(extraction_result["corners"], dtype=np.int32)
        for point in corners:
            x, y = int(point[0]), int(point[1])
            cv2.circle(boundary_img, (x, y), 5, (255, 0, 0), -1)
            
        if len(extraction_result["boundary"]) > 2:
            boundary = np.array(extraction_result["boundary"], dtype=np.int32)
            pts = boundary.reshape((-1, 1, 2))
            cv2.polylines(boundary_img, [pts], True, (0, 0, 255), 2)
            
        axs[1, 1].imshow(boundary_img)
        axs[1, 1].set_title("Extracted Boundary")
        axs[1, 1].axis("off")
        
        plt.tight_layout()
        
        # Save or show the visualization
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return output_path
        else:
            plt.show()
            plt.close(fig)
            return None