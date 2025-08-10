import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import traceback
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

class BoundaryProcessor:
    """
    Specialized processor for handling hand-drawn boundary inputs.
    Revised to directly use detected lines and connect the gaps between them.
    """
    
    def __init__(self):
        # Parameters for boundary detection
        self.edge_threshold1 = 30
        self.edge_threshold2 = 150
        self.hough_threshold = 25
        self.min_line_length = 30
        self.max_line_gap = 20
        self.intersection_distance_threshold = 20
        self.line_extension_factor = 2.0 # How much to extend lines for better intersection
        
        # Parameters for connecting lines
        self.max_gap_to_connect = 100  # Maximum gap to fill between lines
        self.min_structural_line_length = 40  # Minimum length for structural lines
        
        # Parameters for boundary simplification
        self.simplification_enabled = True
        self.simplification_tolerance = 5  # Tolerance for Douglas-Peucker algorithm
        self.angle_threshold = 160  # Angle threshold for corner detection (degrees)
        
        # Text detection parameters
        self.text_detection_enabled = True
        self.text_area_threshold = 100
        
        # Processing timeouts
        self.processing_timeout = 30  # seconds
    
    def is_sketch_input(self, image):
        """
        Determine if an image is likely a hand-drawn sketch rather than a photograph
        """
        try:
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
        except Exception as e:
            print(f"Error detecting if image is a sketch: {e}")
            # Default to False
            return False
    
    def preprocess_sketch(self, image):
        """
        Preprocess a sketch input to enhance boundaries
        """
        try:
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
        except Exception as e:
            print(f"Error preprocessing sketch: {e}")
            # Return original grayscale image as fallback
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image.copy()
    
    def calculate_line_length(self, line):
        """
        Calculate the length of a line segment
        """
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def extract_lines_from_sketch(self, preprocessed_image):
        """
        Extract line segments from a preprocessed sketch
        """
        try:
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
            
            if lines is None or len(lines) == 0:
                print("No lines detected in the sketch")
                return []
            
            # Filter short lines and sort by length
            structural_lines = []
            for line in lines:
                if self.calculate_line_length(line[0]) >= self.min_structural_line_length:
                    structural_lines.append(line[0])
            
            # Sort by length (longest first)
            structural_lines.sort(key=self.calculate_line_length, reverse=True)
            
            # Limit to reasonable number to prevent processing issues
            max_lines = 200
            if len(structural_lines) > max_lines:
                print(f"Too many lines detected ({len(structural_lines)}), limiting to {max_lines}")
                structural_lines = structural_lines[:max_lines]
            
            return structural_lines
        except Exception as e:
            print(f"Error extracting lines from sketch: {e}")
            return []
    
    def detect_corners(self, boundary_points, angle_threshold=160):
        """
        Detect corners in the boundary polygon based on angle threshold
        """
        if len(boundary_points) < 3:
            return []
            
        try:
            corners = []
            n = len(boundary_points)
            
            for i in range(n):
                prev_idx = (i - 1) % n
                next_idx = (i + 1) % n
                
                # Get current point and neighbors
                p1 = boundary_points[prev_idx]
                p2 = boundary_points[i]
                p3 = boundary_points[next_idx]
                
                # Calculate vectors
                v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                
                # Normalize vectors
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    v1 = v1 / v1_norm
                    v2 = v2 / v2_norm
                    
                    # Calculate dot product and angle
                    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle_rad = np.arccos(dot_product)
                    angle_deg = np.degrees(angle_rad)
                    
                    # Check if this is a corner (angle less than threshold)
                    if angle_deg < angle_threshold:
                        corners.append(i)
            
            return [boundary_points[i] for i in corners]
        except Exception as e:
            print(f"Error detecting corners: {e}")
            return []
    
    def order_points_clockwise(self, endpoints):
        """
        Order a set of points clockwise around their centroid
        """
        if len(endpoints) < 3:
            return endpoints
        
        try:
            # Convert to numpy array if not already
            if not isinstance(endpoints, np.ndarray):
                endpoints = np.array(endpoints)
                
            # Find centroid
            centroid = np.mean(endpoints, axis=0)
            
            # Calculate angles
            angles = np.arctan2(
                endpoints[:, 1] - centroid[1],
                endpoints[:, 0] - centroid[0]
            )
            
            # Sort points by angle
            sorted_indices = np.argsort(angles)
            ordered_points = endpoints[sorted_indices]
            
            return ordered_points
        except Exception as e:
            print(f"Error ordering points clockwise: {e}")
            return endpoints
    
    def create_direct_boundary_from_lines(self, lines):
        """
        Create a boundary directly from the detected lines by clustering endpoints
        """
        if not lines or len(lines) < 2:
            print("Not enough lines to create boundary")
            return None
        
        try:
            # Extract endpoints from all lines
            endpoints = []
            for line in lines:
                x1, y1, x2, y2 = line
                endpoints.append((x1, y1))
                endpoints.append((x2, y2))
            
            # Convert to numpy array
            endpoints = np.array(endpoints)
            
            # Cluster nearby endpoints to merge them
            clustering = DBSCAN(eps=20, min_samples=1).fit(endpoints)
            labels = clustering.labels_
            
            # Create single point for each cluster (using mean of points in cluster)
            unique_labels = np.unique(labels)
            merged_points = []
            for label in unique_labels:
                cluster_points = endpoints[labels == label]
                merged_point = np.mean(cluster_points, axis=0)
                merged_points.append(merged_point)
            
            # Order points clockwise to form boundary
            if len(merged_points) >= 3:
                ordered_boundary = self.order_points_clockwise(merged_points)
                return np.array(ordered_boundary)
            else:
                print(f"Not enough points to form boundary (found {len(merged_points)}, need at least 3)")
                return None
        except Exception as e:
            print(f"Error creating direct boundary from lines: {e}")
            return None
    
    def connect_lines_to_form_boundary(self, lines):
        """
        Connect the lines directly to form a continuous boundary with improved error handling
        """
        if not lines or len(lines) < 2:
            print("Not enough lines to connect")
            return None
        
        try:
            # First try to connect the lines to form a boundary
            # Start with the longest line
            boundary_lines = [lines[0]]
            remaining_lines = lines[1:]
            
            # Keep track of the current start and end of the growing boundary
            current_start = (lines[0][0], lines[0][1])
            current_end = (lines[0][2], lines[0][3])
            
            # Keep connecting lines until we form a loop or run out of lines
            max_iterations = min(len(lines) * 2, 50)  # Limit iterations to prevent infinite loops
            iteration = 0
            
            while remaining_lines and iteration < max_iterations:
                iteration += 1
                
                # Find the closest line to either end of the current boundary
                best_distance = float('inf')
                best_line_idx = -1
                best_connection = None
                
                for i, line in enumerate(remaining_lines):
                    line_start = (line[0], line[1])
                    line_end = (line[2], line[3])
                    
                    # Check all possible connections
                    connections = [
                        (current_start, line_start, distance.euclidean(current_start, line_start), "start-start"),
                        (current_start, line_end, distance.euclidean(current_start, line_end), "start-end"),
                        (current_end, line_start, distance.euclidean(current_end, line_start), "end-start"),
                        (current_end, line_end, distance.euclidean(current_end, line_end), "end-end")
                    ]
                    
                    # Find the best connection
                    for conn_start, conn_end, dist, conn_type in connections:
                        if dist < best_distance and dist < self.max_gap_to_connect:
                            best_distance = dist
                            best_line_idx = i
                            best_connection = (conn_start, conn_end, conn_type)
                
                # If we found a good connection
                if best_line_idx >= 0:
                    best_line = remaining_lines[best_line_idx]
                    
                    # Connect based on the connection type
                    if best_connection[2] == "start-start":
                        # Reverse current boundary and add the new line
                        boundary_lines = [(l[2], l[3], l[0], l[1]) for l in reversed(boundary_lines)]
                        boundary_lines.append(best_line)
                        current_start = (best_line[2], best_line[3])
                        # Current end doesn't change (was the start, now the end of the reversed boundary)
                        
                    elif best_connection[2] == "start-end":
                        # Reverse current boundary and add reversed new line
                        boundary_lines = [(l[2], l[3], l[0], l[1]) for l in reversed(boundary_lines)]
                        boundary_lines.append((best_line[2], best_line[3], best_line[0], best_line[1]))
                        current_start = (best_line[0], best_line[1])
                        # Current end doesn't change
                        
                    elif best_connection[2] == "end-start":
                        # Just add the new line
                        boundary_lines.append(best_line)
                        # Current start doesn't change
                        current_end = (best_line[2], best_line[3])
                        
                    elif best_connection[2] == "end-end":
                        # Add the reversed new line
                        boundary_lines.append((best_line[2], best_line[3], best_line[0], best_line[1]))
                        # Current start doesn't change
                        current_end = (best_line[0], best_line[1])
                    
                    # Remove the used line
                    remaining_lines.pop(best_line_idx)
                    
                    # Check if we've formed a loop
                    if distance.euclidean(current_start, current_end) < 20:
                        break
                else:
                    # If no good connection found, break
                    break
            
            # Check if we have enough lines for a boundary
            if len(boundary_lines) < 3:
                print(f"Not enough connected lines to form boundary (found {len(boundary_lines)}, need at least 3)")
                return None
            
            # Create a list of points representing the boundary
            boundary_points = []
            for line in boundary_lines:
                boundary_points.append((line[0], line[1]))
            
            # Add the last point to close the boundary if needed
            if distance.euclidean(boundary_points[0], boundary_points[-1]) > 20:
                boundary_points.append(boundary_points[0])
            
            return np.array(boundary_points)
        except Exception as e:
            print(f"Error connecting lines to form boundary: {e}")
            return None
    
    def simplify_boundary(self, boundary_points):
        """
        Simplify boundary using Douglas-Peucker algorithm with improved MultiPolygon handling
        """
        if len(boundary_points) < 3 or not self.simplification_enabled:
            return boundary_points
        
        try:
            # Create Shapely polygon
            from shapely.geometry import Polygon, MultiPolygon
            poly = Polygon(boundary_points)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Try to fix invalid polygon
                if not poly.is_valid:
                    print("Cannot create valid polygon from boundary for simplification")
                    return boundary_points
            
            # Simplify polygon
            simplified = poly.simplify(self.simplification_tolerance, preserve_topology=True)
            
            # Extract coordinates, handling both Polygon and MultiPolygon cases
            if simplified.geom_type == 'Polygon':
                # Remove duplicate last point
                coords = np.array(simplified.exterior.coords)[:-1] 
                if len(coords) >= 3:
                    return coords
                else:
                    print(f"Simplification resulted in too few points ({len(coords)}), returning original")
                    return boundary_points
            elif simplified.geom_type == 'MultiPolygon':
                # Get the largest polygon from the multipolygon
                largest_poly = max(simplified.geoms, key=lambda x: x.area)
                coords = np.array(largest_poly.exterior.coords)[:-1]  # Remove duplicate last point
                if len(coords) >= 3:
                    return coords
                else:
                    print(f"Simplification resulted in too few points ({len(coords)}), returning original")
                    return boundary_points
            
            return boundary_points
        except Exception as e:
            print(f"Error simplifying boundary: {e}")
            return boundary_points
    
    def extract_boundary_from_sketch(self, image):
        """
        Main function to extract a clean boundary from a hand-drawn sketch
        Enhanced to use detected lines directly and connect gaps
        """
        if not self.is_sketch_input(image):
            print("Input doesn't appear to be a sketch.")
            return None
        
        try:
            # Preprocess the sketch
            preprocessed = self.preprocess_sketch(image)
            
            # Extract lines from the sketch
            lines = self.extract_lines_from_sketch(preprocessed)
            if not lines or len(lines) < 3:
                print("Not enough structural lines detected in the sketch.")
                return None
            
            # Method 1: Try to connect lines sequentially to form a continuous boundary
            boundary = self.connect_lines_to_form_boundary(lines)
            
            # Method 2: If that fails, try clustering line endpoints
            if boundary is None or len(boundary) < 3:
                print("Sequential line connection failed, trying endpoint clustering.")
                boundary = self.create_direct_boundary_from_lines(lines)
            
            if boundary is None or len(boundary) < 3:
                print("Failed to create boundary from detected lines.")
                return None
            
            # Simplify the boundary
            simplified_boundary = self.simplify_boundary(boundary)
            
            # Detect corners in the boundary for visualization and analysis
            corners = self.detect_corners(simplified_boundary, self.angle_threshold)
            
            return {
                "boundary": simplified_boundary,
                "preprocessed_image": preprocessed,
                "detected_lines": lines,
                "original_boundary": boundary,
                "corners": corners
            }
        except Exception as e:
            print(f"Error extracting boundary from sketch: {e}")
            traceback.print_exc()
            return None
    
    def visualize_boundary_extraction(self, image, extraction_result, output_path=None):
    
        if extraction_result is None:
            print("No extraction results to visualize.")
            return None
        
        try:    
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
            
            # Draw original boundary if available
            if "original_boundary" in extraction_result and len(extraction_result["original_boundary"]) > 2:
                orig_boundary = np.array(extraction_result["original_boundary"], dtype=np.int32)
                cv2.polylines(boundary_img, [orig_boundary], True, (0, 255, 0), 1)
            
            # Draw final boundary
            if len(extraction_result["boundary"]) > 2:
                boundary = np.array(extraction_result["boundary"], dtype=np.int32)
                cv2.polylines(boundary_img, [boundary], True, (0, 0, 255), 2)
                
                # Add points at each vertex
                for point in boundary:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(boundary_img, (x, y), 5, (255, 0, 0), -1)
                
                # Draw corners with a different color if available
                if "corners" in extraction_result and extraction_result["corners"]:
                    for corner in extraction_result["corners"]:
                        x, y = int(corner[0]), int(corner[1])
                        cv2.circle(boundary_img, (x, y), 7, (255, 255, 0), -1)
            
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
        except Exception as e:
            print(f"Error visualizing boundary extraction: {e}")
            traceback.print_exc()
            return None