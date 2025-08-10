import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from matplotlib.patches import Patch
import random
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Union
import traceback

@dataclass
class UrbanZoneParameters:
    """Parameters defining an urban zone type"""
    name: str
    color: str
    min_density: float = 0.0  # DN - density factor
    max_density: float = 1.0
    min_far: float = 0.0  # FAR - floor area ratio
    max_far: float = 10.0
    min_height: float = 0.0  # AH - average height (meters)
    max_height: float = 50.0
    height_variance: float = 0.0  # HV - height variance
    massing_variance: float = 0.0  # MV - building massing variance
    green_proportion: float = 0.0  # GP - green area proportion
    water_proportion: float = 0.0  # WP - water surface proportion
    hardscape_proportion: float = 0.0  # HP - hardscape proportion
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class UrbanLayoutConstraints:
    """Urban design constraints and preferences"""
    layout_shape: str = "Regular"  # "Regular" or "Irregular"
    building_density: float = 0.5  # 0.0 to 1.0
    building_count: int = 10
    road_network_type: str = "Grid"  # "Grid", "Organic", or "Radial"
    sustainability_score: float = 0.5  # 0.0 to 1.0
    zoning_distribution: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)

class VoronoiPlanner:
    """
    Voronoi-based urban planner for the Streamlit interface.
    This class wraps the EnhancedVoronoiPlanner from voronoi11.py
    """
    
    def __init__(self):
        """Initialize the Voronoi planner with default parameters"""
        # Image processing parameters
        self.min_line_length = 30
        self.max_line_gap = 10
        self.edge_threshold = (50, 150)
        
        # Default urban zone types with standard parameters
        self.zone_types = {
            "Residential": UrbanZoneParameters(
                name="Residential",
                color="lightgreen",
                min_density=0.3,
                max_density=0.6,
                min_far=0.5,
                max_far=2.0,
                min_height=3.0,
                max_height=15.0,
                height_variance=2.0,
                massing_variance=0.3,
                green_proportion=0.2,
                water_proportion=0.0,
                hardscape_proportion=0.2
            ),
            "Commercial": UrbanZoneParameters(
                name="Commercial",
                color="red",
                min_density=0.5,
                max_density=0.8,
                min_far=1.0,
                max_far=5.0,
                min_height=4.0,
                max_height=30.0,
                height_variance=5.0,
                massing_variance=0.4,
                green_proportion=0.1,
                water_proportion=0.0,
                hardscape_proportion=0.6
            ),
            "Mixed": UrbanZoneParameters(
                name="Mixed",
                color="purple",
                min_density=0.4,
                max_density=0.7,
                min_far=0.8,
                max_far=4.0,
                min_height=4.0,
                max_height=25.0,
                height_variance=4.0,
                massing_variance=0.5,
                green_proportion=0.15,
                water_proportion=0.0,
                hardscape_proportion=0.4
            ),
            "Green": UrbanZoneParameters(
                name="Green",
                color="green",
                min_density=0.0,
                max_density=0.1,
                min_far=0.0,
                max_far=0.2,
                min_height=0.0,
                max_height=3.0,
                height_variance=0.5,
                massing_variance=0.1,
                green_proportion=0.8,
                water_proportion=0.1,
                hardscape_proportion=0.1
            ),
            "Water": UrbanZoneParameters(
                name="Water",
                color="blue",
                min_density=0.0,
                max_density=0.0,
                min_far=0.0,
                max_far=0.0,
                min_height=0.0,
                max_height=0.0,
                height_variance=0.0,
                massing_variance=0.0,
                green_proportion=0.0,
                water_proportion=1.0,
                hardscape_proportion=0.0
            ),
            "Civic": UrbanZoneParameters(
                name="Civic",
                color="orange",
                min_density=0.2,
                max_density=0.5,
                min_far=0.5,
                max_far=3.0,
                min_height=4.0,
                max_height=20.0,
                height_variance=3.0,
                massing_variance=0.6,
                green_proportion=0.3,
                water_proportion=0.0,
                hardscape_proportion=0.3
            ),
            "Industrial": UrbanZoneParameters(
                name="Industrial",
                color="gray",
                min_density=0.4,
                max_density=0.7,
                min_far=0.6,
                max_far=2.5,
                min_height=6.0,
                max_height=15.0,
                height_variance=2.0,
                massing_variance=0.7,
                green_proportion=0.1,
                water_proportion=0.0,
                hardscape_proportion=0.7
            )
        }
        
        self.pixels_to_meters = 1.0  # Default scale factor
        self.constraints = UrbanLayoutConstraints()  # Default constraints
    
    def set_scale(self, pixels_to_meters):
        """Set scale factor to convert pixels to meters"""
        self.pixels_to_meters = pixels_to_meters
    
    def set_constraints(self, constraints: Union[Dict, UrbanLayoutConstraints]):
        """Set urban design constraints from dictionary or object"""
        if isinstance(constraints, dict):
            # Convert dictionary to UrbanLayoutConstraints
            self.constraints = UrbanLayoutConstraints(
                layout_shape=constraints.get('layout_shape', 'Regular'),
                building_density=constraints.get('building_density', 0.5),
                building_count=constraints.get('building_count', 10),
                road_network_type=constraints.get('road_network_type', 'Grid'),
                sustainability_score=constraints.get('sustainability_score', 0.5),
                zoning_distribution=constraints.get('zoning_distribution', {})
            )
        else:
            self.constraints = constraints
    
    def extract_boundary_from_image(self, image_path):
        """Extract boundary from image using edge detection and line detection"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
            
            # Process image to get edges
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, *self.edge_threshold)
            
            # Detect lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            # Extract points from lines
            points = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    points.append((x1, y1))
                    points.append((x2, y2))
            
            # Get boundary from points
            boundary = self.get_plot_boundary(points, img.shape)
            
            return {
                "image": img,
                "boundary": boundary,
                "edges": edges
            }
        except Exception as e:
            print(f"Error extracting boundary from image: {e}")
            traceback.print_exc()
            return None
        
    def get_plot_boundary(self, points, img_shape):
        """Compute the boundary polygon (convex hull) around extracted points"""
        if not points:
            return np.array([
                [0, 0],
                [img_shape[1], 0],
                [img_shape[1], img_shape[0]],
                [0, img_shape[0]]
            ], dtype=np.float32)

        pts = np.array(points, dtype=np.float32)
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        return hull.squeeze()
    
    def process_and_plan(self, image_path, output_folder, num_zones=10, pixels_to_meters=1.0):
        """Complete pipeline for processing an image and generating an urban plan with improved error handling"""
        try:
            # Create output folder
            os.makedirs(output_folder, exist_ok=True)
            
            # Set scale
            self.set_scale(pixels_to_meters)
            
            # 1) Extract boundary from image or use unet boundary if available
            print("Extracting boundary from image...")
            start_time = time.time()
            boundary_data = self.extract_boundary_from_image(image_path)
            
            if boundary_data is None:
                print("Failed to extract boundary from image")
                return None
                
            boundary = boundary_data["boundary"]
            print(f"Boundary extraction completed in {time.time() - start_time:.2f} seconds")
            
            # Verify boundary has enough points and is valid
            if len(boundary) < 3:
                print("Invalid boundary: needs at least 3 points")
                return None
                
            # Verify boundary is a valid polygon
            try:
                poly = Polygon(boundary)
                if not poly.is_valid:
                    # Try to fix invalid polygon
                    poly = poly.buffer(0)
                    if not poly.is_valid:
                        print("Cannot create valid polygon from boundary")
                        return None
                    
                    # Handle MultiPolygon case
                    if isinstance(poly, MultiPolygon):
                        print("Boundary resulted in a MultiPolygon. Selecting largest polygon.")
                        largest_poly = max(poly.geoms, key=lambda x: x.area)
                        boundary = np.array(largest_poly.exterior.coords)[:-1]
                    else:
                        boundary = np.array(poly.exterior.coords)[:-1]
                        
            except Exception as e:
                print(f"Error validating boundary: {e}")
                traceback.print_exc()
                return None
            
            # 2) Generate road network based on constraints
            print("Generating road network...")
            start_time = time.time()
            road_network = self.generate_road_network(
                boundary, 
                num_zones, 
                road_type=self.constraints.road_network_type
            )
            print(f"Road network generation completed in {time.time() - start_time:.2f} seconds")
            
            # 3) Generate Voronoi zones
            print("Generating Voronoi zones...")
            start_time = time.time()
            voronoi_data = self.generate_voronoi_zones(boundary, num_zones)
            
            if voronoi_data is None:
                print("Failed to generate Voronoi zones")
                return None
                
            regions = voronoi_data["regions"]
            print(f"Voronoi zone generation completed in {time.time() - start_time:.2f} seconds")
            
            # 4) Assign zone types
            print("Assigning zone types...")
            start_time = time.time()
            zone_assignments = self.assign_zone_types(regions)
            print(f"Zone assignment completed in {time.time() - start_time:.2f} seconds")
            
            # 5) Visualize the plan
            print("Creating visualization...")
            start_time = time.time()
            vis_output = self.visualize_urban_plan(
                boundary,
                regions,
                zone_assignments,
                road_network,
                image_path,
                os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_urban_plan.png")
            )
            print(f"Visualization completed in {time.time() - start_time:.2f} seconds")
            
            # 6) Extract morphological parameters
            print("Calculating morphological parameters...")
            start_time = time.time()
            morph_params = self.extract_morphological_parameters(regions, zone_assignments)
            print(f"Parameter calculation completed in {time.time() - start_time:.2f} seconds")
            
            # 7) Create tensor for UDGAN
            print("Creating data tensor...")
            start_time = time.time()
            tensor_data = self.create_tensor_for_udgan(regions, zone_assignments, road_network, boundary)
            print(f"Tensor creation completed in {time.time() - start_time:.2f} seconds")
            
            # 8) Save parameters to JSON
            params_output = os.path.join(
                output_folder,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_params.json"
            )
            
            with open(params_output, 'w') as f:
                json.dump({
                    "morphological_parameters": morph_params,
                    "road_network": {
                        "type": road_network["road_type"],
                        "num_roads": road_network["num_roads"]
                    },
                    "zones": {
                        "num_zones": len(regions),
                        "zone_distribution": {zone: zone_assignments.count(zone) for zone in set(zone_assignments)}
                    },
                    "scale": pixels_to_meters,
                    "runtime_info": {
                        "timestamp": datetime.now().isoformat(),
                        "num_regions": len(regions)
                    }
                }, f, indent=2)
            
            print(f"Urban plan processing complete. Results saved to {output_folder}")
            
            return {
                "boundary": boundary,
                "regions": regions,
                "zone_assignments": zone_assignments,
                "road_network": road_network,
                "morphological_parameters": morph_params,
                "visualization_output": vis_output,
                "parameters_output": params_output,
                "tensor_data": tensor_data
            }
            
        except Exception as e:
            print(f"Error in urban planning process: {e}")
            traceback.print_exc()
            return None
    
    def generate_road_network(self, boundary, num_zones, road_type="Grid"):
        """Generate a road network based on the specified road type and boundary"""
        try:
            # Convert boundary to shapely polygon
            boundary_poly = Polygon(boundary)
            minx, miny, maxx, maxy = boundary_poly.bounds
            
            # Generate road network based on type
            if road_type == "Grid":
                # Create a grid of roads
                x_divisions = max(2, int(np.sqrt(num_zones)) + 1)
                y_divisions = max(2, int(num_zones / x_divisions) + 1)
                
                x_steps = np.linspace(minx, maxx, x_divisions)
                y_steps = np.linspace(miny, maxy, y_divisions)
                
                roads = []
                # Horizontal roads
                for y in y_steps:
                    road = LineString([(minx, y), (maxx, y)])
                    road_clipped = road.intersection(boundary_poly)
                    if not road_clipped.is_empty and not road_clipped.length == 0:
                        roads.append(road_clipped)
                
                # Vertical roads
                for x in x_steps:
                    road = LineString([(x, miny), (x, maxy)])
                    road_clipped = road.intersection(boundary_poly)
                    if not road_clipped.is_empty and not road_clipped.length == 0:
                        roads.append(road_clipped)
                        
            elif road_type == "Radial":
                # Create radial road network
                center_x = (minx + maxx) / 2
                center_y = (miny + maxy) / 2
                radius_max = max(maxx - minx, maxy - miny) / 1.5
                
                # Radial roads
                roads = []
                num_radial = max(3, int(num_zones / 3))
                for i in range(num_radial):
                    angle = 2 * np.pi * i / num_radial
                    dx, dy = np.cos(angle) * radius_max, np.sin(angle) * radius_max
                    road = LineString([(center_x, center_y), (center_x + dx, center_y + dy)])
                    road_clipped = road.intersection(boundary_poly)
                    if not road_clipped.is_empty and not road_clipped.length == 0:
                        roads.append(road_clipped)
                        
                # Concentric roads
                num_rings = max(1, int(np.sqrt(num_zones) / 2))
                for i in range(1, num_rings + 1):
                    radius = radius_max * i / num_rings
                    # Approximate circle with LineString
                    circle_points = []
                    for j in range(60):
                        angle = 2 * np.pi * j / 60
                        circle_points.append((
                            center_x + np.cos(angle) * radius,
                            center_y + np.sin(angle) * radius
                        ))
                    circle_points.append(circle_points[0])  # Close the circle
                    
                    road = LineString(circle_points)
                    road_clipped = road.intersection(boundary_poly)
                    if not road_clipped.is_empty and not road_clipped.length == 0:
                        roads.append(road_clipped)
                        
            else:  # "Organic" or default
                # Create organic road network using randomized Voronoi
                num_seeds = max(5, int(num_zones / 2))
                seed_points = []
                for _ in range(num_seeds):
                    x = np.random.uniform(minx, maxx)
                    y = np.random.uniform(miny, maxy)
                    if boundary_poly.contains(Point(x, y)):
                        seed_points.append((x, y))
                
                if len(seed_points) >= 3:
                    vor = Voronoi(seed_points)
                    roads = []
                    
                    # Extract Voronoi ridges as roads
                    for ridge_vertices in vor.ridge_vertices:
                        if -1 not in ridge_vertices:
                            p1 = vor.vertices[ridge_vertices[0]]
                            p2 = vor.vertices[ridge_vertices[1]]
                            road = LineString([p1, p2])
                            road_clipped = road.intersection(boundary_poly)
                            if not road_clipped.is_empty and not road_clipped.length == 0:
                                roads.append(road_clipped)
                else:
                    # Fallback to simple roads
                    roads = [
                        LineString([(minx, (miny + maxy) / 2), (maxx, (miny + maxy) / 2)]),
                        LineString([((minx + maxx) / 2, miny), ((minx + maxx) / 2, maxy)])
                    ]
            
            return {
                "roads": roads,
                "road_type": road_type,
                "num_roads": len(roads)
            }
        except Exception as e:
            print(f"Error generating road network: {e}")
            traceback.print_exc()
            return {"roads": [], "road_type": road_type, "num_roads": 0}
        
    def generate_voronoi_zones(self, boundary, num_zones, seed_points=None):
        """Generate Voronoi cells for urban zoning with exact number of regions"""
        try:
            boundary_poly = Polygon(boundary)
            max_attempts = 10  # Maximum number of attempts to get the right number of zones
            
            for attempt in range(max_attempts):
                # If no seed points provided, generate them
                if seed_points is None or attempt > 0:
                    # Generate seed points within the boundary
                    # For subsequent attempts, increase the number of candidates
                    candidate_multiplier = 1 + (attempt * 0.5)
                    seed_candidate_count = int(num_zones * candidate_multiplier)
                    seed_points = self.sample_points_in_polygon(boundary, seed_candidate_count)
                
                # Create Voronoi diagram
                vor = Voronoi(seed_points)
                
                # Get clipped regions
                clipped_regions = self.get_voronoi_polygons(vor, boundary)
                
                # Check if we have the right number of regions
                if len(clipped_regions) == num_zones:
                    break
                elif len(clipped_regions) > num_zones:
                    # Too many regions - reduce by keeping larger ones
                    clipped_regions = sorted(clipped_regions, key=lambda r: r.area, reverse=True)[:num_zones]
                    break
                else:
                    # Too few regions - will try again with more seed points
                    print(f"Attempt {attempt+1}: Generated {len(clipped_regions)} regions, need {num_zones}")
                    continue
            
            # Fill any gaps (white spaces) between regions and boundary
            filled_regions = self.fill_boundary_gaps(clipped_regions, boundary_poly)
            
            return {
                "voronoi": vor,
                "regions": filled_regions,
                "seed_points": seed_points,
                "boundary": boundary
            }
        except Exception as e:
            print(f"Error generating Voronoi zones: {e}")
            traceback.print_exc()
            return None
        
    def sample_points_in_polygon(self, boundary_polygon, num_samples):
        """Sample points within a polygon for Voronoi seed generation with improved distribution"""
        try:
            boundary = Polygon(boundary_polygon)
            minx, miny, maxx, maxy = boundary.bounds
            
            # Strategy depends on layout shape
            if self.constraints.layout_shape == "Regular":
                # For regular layouts, use a more precise grid distribution
                # This ensures more even cell sizes
                points = []
                
                # Calculate grid dimensions to get exactly num_samples points
                side = max(1, int(np.sqrt(num_samples)))
                if side * side < num_samples:
                    side += 1
                    
                x_step = (maxx - minx) / (side + 1)
                y_step = (maxy - miny) / (side + 1)
                
                # Generate grid points with slight jitter for better Voronoi cells
                for i in range(1, side + 1):
                    for j in range(1, side + 1):
                        # Add small random jitter to avoid perfectly regular cells
                        jitter_x = x_step * 0.3 * (np.random.random() - 0.5)
                        jitter_y = y_step * 0.3 * (np.random.random() - 0.5)
                        
                        x = minx + i * x_step + jitter_x
                        y = miny + j * y_step + jitter_y
                        
                        if boundary.contains(Point(x, y)) and len(points) < num_samples:
                            points.append((x, y))
                        
                        if len(points) >= num_samples:
                            break
                    if len(points) >= num_samples:
                        break
            else:
                # For irregular layouts, use clusters with controlled distribution
                points = []
                
                # Create cluster centers based on boundary shape
                cluster_count = max(3, min(8, int(num_samples / 10) + 1))
                
                # Sample points around polygon to better respect the boundary shape
                boundary_points = list(zip(boundary_polygon[:,0], boundary_polygon[:,1]))
                
                # Select cluster centers - mix of boundary points and interior points
                cluster_centers = []
                
                # Add some boundary-based centers
                for _ in range(cluster_count // 2):
                    idx = np.random.randint(0, len(boundary_points))
                    # Move slightly inward from boundary
                    px, py = boundary_points[idx]
                    center_x = px * 0.9 + 0.1 * ((minx + maxx) / 2)
                    center_y = py * 0.9 + 0.1 * ((miny + maxy) / 2)
                    if boundary.contains(Point(center_x, center_y)):
                        cluster_centers.append((center_x, center_y))
                
                # Add interior centers
                for _ in range(cluster_count - len(cluster_centers)):
                    attempts = 0
                    while attempts < 20:
                        cx = minx + (maxx - minx) * np.random.random()
                        cy = miny + (maxy - miny) * np.random.random()
                        if boundary.contains(Point(cx, cy)):
                            cluster_centers.append((cx, cy))
                            break
                        attempts += 1
                        
                # Add points around each cluster center
                points_per_cluster = num_samples // len(cluster_centers)
                remaining = num_samples - (points_per_cluster * len(cluster_centers))
                
                for i, center in enumerate(cluster_centers):
                    # Add extra point to first few clusters if we have remainder
                    current_cluster_points = points_per_cluster + (1 if i < remaining else 0)
                    
                    # Calculate appropriate radius based on boundary size and number of points
                    boundary_size = max(maxx - minx, maxy - miny)
                    radius = boundary_size * (0.2 + 0.1 * np.random.random())
                    
                    for _ in range(current_cluster_points):
                        # Generate points with distance from center increasing with each point
                        # This creates a more natural distribution
                        angle = np.random.uniform(0, 2 * np.pi)
                        # Use square root distribution for more even spacing
                        dist_factor = np.sqrt(np.random.random())
                        dist = radius * dist_factor
                        
                        x = center[0] + np.cos(angle) * dist
                        y = center[1] + np.sin(angle) * dist
                        
                        # Check if point is inside boundary
                        if boundary.contains(Point(x, y)):
                            points.append((x, y))
                        else:
                            # If outside, try projecting it back inside
                            center_x, center_y = boundary.centroid.coords[0]
                            # Project toward centroid
                            t = 0.7  # Interpolation factor
                            proj_x = t * center_x + (1-t) * x
                            proj_y = t * center_y + (1-t) * y
                            
                            if boundary.contains(Point(proj_x, proj_y)):
                                points.append((proj_x, proj_y))
            
            # If we still don't have enough points, add random points
            while len(points) < num_samples:
                x = minx + (maxx - minx) * np.random.random()
                y = miny + (maxy - miny) * np.random.random()
                if boundary.contains(Point(x, y)):
                    points.append((x, y))
            
            # If we have too many points, trim the list
            points = points[:num_samples]
            
            return np.array(points)
        except Exception as e:
            print(f"Error sampling points in polygon: {e}")
            traceback.print_exc()
            # Fallback to simple grid sampling
            return self._fallback_grid_sampling(boundary_polygon, num_samples)

    def _fallback_grid_sampling(self, boundary_polygon, num_samples):
        """Fallback grid sampling method when other methods fail"""
        minx, miny = np.min(boundary_polygon, axis=0)
        maxx, maxy = np.max(boundary_polygon, axis=0)
        
        points = []
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        # Create grid points
        for i in range(rows):
            for j in range(cols):
                x = minx + (maxx - minx) * (j + 0.5) / cols
                y = miny + (maxy - miny) * (i + 0.5) / rows
                points.append((x, y))
                if len(points) >= num_samples:
                    break
            if len(points) >= num_samples:
                break
                
        return np.array(points[:num_samples])

    def get_voronoi_polygons(self, vor, boundary_polygon):
        """Clip Voronoi cells to the boundary polygon with improved handling"""
        try:
            boundary = Polygon(boundary_polygon)
            regions = []

            # Process all regions for better boundary coverage
            for region_idx in vor.point_region:
                region = vor.regions[region_idx]
                
                # Skip empty regions
                if not region:
                    continue
                    
                # Handle infinite regions by constructing a large bounding box
                if -1 in region:
                    # Create a bounding box much larger than our boundary
                    minx, miny, maxx, maxy = boundary.bounds
                    boundary_size = max(maxx - minx, maxy - miny)
                    buffer_size = boundary_size * 5
                    
                    # Create extended bounding box
                    big_box = Polygon([
                        [minx - buffer_size, miny - buffer_size],
                        [maxx + buffer_size, miny - buffer_size],
                        [maxx + buffer_size, maxy + buffer_size],
                        [minx - buffer_size, maxy + buffer_size]
                    ])
                    
                    # Get the site point for this region
                    site_idx = list(vor.point_region).index(region_idx)
                    site = vor.points[site_idx]
                    
                    # Find all ridges connected to this site
                    connected_vertices = []
                    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
                        if p1 == site_idx or p2 == site_idx:
                            if v1 != -1:
                                connected_vertices.append(vor.vertices[v1])
                            if v2 != -1:
                                connected_vertices.append(vor.vertices[v2])
                    
                    # If we have enough vertices, try to form a polygon
                    if len(connected_vertices) >= 3:
                        try:
                            site_poly = Polygon(connected_vertices)
                            if not site_poly.is_valid:
                                continue
                            site_region = site_poly.intersection(big_box)
                            site_region = site_region.intersection(boundary)
                            if not site_region.is_empty and site_region.area > 0:
                                regions.append(site_region)
                        except:
                            continue
                    continue
                
                # Process normal regions
                polygon_points = []
                for vert_index in region:
                    polygon_points.append(vor.vertices[vert_index])
                    
                if len(polygon_points) < 3:
                    continue
                
                try:
                    voronoi_polygon = Polygon(polygon_points)
                    if not voronoi_polygon.is_valid:
                        continue
                        
                    clipped_polygon = voronoi_polygon.intersection(boundary)
                    
                    if not clipped_polygon.is_empty and clipped_polygon.area > 0:
                        regions.append(clipped_polygon)
                except Exception as e:
                    print(f"Error creating polygon: {e}")
                    continue

            # Ensure we don't have overlapping regions
            non_overlapping_regions = []
            for i, region in enumerate(regions):
                # Check for overlaps with already processed regions
                is_valid = True
                for j, other_region in enumerate(non_overlapping_regions):
                    intersection = region.intersection(other_region)
                    if not intersection.is_empty and intersection.area / region.area > 0.05:
                        # Significant overlap, resolve by keeping the larger region
                        if region.area > other_region.area:
                            non_overlapping_regions[j] = region.difference(other_region)
                            is_valid = False
                        else:
                            region = region.difference(other_region)
                            if region.area / other_region.area < 0.05:
                                is_valid = False
                
                if is_valid and not region.is_empty and region.area > 0:
                    non_overlapping_regions.append(region)

            return non_overlapping_regions
        except Exception as e:
            print(f"Error getting Voronoi polygons: {e}")
            traceback.print_exc()
            return []

    def fill_boundary_gaps(self, regions, boundary):
        """Fill gaps between regions and boundary to eliminate white spaces"""
        try:
            # Calculate the union of all regions
            if not regions:
                return [boundary]
                
            union = regions[0]
            for region in regions[1:]:
                union = union.union(region)
                
            # Calculate the difference to find gaps
            gaps = boundary.difference(union)
            
            # If there are no gaps, return original regions
            if gaps.is_empty or gaps.area < 0.001 * boundary.area:
                return regions
                
            filled_regions = list(regions)
            
            # Process gaps
            if gaps.geom_type == 'Polygon':
                gaps = [gaps]
            elif gaps.geom_type == 'MultiPolygon':
                gaps = list(gaps.geoms)
            else:
                return regions  # Unexpected geometry type
                
            # For each gap, either create a new region or merge with closest region
            for gap in gaps:
                # Skip tiny gaps
                if gap.area < 0.001 * boundary.area:
                    continue
                    
                gap_centroid = gap.centroid
                
                # Find closest region
                min_dist = float('inf')
                closest_idx = -1
                
                for i, region in enumerate(filled_regions):
                    dist = gap_centroid.distance(region.centroid)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                if closest_idx >= 0:
                    # Merge gap with closest region
                    filled_regions[closest_idx] = filled_regions[closest_idx].union(gap)
                else:
                    # Add as new region if no close regions found
                    filled_regions.append(gap)
            
            return filled_regions
        except Exception as e:
            print(f"Error filling boundary gaps: {e}")
            traceback.print_exc()
            return regions
        
    def assign_zone_types(self, regions, zoning_distribution=None):
        """Assign urban zone types to Voronoi regions based on distribution"""
        try:
            if zoning_distribution is None:
                zoning_distribution = self.constraints.zoning_distribution
                
            if not zoning_distribution:
                # Default distribution if none provided
                zoning_distribution = {
                    "Residential": 50,
                    "Commercial": 20,
                    "Green": 15,
                    "Mixed": 10,
                    "Civic": 5
                }
                
            # Ensure all zone types are valid
            zone_types = list(self.zone_types.keys())
            for zone in list(zoning_distribution.keys()):
                if zone not in zone_types:
                    print(f"Warning: Unknown zone type '{zone}'. Removing from distribution.")
                    del zoning_distribution[zone]
                    
            # Normalize distribution to sum to 100%
            total = sum(zoning_distribution.values())
            if total == 0:
                # Default to equal distribution if all values are 0
                zoning_distribution = {k: 100/len(zoning_distribution) for k in zoning_distribution}
            else:
                zoning_distribution = {k: v/total*100 for k, v in zoning_distribution.items()}
                
            # Calculate how many regions of each type
            num_regions = len(regions)
            type_counts = {}
            remaining = num_regions
            
            # Assign counts based on percentages
            for zone_type, percentage in zoning_distribution.items():
                count = int(np.round(percentage / 100 * num_regions))
                type_counts[zone_type] = count
                remaining -= count
                
            # Adjust for rounding errors
            if remaining > 0:
                # Add remaining to the largest category
                max_type = max(type_counts, key=type_counts.get)
                type_counts[max_type] += remaining
            elif remaining < 0:
                # Remove from largest until balanced
                while remaining < 0:
                    max_type = max(type_counts, key=type_counts.get)
                    type_counts[max_type] -= 1
                    remaining += 1
                    
            # Create the zone type assignments
            zone_assignments = []
            for zone_type, count in type_counts.items():
                zone_assignments.extend([zone_type] * count)
                
            # If we have too many, truncate
            zone_assignments = zone_assignments[:num_regions]
            
            # If we have too few, add default type
            while len(zone_assignments) < num_regions:
                zone_assignments.append("Mixed")
                
            # Optimize placement based on zone compatibility
            # For now, just shuffle randomly - this could be enhanced with
            # a more sophisticated algorithm that considers adjacency rules
            random.shuffle(zone_assignments)
            
            return zone_assignments
        except Exception as e:
            print(f"Error assigning zone types: {e}")
            traceback.print_exc()
            # Fallback to simple random assignment
            return [random.choice(list(self.zone_types.keys())) for _ in range(len(regions))]
        
    def extract_morphological_parameters(self, regions, zone_assignments):
        """Extract urban morphological parameters"""
        try:
            total_area = sum(region.area for region in regions)
            building_footprint_area = 0
            floor_area = 0
            heights = []
            building_sizes = []
            green_area = 0
            water_area = 0
            hardscape_area = 0
            
            # Process each region based on its zone type
            for i, (region, zone_type) in enumerate(zip(regions, zone_assignments)):
                zone_params = self.zone_types.get(zone_type, self.zone_types["Mixed"])
                region_area = region.area
                
                # Calculate building footprint for this region
                density = np.random.uniform(zone_params.min_density, zone_params.max_density)
                footprint = region_area * density
                building_footprint_area += footprint
                
                # Calculate height and floor area
                height = np.random.uniform(zone_params.min_height, zone_params.max_height)
                heights.append(height)
                
                # Add some variance to building heights
                if zone_params.height_variance > 0:
                    num_buildings = max(1, int(np.random.normal(5, 2)))
                    building_heights = np.random.normal(
                        height, zone_params.height_variance, num_buildings
                    )
                    building_heights = np.clip(
                        building_heights, zone_params.min_height, zone_params.max_height
                    )
                    
                    # Calculate floor area ratio
                    far = np.random.uniform(zone_params.min_far, zone_params.max_far)
                    total_floor_area = region_area * far
                    floor_area += total_floor_area
                    
                    # Distribute building sizes with variance
                    mean_size = footprint / num_buildings
                    building_area_variance = mean_size * zone_params.massing_variance
                    for _ in range(num_buildings):
                        bsize = np.random.normal(mean_size, building_area_variance)
                        bsize = max(10, bsize)  # Minimum building size
                        building_sizes.append(bsize)
                
                # Calculate green, water, and hardscape areas
                green_area += region_area * zone_params.green_proportion
                water_area += region_area * zone_params.water_proportion
                hardscape_area += region_area * zone_params.hardscape_proportion
            
            # Calculate the morphological parameters
            params = {
                "DN": building_footprint_area / total_area if total_area > 0 else 0,  # Density factor
                "FAR": floor_area / total_area if total_area > 0 else 0,  # Floor area ratio
                "AH": np.mean(heights) if heights else 0,  # Average height
                "HV": np.var(heights) if len(heights) > 1 else 0,  # Height variance
                "MV": np.var(building_sizes) / np.mean(building_sizes) if building_sizes and np.mean(building_sizes) > 0 else 0,  # Building massing variance
                "GP": green_area / total_area if total_area > 0 else 0,  # Green area proportion
                "WP": water_area / total_area if total_area > 0 else 0,  # Water surface proportion
                "HP": hardscape_area / total_area if total_area > 0 else 0,  # Hardscape proportion
                "total_area": total_area * self.pixels_to_meters ** 2,  # Convert to real-world units
                "building_footprint_area": building_footprint_area * self.pixels_to_meters ** 2,
                "floor_area": floor_area * self.pixels_to_meters ** 2
            }
            
            return params
        except Exception as e:
            print(f"Error extracting morphological parameters: {e}")
            traceback.print_exc()
            # Return default values
            return {
                "DN": 0.3,
                "FAR": 1.0,
                "AH": 10.0,
                "HV": 5.0,
                "MV": 0.2,
                "GP": 0.2,
                "WP": 0.05,
                "HP": 0.4,
                "total_area": 10000,
                "building_footprint_area": 3000,
                "floor_area": 10000
            }
        
    def visualize_urban_plan(self, boundary, regions, zone_assignments, road_network=None,
                           image_path=None, output_file=None):
        """Visualize the urban plan with zones, roads, and statistics"""
        try:
            # Extract morphological parameters
            morph_params = self.extract_morphological_parameters(regions, zone_assignments)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Draw boundary
            boundary_plot = np.vstack([boundary, boundary[0]])
            ax.plot(boundary_plot[:, 0], boundary_plot[:, 1],
                   'k--', label="Boundary", linewidth=2)
            
            # Fill each region with zone-specific color
            zone_counts = {}
            for i, (region, zone_type) in enumerate(zip(regions, zone_assignments)):
                # Count zones by type
                zone_counts[zone_type] = zone_counts.get(zone_type, 0) + 1
                
                # Get color for this zone type
                zone_params = self.zone_types.get(zone_type, self.zone_types["Mixed"])
                color = zone_params.color
                
                # Fill the region
                if not region.is_empty:
                    if hasattr(region, 'exterior'):
                        # Single polygon
                        x, y = region.exterior.xy
                        ax.fill(x, y, alpha=0.6, color=color)
                        
                        # Add label with zone number and type
                        centroid = region.centroid
                        ax.text(centroid.x, centroid.y,
                                f"#{i+1}\n{zone_type}",
                                ha='center', va='center', fontsize=8, fontweight='bold')
                    else:
                        # MultiPolygon - handle each part
                        for part in region.geoms:
                            x, y = part.exterior.xy
                            ax.fill(x, y, alpha=0.6, color=color)
            
            # Draw road network if provided
            if road_network and 'roads' in road_network:
                for road in road_network['roads']:
                    if not road.is_empty:
                        if isinstance(road, LineString):
                            x, y = road.xy
                            ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.8)
                        else:
                            # Handle other geometries like MultiLineString
                            try:
                                for line in road.geoms:
                                    x, y = line.xy
                                    ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.8)
                            except Exception as e:
                                print(f"Error plotting road: {e}")
            
            # Create legend items for zone types
            legend_items = []
            for zone_type, count in zone_counts.items():
                zone_params = self.zone_types.get(zone_type, self.zone_types["Mixed"])
                patch = Patch(
                    facecolor=zone_params.color,
                    alpha=0.6,
                    label=f"{zone_type}: {count} zones"
                )
                legend_items.append(patch)
            
            # Add morphological parameter information
            ax.text(
                0.02, 0.02,
                f"Morphological Parameters:\n"
                f"DN (Density): {morph_params['DN']:.2f}\n"
                f"FAR (Floor Area Ratio): {morph_params['FAR']:.2f}\n"
                f"AH (Avg Height): {morph_params['AH']:.1f}m\n"
                f"GP (Green Proportion): {morph_params['GP']:.2f}\n"
                f"WP (Water Proportion): {morph_params['WP']:.2f}\n"
                f"HP (Hardscape Proportion): {morph_params['HP']:.2f}\n"
                f"Total Area: {morph_params['total_area']:.1f} m²",
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
                fontsize=9,
                verticalalignment='bottom'
            )
            
            # Set title
            title = "Urban Plan"
            if image_path:
                title += f": {os.path.basename(image_path)}"
            ax.set_title(title)
            
            # Add legend
            ax.legend(handles=legend_items, loc='upper right')
            
            # Equal aspect ratio
            ax.set_aspect('equal')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Invert y-axis to match image coordinates
            ax.invert_yaxis()
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if output_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                return output_file
            else:
                plt.show()
                plt.close()
                return None
                
        except Exception as e:
            print(f"Error in visualization: {e}")
            traceback.print_exc()
            return None
            
    def create_tensor_for_udgan(self, regions, zone_assignments, road_network=None, boundary=None):
        """Create multi-channel tensor representation for UDGAN input"""
        try:
            # Define output dimensions (256x256 is standard for many GANs)
            width, height = 256, 256
            
            # Initialize output channels
            channels = {
                "buildings_low": np.zeros((height, width), dtype=np.float32),   # Low-rise buildings
                "buildings_mid": np.zeros((height, width), dtype=np.float32),   # Mid-rise buildings
                "buildings_high": np.zeros((height, width), dtype=np.float32),  # High-rise buildings
                "roads": np.zeros((height, width), dtype=np.float32),           # Road network
                "green": np.zeros((height, width), dtype=np.float32),           # Green spaces
                "water": np.zeros((height, width), dtype=np.float32),           # Water bodies
                "zone_types": np.zeros((height, width, len(self.zone_types)), dtype=np.float32)  # One-hot encoded zone types
            }
            
            # Get bounding box for normalization
            if boundary is not None:
                bpoly = Polygon(boundary)
                minx, miny, maxx, maxy = bpoly.bounds
            else:
                # Compute bounds from regions
                bounds = [region.bounds for region in regions]
                minx = min(b[0] for b in bounds)
                miny = min(b[1] for b in bounds)
                maxx = max(b[2] for b in bounds)
                maxy = max(b[3] for b in bounds)
            
            # Function to normalize coordinates to image space
            def norm_coords(x, y):
                nx = int((x - minx) / (maxx - minx) * (width - 1))
                ny = int((y - miny) / (maxy - miny) * (height - 1))
                return min(max(nx, 0), width-1), min(max(ny, 0), height-1)
            
            # Create rasterized representation of regions with zone types
            for i, (region, zone_type) in enumerate(zip(regions, zone_assignments)):
                # Get zone parameters
                zone_params = self.zone_types.get(zone_type, self.zone_types["Mixed"])
                
                # Sample points within the region
                if hasattr(region, 'exterior'):
                    # Get polygon bounds
                    rx_min, ry_min, rx_max, ry_max = region.bounds
                    
                    # Sample grid points within bounds
                    x_steps = np.linspace(rx_min, rx_max, int((rx_max - rx_min) / (maxx - minx) * width) + 1)
                    y_steps = np.linspace(ry_min, ry_max, int((ry_max - ry_min) / (maxy - miny) * height) + 1)
                    
                    for x in x_steps:
                        for y in y_steps:
                            if region.contains(Point(x, y)):
                                nx, ny = norm_coords(x, y)
                                
                                # Set zone type (one-hot encoded)
                                zone_idx = list(self.zone_types.keys()).index(zone_type)
                                channels["zone_types"][ny, nx, zone_idx] = 1.0
                                
                                # Set building density based on zone parameters
                                density = np.random.uniform(zone_params.min_density, zone_params.max_density)
                                
                                # Determine building height category
                                height_val = np.random.uniform(zone_params.min_height, zone_params.max_height)
                                if height_val < 10:  # Low-rise
                                    channels["buildings_low"][ny, nx] = density
                                elif height_val < 25:  # Mid-rise
                                    channels["buildings_mid"][ny, nx] = density
                                else:  # High-rise
                                    channels["buildings_high"][ny, nx] = density
                                
                                # Set green and water based on zone parameters
                                channels["green"][ny, nx] = zone_params.green_proportion
                                channels["water"][ny, nx] = zone_params.water_proportion
            
            # Add road network if provided
            if road_network and 'roads' in road_network:
                for road in road_network['roads']:
                    if not road.is_empty:
                        try:
                            if isinstance(road, LineString):
                                # Sample points along the line
                                points = []
                                length = road.length
                                num_points = max(2, int(length / (maxx - minx) * width / 10))
                                for i in range(num_points):
                                    point = road.interpolate(i / (num_points - 1) * length)
                                    points.append((point.x, point.y))
                            else:
                                # Handle other geometries by extracting coordinates
                                points = []
                                for line in road.geoms:
                                    points.extend(list(line.coords))
                            
                            # Draw road points
                            for x, y in points:
                                nx, ny = norm_coords(x, y)
                                # Draw with thickness
                                for dx in range(-1, 2):
                                    for dy in range(-1, 2):
                                        nx2, ny2 = nx + dx, ny + dy
                                        if 0 <= nx2 < width and 0 <= ny2 < height:
                                            channels["roads"][ny2, nx2] = 1.0
                        except Exception as e:
                            print(f"Error processing road for tensor: {e}")
            
            # Combine channels into a single tensor
            tensor = np.zeros((height, width, 7 + len(self.zone_types)), dtype=np.float32)
            tensor[:, :, 0] = channels["buildings_low"]
            tensor[:, :, 1] = channels["buildings_mid"]
            tensor[:, :, 2] = channels["buildings_high"]
            tensor[:, :, 3] = channels["roads"]
            tensor[:, :, 4] = channels["green"]
            tensor[:, :, 5] = channels["water"]
            tensor[:, :, 6:6+len(self.zone_types)] = channels["zone_types"]
            
            return {
                "tensor": tensor,
                "metadata": {
                    "width": width,
                    "height": height,
                    "channels": {
                        "buildings_low": 0,
                        "buildings_mid": 1,
                        "buildings_high": 2,
                        "roads": 3,
                        "green": 4, 
                        "water": 5,
                        "zone_types": list(range(6, 6 + len(self.zone_types)))
                    },
                    "zone_type_mapping": {i: zone for i, zone in enumerate(self.zone_types.keys())}
                }
            }
        except Exception as e:
            print(f"Error creating tensor for UDGAN: {e}")
            traceback.print_exc()
            # Return dummy tensor with correct shape
            dummy_tensor = np.zeros((256, 256, 7 + len(self.zone_types)), dtype=np.float32)
            return {"tensor": dummy_tensor, "metadata": {"error": str(e)}}

    def process_and_plan(self, image_path, output_folder, num_zones=10, pixels_to_meters=1.0):
        """Complete pipeline for processing an image and generating an urban plan"""
        try:
            # Create output folder
            os.makedirs(output_folder, exist_ok=True)
            
            # Set scale
            self.set_scale(pixels_to_meters)
            
            # 1) Extract boundary from image or use unet boundary if available
            boundary_data = self.extract_boundary_from_image(image_path)
            if boundary_data is None:
                return None
                
            boundary = boundary_data["boundary"]
            
            # Verify boundary has enough points and is valid
            if len(boundary) < 3:
                print("Invalid boundary: needs at least 3 points")
                return None
                
            # Verify boundary is a valid polygon
            try:
                poly = Polygon(boundary)
                if not poly.is_valid:
                    # Try to fix invalid polygon
                    poly = poly.buffer(0)
                    if not poly.is_valid:
                        print("Cannot create valid polygon from boundary")
                        return None
                    boundary = np.array(poly.exterior.coords)[:-1]
            except Exception as e:
                print(f"Error validating boundary: {e}")
                return None
            
            # 2) Generate road network based on constraints
            road_network = self.generate_road_network(
                boundary, 
                num_zones, 
                road_type=self.constraints.road_network_type
            )
            
            # 3) Generate Voronoi zones
            voronoi_data = self.generate_voronoi_zones(boundary, num_zones)
            if voronoi_data is None:
                return None
                
            regions = voronoi_data["regions"]
            
            # 4) Assign zone types
            zone_assignments = self.assign_zone_types(regions)
            
            # 5) Visualize the plan
            vis_output = self.visualize_urban_plan(
                boundary,
                regions,
                zone_assignments,
                road_network,
                image_path,
                os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_urban_plan.png")
            )
            
            # 6) Extract morphological parameters
            morph_params = self.extract_morphological_parameters(regions, zone_assignments)
            
            # 7) Create tensor for UDGAN
            tensor_data = self.create_tensor_for_udgan(regions, zone_assignments, road_network, boundary)
            
            # 8) Save parameters to JSON
            params_output = os.path.join(
                output_folder,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_params.json"
            )
            
            with open(params_output, 'w') as f:
                json.dump({
                    "morphological_parameters": morph_params,
                    "road_network": {
                        "type": road_network["road_type"],
                        "num_roads": road_network["num_roads"]
                    },
                    "zones": {
                        "num_zones": len(regions),
                        "zone_distribution": {zone: zone_assignments.count(zone) for zone in set(zone_assignments)}
                    },
                    "scale": pixels_to_meters
                }, f, indent=2)
            
            return {
                "boundary": boundary,
                "regions": regions,
                "zone_assignments": zone_assignments,
                "road_network": road_network,
                "morphological_parameters": morph_params,
                "visualization_output": vis_output,
                "parameters_output": params_output,
                
            }
            
        except Exception as e:
            print(f"Error in urban planning process: {e}")
            traceback.print_exc()
            return None