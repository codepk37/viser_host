#the animation will play through all the grasps and then automatically restart from the first grasp


import numpy as np
import open3d as o3d
import os
from makegripper_points import plot_gripper_pro_max
from pathlib import Path
import time
import viser

# --- Setup and Data Loading ---
server = viser.ViserServer()
data_dir = './antipodal/blue_cylinder/inputs/'
cloud = o3d.io.read_point_cloud(os.path.join(data_dir, "cloud.ply"))
grasp_data = np.load(os.path.join(data_dir, "grasps.npy"), allow_pickle=True).item()

# --- List to hold geometry for the final Open3D visualization ---
geometries_for_open3d = []

# --- Plane Segmentation and Coloring Logic ---
print("Detecting plane to color the table...")
if not cloud.has_colors():
    cloud.paint_uniform_color([0.5, 0.5, 0.5])

plane_model, inliers = cloud.segment_plane(
    distance_threshold=0.005,
    ransac_n=3,
    num_iterations=1000
)
print(f"  - Found {len(inliers)} points belonging to the plane.")
final_colors = np.asarray(cloud.colors)
final_colors[inliers] = [0.9, 0.8, 0.9] 

# Create a new Open3D PointCloud object with the final colors for the preview
o3d_colored_cloud = o3d.geometry.PointCloud()
o3d_colored_cloud.points = cloud.points
o3d_colored_cloud.colors = o3d.utility.Vector3dVector(final_colors)
geometries_for_open3d.append(o3d_colored_cloud)
geometries_for_open3d.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

# --- Add Static Scene Elements to Viser (Always Visible) ---
server.add_point_cloud(
    name="/scene/point_cloud",
    points=np.asarray(cloud.points),
    colors=final_colors,
    point_size=0.002
)

# --- Create Grasp Geometry for Viser (Initially Hidden) ---
# This loop populates both the Viser scene and the Open3D preview list
grasp_indices = [11, 2, 4, 9]
all_grasp_handles = []
for idx in grasp_indices:
    t = grasp_data['translations'][idx]
    R_mat = grasp_data['rotations'][idx]
    
    gripper_mesh, _ = plot_gripper_pro_max(t, R_mat, width=grasp_data['widths'][idx], depth=grasp_data['heights'][idx])
    
    # Add the mesh to our Open3D list for the final preview
    geometries_for_open3d.append(gripper_mesh)
    
    # Add the mesh to the Viser scene but keep it hidden
    handle = server.add_mesh(
        name=f"/grasps/grasp_{idx}",
        vertices=np.asarray(gripper_mesh.vertices),
        faces=np.asarray(gripper_mesh.triangles),
        color=(0.0, 1.0, 0.0),
        visible=False,
    )
    all_grasp_handles.append(handle)

# --- NEW ORDER: Dynamic Scene Export (Animation) FIRST ---
print("\nRecording Viser animation...")
recorder = server._start_scene_recording()
recorder.set_loop_start()

for i, handle in enumerate(all_grasp_handles):
    print(f"  - Frame {i+1}/{len(all_grasp_handles)}")
    handle.visible = True
    recorder.insert_sleep(1.0)
    handle.visible = False

# Save the Final Animation
output_filename = "grasp_animation_colored_plane.viser"
print(f"Saving animation to {output_filename}...")
Path(output_filename).write_bytes(recorder.end_and_serialize())
print("File saved successfully.")

# --- NEW ORDER: Show the Open3D Preview Window LAST ---
print("\n.viser file saved. Now showing static preview in Open3D window.")
print("Close the Open3D window to exit the script.")
o3d.visualization.draw_geometries(
    geometries_for_open3d,
    window_name="Static Preview (after saving .viser)"
)

print("Script finished.")
