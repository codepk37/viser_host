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

# --- Add Static Scene Elements (Always Visible) ---
server.add_point_cloud(
    name="/scene/point_cloud",
    points=np.asarray(cloud.points),
    colors=np.asarray(cloud.colors),
    point_size=0.002
)

# --- Create Grasp Geometry (Initially Hidden) ---
grasp_indices = [11, 2, 4, 9]
all_grasp_handles = []
for idx in grasp_indices:
    t = grasp_data['translations'][idx]
    R_mat = grasp_data['rotations'][idx]
    
    # Create the gripper mesh using your helper function
    gripper_mesh, _ = plot_gripper_pro_max(t, R_mat, width=grasp_data['widths'][idx], depth=grasp_data['heights'][idx])
    
    # Add the mesh to the scene but keep it hidden, and store its handle
    handle = server.add_mesh(
        name=f"/grasps/grasp_{idx}",
        vertices=np.asarray(gripper_mesh.vertices),
        faces=np.asarray(gripper_mesh.triangles),
        color=(0.0, 1.0, 0.0),
        visible=False,
    )
    all_grasp_handles.append(handle)

# --- Dynamic Scene Export (Animation) ---
print("Recording animation...")

# 1. Get the recorder object
recorder = server._start_scene_recording()

# 2. THE FIX: Set the loop point right after starting the recording.
recorder.set_loop_start()

# 3. Loop through the handles to create the animation frames
for i, handle in enumerate(all_grasp_handles):
    print(f"  - Frame {i+1}/{len(all_grasp_handles)}")
    
    # Make the current grasp visible
    handle.visible = True
    
    # Record a 1.0 second pause in the animation
    recorder.insert_sleep(1.0)
    
    # Make the grasp invisible again for the next frame
    handle.visible = False

# 4. Serialize the entire animation sequence and save to a file
output_filename = "grasp_animation_looping.viser"
print(f"Saving animation to {output_filename}...")
Path(output_filename).write_bytes(recorder.end_and_serialize())

print("Done. The new file will now loop automatically.")
time.sleep(2)
