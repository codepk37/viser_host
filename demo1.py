# demo code to visualizwe all at time 
import numpy as np
import open3d as o3d
import os
from makegripper_points import plot_gripper_pro_max
from scipy.spatial.transform import Rotation as R

# --- New imports for Viser ---
import time
import viser
import viser.transforms as tf

# Load saved data
data_dir = './antipodal/blue_cylinder/inputs/'
cloud_path = os.path.join(data_dir, "cloud.ply")
grasps_path = os.path.join(data_dir, "grasps.npy")

cloud = o3d.io.read_point_cloud(cloud_path)
grasp_data = np.load(grasps_path, allow_pickle=True).item()

translations = grasp_data['translations']
rotations = grasp_data['rotations']
widths = grasp_data['widths']
heights = grasp_data['heights']
scores = grasp_data['scores']
gripper_points = grasp_data['gripper_points']

# --- Plane segmentation to detect the table ---
plane_model, inliers = cloud.segment_plane(
    distance_threshold=0.005,
    ransac_n=3,
    num_iterations=1000
)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

# Copy cloud and color inliers (table points) white
colored_cloud = o3d.geometry.PointCloud()
colored_cloud.points = o3d.utility.Vector3dVector(np.asarray(cloud.points).copy())
# Handle case where input cloud has no colors
if cloud.has_colors():
    colored_cloud.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors).copy())
else:
    colored_cloud.paint_uniform_color([0.5, 0.5, 0.5])

colors = np.asarray(colored_cloud.colors)
colors[inliers] = [.9, .8, .7]  # white for table points

# --- Viser Setup ---
server = viser.ViserServer()

server.add_point_cloud(
    name="/scene/point_cloud",
    points=np.asarray(colored_cloud.points),
    colors=np.asarray(colored_cloud.colors),
    point_size=0.002
)
server.add_frame(name="/scene/world_origin", show_axes=True, axes_length=0.1, axes_radius=0.002)


def visualize_selected_grasps(grasp_indices):
    selected_index = []

    for idx in grasp_indices:
        if idx >= len(translations):
            print(f"Index {idx} out of range.")
            continue

        t = translations[idx]
        R_mat = rotations[idx]
        width = widths[idx]
        height = heights[idx]
        score = scores[idx]
        gripper_point = gripper_points[idx]

        # This function returns an open3d.TriangleMesh object.
        gripper_mesh, _ = plot_gripper_pro_max(t, R_mat, width=width, depth=height, score=score, color=[0, 1, 0])

        # Reorient for xArm
        R_mat_XG = R_mat.copy()
        R_mat_XG[:, [0, 2]] = R_mat_XG[:, [2, 0]]
        R_mat_XG[:, 1] *= -1

        # Pregrasp and final grasp transforms
        z_axis = R_mat_XG[:, 2]
        t_moved_pregrasp = t - 0.18 * z_axis
        t_moved = t - 0.13 * z_axis
        x_axis = R_mat_XG[:, 0]
        t_moved += 0.01 * x_axis

        if t_moved[2] < 0.16:
            continue

        selected_index.append(idx)

        # --- CORRECTED BLOCK ---
        # Use server.add_mesh with the 'color' argument for a uniform color.
        server.add_mesh(
            name=f"/grasps/grasp_{idx}/gripper_mesh",
            vertices=np.asarray(gripper_mesh.vertices),
            faces=np.asarray(gripper_mesh.triangles),
            color=(0.0, 1.0, 0.0)  # Use a uniform green color
        )
        # --- END CORRECTED BLOCK ---

        # Add the coordinate frame
        # server.add_frame(
        #     name=f"/grasps/grasp_{idx}/frame",
        #     position=t_moved,
        #     wxyz=tf.SO3.from_matrix(R_mat_XG).wxyz,
        #     show_axes=True,
        #     axes_length=0.05,
        #     axes_radius=0.001
        # )

        # Add the contact points as small spheres
        for i, p in enumerate(gripper_point):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
            sphere.translate(p)
            server.add_mesh(
                name=f"/grasps/grasp_{idx}/contact_{i}",
                vertices=np.asarray(sphere.vertices),
                faces=np.asarray(sphere.triangles),
                color=(1.0, 0.0, 0.0) # Red
            )

        # Print grasp info
        rpy = R.from_matrix(R_mat_XG).as_euler('xyz', degrees=True)
        print(f"\nGrasp Index: {idx}")
        print(f"  Score: {score:.4f}")
        print(f"  Rotation (roll, pitch, yaw) [deg]: {rpy}")
        print(f"  Pregrasp Translation (x, y, z): {t_moved_pregrasp * 1000}")
        print(f"  Translation (x, y, z): {t_moved * 1000}")
        print(f"  Width: {width:.4f}  | Depth: {height:.4f}")

    print("\nSelected grasp indices added to scene:", selected_index)


# Example grasp indices (you can modify this)
visualize_selected_grasps(grasp_indices=[11, 2, 4, 9])

# --- Save the entire scene to a .viser file ---
print("\nSaving scene to grasp_visualization.viser...")
rec = server._start_scene_recording()
with open("grasp_visualization.viser", "wb") as f:
    f.write(rec.end_and_serialize())
print("Done. You can now drag-and-drop 'grasp_visualization.viser' into the Viser viewer.")

# Keep the script alive for a few seconds to see the URL in the console.
time.sleep(3)
