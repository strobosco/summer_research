import bpy
import mathutils

# Clear existing objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

filepath = "no_vt.obj"

# Import the mesh
bpy.ops.wm.obj_import(filepath=filepath)

####
# Find the imported object
obj = bpy.context.selected_objects[0]

# Ensure the object is selected
bpy.context.view_layer.objects.active = obj

bbox_vertices = obj.bound_box
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom rectangle
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top rectangle
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting lines
]

def create_line_mesh(name, vertices, edges):
    # Create a new mesh
    mesh = bpy.data.meshes.new(name)
    
    # Create a new object
    obj = bpy.data.objects.new(name, mesh)
    
    # Link the object to the current collection
    bpy.context.collection.objects.link(obj)
    
    # Create mesh from given vertices and edges
    # mesh.from_pydata([v[:] for v in vertices], edges, [])
    mesh.from_pydata([v[:] for v in vertices], [], edges)
    print("Vertices:")
    for vertex in mesh.vertices:
        print(f"Vertex {vertex.index}: {vertex.co}")

    # Print edges
    print("\nEdges:")
    for edge in mesh.edges:
        print(f"Edge {edge.index}: {edge.vertices[:]}")

    # Print faces
    print("\nFaces:")
    for face in mesh.polygons:
        print(f"Face {face.index}: {face.vertices[:]}")

    # If you need to see the bounding box
    print("\nBounding Box:")
    for i, corner in enumerate(obj.bound_box):
        world_corner = obj.matrix_world @ mathutils.Vector(corner)
        print(f"Corner {i}: {world_corner}")
    # Update mesh with new data
    mesh.update()
    
    return obj

create_line_mesh("LineMesh", bbox_vertices, edges)

# Update the scene
bpy.context.view_layer.update()
###

# Setup render scene
scene = bpy.context.scene
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = "face.png"

# Render the scene
bpy.ops.render.render(write_still=True)

# Save it for 3D visualization
output_filepath = "face_mesh.glb"
bpy.ops.export_scene.gltf(filepath=output_filepath, export_draco_mesh_compression_enable=False)