import numpy as np
import networkx as nx
import os
import pygmsh, gmsh
from collections import Counter
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint, Point, LineString
import matplotlib as mpl
from matplotlib.collections import PatchCollection, LineCollection
from typing import List, Tuple
import pickle
from copy import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.linalg import lstsq
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, cKDTree
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, from_networkx
from torch_geometric.utils import scatter
from shapely.geometry import Polygon
from shapely.prepared import prep
import xarray as xr
from meshkernel import MeshKernel, GeometryList, OrthogonalizationParameters, ProjectToLandBoundaryOption, MeshRefinementParameters
# from meshkernel import Mesh2d, py_structures, DeleteMeshOption

def center_grid_graph(dim1, dim2, grid_size=1):
    """
    Create a graph from a rectangular grid of dimensions dim1 x dim2.

    Args:
        dim1 (int): Number of grids in the x direction.
        dim2 (int): Number of grids in the y direction.
        grid_size (int, optional): Size of each grid. Defaults to 1.

    Returns:
        Tuple[nx.Graph, dict]: A tuple containing the graph connecting the grid centers and the corresponding node positions.
    """
    G = nx.grid_2d_graph(dim1, dim2, create_using=nx.DiGraph)
    # for the position, it is assumed that they are located in the centre of each grid
    pos = {i:((x+0.5)*grid_size,(y+0.5)*grid_size) for i, (x,y) in enumerate(G.nodes())}
    
    #change keys from (x,y) format to i format
    mapping = dict(zip(G, range(0, G.number_of_nodes())))
    G = nx.relabel_nodes(G, mapping)

    return G, pos

def get_coords(pos: dict):
    """
    Returns x and y coordinates of each node in a dictionary {node: (x, y)}.

    Args:
        pos (dict): keys - nodes, values - (x, y) positions of each node.

    Returns:
        np.ndarray: Array of shape (n_nodes, 2) containing x and y coordinates of each node.
    """
    return np.array([xy for xy in pos.values()])

def mesh_radius_graph(face_xy, max_radius=100):
    """
    Create a radius neighbors graph from face coordinates.

    Args:
        face_xy (np.ndarray): Array of face coordinates.
        max_radius (int, optional): Maximum radius to consider for neighbors. Defaults to 100.

    Returns:
        np.ndarray: Array of shape (2, n_edges) containing the edge indices.
    """
    radius_graph = radius_neighbors_graph(face_xy, max_radius, mode='connectivity', include_self=False)
    new_edge_index = np.stack(radius_graph.nonzero())
    assert new_edge_index.shape[1] > 0, 'There are no edges with the selected radius, try increasing it'

    return new_edge_index

def dual_graph_from_mesh(mesh):
    """
    Create a dual graph from a mesh.

    Args:
        mesh (Mesh): Mesh object containing the dual edge index and face coordinates.

    Returns:
        Tuple[nx.Graph, np.ndarray]: A tuple containing the dual graph and the face coordinates.
    """
    graph = nx.from_edgelist(mesh.dual_edge_index.T)
    pos = mesh.face_xy
    return graph, pos

def graph_from_mesh(mesh):
    """
    Create a graph from a mesh.

    Args:
        mesh (Mesh): Mesh object containing the edge index and node coordinates.

    Returns:
        Tuple[nx.Graph, np.ndarray]: A tuple containing the graph and the node coordinates.
    """
    graph = nx.from_edgelist(mesh.edge_index.T)
    pos = mesh.node_xy
    return graph, pos

def get_graph_from_geodataframe(geodataframe):
    """Create a graph from a polygon file. The polygon file should be a .gpkg object and contain subpolygons."""
    # Calculate the centroids of each polygon
    geodataframe['centroid'] = geodataframe.centroid

    # Spatial join to find touching polygons (i.e., neighbors)
    neighbors = gpd.sjoin(geodataframe, geodataframe, how="inner", predicate="touches")
    
    def get_shared_boundary_length(poly1, poly2):
        """Get the lenght of the part of the boundary that is shared between two polygons"""
        shared_boundary = poly1.intersection(poly2)
        if shared_boundary.is_empty:
            return 0
        return shared_boundary.length
    
    # calculate the shared boundary length between each pair of neighbors
    shared_boundary_lengths = []
    for i, row in neighbors.iterrows():
        poly1 = row['geometry']
        poly2 = geodataframe.loc[row['index_right']].geometry
        shared_boundary_lengths.append(get_shared_boundary_length(poly1, poly2))

    neighbors['shared_boundary_length'] = shared_boundary_lengths

    # remove edges of polygons that do not share a boundary
    neighbors = neighbors.iloc[np.where(neighbors.shared_boundary_length != 0)]

    # Initialize an empty undirected graph
    G = nx.Graph()

    # Add nodes to the graph using the centroid coordinates
    for idx, row in geodataframe.iterrows():
        G.add_node(row.name, pos=(row['centroid'].x, row['centroid'].y), area=row.geometry.area)

    # Add edges based on neighboring polygons and as edge weight the shared boundary length
    for idx, row in neighbors.iterrows():
        G.add_edge(row.name, row['index_right'], edge_length=row.shared_boundary_length)

    return G

def sample_points_from_grid(points, grid_size):
    """
    Sample points from a regular square grid. Each square in the grid will have a single point selected at random.

    Args:
        points (np.ndarray): Array of points to sample from, shape (n_points, 2).
        grid_size (float): Size of each square in the grid.

    Returns:
        np.ndarray: Array of sampled points.
    """
    num_points = 1  # Number of points to select in each square

    # Generate the regular grid
    x = np.arange(points[:,0].min(), points[:,0].max() + grid_size, grid_size)
    y = np.arange(points[:,1].min(), points[:,1].max() + grid_size, grid_size)

    selected_points = []
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            # Create a mask for each square
            square_mask = ((points[:,0] > x[i]) & (points[:,0] < x[i+1])) & \
                          ((points[:,1] > y[j]) & (points[:,1] < y[j+1]))
            possible_points = points[square_mask]

            if len(possible_points) > 0:
                # Generate random indices within the square
                indices = np.random.randint(0, len(possible_points), size=min(num_points, len(possible_points)))
                
                # Add the selected points to the list
                selected_points.extend(possible_points[indices])

    selected_points = np.array(selected_points)

    print(f"Sampled {(selected_points.shape[0] / points.shape[0])*100:0.1f}% of the nodes")

    return selected_points

def order_boundary_nodes(boundary_node_xy):
    """
    Order the boundary nodes in a clockwise fashion starting from 180 degrees.

    Args:
        boundary_node_xy (np.ndarray): Array of boundary node coordinates.

    Returns:
        np.ndarray: Ordered boundary node coordinates.
    """
    # get center of the boundary nodes and shift the coordinates
    center = boundary_node_xy.mean(0)
    bnd_node_xy = boundary_node_xy - center

    # get complex numbers
    zs = bnd_node_xy[:,0] + 1j * bnd_node_xy[:,1]
    
    # sort by angle to obtain the boundary nodes in a clockwise fashion
    boundary_nodes_sorted = boundary_node_xy[np.angle(zs).argsort()]
    return boundary_nodes_sorted

def get_boundary_edges(mesh):
    boundary_edges = []
    face_nodes_matrix = get_face_nodes_matrix(mesh)

    for edge_id in range(mesh.num_edges):
        is_internal_edge = sum([(mesh.edge_index.T[edge_id] == face_nodes_matrix[:,i:i+2]).all(1).sum() + 
                                (mesh.edge_index.T[edge_id][::-1] == face_nodes_matrix[:,i:i+2]).all(1).sum() 
                                for i in range(mesh.nodes_per_face.max())]) == 2
        if not is_internal_edge:
            boundary_edges.append(edge_id)
    return np.array(boundary_edges)

def get_concave_hull(points, alpha, only_outer=True):
    """
    Answer from https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
    
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return np.array(list(edges))

def close_polygon(points):
    """
    Close a polygon by ensuring the first and last points are the same.

    Args:
        points (np.ndarray): Array of polygon points.

    Returns:
        np.ndarray: Array of closed polygon points.
    """
    if not np.all(points[0] == points[-1]):
        points = np.vstack((points, points[0]))
    return points

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float, 
                     num_vertices: int, seed: float, ellipticality: float=1):
    """
    TAKEN FROM: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]): 
            A pair representing the center of the circumference used to generate the polygon.
        avg_radius (float): 
            The average radius (distance of each generated vertex to the center of the circumference) 
            used to generate points with a normal distribution.
        irregularity (float): 
            Variance of the spacing of the angles between consecutive vertices.
        spikiness (float): 
            Variance of the distance of each vertex to the center of the circumference.
        ellipticality (float): 
            Ratio between the major and minor axis of the ellipse used to generate the polygon.
        num_vertices (int): 
            The number of vertices of the polygon.
        seed (float): 
            Seed for the random number generator.

    Returns:
        Polygon: The generated polygon.
    """
    np.random.seed(seed)

    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * np.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity, seed)

    points = []
    angle = np.random.uniform(0, 2 * np.pi)
    for i in range(num_vertices):
        radius = np.clip(np.random.normal(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * np.cos(angle) * ellipticality,
                 center[1] + radius * np.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    polygon = Polygon(points)

    return polygon

def random_angle_steps(steps: int, irregularity: float, seed: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    np.random.seed(seed)

    # generate n angle steps
    angles = []
    lower = (2 * np.pi / steps) - irregularity
    upper = (2 * np.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = np.random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * np.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def get_equidistant_perimiter(vertices):
    """Add more vertices to the perimiter of the polygon to make the segments equidistant.
    
    Args:
        vertices (np.ndarray): Array of polygon vertices.
        
    Returns:
        np.ndarray: Array of vertices with equidistant segments.
    """
    segments_lengths = np.array([np.linalg.norm(vertices[i+1,:] - vertices[i,:]) for i in range(len(vertices)-1)])
    min_length = segments_lengths.min()

    new_vertices = copy(vertices)
    for i in range(len(vertices)-1):
        segment_ratio = segments_lengths[i] / min_length
        if segment_ratio > 2:
            more_segments = np.linspace(vertices[i,:], vertices[i+1,:], int(np.ceil(segment_ratio/2)))
            index = i + len(new_vertices) - len(vertices)
            new_vertices = np.concatenate((new_vertices[:index+1,:], more_segments[1:-1], new_vertices[index+1:,:]))

    return new_vertices

def save_polygon_to_file(polygon, filename):
    """Save a polygon to a file.
    
    Args:
        polygon (Polygon): Polygon to save.
        filename (str): Path to the file
    """
    assert isinstance(polygon, Polygon), "polygon must be a shapely.geometry.Polygon object"

    with open(filename, 'w') as f:
        f.write(f'# Extent: {polygon.bounds}\n')
        f.write('# Coordinates (x, y):\n')
        for coord in polygon.exterior.coords:
            f.write(f'{coord[0]}, {coord[1]}\n')
    
    return None

def generate_random_polygon(save_polygon=False, avg_radius=100, irregularity=0.5, 
                            spikiness=0.2, seed=42, num_vertices=(20,30), ellipticality=(0.5,2)):
    """Generates a polygon with random vertices.
    
    Args:
        save_polygon (bool): If True, saves the polygon to 'random_polygon.pol' file.
        avg_radius (float): Average radius of the polygon.
        irregularity (float): Variance of the spacing of the angles between consecutive vertices.
        spikiness (float): Variance of the distance of each vertex to the center of the circumference.
        seed (int): Seed for the random number generator.
        num_vertices (Tuple[int, int]): Range of the number of vertices of the polygon.
        ellipticality (Tuple[float, float]): Range of the ellipticality of the polygon.
    
    Returns:
        Polygon: The generated polygon."""
    np.random.seed(seed)

    num_vertices = np.random.randint(num_vertices[0], num_vertices[1])
    ellipticality = np.random.uniform(ellipticality[0], ellipticality[1])
    avg_radius = avg_radius/ellipticality
    polygon = generate_polygon(center=(avg_radius, avg_radius), avg_radius=avg_radius, 
                               irregularity=irregularity, spikiness=spikiness,
                               ellipticality=ellipticality,
                               num_vertices=num_vertices, seed=seed)

    if save_polygon:
        save_polygon_to_file(polygon, 'random_polygon.pol')

    vertices = np.array(polygon.exterior.coords)
    vertices = get_equidistant_perimiter(vertices)

    return polygon

def plot_faces(mesh, face_value=None, ax=None, remove_ticks=True, linewidths=0.01, **kwargs):
    """Plots the mesh with face values if specified
    
    Args:
        mesh (Mesh): Mesh object
        ax (matplotlib.axes.Axes): Axes to plot the mesh
        face_value (np.ndarray): Array of face values to plot
        kwargs: Additional keyword arguments for the plot
    
    Returns:
        matplotlib.axes.Axes: Axes with the plot
    """
    ax = ax or plt.gca()

    if hasattr(mesh, 'gdf'):
        mesh.gdf['map'] = face_value
        mesh.gdf.plot(ax=ax, edgecolor="black", linewidth=0.2, column='map', missing_kwds={"color": "none"}, **kwargs)
    else:
        # iterate over the faces and add them to a collection of patches
        node_position = 0
        patches = []
        for num_nodes in mesh.nodes_per_face:
            face_node = mesh.face_nodes[node_position : (node_position + num_nodes)]
            face_nodes_x = mesh.node_x[face_node]
            face_nodes_y = mesh.node_y[face_node]
            face = np.stack((face_nodes_x, face_nodes_y)).T
            node_position += num_nodes
            patches.append(mpl.patches.Polygon(face, closed=True))
            
        collection = PatchCollection(patches, edgecolor='k', linewidths=linewidths, **kwargs)
        collection.set_array(face_value)
        # mesh.plot_boundary(ax=ax, c='k', lw=0.25)
        ax.add_collection(collection)
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlim(mesh.node_x.min(), mesh.node_x.max())
    ax.set_ylim(mesh.node_y.min(), mesh.node_y.max())

    return ax

def plot_mesh(mesh, ax=None, node_size=2, lw=0.7, **plt_kwargs):
    """Plot the mesh.
    
    Args:
        mesh (Mesh): Mesh object
        ax (matplotlib.axes.Axes): Axes to plot the mesh
        node_size (int): Size of the nodes
        lw (float): Line width of the mesh
        plt_kwargs: Additional keyword arguments for the plot
    
    Returns:
        matplotlib.axes.Axes: Axes with the plot
    """
    ax = ax or plt.gca()
    
    if hasattr(mesh, 'gdf'):
        mesh.gdf.exterior.plot(color='black', lw=lw, ax=ax, **plt_kwargs)
    else:
        graph, pos = graph_from_mesh(mesh)
        nx.draw(graph, pos, ax=ax, node_size=node_size, **plt_kwargs)

    return ax

def plot_mesh_and_dual(mesh, ax=None, **plt_kwargs):
    """Plot the mesh and its dual graph.
    
    Args:
        mesh (Mesh): Mesh object
        ax (matplotlib.axes.Axes): Axes to plot the mesh
        plt_kwargs: Additional keyword arguments for the plot
    
    Returns:
        matplotlib.axes.Axes: Axes with the plot
    """
    ax = ax or plt.gca()
    
    if hasattr(mesh, 'gdf'):
        mesh_kwargs = dict(linestyle=':', lw=0.5)
    else:
        mesh_kwargs = dict(style='dotted', node_size=0, width=0.2)
    plot_mesh(mesh, ax, **mesh_kwargs)

    dual_graph, pos = dual_graph_from_mesh(mesh)
    
    node_size = 4000/dual_graph.number_of_nodes()**0.95
    nx.draw(dual_graph, pos, ax=ax, node_size=node_size, width=0.7, **plt_kwargs)

    if hasattr(mesh, 'ghost_cells_ids'):
        plt.scatter(*mesh.face_xy[mesh.ghost_cells_ids].T, s=20, c='r', zorder=3, marker='x')
    elif hasattr(mesh, 'face_BC'):
        plt.scatter(*mesh.face_xy[mesh.face_BC].T, s=20, c='b', zorder=3, marker='x')
    
    ax.set_xlim(mesh.node_x.min(), mesh.node_x.max())
    ax.set_ylim(mesh.node_y.min(), mesh.node_y.max())

    return ax

def plot_edges(mesh, edge_value, ax=None, colorbar=True, **kwargs):
    """
    Plots the edges of a mesh colored by a given variable per edge.

    Args:
        mesh : Mesh
            Mesh object with attributes node_xy and edge_index.
        edge_value : array-like
            Array of values per edge (length = number of edges).
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new figure and axis are created.
        colorbar : bool, optional
            Whether to add a colorbar.
        kwargs : dict
            Additional arguments passed to LineCollection.
    """
    ax = ax or plt.gca()

    assert mesh.dual_edge_index.shape[1] == edge_value.shape[0]

    # Get edge coordinates
    edge_nodes = mesh.dual_edge_index.T
    lines = [mesh.face_xy[nodes] for nodes in edge_nodes]

    lc = LineCollection(lines, array=np.asarray(edge_value), **kwargs)
    ax.add_collection(lc)
    ax.autoscale()

    if colorbar:
        plt.colorbar(lc, ax=ax, label='Edge value')

    return ax

def plot_multiscale_mesh_properties(meshes, with_area=True, **kwargs):
    """Plot the mesh properties of a multiscale mesh.

    Args:
        meshes (List[Mesh]): List of Mesh objects
        with_area (bool): If True, plot the face area histogram
        kwargs: Additional keyword arguments for the plot
    """
    assert isinstance(meshes, list), "meshes must be a list of Mesh objects"

    number_of_multiscales = len(meshes)
    
    height_ratios = [1, 0.65] if with_area else [1]

    fig, axs = plt.subplots(1+with_area, number_of_multiscales, figsize=(number_of_multiscales*4,4+with_area*2), 
                            gridspec_kw={'height_ratios': height_ratios})
    fig.suptitle("Mesh faces properties", fontsize=16)

    for i, mesh in enumerate(meshes):

        if with_area:
            ax0 = axs[0, i] if number_of_multiscales > 1 else axs[0]
            ax1 = axs[1, i] if number_of_multiscales > 1 else axs[1]
            ax1.hist(mesh.face_area, bins=30)
            ax1.set_xlabel("Face Area")
        else:
            ax0 = axs[i] if number_of_multiscales > 1 else axs
        plot_mesh(mesh, ax=ax0, **kwargs)
        ax0.set_title(f"Num faces: {mesh.face_xy.shape[0]}")

    plt.show()

def mesh_to_gdf(mesh, crs=None):
    """Convert mesh to GeoDataFrame for spatial operations"""
    gdf =  gpd.GeoSeries([Polygon(mesh.node_xy[face_nodes[~np.isnan(face_nodes)].astype(np.int64)]) 
                          for face_nodes in get_face_nodes_matrix(mesh)], crs=crs).to_frame('geometry')
    gdf['centroid'] = gdf.centroid
    return gdf

def connect_coarse_to_fine_mesh(coarse_mesh, fine_mesh):
    """Connects the coarse mesh to the fine mesh by creating a dual edge index between the two meshes.
    An edge is created if the center of the fine mesh face is contained in the coarse mesh face.
    
    Args:
        coarse_mesh (Mesh): Coarse mesh object
        fine_mesh (Mesh): Fine mesh object
    
    Returns:
        np.ndarray: Array of shape (2, n_edges) containing the dual edge index that connects the coarse mesh to the fine mesh.
    """
    assert isinstance(coarse_mesh, Mesh) and isinstance(fine_mesh, Mesh), "coarse_mesh and fine_mesh must be Mesh objects"
    assert coarse_mesh.num_faces <= fine_mesh.num_faces, "This function should apply coarse to fine mesh connections only"

    coarse_indices = []
    fine_indices = []
    fine_to_coarse = {}

    coarse_gdf = coarse_mesh.gdf if hasattr(coarse_mesh, 'gdf') else mesh_to_gdf(coarse_mesh)
    fine_points = gpd.GeoSeries([Point(x, y) for x, y in fine_mesh.face_xy], crs=coarse_gdf.crs).to_frame('geometry')

    join = gpd.sjoin(fine_points, coarse_gdf.reset_index(), predicate='within', how='left')
    
    if len(join) != len(fine_points):
        # There must be some double        
        duplicate_nodes = [item for item, count in Counter(join.index).items() if count > 1]

        for node in duplicate_nodes:
            # For each duplicate node, keep only the row with the minimum distance
            candidates = join[join.index == node]

            # Compute distances from centroid to geometry
            distances = candidates.apply(lambda row: row['centroid'].distance(row['geometry']), axis=1)

            join = join[~((join.index == node) & (join['index_right0'] != candidates.iloc[distances.argmin()]['index_right0']))]
            
    for fine_idx, row in join.dropna(subset=['index']).iterrows():
        fine_to_coarse[fine_idx] = int(row['index'])
    # Handle fine faces not contained in any coarse face (assign to nearest coarse face)
    missing = join['index'].isna()
    if missing.any():
        tree = cKDTree(coarse_mesh.face_xy)
        dists, idxs = tree.query(fine_mesh.face_xy[missing], k=1)
        for fine_idx, coarse_idx in zip(np.where(missing)[0], idxs):
            fine_to_coarse[fine_idx] = coarse_idx
    
    # Check if all coarse faces have at least one fine face connected
    missing_coarse_faces = set(range(coarse_mesh.num_faces)) - set(coarse_indices)
    if len(missing_coarse_faces) > 0:
        tree = cKDTree(fine_mesh.face_xy)
        for coarse_idx in missing_coarse_faces:
            # find the closest fine face to the coarse face center
            coarse_center = coarse_mesh.face_xy[coarse_idx]
            _, fine_idx = tree.query(coarse_center, k=1)
            fine_to_coarse[fine_idx] = coarse_idx

    # Now ensure each fine index is present only once
    for fine_idx, coarse_idx in fine_to_coarse.items():
        coarse_indices.append(coarse_idx)
        fine_indices.append(fine_idx)

    return np.vstack((coarse_indices, fine_indices)).T

def intersect_line_elements_and_mesh(line_elements, mesh):
    """Intersect line elements and mesh edges.
    Returns a Dataframe of the intersections"""    
    # Create a GeoDataFrame from the line elements
    line_gdf = gpd.GeoDataFrame(geometry=[LineString(line) for line in line_elements], crs="EPSG:28992")

    # Create a GeoDataFrame from the mesh edges
    mesh_gdf = gpd.GeoDataFrame(geometry=[LineString(edge) for edge in mesh.face_xy[mesh.dual_edge_index.T]], crs="EPSG:28992")

    # Perform a spatial join to find intersections
    intersection = gpd.sjoin(line_gdf, mesh_gdf)

    # add the intersection as an additional attribute
    intersection['intersected_point'] = [row['geometry'].intersection(mesh_gdf.loc[row['index_right'], 'geometry']) for idx, row in intersection.iterrows()]
    intersection = intersection.rename(columns={"index_right": "mesh_edge_index", "geometry": "line_element"})

    # remove multipoints and keep only the maximum z one
    mulit_points_ids = intersection.intersected_point.apply(lambda x: isinstance(x, MultiPoint))
    for idx, mp in intersection[mulit_points_ids].iterrows():
        max_z_geom = mp.intersected_point.geoms[np.argmax([point.z for point in mp.intersected_point.geoms])]
        intersection.at[idx, 'intersected_point'] = max_z_geom

    return intersection

def get_edge_weirs(line_elements, mesh):
    """Returns the weir height as an edge feature. It is 0 where there are no weirs"""
    intersection = intersect_line_elements_and_mesh(line_elements, mesh)

    edge_weir = np.zeros(mesh.dual_edge_index.shape[-1])
    edge_weir[intersection.mesh_edge_index] = intersection.intersected_point.apply(lambda x: x.coords[0][2])

    return edge_weir   

def create_mesh_dhydro(boundary_nodes, number_of_multiscales=4, for_simulation=True):
    '''Creates a fine mesh or a multiscale mesh using meshkernel
    
    Args:
        boundary_nodes(np.ndarray): Array of shape (n_nodes, 2) containing the boundary nodes
        number_of_multiscales (int): Number of multiscales to create
        for_simulation (bool): If True, returns the mesh as a Mesh2d object (from meshkernel)

    Returns:
        Mesh2d or List[Mesh]: Mesh object or list of Mesh objects
    '''
    boundary_polygon = GeometryList(boundary_nodes[:,0].copy(), boundary_nodes[:,1].copy())
    
    mk = MeshKernel()
    mk.mesh2d_make_triangular_mesh_from_polygon(boundary_polygon)

    meshes = []
    for i in range(number_of_multiscales):
        mk.mesh2d_compute_orthogonalization(ProjectToLandBoundaryOption(0), OrthogonalizationParameters(
                    outer_iterations=25, boundary_iterations=25, inner_iterations=25, 
                    orthogonalization_to_smoothing_factor=0.975),
                    boundary_polygon, boundary_polygon)
        
        if i == number_of_multiscales-1:
            mk.mesh2d_delete_small_flow_edges_and_small_triangles(
            small_flow_edges_length_threshold=0.1, min_fractional_area_triangles=2.0)
            
        mesh = Mesh()
        mesh._import_from_meshkernel(mk)
        meshes.append(mesh)

        if i < number_of_multiscales-1:
            refinement_parameters = MeshRefinementParameters(refine_intersected=True, min_edge_size=0.5, 
                                                        max_refinement_iterations=1, smoothing_iterations=5)
            mk.mesh2d_refine_based_on_polygon(boundary_polygon, refinement_parameters)
        
    if for_simulation:
        output_mesh2d = mk.mesh2d_get()

        output_mesh2d.mesh_nodes = np.stack((output_mesh2d.node_x, output_mesh2d.node_y), -1)
        output_mesh2d.face_xy = np.stack((output_mesh2d.face_x, output_mesh2d.face_y), -1)

        return output_mesh2d
    else:
        return meshes
    
def create_mesh_from_polygon(polygons_file):
    """Create a mesh that preserves the connecctivity taken from a polygon file.
    
    Args:
        polygons_file (str): Path to the polygon file.
        
    Returns:
        mesh (Mesh)
    """
    gdf = gpd.read_file(polygons_file)

    rings = []
    for poly in gdf.geometry:
        rings.append(np.array(poly.exterior.coords))
        for ihole in poly.interiors:
            rings.append(np.array(ihole.coords))

    def close_ring(r):
        return r if np.all(r[0] == r[-1]) else np.vstack((r, r[0]))
    rings = [close_ring(r) for r in rings]

    mk = MeshKernel()

    for ring in rings:
        geom = GeometryList(ring[:,0].copy(), ring[:,1].copy())
        mk.mesh2d_make_triangular_mesh_from_polygon(geom, scale_factor=2)

    mk.mesh2d_connect_meshes(mk.mesh2d_get(), search_fraction=0.4, connect=True)

    mesh = Mesh()
    mesh._import_from_meshkernel(mk)

    return mesh

def create_polygon_meshes(polygons_file, include_polygon_mesh=False):
    """
    Create list of meshes from a polygon file.

    Args:
        polygons_file (str): Path to the polygon file.
        include_polygon_mesh (bool): Whether to include the polygon mesh in the list of meshes.

    Returns:
        list: List of created meshes.
    """
    gdf = gpd.read_file(polygons_file)

    # Create polygon mesh
    geomesh = Mesh()
    geomesh._import_from_geodataframe(gdf)
    meshes = []
    if include_polygon_mesh:
        meshes.append(geomesh)

    # Create triangular mesh from polygon
    polygons_boundary = np.concatenate((geomesh.boundary_node_xy, geomesh.boundary_node_xy[:1])).T.copy()
    polygons_geometry = GeometryList(polygons_boundary[0], polygons_boundary[1])

    mk = MeshKernel()
    mk.mesh2d_make_triangular_mesh_from_polygon(polygons_geometry, scale_factor=1)
    mesh = Mesh()
    mesh._import_from_meshkernel(mk)

    # Remove elements outside the boundary
    boundary_polygon = gdf.union_all()
    mesh = remove_elements_outside_boundary(mesh, boundary_polygon)
    meshes.append(mesh)

    return meshes

def resample_line(line: LineString, distance: float):
    if distance is None:
        return line
    if distance <= 0:
        raise ValueError("Distance must be greater than 0")
    # If line is MultiLineString, convert to list of LineStrings
    if hasattr(line, 'geoms'):
        lines = list(line.geoms)
    else:
        lines = [line]

    densified_points = []
    for line in lines:
        num_points = int(line.length // distance)
        # Vectorized interpolation using numpy
        distances = np.linspace(0, line.length, num_points + 1)
        densified_points.extend([line.interpolate(d) for d in distances])
    
    densified_line = LineString([pt.coords[0] for pt in densified_points])

    return densified_line

def create_boundary_meshes(polygons_file, distance_points=[]):
    """
    Create list of meshes from a polygon file.

    Args:
        polygons_file (str): Path to the polygon file.
        include_polygon_mesh (bool): Whether to include the polygon mesh in the list of meshes.

    Returns:
        list: List of created meshes.
    """
    gdf = gpd.read_file(polygons_file)

    meshes = []

    for distance in distance_points:
        new_boundary = resample_line(gdf.union_all().boundary, distance)
        boundary_node_xy = filter_unique_nodes(np.array(new_boundary.xy).T)

        # Create triangular mesh from polygon
        polygons_boundary = np.concatenate((boundary_node_xy, boundary_node_xy[:1])).T.copy()
        polygons_geometry = GeometryList(polygons_boundary[0], polygons_boundary[1])

        mk = MeshKernel()
        mk.mesh2d_make_triangular_mesh_from_polygon(polygons_geometry, scale_factor=1)
        mesh = Mesh()
        mesh._import_from_meshkernel(mk)
        
        meshes.append(mesh)

    return meshes

def create_gmesh(polygons_file, with_interior_lines=True, max_distance=None, border_resample_distance=None):
    """
    Creates a gmsh mesh from a polygon file.

    Args:
        polygons_file (str): Path to the polygon file.
        with_interior_lines (bool): Whether to include interior lines in the mesh.
        max_distance (float): Maximum distance between points in the mesh.
        resample_distance (float): Distance for resampling the polygon boundary.

    Returns:
        meshio.Mesh: The created mesh.
    """
    gdf = gpd.read_file(polygons_file)

    boundary_coords = np.array(resample_line(gdf.union_all().boundary, border_resample_distance).coords)

    # Create mesh from boundary_coords
    with pygmsh.occ.Geometry() as geom:
        # Outer boundary loop
        boundary_points = [geom.add_point(p) for p in boundary_coords[:-1]]
        boundary_lines = [geom.add_line(boundary_points[i], boundary_points[i + 1])
                        for i in range(len(boundary_points) - 1)]
        # Close the loop by connecting the last point to the first
        boundary_lines.append(geom.add_line(boundary_points[-1], boundary_points[0]))
        
        loop = geom.add_curve_loop(boundary_lines)
        surface = geom.add_plane_surface(loop)

        if with_interior_lines:
            split_curves = []
            for line in gdf.geometry:
                coords = list(line.exterior.coords)
                if len(coords) < 2:
                    continue # Skip if the line has less than 2 points
                pts = [geom.add_point(p) for p in coords]
                for i in range(len(pts)-1):
                    # if np.linalg.norm(np.array(pts[i].x) - np.array(pts[i+1].x)) > 100:
                    split_curves.append(geom.add_line(pts[i], pts[i+1]))

            # Boolean fragments will ensure internal edges are respected
            geom.boolean_fragments([surface], split_curves)

        gmsh.option.setNumber("Mesh.Algorithm", 8)               # Frontal-Delaunay for quads
        # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Simple quad recombination
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)            # Try recombining all surfaces
        if max_distance is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_distance)

        gmesh = geom.generate_mesh(dim=2)

    mesh = Mesh()
    mesh._import_from_gmsh(gmesh)

    return mesh

def get_face_nodes_matrix(mesh):
    '''Returns a matrix with the nodes that define each face. NaN values are used to fill the empty spaces.

    Args:
        mesh (Mesh): mesh object

    Returns:
        np.ndarray: Array of shape (num_faces, max_nodes_per_face) containing the nodes that define each face
    '''
    max_nodes_per_face = mesh.nodes_per_face.max()
    face_nodes = np.zeros((mesh.num_faces, max_nodes_per_face)) * np.nan
    node_position = 0

    for i, num_nodes in enumerate(mesh.nodes_per_face):
        face_nodes[i,:num_nodes] = mesh.face_nodes[node_position : (node_position + num_nodes)]
        node_position += num_nodes

    return face_nodes

def get_outside_elements(mesh, boundary_polygon):
    """Determines which elements (nodes, faces, and edges) are outside of the boundary polygon."""
    prepared_polygon = prep(boundary_polygon)

    # Nodes
    outside_nodes = np.where(~np.array([prepared_polygon.contains(Point(node)) for node in mesh.node_xy]))[0]

    # Edges
    edge_midpoints = mesh.node_xy[mesh.edge_index].mean(axis=0)
    outside_edges = np.where(~np.array([prepared_polygon.contains(Point(edge)) for edge in edge_midpoints]))[0]

    outside_edges_ids = np.unique(np.where(np.isin(mesh.edge_index, outside_nodes))[1])
    outside_edges = np.unique(np.concatenate((outside_edges, outside_edges_ids)))
    
    # Faces
    outside_faces = np.where(~np.array([prepared_polygon.contains(Point(face)) for face in mesh.face_xy]))[0]

    outside_faces_nodes_ids = np.unique(np.where(np.isin(get_face_nodes_matrix(mesh), outside_nodes))[0])
    outside_faces = np.unique(np.concatenate((outside_faces, outside_faces_nodes_ids)))

    # Dual edges
    dual_edge_midpoints = mesh.face_xy[mesh.dual_edge_index].mean(axis=0)
    outside_dual_edges = np.where(~np.array([prepared_polygon.contains(Point(edge)) for edge in dual_edge_midpoints]))[0]

    outside_dual_edges_ids = np.unique(np.where(np.isin(mesh.dual_edge_index, outside_faces))[1])
    outside_dual_edges = np.unique(np.concatenate((outside_dual_edges, outside_dual_edges_ids)))

    return outside_edges, outside_nodes, outside_dual_edges, outside_faces

def remove_elements_outside_boundary(mesh, boundary_polygon):
    """
    Masks the elements of the mesh that are outside the specified boundaries.

    Args:
        mesh (Mesh): The mesh object to be modified.
        boundary_polygon (shapely.geometry.Polygon): The boundary polygon.

    Returns:
        Mesh: The modified mesh object with masked elements.
    """
    # use a small buffer to exclude elements on the boundary
    boundary_polygon_buffer = boundary_polygon.buffer(20)

    outside_edges, outside_nodes, outside_dual_edges, outside_faces = get_outside_elements(mesh, boundary_polygon_buffer)

    new_mesh = copy(mesh)

    # Mask the edges
    mask_edges = np.ones(mesh.edge_index.shape[1], dtype=bool)
    mask_edges[outside_edges] = False
    new_mesh.edge_index = mesh.edge_index[:, mask_edges]
    new_mesh.edge_type = mesh.edge_type[mask_edges]

    node_id_adapter = np.zeros_like(mesh.node_x, dtype=int)
    increment = 0
    for i in range(len(mesh.node_x)):
        if increment < len(outside_nodes) and i == outside_nodes[increment]:
            increment += 1
        node_id_adapter[i] = increment
    new_mesh.edge_index -= node_id_adapter[new_mesh.edge_index]
    # if hasattr(mesh, 'boundary_edges'):
    #     new_mesh.boundary_edges = get_boundary_edges(new_mesh)
    new_mesh.boundary_node_xy = np.array(boundary_polygon.exterior.xy).T

    # Mask the nodes
    mask_nodes = np.ones(mesh.node_xy.shape[0], dtype=bool)
    mask_nodes[outside_nodes] = False
    new_mesh.node_x = mesh.node_x[mask_nodes]
    new_mesh.node_y = mesh.node_y[mask_nodes]

    # Mask the dual edges
    mask_dual_edges = np.ones(mesh.dual_edge_index.shape[1], dtype=bool)
    mask_dual_edges[outside_dual_edges] = False
    new_mesh.dual_edge_index = mesh.dual_edge_index[:, mask_dual_edges]

    face_id_adapter = np.zeros_like(mesh.face_x, dtype=int)
    increment = 0
    for i in range(len(mesh.face_x)):
        if increment < len(outside_faces) and i == outside_faces[increment]:
            increment += 1
        face_id_adapter[i] = increment
    new_mesh.dual_edge_index -= face_id_adapter[new_mesh.dual_edge_index]

    # Mask the faces
    mask_faces = np.ones(mesh.face_xy.shape[0], dtype=bool)
    mask_faces[outside_faces] = False
    new_mesh.face_x = mesh.face_x[mask_faces]
    new_mesh.face_y = mesh.face_y[mask_faces]
    face_nodes = get_face_nodes_matrix(mesh)[mask_faces].flatten()
    face_nodes = face_nodes[~np.isnan(face_nodes)].astype(int)
    new_mesh.face_nodes = face_nodes - node_id_adapter[face_nodes]
    new_mesh.nodes_per_face = mesh.nodes_per_face[mask_faces]

    # Update derived attributes
    new_mesh._get_derived_attributes()

    return new_mesh

def save_mesh(mesh, mesh_file):
    '''Saves mesh as NETCDF file

    Args:    
        mesh (meshkernel.py_structures.Mesh2d): mesh object
        mesh_file (str): path to the output file
    '''
    from netCDF4 import Dataset
    
    # remove file if it already exists
    if os.path.exists(mesh_file): os.remove(mesh_file)

    # save mesh to netcdf file using D-HYDRO format
    with Dataset(mesh_file, mode='w', format='NETCDF4') as ncfile:
        ncfile.createDimension('nNetNode', mesh.node_x.shape[0])
        ncfile.createDimension('nNetLink', mesh.edge_nodes.shape[0]//2)
        ncfile.createDimension('nNetLinkPts', 2)

        ncfile.createVariable('projected_coordinate_system', 'int32', ())
        NetNode_x = ncfile.createVariable('NetNode_x', 'f8', ('nNetNode',))
        NetNode_y = ncfile.createVariable('NetNode_y', 'f8', ('nNetNode',))
        NetNode_z = ncfile.createVariable('NetNode_z', 'f8', ('nNetNode',))
        NetLink = ncfile.createVariable('NetLink', 'int32', ('nNetLink', 'nNetLinkPts'))
        NetLinkType = ncfile.createVariable('NetLinkType', 'int32', ('nNetLink'))

        NetNode_x.units = "m"
        NetNode_x.long_name = "x-coordinate"
        NetNode_y.units = "m"
        NetNode_y.long_name = "y-coordinate"
        NetNode_x.coordinates = "NetNode_x NetNode_y"
        NetNode_y.coordinates = "NetNode_x NetNode_y"
        NetNode_z.coordinates = "NetNode_x NetNode_y"

        NetNode_x[:] = mesh.node_x
        NetNode_y[:] = mesh.node_y
        NetNode_z[:] = np.zeros_like(mesh.node_y)
        NetLink[:] = mesh.edge_nodes.reshape(-1,2)+1
        NetLinkType[:] = np.ones(mesh.edge_nodes.shape[0]//2, dtype=np.int32)-1

    return None

def get_polygon_area(x, y):
    '''Apply shoelace algorithm to evaluate area defined by sequence of points (x,y)'''
    assert x.shape == y.shape, f"Input x and y have incompatible dimensions \n\
                                x: {x.shape}, y: {y.shape}"
    if x.ndim == 1:
        area = 0.5*np.abs(np.dot(x,np.roll(y,1,axis=-1))
            -np.dot(y,np.roll(x,1,axis=-1)))
    elif x.ndim == 2:
        area = 0.5*np.abs(np.multiply(x,np.roll(y,1,axis=-1)).sum(1)
            -np.multiply(y,np.roll(x,1,axis=-1)).sum(1))
    else:
        raise ValueError(f"Input x and y have incorrect dimension ({x.shape})")
    return area

def get_mesh_face_areas(mesh):
    """
    Returns the area of each face in the mesh

    Args:
        mesh: Mesh object

    Returns:
        face_areas: np.array, shape (num_faces,)
    """
    node_position = 0
    face_areas = []
    for num_nodes in mesh.nodes_per_face:
        face_node = mesh.face_nodes[node_position : (node_position + num_nodes)]
        face_nodes_x = mesh.node_x[face_node]
        face_nodes_y = mesh.node_y[face_node]
        node_position += num_nodes
        face_area = get_polygon_area(face_nodes_x, face_nodes_y)
        # if face_area == 0: raise ValueError(f"Face {face_node} has area equal to zero")
        face_areas.append(face_area)

    return np.array(face_areas)

def get_polygon_perimeter(x, y):
    """
    Returns the perimeter of a polygon defined by the points x and y

    Args:
        x: np.array, shape (num_points,)
        y: np.array, shape (num_points,)

    Returns:
        perimeter: float
    """
    dx = np.diff(np.append(x, x[0]))
    dy = np.diff(np.append(y, y[0]))
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))

    return perimeter

def get_mesh_face_perimeters(mesh):
    """
    Returns the perimeter of each face in the mesh

    Args:
        mesh: Mesh object

    Returns:
        face_perimeters: np.array, shape (num_faces,)
    """
    node_position = 0
    face_perimeters = []
    for num_nodes in mesh.nodes_per_face:
        face_node = mesh.face_nodes[node_position : (node_position + num_nodes)]
        face_nodes_x = mesh.node_x[face_node]
        face_nodes_y = mesh.node_y[face_node]
        node_position += num_nodes
        face_perimeter = get_polygon_perimeter(face_nodes_x, face_nodes_y)
        face_perimeters.append(face_perimeter)

    return np.array(face_perimeters)

def get_shape_factor(perimeter, area):
    return (perimeter**2 / (4 * np.pi * area))**0.5


def _where_undirected_edge_in_edge_list(edge, edge_list):
    """Return indices where an edge appears in an undirected edge list.

    Args:
        edge (np.ndarray): Source and destination node ids, shape (2,).
        edge_list (np.ndarray): List of edges, shape (E, 2).

    Returns:
        np.ndarray: Matching indices in edge_list.
    """
    return np.where((edge_list == edge).all(1) | (edge_list[:, ::-1] == edge).all(1))[0]

def where_edge_in_edge_list(edge, edge_list):
    """Returns the index of the edge in the edge list
    
    Args:
        edge (np.array): source and destination nodes (shape: [2])
        edge_list (np.array): list of edges (shape: [E, 2])

    Returns:
        np.array: index of the edge in the edge list
    """
    return _where_undirected_edge_in_edge_list(edge, edge_list)

def is_edge_in_edge_list(edge, edge_list):
    """Returns True if the edge is in the edge list
    
    Args:
        edge (np.array): source and destination nodes (shape: [2])
        edge_list (np.array): list of edges (shape: [E, 2])

    Returns:
        bool: True if the edge is in the edge list
    """
    return len(_where_undirected_edge_in_edge_list(edge, edge_list)) > 0


def _mirror_points_with_normals(points_xy, edge_outward_normals, symmetry_points):
    """Mirror points with respect to boundary symmetry points using outward normals.

    Args:
        points_xy (np.ndarray): Point coordinates to mirror, shape (N, 2).
        edge_outward_normals (np.ndarray): Edge outward normals, shape (N, 2).
        symmetry_points (np.ndarray): Symmetry points, shape (N, 2).

    Returns:
        np.ndarray: Mirrored coordinates, shape (N, 2).
    """
    point_to_symmetry_distance = np.linalg.norm((points_xy - symmetry_points), axis=1, keepdims=True)

    # normal_adapter is used to adapt the normal to the correct axis
    normal_adapter = np.int32([1, 0])
    mirrored_xy = symmetry_points - edge_outward_normals[:, normal_adapter] * point_to_symmetry_distance

    point_to_mirrored_distance = np.linalg.norm((points_xy - mirrored_xy), axis=1, keepdims=True)

    # The mirrored points are not mirrored correctly so we flip the normal
    if (point_to_mirrored_distance < point_to_symmetry_distance).any():
        invalid_normal_mask = (point_to_mirrored_distance < point_to_symmetry_distance).squeeze()
        edge_outward_normals[invalid_normal_mask] *= -1
        mirrored_xy = symmetry_points - edge_outward_normals[:, normal_adapter] * point_to_symmetry_distance

    return mirrored_xy

def filter_unique_edges(edge_index):
    """Filters the unique edges from the edge index
    
    Args:
        edge_index (np.array): Array of shape (n_edges, 2) containing the edge indices
    
    Returns:
        np.array: Array of shape (n_edges, 2) containing the unique edge indices
    """
    unique_edges = set()
    filtered_edges = []

    for edge in edge_index:
        edge_tuple = tuple(edge)
        reverse_edge_tuple = tuple(edge[::-1])

        if reverse_edge_tuple not in unique_edges:
            unique_edges.add(edge_tuple)
            filtered_edges.append(edge)
    
    return np.array(filtered_edges)

def filter_unique_nodes(nodes):
    """Filters the unique nodes from the node index
    
    Args:
        nodes (np.array): Array of shape (n_nodes) containing the node indices
    
    Returns:
        np.array: Array of shape (n_nodes) containing the unique node indices
    """
    unique_nodes = set()
    filtered_nodes = []

    for node in nodes:
        node_tuple = tuple(node) if node.ndim > 0 else node
        if node_tuple not in unique_nodes:
            unique_nodes.add(node_tuple)
            filtered_nodes.append(node)
    
    return np.array(filtered_nodes)

def remove_duplicate_edges(edge_index):
    """Removes duplicate edges from the list of edges
    
    Args:
        edges (np.array): Array of shape (n_edges, 2) containing the edge indices
    
    Returns:
        np.array: Array of shape (n_edges, 2) containing the unique edge indices
    """
    seen = set()
    filtered_edges = []
    
    for edge in edge_index:
        edge_tuple = tuple(edge)
        if edge_tuple not in seen:
            seen.add(edge_tuple)
            filtered_edges.append(edge)

    return np.array(filtered_edges)

class Mesh(object):
    def __init__(self):
        '''Mixed-elements mesh base object
        
        Attributes:
        node_x (np.ndarray): x-coordinates of the nodes
        node_y (np.ndarray): y-coordinates of the nodes
        node_xy (np.ndarray): Array of shape (n_nodes, 2) containing x and y coordinates of each node
        face_x (np.ndarray): x-coordinates of the faces
        face_y (np.ndarray): y-coordinates of the faces
        face_xy (np.ndarray): Array of shape (n_faces, 2) containing x and y coordinates of each face
        edge_index (np.ndarray): Array of shape (2, n_edges) containing the edge indices
        dual_edge_index (np.ndarray): Array of shape (2, n_edges) containing the dual edge indices
        face_nodes (np.ndarray): Array of shape (n_faces, max_nodes_per_face) containing the nodes that define each face
        nodes_per_face (np.ndarray): Array of shape (n_faces,) containing the number of nodes per face
        boundary_node_xy (np.ndarray): Array of shape (n_boundary_nodes, 2) containing the boundary node coordinates
        boundary_edges (np.ndarray): Array of shape (2, n_boundary_edges) containing the boundary edge indices
        edge_type (np.ndarray): Array of shape (n_edges,) containing the edge types

        Methods:
        plot_boundary: plot the boundary of the mesh
        '''
        self.added_ghost_cells = False
        self.node_xy = np.array([])
        self.face_xy = np.array([])
        self.edge_index = np.array([[]])
        self.dual_edge_index = np.array([[]])

    def _import_from_map_netcdf(self, nc_file, import_BC=True):
        """Import mesh from map netcdf file (the output of DHYDRO)

        -------
        Adds the following attributes:
        edge_index_BC: np.array, shape (num_edges_BC, 2)
            index of the edges that have boundary conditions
        face_BC: np.array, shape (num_faces_BC,)
            index of the faces that have boundary conditions
        edge_BC: np.array, shape (num_edges_BC,)
            index of the edges that have boundary conditions
        extra_face_BC: np.array, shape (num_extra_faces_BC,)
            index of the faces that are in the boundary but with no boundary conditions
        
        -------
        Updates the following attributes:
        dual_edge_index: removes the edges that are in the boundary but with no boundary conditions
        """
        nc_dataset = xr.open_dataset(nc_file)
        self.node_x = nc_dataset['mesh2d_node_x'].data
        self.node_y = nc_dataset['mesh2d_node_y'].data

        self.face_x = nc_dataset['mesh2d_face_x'].data
        self.face_y = nc_dataset['mesh2d_face_y'].data

        self.edge_index = nc_dataset['mesh2d_edge_nodes'].data.T - 1
        self.edge_type = nc_dataset['mesh2d_edge_type'].data # 1:normal edges, 2:BC_edge, 3:other boundary edges, 4:ghost cells

        self.DEM = nc_dataset['mesh2d_flowelem_bl'].data if hasattr(nc_dataset, 'mesh2d_flowelem_bl') else None
        self.roughness = nc_dataset['mesh2d_roughness'].data if hasattr(nc_dataset, 'mesh2d_roughness') else None

        if np.isnan(nc_dataset.mesh2d_edge_faces[:,1].values).sum() > 0:
            self.face_bnd_mask = np.isnan(nc_dataset.mesh2d_edge_faces[:,1].values)
            self.face_bnd = nc_dataset.mesh2d_edge_faces.values[self.face_bnd_mask,0]
            self.dual_edge_index = nc_dataset.mesh2d_edge_faces.values[~self.face_bnd_mask].T.astype(int) - 1
        else:
            self.dual_edge_index = nc_dataset['mesh2d_edge_faces'].data.T.astype(int) - 1

            face_bnd_mask = self.dual_edge_index[0,:] == -1
            self.face_BC = self.dual_edge_index[1,face_bnd_mask]

            extra_face_bnd_mask = self.dual_edge_index[1,:] == -1
            self.face_bnd = self.dual_edge_index[0,extra_face_bnd_mask]

            total_face_bnd_mask = extra_face_bnd_mask | face_bnd_mask
            self.dual_edge_index = self.dual_edge_index[:,~total_face_bnd_mask]

        self.face_nodes = nc_dataset['mesh2d_face_nodes'] - 1
        # mixed mesh
        if isinstance(self.face_nodes.to_masked_array().mask, np.ndarray):
            self.nodes_per_face = (~self.face_nodes.to_masked_array().mask).sum(1).astype(int)
            self.face_nodes = self.face_nodes.data[~self.face_nodes.to_masked_array().mask].astype(int)
        # triangular or quadrilateral mesh
        else:
            self.nodes_per_face = np.ones_like(self.face_nodes).sum(1).data.astype(int)
            self.face_nodes = self.face_nodes.reshape(-1).data.astype(int)

        if import_BC:
            self.edge_index_BC = self.edge_index[:,self.edge_type == 2].T
            self.boundary_edges = self.edge_index[:,self.edge_type > 1].T
            self.edge_BC = np.stack([np.where((edge==self.edge_index.T).sum(1) == 2) for edge in self.edge_index_BC]).reshape(-1)

            self.num_bnd_faces = len(self.face_bnd)

        self.dual_edge_index = to_undirected(torch.LongTensor(self.dual_edge_index)).numpy() #convert to undirected graph
        self._get_derived_attributes()

    def _import_from_meshkernel(self, meshkernel_mesh):
        """Import mesh from meshkernel Mesh2d object
        
        Example to create a Mesh2d object:
        mk = MeshKernel()
        mk.mesh2d_make_rectilinear_mesh(0, 0, 100, 100, 10, 10)
        
        mesh = Mesh()
        mesh._import_from_meshkernel(mk)
        """
        assert isinstance(meshkernel_mesh, MeshKernel), 'Input mesh must be a MeshKernel object from meshkernel'
        mesh = meshkernel_mesh.mesh2d_get()

        self.node_x = mesh.node_x
        self.node_y = mesh.node_y
        self.node_xy = np.stack((self.node_x, self.node_y),-1)
        
        self.face_x = mesh.face_x
        self.face_y = mesh.face_y

        self.edge_index = mesh.edge_nodes.reshape(-1,2).T
        self.dual_edge_index = mesh.edge_faces.reshape(-1,2).T if mesh.edge_faces.shape[0] % 2 == 0 else mesh.edge_faces[:-1].reshape(-1,2).T
        
        extra_face_bnd_mask = self.dual_edge_index[1,:] == -1
        self.face_bnd = self.dual_edge_index[0,extra_face_bnd_mask]
        self.dual_edge_index = self.dual_edge_index[:,~extra_face_bnd_mask]
        self.dual_edge_index = to_undirected(torch.LongTensor(self.dual_edge_index)).numpy() #convert to undirected graph

        self.face_nodes = mesh.face_nodes
        self.nodes_per_face = mesh.nodes_per_face

        boundary_polygon = meshkernel_mesh.mesh2d_get_mesh_boundaries_as_polygons()
        self.boundary_node_xy = np.stack((boundary_polygon.x_coordinates, boundary_polygon.y_coordinates),-1)
        self.boundary_node_xy = self.boundary_node_xy[(self.boundary_node_xy != -999).any(1)]
        boundary_nodes_ids = [np.where((self.node_xy == node).all(1))[0][0] for node in self.boundary_node_xy[:-1]]
        boundary_edge = np.stack([boundary_nodes_ids[i:i+2] for i in range(len(boundary_nodes_ids)-1) if len(boundary_nodes_ids[i:i+2]) == 2])
        # boundary_nodes_ids = np.array([i for i in range(len(self.boundary_nodes)-1)])
        # boundary_edge = np.array([[i, (i+1)%len(boundary_nodes_ids)] for i in boundary_nodes_ids]).T
        boundary_edge_id = np.array([where_edge_in_edge_list(edge, self.edge_index.T) 
                                     for edge in boundary_edge if is_edge_in_edge_list(edge, self.edge_index.T)]).squeeze()
        self.edge_type = np.ones(self.edge_index.shape[1])
        self.edge_type[boundary_edge_id] = 3
        self.boundary_edges = self.edge_index[:,self.edge_type > 1].T
        self.boundary_nodes = np.unique(self.boundary_edges.flatten())
        self._get_derived_attributes()

    def _import_from_gmsh(self, mesh):
        """
        Import mesh from a meshio GMSH mesh object and populate Mesh attributes.

        Args:
            mesh: meshio.Mesh object (from a GMSH file)
        """
        # Node coordinates
        self.node_x = mesh.points[:, 0]
        self.node_y = mesh.points[:, 1]
        self.node_xy = mesh.points[:, :2]

        # Face nodes (mixed elements: triangles and quads)
        tri_cells = []
        quad_cells = []
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                tri_cells.extend(cell_block.data.tolist())
            elif cell_block.type == "quad":
                quad_cells.extend(cell_block.data.tolist())

        tri_cells = np.array(tri_cells, dtype=int) if tri_cells else np.empty((0, 3), dtype=int)
        quad_cells = np.array(quad_cells, dtype=int) if quad_cells else np.empty((0, 4), dtype=int)

        # Pad triangles to 4 nodes for mixed mesh
        if len(tri_cells) > 0 and len(quad_cells) > 0:
            tri_cells_padded = np.pad(tri_cells, ((0, 0), (0, 1)), constant_values=-1)
            elements = np.vstack([tri_cells_padded, quad_cells])
        elif len(tri_cells) > 0:
            elements = np.pad(tri_cells, ((0, 0), (0, 1)), constant_values=-1)
        elif len(quad_cells) > 0:
            elements = quad_cells
        else:
            elements = np.empty((0, 4), dtype=int)

        # Remove -1 padding for face_nodes and build nodes_per_face
        self.nodes_per_face = np.array([np.sum(e != -1) for e in elements])
        self.face_nodes = np.concatenate([e[e != -1] for e in elements])

        # Face centroids
        face_centroids = []
        for e in elements:
            valid_nodes = e[e != -1]
            coords = self.node_xy[valid_nodes]
            face_centroids.append(coords.mean(axis=0))
        self.face_xy = np.array(face_centroids)
        self.face_x = self.face_xy[:, 0]
        self.face_y = self.face_xy[:, 1]

        # Edge index
        edge_set = set()
        for e in elements:
            valid_nodes = e[e != -1]
            n = len(valid_nodes)
            for i in range(n):
                a, b = valid_nodes[i], valid_nodes[(i + 1) % n]
                edge_set.add(tuple(sorted((a, b))))
        edge_list = np.array(list(edge_set), dtype=int).T
        self.edge_index = edge_list

        # Edge type (boundary = 3, interior = 1)
        from collections import defaultdict
        edge_to_faces = defaultdict(list)
        for face_idx, e in enumerate(elements):
            valid_nodes = e[e != -1]
            n = len(valid_nodes)
            for i in range(n):
                a, b = valid_nodes[i], valid_nodes[(i + 1) % n]
                edge = tuple(sorted((a, b)))
                edge_to_faces[edge].append(face_idx)
        self.edge_type = np.array([3 if len(edge_to_faces[tuple(edge)]) == 1 else 1 for edge in self.edge_index.T])

        # Dual edge index (face adjacency)
        face_adj = defaultdict(set)
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                a, b = faces
                face_adj[a].add(b)
                face_adj[b].add(a)
        dual_edges = []
        for a, neighbors in face_adj.items():
            for b in neighbors:
                if a < b:
                    dual_edges.append([a, b])
        self.dual_edge_index = np.array(dual_edges).T if dual_edges else np.empty((2, 0), dtype=int)

        # Boundary nodes and edges
        boundary_edges = [edge for edge, faces in edge_to_faces.items() if len(faces) == 1]
        self.boundary_edges = np.array(boundary_edges, dtype=int)
        self.boundary_nodes = np.unique(self.boundary_edges.flatten())
        self.boundary_node_xy = self.node_xy[self.boundary_nodes]

        self._get_derived_attributes()

    def _import_from_Triangle(self, mesh):
        """Import mesh from triangle mesh object.
        The triangulation must have -en flags activated
        
        Example:
        mesh_options = {"vertices": points}
        mesh = tr.triangulate(mesh_options, 'en')
        """
        self.edge_index = mesh['edges'].T
        self.edge_type = mesh['edge_markers'].squeeze() * 2 + 1
        self.node_x = mesh['vertices'][:, 0]
        self.node_y = mesh['vertices'][:, 1]
        self.face_nodes = mesh['triangles'].ravel()

        self.face_x = self.node_x[mesh['triangles']]  # shape [F, 3]
        self.face_y = self.node_y[mesh['triangles']]  # shape [F, 3]

        self.nodes_per_face = np.ones(len(self.face_x), dtype=int) * 3  # times 3 because it's a triangle

        dual_edge_index = np.array([[[face, neighbour]
                                          for neighbour in mesh['neighbors'][face]]
                                         for face in range(len(self.face_x))]
                                        ).reshape(-1, 2).T  # shape [2, E_d]
        self.dual_edge_index = to_undirected(torch.LongTensor(dual_edge_index)).numpy() #convert to undirected graph
        self.boundary_node_xy = mesh['vertex_markers']

    def _import_from_geodataframe(self, gdf):
        '''Import mesh from a geodataframe object that contains polygons

        Example:
        gdf = gpd.read_file('polygons.gpkg')
        mesh = Mesh()
        mesh._import_from_geodataframe(gdf)
        '''
        self.gdf = gdf

        G = get_graph_from_geodataframe(gdf)
        data = from_networkx(G)

        # nodes
        polygons_coords = [np.array(poly.exterior.xy)[:,:-1] for poly in gdf.geometry.values] # remove last point because it is the same as the first
        self.node_xy = filter_unique_nodes(np.concatenate(polygons_coords, axis=1).T)

        self.node_x = self.node_xy[:,0]
        self.node_y = self.node_xy[:,1]
        self.nodes_per_face = np.array([len(poly.T) for poly in polygons_coords])

        self.boundary_node_xy = filter_unique_nodes(np.array(gdf.union_all().boundary.xy).T)
        self.boundary_nodes = np.where(np.isin(self.node_xy, self.boundary_node_xy).all(1))[0]

        # faces
        self.face_xy = data.pos.numpy()
        self.face_x = self.face_xy[:,0]
        self.face_y = self.face_xy[:,1]
        self.face_nodes = np.concat([np.unique(np.where(np.isin(self.node_xy, poly).all(1))[0]) for poly in polygons_coords])
        self.num_faces = self.face_xy.shape[0]

        # dual edges
        self.dual_edge_index = data.edge_index.numpy()
        
        self._get_derived_attributes()

    def _get_derived_attributes(self):
        """Calculate derived attributes from the mesh
        
        Adds the following attributes:
        num_nodes: int, number of nodes
        num_edges: int, number of edges
        num_faces: int, number of faces
        boundary_nodes: np.ndarray, shape (n_boundary_nodes,)
            index of the boundary nodes 
        boundary_node_xy: np.ndarray, shape (n_boundary_nodes, 2)
            x and y coordinates of the boundary nodes
        edge_relative_distance: np.ndarray, shape (n_edges, 2)
            relative distance of the edges
        edge_length: np.ndarray, shape (n_edges,)
            length of the edges
        edge_outward_normal: np.ndarray, shape (n_edges, 2)
            outward normal of the edges
        face_relative_distance: np.ndarray, shape (n_dual_edges, 2)
            relative distance of the dual edges
        dual_edge_length: np.ndarray, shape (n_dual_edges,)
            length of the dual edges
        face_area: np.ndarray, shape (n_faces,)
            area of the faces

        Raises:
        ValueError: if a face area is zero
        """
        # Nodes
        self.node_xy = np.stack((self.node_x, self.node_y),-1)
        self.num_nodes = self.node_x.shape[0]

        if not hasattr(self, 'gdf'):
            self.boundary_nodes = np.array(list(set(self.edge_index.T[(self.edge_type > 1) & (self.edge_type < 4)].flatten())))
            self.boundary_node_xy = self.node_xy[self.boundary_nodes]
            self.boundary_node_xy = order_boundary_nodes(self.boundary_node_xy)
            
            # Edges            
            self.edge_relative_distance = self.node_xy[self.edge_index[1,:]] - self.node_xy[self.edge_index[0,:]]
            self.edge_length = np.linalg.norm(self.edge_relative_distance, axis=1)

            self.edge_outward_normal = self.edge_relative_distance/self.edge_length[:,None]
            self.edge_outward_normal[:,1] = -self.edge_outward_normal[:,1]
            self.num_edges = self.edge_index.shape[1]

            self.face_area = get_mesh_face_areas(self)
            self.perimeter = get_mesh_face_perimeters(self)
        else:
            self.face_area = self.gdf.area.values
            self.perimeter = self.gdf.geometry.length.values

        # Faces
        self.face_xy = np.stack((self.face_x, self.face_y),-1)
        self.face_relative_distance = self.face_xy[self.dual_edge_index[1,:]] - self.face_xy[self.dual_edge_index[0,:]]
        self.dual_edge_length = np.linalg.norm(self.face_relative_distance, axis=1)
        self.num_faces = self.face_x.shape[0]

    def _import_map(self, map_file, method='nearest', delimiter=' '):
        """Imports map file and interpolate it on the mesh
        
        Args:
            map_file: str, path-like
                path to map file
            method: str, optional
                interpolation method 
            delimiter: str, optional
                delimiter used in the map file
        """
        try:
            if map_file.endswith('.xyz'):
                xyz_data = np.loadtxt(map_file, delimiter=delimiter)
                assert xyz_data.shape[1] == 3, "map file must have three columns: x, y, z"

            elif map_file.endswith('.tif'):
                # load raster
                with rasterio.open(map_file) as src:
                    data = src.read(1)  # Read the first band
                    # x coordinates of the center of each pixel
                    x_coords = np.arange(src.transform[2], src.transform[2] + src.width * src.transform[0], src.transform[0])
                    # y coordinates of the center of each pixel
                    y_coords = np.arange(src.transform[5], src.transform[5] + src.height * src.transform[4], src.transform[4])
                    xy_coords = np.meshgrid(x_coords, y_coords, indexing='ij')

                    # convert the xy coordinates and the corresponding values to a xyz file
                    xyz_data = np.column_stack((xy_coords[0].ravel(), xy_coords[1].ravel(), data.T.ravel()))
                    # remove NaN values
                    xyz_data = xyz_data[~np.isnan(xyz_data).any(axis=1)]

            mesh_map = interpolate_variable(self.face_xy, xyz_data[:,:2], xyz_data[:,2], method=method)
        except FileNotFoundError:
            print(f"Could not find the file {map_file}. Setting its values to zeros.")
            mesh_map = np.zeros_like(self.face_area)

        return mesh_map

    def _import_DEM(self, DEM_file, method='nearest', delimiter=' '):
        """Import DEM file and interpolate it on the mesh
    
        Args:
            DEM_file: str, path-like
                path to DEM file. It must be a file with three columns: x, y, z
            method: str, optional
                interpolation method 
            delimiter: str, optional
                delimiter used in the DEM file
        """
        self.DEM = self._import_map(DEM_file, method, delimiter)

    def _import_roughness(self, roughness_file, method='nearest', delimiter=' '):
        """Import roughness file and interpolate it on the mesh
    
        Args:
            roughness_file: str, path-like
                path to roughness file. It must be a file with three columns: x, y, z
            method: str, optional
                interpolation method 
            delimiter: str, optional
                delimiter used in the roughness file
        """
        self.roughness = self._import_map(roughness_file, method, delimiter)

    def plot_boundary(self, ax=None, **plt_kwargs):
        '''Plot the boundary of the mesh'''
        ax = ax or plt.gca()

        if hasattr(self, 'gdf'):
            gpd.GeoSeries(self.gdf.union_all().exterior).plot(ax=ax, **plt_kwargs)
        else:
            # [ax.plot(self.node_xy[edge][:,0], self.node_xy[edge][:,1], **plt_kwargs) for edge in self.boundary_edges];
            # plt.plot(self.boundary_node_xy[:,0], self.boundary_node_xy[:,1], **plt_kwargs)
            print('Not implemented yet. Try it with the gdf object')

        return ax
    
    def _export_to_netcdf(self, nc_file):
        """
        Export mesh properties to a NetCDF (.nc) file.

        Args:
            nc_file: str, path to the output NetCDF file
        """
        import netCDF4 as nc

        with nc.Dataset(nc_file, 'w', format='NETCDF4') as ds:
            # Dimensions
            ds.createDimension('nodes', self.node_x.shape[0])
            ds.createDimension('faces', self.face_x.shape[0])
            ds.createDimension('edges', self.edge_index.shape[1])
            ds.createDimension('dual_edges', self.dual_edge_index.shape[1])
            ds.createDimension('max_nodes_per_face', self.nodes_per_face.max())
            ds.createDimension('Two', 2)

            # Variables
            node_x = ds.createVariable('mesh2d_node_x', 'f8', ('nodes',))
            node_y = ds.createVariable('mesh2d_node_y', 'f8', ('nodes',))
            face_x = ds.createVariable('mesh2d_face_x', 'f8', ('faces',))
            face_y = ds.createVariable('mesh2d_face_y', 'f8', ('faces',))
            edge_nodes = ds.createVariable('mesh2d_edge_nodes', 'i4', ('edges', 'Two'))
            edge_type = ds.createVariable('mesh2d_edge_type', 'i4', ('edges',))
            edge_faces = ds.createVariable('mesh2d_edge_faces', 'i4', ('dual_edges', 'Two'))
            face_nodes = ds.createVariable('mesh2d_face_nodes', 'i4', ('faces', 'max_nodes_per_face'))
            nodes_per_face = ds.createVariable('nodes_per_face', 'i4', ('faces',))

            # Data assignment
            node_x[:] = self.node_x
            node_y[:] = self.node_y
            face_x[:] = self.face_x
            face_y[:] = self.face_y
            edge_nodes[:, :] = self.edge_index.T + 1  # convert to 1-based index
            edge_type[:] = self.edge_type
            edge_faces[:, :] = self.dual_edge_index.T + 1  # convert to 1-based index

            # Handle face_nodes and nodes_per_face
            face_nodes[:, :] = -1  # Use -1 as a fill value for unused nodes
            idx = 0
            for i, n in enumerate(self.nodes_per_face):
                face_nodes[i, :n] = self.face_nodes[idx:idx+n] + 1
                idx += n
            nodes_per_face[:] = self.nodes_per_face

    def __repr__(self) -> str:
        return 'Mesh object with {} nodes, {} edges, {} faces, and {} dual edges'.format(
            self.node_xy.shape[0], self.edge_index.shape[1], self.face_xy.shape[0], self.dual_edge_index.shape[1])    

class MultiscaleMesh(object):
    """Mesh class for multiscale meshes
    
    Attributes:
        num_meshes (int): number of meshes
        meshes (List[Mesh]): list of Mesh objects
        face_ptr (np.ndarray): partition of the faces of each mesh
            
    Methods:
        stack_meshes: stack the meshes to create a multiscale mesh
        get_partitioning: get the partitioning of the multiscale mesh
        get_multiscale_BC: get the boundary conditions of the multiscale mesh
        get_intra_edges: get the dual edges across each multiscale level
        remove_intra_edges: remove the intra mesh dual edges
    """
    def __init__(self):
        super().__init__()
        self.num_meshes = 0

    def stack_meshes(self, meshes):
        """Stack the meshes to create a multiscale mesh

        Args:
            meshes (List[Mesh]): list of Mesh objects
        """
        self.num_meshes = len(meshes)
        self.meshes = meshes

        # stack features
        self.face_xy = np.concatenate([mesh.face_xy for mesh in meshes])
        self.nodes_per_face = np.concatenate([mesh.nodes_per_face for mesh in meshes])
        self.face_area = np.concatenate([mesh.face_area for mesh in meshes])
        self.num_faces = self.face_xy.shape[0]

        dual_edge_index = [meshes[0].dual_edge_index]
        for i, mesh in enumerate(meshes[1:]):
            dual_edge_index.append(mesh.dual_edge_index + dual_edge_index[i].max() + 1)
            
        self.dual_edge_index = np.concatenate(dual_edge_index, 1)
        self.edge_weir = np.concat([mesh.edge_weir for mesh in meshes]) if hasattr(meshes[0], 'edge_weir') else None
        self.face_relative_distance = self.face_xy[self.dual_edge_index[1,:]] - self.face_xy[self.dual_edge_index[0,:]]
        self.dual_edge_length = np.linalg.norm(self.face_relative_distance, axis=1)
        
        # partition and compose the meshes
        self.get_partitioning(meshes)
        self.get_intra_edges(meshes)

    def get_partitioning(self, meshes):
        """Get the partitioning of the meshes in the multiscale mesh
        
        Adds:
            face_ptr: index of first face of each mesh
            dual_edge_ptr: index of first edge of each mesh
        """
        self.face_ptr = np.cumsum([0] + [mesh.face_xy.shape[0] for mesh in meshes])
        self.dual_edge_ptr = np.cumsum([0] + [mesh.dual_edge_index.shape[1] for mesh in meshes])

    def get_intra_edges(self, meshes, add_edges=False):
        """Adds dual edges across each multiscale level
        based on the position of the fine mesh centers in the coarse mesh

        Args:
            meshes (List[Mesh]): list of Mesh objects
            add_edges (bool): if True, adds the intra mesh dual edges to the dual edge index
        
        Adds:
            intra_mesh_dual_edge_index (np.array): dual edge index of the intra mesh edges

        Updates:
            dual_edge_index: adds the intra mesh dual edges
        """
        if meshes[0].num_faces < meshes[1].num_faces:   # coarse to fine
            intra_mesh_dual_edge_index = [connect_coarse_to_fine_mesh(meshes[i], meshes[i+1]) + [self.face_ptr[i], self.face_ptr[i+1]] for i in range(len(meshes)-1)]
        else:   # fine to coarse
            intra_mesh_dual_edge_index = [connect_coarse_to_fine_mesh(meshes[i+1], meshes[i]) + [self.face_ptr[i+1], self.face_ptr[i]] for i in range(len(meshes)-1)]
        self.intra_edge_ptr = np.cumsum([0] + [edge.shape[0] for edge in intra_mesh_dual_edge_index])
        self.intra_mesh_dual_edge_index = np.concatenate(intra_mesh_dual_edge_index).T

        if add_edges:
            self.with_intra_edges = True
            self.dual_edge_index = np.concatenate([self.dual_edge_index, self.intra_mesh_dual_edge_index], 1)

    def remove_intra_edges(self):
        """Removes the intra mesh dual edges"""
        if self.with_intra_edges:
            self.with_intra_edges = False
            if self.added_ghost_cells:
                self.dual_edge_index = np.concatenate((self.dual_edge_index[:,:-self.intra_mesh_dual_edge_index.shape[1]-len(self.face_BC)], 
                                                    self.dual_edge_index_BC), 1)
            else:
                self.dual_edge_index = self.dual_edge_index[:,:-self.intra_mesh_dual_edge_index.shape[1]]
        else:
            print("The mesh does not have intra mesh dual edges. You can add them with add_intra_edges(meshes)")

    def get_multiscale_BC(self, meshes):
        """Get the boundary conditions for the multiscale mesh by 
        stacking the boundary conditions of the meshes
        
        Adds:
            edge_BC: index of boundary edges
            face_BC: index of boundary faces
            edge_index_BC: index of boundary edges in edge_index
        """
        if not all([hasattr(mesh, 'edge_BC') for mesh in meshes]):
            # add ghost cells to the coarse meshes if the fine mesh has them
            if hasattr(meshes[0], 'edge_index_BC'):
                edge_BC_mid = meshes[0].node_xy[meshes[0].edge_index_BC].mean(1)
                meshes = interpolate_BC_location_multiscale(meshes, edge_BC_mid)
                meshes = [add_ghost_cells_mesh(mesh) for mesh in meshes]
            else:
                raise ValueError("The meshes must have boundary conditions")            

        self.edge_BC = np.concatenate([mesh.edge_BC + self.edge_ptr[i] for i, mesh in enumerate(meshes)])
        self.face_BC = np.concatenate([mesh.face_BC + self.face_ptr[i] for i, mesh in enumerate(meshes)])
        self.edge_index_BC = self.edge_index[:,self.edge_type == 2].T

        dual_edge_index_BC = [meshes[0].dual_edge_index_BC]
        for i, mesh in enumerate(meshes[1:]):
            dual_edge_index_BC.append(mesh.dual_edge_index_BC + dual_edge_index_BC[i].max() + 1)
        self.dual_edge_index_BC = np.concatenate(dual_edge_index_BC, 1)

    def correct_BC(self, meshes):
        """Correct the connection of boundary conditions elements of the multiscale mesh"""                
        self.added_ghost_cells = True
        self.face_BC = np.concatenate([mesh.face_BC + self.face_ptr[i] for i, mesh in enumerate(meshes)])
        self.ghost_cells_ids = np.concatenate([mesh.ghost_cells_ids + self.face_ptr[i] for i, mesh in enumerate(meshes)])
        
        # check that BC nodes across between scales are only connected to other BC nodes
        for i in range(self.num_meshes-1):
            # find intra scale edges that are connected to fine ghost cells
            where_ghost_cells_loc = np.concat([np.where(ghost_cell+self.face_ptr[i] == self.intra_mesh_dual_edge_index) for ghost_cell in meshes[i].ghost_cells_ids], axis=1)
            # we are interested only in the edges in the fine scale (axis=1)
            fine_ghost_cells_loc = where_ghost_cells_loc[:,np.where(where_ghost_cells_loc == 1)[1]]
            intra_ghost_cell_neighbour = self.intra_mesh_dual_edge_index[0, fine_ghost_cells_loc[1]]
            
            num_ghost_cell_intra_edges = max(len(meshes[i].ghost_cells_ids), len(meshes[i+1].ghost_cells_ids)) # there should be an edge for each ghost cell in the finest scale (axis=1)
            # assert num_ghost_cell_intra_edges == fine_ghost_cells_loc.shape[1], "The number of ghost cell edges was not correct, which means you would have to change also edge_ptr"

            # if the neighbour is not a ghost cell, then replace it with the correct one
            wrong_edges = np.isin(intra_ghost_cell_neighbour, self.ghost_cells_ids, invert=True)
            
            # Update the wrong edges with the correct ghost cell ids
            if len(meshes[i+1].ghost_cells_ids) == 1:
                self.intra_mesh_dual_edge_index[fine_ghost_cells_loc[0, wrong_edges]-1, fine_ghost_cells_loc[1, wrong_edges]] = meshes[i+1].ghost_cells_ids + self.face_ptr[i+1]
            else:
                self.intra_mesh_dual_edge_index[fine_ghost_cells_loc[0, wrong_edges]-1, fine_ghost_cells_loc[1, wrong_edges]] = meshes[i+1].ghost_cells_ids[0] + self.face_ptr[i+1]
                print("Attention to how the ghost cells are connected in the coarse mesh. I cannot bother right now to fix this for you.")
        
    def __repr__(self) -> str:
        return 'MultiscaleMesh object with {} meshes'.format(self.num_meshes)
    
def rotate_mesh(mesh, angle):
    """Data augmentation: rotate the mesh by a given angle
    
    Args:
        mesh (Mesh): mesh object
        angle (float): angle in degrees
    """
    rotated_mesh = copy(mesh)

    angle = np.deg2rad(angle)

    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    rotated_mesh.node_x, rotated_mesh.node_y = np.dot(rot_matrix, np.array([mesh.node_x, mesh.node_y]))
    rotated_mesh.face_x, rotated_mesh.face_y = np.dot(rot_matrix, np.array([mesh.face_x, mesh.face_y]))
    
    rotated_mesh._get_derived_attributes()

    return rotated_mesh

def get_slopes(coords, DEM, neighborhood_size=200, min_neighbours=5):
    """
    Calculate the slope from points with x, y coordinates and z elevation (DEM).

    Args:
        coords: np.array, shape (n, 2) containing the x and y coordinates of the points
        DEM: np.array, shape (n,) containing the elevation of the points
        neighborhood_size: int, size of the radius graph used determine which points affect the slope
        min_neighbours: int, minimum number of neighbours to calculate the slope

    Returns:
        slope_x: np.array, shape (n,) containing the slope in the x direction
        slope_y: np.array, shape (n,) containing the slope in the y direction
    """
    slope_x = []
    slope_y = []

    radius_graph = radius_neighbors_graph(coords, neighborhood_size, mode='connectivity', include_self=False)
    KNN = kneighbors_graph(coords, min_neighbours, mode='connectivity', include_self=False)

    for row in ((radius_graph.todense() + KNN.todense()) > 0):    
        A = np.column_stack((np.ones((row.sum(), 1)), coords[np.where(row)[1]]))
        b = DEM[np.where(row)[1]]
        coefficients, _, _, _ = lstsq(A, b)

        # The gradient of the plane is the coefficients for x and y
        dz_dx = coefficients[1]
        dz_dy = coefficients[2]

        slope_x.append(dz_dx)
        slope_y.append(dz_dy)

    return np.array(slope_x), np.array(slope_y)

def interpolate_variable(interpolated_points, points, value, method='nearest'):
    '''Interpolate variable at specific interpolated_points contained in 
    
    Args:
        interpolated_points: np.array, shape (n, 2)
            points at which to interpolate data
        points: np.array, shape (m, 2)
            points at which the data is known
        variable: np.array, shape (m,)
            value of a variable for each point in the domain
        method: str
            choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)

    Returns:
        interpolated_variable: np.array, shape (n,) containing the interpolated variable
    '''
    if isinstance(points, dict):
        points = get_coords(points)

    interpolated_variable = griddata(points, value, interpolated_points, method=method)
    
    # interpolate nan values
    mask = np.isnan(interpolated_variable)
    interpolated_variable[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), interpolated_variable[~mask])

    return interpolated_variable

def interpolate_temporal_variable(interpolated_points, points, temporal_value, method='nearest'):
    '''Interpolate temporal variable at specific interpolated_points contained in points
    
    Args:
        interpolated_points: np.array, shape (n, 2)
            points at which to interpolate data
        points: np.array, shape (m, 2)
            points at which the data is known
        variable: np.array, shape (m, T)
            value of a variable for each point in the domain
        method: str
            choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)

    Returns:
        interpolated_time_variable: np.array, shape (n, T) containing the interpolated variable at each time step
    '''
    total_time = temporal_value.shape[1]

    interpolated_time_variable = np.stack([interpolate_variable(interpolated_points, points, temporal_value[:,time_step], method=method) for time_step in range(total_time)], 1)

    return interpolated_time_variable

def interpolate_mesh_attributes(fine_mesh, coarse_mesh, attribute, method='nearest'):
    """Interpolate the attribute from the fine mesh to the coarse mesh
    
    Args:
        fine_mesh, coarse_mesh: Mesh
            fine and coarse meshes
        attribute: np.array (n,) or (n,T)
            attribute to interpolate
        method: str
            choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)

    Returns:
        interpolated_attribute: np.array (m,) or (m,T) containing the interpolated attribute
    """
    assert isinstance(fine_mesh, Mesh) and isinstance(coarse_mesh, Mesh), "Meshes must be of type Mesh"
    assert attribute.shape[0] == fine_mesh.num_faces, "Attribute must have the same number of nodes as the fine mesh"

    if attribute.ndim == 1:
        interpolated_attribute = interpolate_variable(coarse_mesh.face_xy, fine_mesh.face_xy, attribute, method=method)
    elif attribute.ndim == 2:
        interpolated_attribute = interpolate_temporal_variable(coarse_mesh.face_xy, fine_mesh.face_xy, attribute, method=method)
    else:
        raise ValueError("Attribute must be of shape (n,) or (n,T)")

    return interpolated_attribute

def interpolate_multiscale_attributes(meshes, *attributes, method='nearest'):
    """Interpolate and stack the attributes from the fine mesh to all coarse meshes in the list
    
    Args:
        meshes: list of Mesh
            list of meshes from the coarse to finest
        attributes: list of np.array (n,) or (n,T)
            attributes to interpolate (must have the same number of nodes as the finest mesh)
        method: str
            choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)

    Returns:
        interpolated_attributes: list of np.array (m,) or (m,T) containing the interpolated attributes
    """
    assert isinstance(meshes, list), "meshes must be a list of meshes"
    assert len(meshes) > 0, "meshes must not be empty"
    assert all(isinstance(mesh, Mesh) for mesh in meshes), "meshes must be a list of meshes"

    if len(meshes) == 1:
        print("Only one mesh in the list. No multiscale interpolation needed.")
        return attributes
    
    fine_mesh = meshes[-1]

    interpolated_attributes = [np.concatenate([interpolate_mesh_attributes(fine_mesh, coarse_mesh, attribute, method=method) 
                                for coarse_mesh in meshes]) for attribute in attributes]
    
    return interpolated_attributes

def pool_multiscale_attributes(mesh, *attributes, reduce='mean'):
    """Pool and stack the attributes from the fine mesh to all coarse meshes in the multiscale mesh."""
    assert isinstance(mesh, MultiscaleMesh), "mesh must be a MultiscaleMesh"
    assert mesh.num_meshes > 1, "mesh must contain at least 2 meshes"
    assert mesh.meshes[0].num_faces > mesh.meshes[1].num_faces, "the first mesh must be the finest mesh"
    assert reduce in ['mean', 'sum'], "reduce must be either 'mean' or 'sum'"

    attrs = [torch.as_tensor(attr, dtype=torch.float32) for attr in attributes]
    for i in range(1, mesh.num_meshes):
        if i == 1:
            # For the first coarse mesh, we directly use the stored intra mesh dual edge index
            intra_mesh_dual_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index)[:, mesh.intra_edge_ptr[0]:mesh.intra_edge_ptr[1]] 
            row, col = intra_mesh_dual_edge_index - torch.LongTensor([mesh.face_ptr[1], mesh.face_ptr[0]]).unsqueeze(1)
        else:
            intra_mesh_dual_edge_index = torch.as_tensor(connect_coarse_to_fine_mesh(mesh.meshes[i], mesh.meshes[0]), dtype=torch.long)
            row, col = intra_mesh_dual_edge_index.T

        for j in range(len(attributes)):
            attr_tensor = torch.as_tensor(attrs[j], dtype=torch.float32)
            pooled = scatter(src=attr_tensor[col], index=row, dim=0,
                            dim_size=mesh.meshes[i].num_faces, reduce=reduce)
            attrs[j] = torch.cat((attrs[j], pooled), dim=0)

            if attributes[j].dtype == 'bool' and i == mesh.num_meshes - 1:
                attrs[j] = attrs[j] > 0.5

    return attrs[0] if len(attrs) == 1 else attrs

def extract_single_scale_features_in_multimesh(mesh, scale, *features):
    """Extracts specified features at a single scale from a multiscale mesh.

    Args:
        mesh (MultiscaleMesh) : multiscale mesh object.
        scale (int) : scale at which the features are extracted.
        features (torch.tensor): features to extract (e.g., WD, DEM, etc.)

    Returns:
        features_at_scale (list): list of features at the specified scale.

    Example:
    WD, DEM = extract_single_scale_features_in_multimesh(mesh, scale, data.WD, data.DEM)
    """
    assert isinstance(mesh, MultiscaleMesh), "mesh must be a MultiscaleMesh object"
    assert scale < mesh.num_meshes, "scale must be smaller than the number of meshes in the multiscale mesh"

    scale = scale % mesh.num_meshes

    if len(features) == 1:
        features_at_scale = features[0][mesh.face_ptr[scale]:mesh.face_ptr[scale+1]]
    else:
        features_at_scale = [feature[mesh.face_ptr[scale]:mesh.face_ptr[scale+1]] for feature in features]

    return features_at_scale

def find_closest_nodes(all_points, reference_point, top_n=3):
    """Find the closest top_n nodes from all_points to the reference_point
    
    Args:
        all_points: (np.array), shape (n, 2) all points in the domain
        reference_point: (np.array), shape (2,) reference point
        top_n: (int) number of closest nodes to find
            
    Returns:
        top: np.array, shape (top_n,) index of the closest nodes
    """
    dist = np.sqrt(np.sum((all_points - reference_point)**2, axis=1))
    top = np.argsort(dist)[:top_n]
    return top

def interpolate_BC_location_multiscale(meshes, edge_BC_mid):
    """Find the location of the boundary condition edge for each multiscale mesh 
    by interpolation from the edge midpoints in the finest mesh
    
    Updates:
    edge_index_BC: np.array, shape (num_edges_BC, 2)
        index of the edges that have boundary conditions
    edge_BC: np.array, shape (num_edges_BC,)
        index of the edges that have boundary conditions
    edge_type: np.array, shape (num_edges_BC,)
        type of each edge (1:normal edges, 2:edge with boundary condition, 3:other boundary edges)
    face_BC: np.array, shape (num_faces_BC,)
        index of the faces that have boundary conditions
    """

    for mesh in copy(meshes):
        if not hasattr(mesh, 'boundary_nodes'):
            continue
        all_possible_nodes = []
        for edge in edge_BC_mid:
            for top_n in range(1, 25):
                # find the closest nodes to the edge midpoints for each edge midpoint (shape: (num_edges, top_n))
                possible_nodes = find_closest_nodes(mesh.node_xy, edge, top_n=top_n)
                # check if the closest nodes are boundary nodes
                boundary_nodes_mask = np.isin(possible_nodes, mesh.boundary_nodes)
                if boundary_nodes_mask.sum() >= 2:
                    boundary_possible_nodes = possible_nodes[boundary_nodes_mask]
                    if not hasattr(mesh, 'gdf'):
                        boundary_edge_index = mesh.edge_index.T[mesh.edge_type > 1]
                        if boundary_possible_nodes.shape[0] > 2:
                            boundary_possible_nodes = np.concat((boundary_possible_nodes, boundary_possible_nodes[:1]))
                            for i in range(boundary_possible_nodes.shape[0] -1):
                                is_also_boundary_edge = any(((boundary_possible_nodes[i:i+2] == boundary_edge_index).sum(1) == 2) | 
                                                            ((boundary_possible_nodes[i:i+2][::-1] == boundary_edge_index).sum(1) == 2))
                                if is_also_boundary_edge:
                                    all_possible_nodes.append(boundary_possible_nodes[i:i+2])
                                    break
                        else:
                            is_also_boundary_edge = any(((boundary_possible_nodes == boundary_edge_index).sum(1) == 2) | 
                                                        ((boundary_possible_nodes[::-1] == boundary_edge_index).sum(1) == 2))
                            if is_also_boundary_edge:
                                all_possible_nodes.append(boundary_possible_nodes)
                                break
                    else:
                        if boundary_possible_nodes.shape[0] > 2:
                            boundary_possible_nodes = np.concat((boundary_possible_nodes, boundary_possible_nodes[:1]))
                            for i in range(boundary_possible_nodes.shape[0] -1):
                                is_also_boundary_edge = (is_edge_in_polygon(boundary_possible_nodes[i:i+2], mesh) or 
                                                            is_edge_in_polygon(boundary_possible_nodes[i:i+2][::-1], mesh))
                                if is_also_boundary_edge:
                                    all_possible_nodes.append(boundary_possible_nodes[i:i+2])
                                    break
                        else:
                            if is_edge_in_polygon(boundary_possible_nodes, mesh):
                                all_possible_nodes.append(boundary_possible_nodes)
                                break
        edge_index_BC = remove_duplicate_edges(filter_unique_edges(all_possible_nodes))

        # find the edge index that are on the boundary
        edge_BC_ids = [np.where(((edge == mesh.edge_index.T).sum(1) == 2) | 
                                ((edge[::-1] == mesh.edge_index.T).sum(1) == 2))[0] 
                                        for edge in edge_index_BC]
        correct_edge_BC_mask = np.array([eid.size == 1 for eid in edge_BC_ids])

        # special case for polygonal meshes
        if hasattr(mesh, 'gdf'):
            mesh.edge_index_BC = edge_index_BC
        else:
            mesh.edge_BC = np.array([eid for eid, mask in zip(edge_BC_ids, correct_edge_BC_mask) if mask]).flatten()
            mesh.edge_index_BC = edge_index_BC[correct_edge_BC_mask]

            mesh.edge_type[mesh.edge_BC] = 2
        mesh.face_BC = find_face_BC(mesh)

        # overwrite edge_BC_mid to ensure that the ghost cells don't derail in the coarse meshes
        edge_BC_mid = mesh.node_xy[mesh.edge_index_BC].mean(1)

    return meshes

def get_angle_between_edges(edge1, edge2):
    """Returns the angle between two edges in degrees.
    
    Args:
        edge1 (np.array): shape (2,) representing the x,y coordinates of an edge
        edge2 (np.array): shape (2,) representing the x,y coordinates of an edge
    
    Returns:
        float: The angle between the edges in degrees
    """
    # Calculate the dot product
    dot_product = np.dot(edge1, edge2)
    
    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(edge1)
    magnitude2 = np.linalg.norm(edge2)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Calculate the angle in radians
    angle = np.arccos(cos_angle)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def get_BC_edge_index(dual_edge_index, face_BC, undirected_BC=False):
    """
    Adds ghost cells to existing graph in correspondance of boundary condition (BC) faces

    Args:
        dual_edge_index (np.array): contains a list of edges of the dual graph
        face_BC (np.array): contains a list of boundary faces (faces with boundary conditions)
        undirected_BC (bool): if True, the information flow can go also to ghost nodes

    Returns:
        updated dual_edge_index (with ghost cells)
        ghost_cells_ids: np.array of ghost cells ids
    """
    num_faces = dual_edge_index.max() + 1
    dual_edge_index_BC = []
    ghost_cells_ids = []

    for i, face in enumerate(face_BC):
        dual_edge_index_BC.append([num_faces+i, face])
        ghost_cells_ids.append(num_faces+i)
        if undirected_BC:
            dual_edge_index_BC.append([face, num_faces+i])

    return np.array(dual_edge_index_BC).T, np.array(ghost_cells_ids)

def get_ghost_nodes(mesh):
    """Returns the ghost nodes ids"""
    num_BC_faces = len(mesh.face_BC)
    ghost_edge_index = []
    ghost_face_nodes = []

    ghost_nodes = mesh.nodes_per_face[mesh.face_BC]-2 if not hasattr(mesh, 'gdf') else np.ones(len(mesh.face_BC), dtype=np.int32)*2
    mesh.ghost_node_ids = [mesh.node_x.shape[0]-j-1 for j in range(ghost_nodes.sum())][::-1]

    for i in range(num_BC_faces):
        ghost_i = ghost_nodes[:i].sum()
        ghost_edge_index.append([mesh.ghost_node_ids[ghost_i], mesh.edge_index_BC[i,0]])

        # loop for polygons with more than 3 nodes
        for j in range(ghost_nodes[i]-1):
            ghost_edge_index.append([mesh.ghost_node_ids[i+j], mesh.ghost_node_ids[i+j+1]])

        ghost_edge_index.append([mesh.edge_index_BC[i,1], mesh.ghost_node_ids[ghost_nodes[:i+1].sum()-1]])
        ghost_face_nodes.append(mesh.ghost_node_ids[ghost_i:ghost_i+ghost_nodes[i]][::-1] + mesh.edge_index_BC[i].tolist())

    ghost_edge_index = np.array(ghost_edge_index).T
    ghost_face_nodes = np.concatenate(ghost_face_nodes)

    if not hasattr(mesh, 'gdf'):
        assert ghost_edge_index.shape[1] == (mesh.nodes_per_face[mesh.face_BC]-1).sum(), "The number of ghost edges is not correct"
        assert ghost_face_nodes.shape[0] == (mesh.nodes_per_face[mesh.face_BC]).sum(), "The number of ghost nodes is not correct"
    else:
        assert ghost_edge_index.shape[1] == 3*num_BC_faces, f"The number of ghost edges is not correct.\
            It should be 3*num_BC_faces (because we add 2 nodes for each face) but is {ghost_edge_index.shape[1]}"
        assert ghost_face_nodes.shape[0] == 4*num_BC_faces, f"The number of ghost nodes is not correct.\
            It should be 4*num_BC_faces (because we add 2 nodes for each face) but is {ghost_face_nodes.shape[0]}"

    return ghost_edge_index, ghost_face_nodes

def find_BC_other_nodes(mesh):
    """Returns the coordinates of the nodes that are in the boundary faces but not in the boundary edges
    In case of polygons of more than 4 nodes, we create fake nodes so that the polygon is a quadrilateral
    
    Returns:
    the_other_nodes: list (len = num_BC) of np.array (shape (num_outside_nodes, 2))
        The coordinates of the nodes that are in the boundary faces but not in the boundary edges
    """
    assert mesh.face_BC is not [], "The boundary faces face_BC must be known"
    if mesh.added_ghost_cells:
        the_other_nodes = mesh.node_xy[mesh.ghost_node_ids]
    else:
        # the nodes which are not in the BC edge
        the_other_nodes = []

        for edge, face in zip(mesh.edge_index_BC, mesh.face_BC):
            face_nodes = get_face_nodes_matrix(mesh)[face,:mesh.nodes_per_face[face]].astype(int)

            assert len(face_nodes) >= 3, "The face must have at least 3 nodes"

            if len(face_nodes) <= 4:
                # take the nodes that are in face_nodes but not in edge
                other_nodes = np.setdiff1d(face_nodes, edge)

                # add node for circularity
                face_nodes = np.append(face_nodes, face_nodes[0])

                # we need to order the nodes in the same order as the edge for the quadrilateral
                if np.where(face_nodes == edge[0])[0][-1] > np.where(face_nodes == edge[1])[0][0]:
                    other_nodes = other_nodes[::-1]
                the_other_nodes.append(mesh.node_xy[other_nodes])
            else:
                # we need to create 2 fake nodes outside of the polygon so that the face is a quadrilateral
                # we compute the outward normal of the edge
                edge_nodes = mesh.node_xy[edge]
                edge_vector = edge_nodes[1] - edge_nodes[0]
                edge_normal = np.array([edge_vector[1], -edge_vector[0]])
                edge_normal = edge_normal / np.linalg.norm(edge_normal)

                # we symmetrize two points with length equal to the edge index BC length outside of the polygon
                fake_node1 = edge_nodes[0] + edge_normal * np.linalg.norm(edge_vector)
                fake_node2 = edge_nodes[1] + edge_normal * np.linalg.norm(edge_vector)
                
                # we symmetrize two points with length equal to the edge index BC length outside of the polygon
                check_fake_node1 = edge_nodes[0] + edge_normal
                check_fake_node2 = edge_nodes[1] + edge_normal

                # check that the fake nodes are outside of the polygon
                if mesh.gdf.iloc[face].geometry.contains(Point(check_fake_node1)) or \
                   mesh.gdf.iloc[face].geometry.contains(Point(check_fake_node2)):
                    fake_node1 = edge_nodes[0] - edge_normal * np.linalg.norm(edge_vector)
                    fake_node2 = edge_nodes[1] - edge_normal * np.linalg.norm(edge_vector)

                the_other_nodes.append(np.array([fake_node2, fake_node1]))

        # we return a list since the number of nodes outside of the polygon can be different for each face
        assert len(the_other_nodes) == len(mesh.face_BC), "The number of faces must be equal to the number of other nodes"
    return the_other_nodes

def is_edge_in_polygon(edge, mesh):
    """Find the edges that are inside a polygon (polygon)"""
    return any([np.isin(edge, face_nodes).all() for face_nodes in get_face_nodes_matrix(mesh)])

def find_face_BC(mesh):
    """Find the faces that have boundary conditions (face_BC), knowing the edges with boundary conditions (edge_index_BC)"""
    face_BC = []
    face_nodes_matrix = get_face_nodes_matrix(mesh)

    assert mesh.edge_index_BC is not [], "The boundary edges edge_index_BC must be known"
    # assert mesh.edge_BC.size == 1 and mesh.edge_index_BC.size == 2, "The dimension of the BC edge are wrong"
    
    if hasattr(mesh, 'gdf'):
        return np.array([np.where([np.isin(edge, face_nodes).all() for face_nodes in face_nodes_matrix])[0] 
                       for edge in mesh.edge_index_BC]).reshape(-1)
    else:
        for edge_index_BC in mesh.edge_index_BC:
            for i, face_nodes in enumerate(face_nodes_matrix):
                face_nodes = face_nodes[~np.isnan(face_nodes)].astype(int)
                
                # Add first element to simplify circular iterations
                face_nodes = np.concatenate((face_nodes, face_nodes[:1]))

                # Check if the edge is part of the face
                if any((edge_index_BC == face_nodes[j:j+2]).all() or
                    (edge_index_BC[::-1] == face_nodes[j:j+2]).all()
                    for j in range(len(face_nodes)-1)):
                    face_BC.append(i)

    assert len(face_BC) == len(mesh.edge_index_BC), f"The number of faces with boundary conditions {len(face_BC)} must be equal to the number of edges with boundary conditions {len(mesh.edge_index_BC)}"
    
    return np.array(face_BC)

def mirror_BC_nodes(node_BC_xy, edge_outward_normal_nodes, node_symmetry_point):
    """Mirrors the BC nodes w.r.t. the edge center (if it's a triangle) or the edge nodes (if it's a quadrilateral)
    
    Args:
        node_BC_xy: np.array, shape (num_BC_nodes, 2)
            The coordinates of the nodes in the boundary faces not in the boundary edges
        edge_outward_normal_nodes: np.array, shape (num_BC_nodes, 2)
            The outward normal of the BC edge 
        node_symmetry_point: np.array, shape (num_BC_nodes, 2)
            The coordinates of the edge center (if it's a triangle) or the edge nodes (if it's a quadrilateral)
    
    Returns:
        ghost_node_BC_xy: np.array, shape (num_BC_nodes, 2)
            The coordinates of the ghost nodes
    """
    ghost_node_BC_xy = _mirror_points_with_normals(
        node_BC_xy,
        edge_outward_normal_nodes,
        node_symmetry_point,
    )

    assert ghost_node_BC_xy.shape[0] == len(node_BC_xy), "The number of ghost nodes must be equal to the number of BC nodes"

    return ghost_node_BC_xy

def mirror_BC_faces(face_BC_xy, edge_outward_normal_faces, face_symmetry_point):
    """Mirrors the BC faces w.r.t. the edge center
    
    Args:
        face_BC_xy: np.array, shape (num_BC, 2)
            The coordinates of the BC faces
        edge_outward_normal_faces: np.array, shape (num_BC, 2)
            The outward normal of the BC edge 
        face_symmetry_point: np.array, shape (num_BC, 2)
            The coordinates of the edge center
    
    Returns:
        ghost_face_BC_xy: np.array, shape (num_BC, 2)
            The coordinates of the ghost faces
    """
    ghost_face_BC_xy = _mirror_points_with_normals(
        face_BC_xy,
        edge_outward_normal_faces,
        face_symmetry_point,
    )

    assert ghost_face_BC_xy.shape[0] == len(face_BC_xy), "The number of ghost faces must be equal to the number of BC faces"

    return ghost_face_BC_xy

def add_ghost_cells_mesh(mesh):
    """
    PERFORMS IN-PLACE MODIFICATION OF MESH
    
    Adds ghost cells to the mesh by mirroring the boundary faces and nodes w.r.t. the boundary edges
    
    Updates:
    node_x, node_y: np.array, shape (num_nodes+num_ghost_nodes,)
    face_x, face_y: np.array, shape (num_faces+num_ghost_faces,)
    nodes_per_face: np.array, shape (num_faces+num_ghost_faces,)
    edge_index: np.array, shape (2, num_edges+num_ghost_edges)
    edge_type: np.array, shape (num_edges+num_ghost_edges,)
    dual_edge_index: np.array, shape (2, num_dual_edges+num_ghost_dual_edges)
        this is also converted to undirected
    face_nodes: np.array, shape (num_faces+num_ghost_faces, max_nodes_per_face)
    added_ghost_cells: bool, True if ghost cells have been added
    """
    if not mesh.added_ghost_cells:
        # nodes that are in the boundary faces but not in the boundary edges
        node_BC_xy = find_BC_other_nodes(mesh) # len = num_BC, shape (num_outside_nodes, 2)

        if hasattr(mesh, 'gdf'):
            # we already have the correct ghost cells nodes
            ghost_node_BC_xy = np.concat(node_BC_xy)
            mesh.nodes_per_face = np.concatenate((mesh.nodes_per_face, np.ones(len(mesh.face_BC), dtype=np.int32)*4))

            ghost_face_nodes_xy = np.concat((np.array(node_BC_xy), mesh.node_xy[mesh.edge_index_BC]), axis=1) # shape (num_BC, 4, 2)
            ghost_face_BC_xy = ghost_face_nodes_xy.mean(1)

            polygons_to_add = [Polygon(close_polygon(face_nodes)) for face_nodes in ghost_face_nodes_xy]
            polygons_to_add = gpd.GeoDataFrame(geometry=polygons_to_add, crs=mesh.gdf.crs)
            polygons_to_add['centroid'] = polygons_to_add.geometry.centroid

            # concatenate the new polygon to the existing GeoDataFrame
            mesh.gdf = pd.concat([mesh.gdf, polygons_to_add], ignore_index=True)

        else:
            node_BC_xy = np.concat(node_BC_xy)
            face_BC_xy = mesh.face_xy[mesh.face_BC]

            edge_outward_normal_nodes = np.concat([np.repeat(item.reshape(1,-1), 2, axis=0) if mesh.nodes_per_face[mesh.face_BC][i] == 4
                                            else [item] for i, item in enumerate(mesh.edge_outward_normal[mesh.edge_BC])])
            edge_outward_normal_faces = mesh.edge_outward_normal[mesh.edge_BC]

            # nodes are mirrored w.r.t. edge center (triangles) or edge nodes (quatrilaterals)
            node_symmetry_point = np.concat([[item.mean(0)] if mesh.nodes_per_face[mesh.face_BC][i] == 3
                                            else item for i, item in enumerate(mesh.node_xy[mesh.edge_index_BC])])
            # faces are mirrored w.r.t. edge center
            face_symmetry_point = mesh.node_xy[mesh.edge_index_BC].mean(1)

            ghost_node_BC_xy = mirror_BC_nodes(node_BC_xy, edge_outward_normal_nodes, node_symmetry_point)
            ghost_face_BC_xy = mirror_BC_faces(face_BC_xy, edge_outward_normal_faces, face_symmetry_point)

            mesh.nodes_per_face = np.concatenate((mesh.nodes_per_face, mesh.nodes_per_face[mesh.face_BC]))

        mesh.node_x = np.concatenate((mesh.node_x, ghost_node_BC_xy[:,0]))
        mesh.node_y = np.concatenate((mesh.node_y, ghost_node_BC_xy[:,1]))

        mesh.face_x = np.concatenate((mesh.face_x, ghost_face_BC_xy[:,0]))
        mesh.face_y = np.concatenate((mesh.face_y, ghost_face_BC_xy[:,1]))

        # update edge_index and dual_edge_index after adding ghost cells
        # dual_edge_index is converted to undirected
        mesh.dual_edge_index_BC, mesh.ghost_cells_ids = get_BC_edge_index(mesh.dual_edge_index, 
                                                                mesh.face_BC, undirected_BC=False)
        mesh.dual_edge_index = np.concatenate((mesh.dual_edge_index, mesh.dual_edge_index_BC), 1)
        mesh.edge_weir = np.concatenate((mesh.edge_weir, np.zeros(mesh.dual_edge_index_BC.shape[1], dtype=np.int32))) if \
            hasattr(mesh, 'edge_weir') else np.zeros(mesh.dual_edge_index.shape[1], dtype=np.int32)
        ghost_edge_index, ghost_face_nodes = get_ghost_nodes(mesh)
        if hasattr(mesh, 'edge_type'): #this is not the case for geodataframes
            mesh.edge_index = np.concatenate((mesh.edge_index, ghost_edge_index), 1)
            mesh.edge_type = np.concatenate((mesh.edge_type, np.ones(ghost_edge_index.shape[1], dtype=np.int32)*4))
        mesh.face_nodes = np.concatenate((mesh.face_nodes, ghost_face_nodes))

        mesh._get_derived_attributes()
        mesh.added_ghost_cells = True
    
    else:
        print("Ghost cells already added. Skipping...")

    return mesh

def remove_ghost_cells(mesh):
    """Remove all ghost cells from the mesh

    PERFORMS IN-PLACE MODIFICATION OF THE MESH
    
    Updates:
    node_x, node_y: np.array, shape (num_nodes,)
    face_x, face_y: np.array, shape (num_faces,)
    nodes_per_face: np.array, shape (num_faces,)
    edge_index: np.array, shape (2, num_edges)
    edge_type: np.array, shape (num_edges,)
    dual_edge_index: np.array, shape (2, num_dual_edges)
    face_nodes
    """
    if not mesh.added_ghost_cells:
        print("No ghost cells present in the mesh")
    else:
        num_ghost_cells = len(mesh.ghost_cells_ids)
        num_ghost_nodes = len(mesh.ghost_node_ids)
        num_face_nodes = mesh.nodes_per_face[-num_ghost_cells:].sum()

        mesh.node_x = mesh.node_x[:-num_ghost_nodes]
        mesh.node_y = mesh.node_y[:-num_ghost_nodes]

        mesh.face_x = mesh.face_x[:-num_ghost_cells]
        mesh.face_y = mesh.face_y[:-num_ghost_cells]

        mesh.nodes_per_face = mesh.nodes_per_face[:-num_ghost_cells]
        
        mesh.dual_edge_index = mesh.dual_edge_index[:,:-num_ghost_cells]
        
        mesh.edge_index = mesh.edge_index[:,:-(num_ghost_nodes+num_ghost_cells)]
        if hasattr(mesh, 'edge_type'):
            mesh.edge_type = mesh.edge_type[:-(num_ghost_nodes+num_ghost_cells)]
            mesh.edge_type[mesh.edge_type == 2] = 3

        mesh.face_nodes = mesh.face_nodes[:-num_face_nodes]
        
        mesh._get_derived_attributes()
        mesh.added_ghost_cells = False
    
    return mesh

def remove_ghost_cells_multiscale(mesh):
    """Remove all ghost cells from a Multiscale mesh"""
    assert isinstance(mesh, MultiscaleMesh), "Input mesh must be a MultiscaleMesh"
    
    new_meshes = [remove_ghost_cells(copy(m)) for m in mesh.meshes]

    new_mesh = MultiscaleMesh()
    new_mesh.stack_meshes(new_meshes)

    return new_mesh

def add_ghost_cells_attributes(mesh, *attributes):
    '''Add attribute value at ghost cells'''
    assert mesh.added_ghost_cells, "This function must be executed after add_ghost_cells_mesh"
    
    attribute_BC = [np.concatenate((attr, attr[mesh.face_BC]), axis=0) for attr in attributes]
    
    return attribute_BC[0] if len(attribute_BC) == 1 else attribute_BC

def copy_face_BC_attributes_to_ghost_cell(mesh, *attributes):
    '''Corrects attribute value at ghost cells by mirroring the boundary faces and nodes w.r.t. the boundary edges'''
    assert mesh.added_ghost_cells, "This function must be executed after add_ghost_cells_mesh"
    
    for attr in attributes:
        attr[mesh.ghost_cells_ids] = attr[mesh.face_BC]

    return attributes

def convert_simulation_to_pyg(output_map, BC, type_BC, meshes, roughness_file=None):
    '''Creates a pytorch geometric Data object of a mesh simulation
    
    Args:
        output_map (str): path to the netcdf file with the simulation output
        roughness_file (str): path to the roughness file
        BC (np.array): boundary conditions
        type_BC (int): type of boundary condition
        meshes (list[Mesh]): list of Mesh objects

    Returns:
        data (Data): pytorch geometric Data object
    '''
    assert len(meshes) > 1, 'Only multiscale meshes are supported in this version'
    assert os.path.exists(output_map), f'File {output_map} does not exist'

    # create multiscale meshes
    meshes = meshes[::-1] if meshes[0].face_xy.shape[0] < meshes[-1].face_xy.shape[0] else meshes
    fine_mesh = meshes[0]

    # Import mesh attributes (water depth, velocity, DEM, roughness)
    nc_dataset = xr.open_dataset(output_map)

    DEM = nc_dataset.mesh2d_flowelem_bl.values
    WD = nc_dataset.mesh2d_waterdepth.values.T
    VX = nc_dataset.mesh2d_ucx.values.T
    VY = nc_dataset.mesh2d_ucy.values.T
    fine_mesh._import_roughness(roughness_file, delimiter=',')

    # right now we are removing the part of the mesh for the boundary conditions
    if hasattr(fine_mesh, 'inside_faces'):
        inside_faces = fine_mesh.inside_faces

        # correct node features
        DEM = fine_mesh.DEM if hasattr(fine_mesh, 'DEM') else DEM[inside_faces]
        WD = WD[inside_faces]
        VX = VX[inside_faces]
        VY = VY[inside_faces]
        
        if hasattr(fine_mesh, 'init_WD'):
            WD[:,0] = fine_mesh.init_WD

    WD[WD > 9] = 0
    WD[fine_mesh.face_BC, 0] = 0.01

    # Add boundary conditions to coarser multiscale meshes
    edge_BC_mid = fine_mesh.node_xy[fine_mesh.edge_index_BC].mean(1)
    meshes[1:] = interpolate_BC_location_multiscale(meshes[1:], edge_BC_mid)
    for m in meshes:
        m.edge_BC = m.edge_BC[:1]
        m.edge_index_BC = m.edge_index_BC[:1]
        m.face_BC = m.face_BC[:1]

    meshes = [add_ghost_cells_mesh(mesh) for mesh in meshes]
    DEM, WD, VX, VY, roughness, canal_mask, lakes_mask = add_ghost_cells_attributes(
        meshes[0], DEM, WD, VX, VY, fine_mesh.roughness, fine_mesh.canal_mask, fine_mesh.lakes_mask)

    # create multiscale mesh
    mesh = MultiscaleMesh()
    mesh.stack_meshes(meshes)
    mesh.correct_BC(meshes)

    # get multiscale attributes
    DEM, WD, VX, VY, mesh.roughness, mesh.canal_mask, mesh.lakes_mask = pool_multiscale_attributes(mesh, DEM, WD, VX, VY, roughness, canal_mask, lakes_mask, reduce='mean')
    mesh.DEM, mesh.roughness = copy_face_BC_attributes_to_ghost_cell(mesh, DEM, mesh.roughness) #correct ghost cells values after pooling

    # Convert mesh to pytorch geometric Data object
    data = convert_mesh_to_pyg(mesh)

    data.WD = torch.FloatTensor(WD)
    data.VX = torch.FloatTensor(VX)
    data.VY = torch.FloatTensor(VY)

    fine_mesh_ghost_cells = meshes[0].ghost_cells_ids
    fine_mesh_BC_edges = meshes[0].edge_BC

    assert fine_mesh_BC_edges.shape == fine_mesh_ghost_cells.shape, "There's something wrong with the number of BC faces and edges"

    data.node_BC = torch.IntTensor(fine_mesh_ghost_cells)
    data.edge_BC_length = torch.FloatTensor(meshes[0].edge_length[fine_mesh_BC_edges])

    data.BC = torch.FloatTensor(BC).unsqueeze(0).repeat(len(data.node_BC), 1, 1) # This repeats the same BC
    data.type_BC = torch.tensor(type_BC, dtype=torch.int).repeat(len(fine_mesh_ghost_cells)) # This repeats the same BC type

    return data

def convert_mesh_to_pyg(mesh):
    """Convert mesh to pytorch geometric Data object"""
    data = Data()

    data.node_ptr = torch.LongTensor(mesh.face_ptr)
    data.edge_ptr = torch.LongTensor(mesh.dual_edge_ptr)
    data.intra_edge_ptr = torch.LongTensor(mesh.intra_edge_ptr)
    data.intra_mesh_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index)

    data.DEM = torch.FloatTensor(mesh.DEM)
    data.roughness = torch.FloatTensor(mesh.roughness)

    data.canal_mask = torch.BoolTensor(mesh.canal_mask) if hasattr(mesh, 'canal_mask') else None
    data.lakes_mask = torch.BoolTensor(mesh.lakes_mask) if hasattr(mesh, 'lakes_mask') else None
    
    # Assign other data properties
    data.edge_index = torch.LongTensor(mesh.dual_edge_index)
    data.face_distance = torch.FloatTensor(mesh.dual_edge_length)
    # data.face_relative_distance = torch.FloatTensor(mesh.face_relative_distance)
    data.edge_weir = torch.FloatTensor(mesh.edge_weir) if hasattr(mesh, 'edge_weir') else None
    # data.normal = torch.FloatTensor(mesh.edge_outward_normal[mesh.edge_type < 3])
    data.num_nodes = mesh.num_faces
    data.area = torch.FloatTensor(mesh.face_area)

    data.mesh = mesh

    return data

def invert_scale_ordering(data):
    """Invert the ordering of the node and edge features in the multiscale mesh (from coarse to fine or viceversa).
    Use this function on the pyg_dataset obtained in create_datasets"""

    assert isinstance(data.mesh, MultiscaleMesh), "This function is valid only for MultiscaleMesh datasets."
    
    temp = Data()

    node_ptr = data.node_ptr
    edge_ptr = data.edge_ptr
    intra_edge_ptr = data.intra_edge_ptr

    temp.node_ptr = torch.LongTensor(np.cumsum([0]+[node_ptr[-i-1]-node_ptr[-i-2] for i in range(len(node_ptr)-1)]))
    temp.edge_ptr = torch.LongTensor(np.cumsum([0]+[edge_ptr[-i-1]-edge_ptr[-i-2] for i in range(len(edge_ptr)-1)]))
    temp.intra_edge_ptr = torch.LongTensor(np.cumsum([0]+[intra_edge_ptr[-i-1]-intra_edge_ptr[-i-2] for i in range(len(intra_edge_ptr)-1)]))

    temp.WD = torch.cat([data.WD[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.VX = torch.cat([data.VX[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.VY = torch.cat([data.VY[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.slopex = torch.cat([data.slopex[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.slopey = torch.cat([data.slopey[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.DEM = torch.cat([data.DEM[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.area = torch.cat([data.area[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])

    temp.BC = torch.flip(data.BC, [0])
    temp.node_BC = torch.stack([data.node_BC[i]-node_ptr[i+1]+temp.node_ptr[-i-1] for i in range(len(data.node_BC))])
    temp.type_BC = data.type_BC

    temp.edge_index = torch.cat([data.edge_index[:,edge_ptr[i]:edge_ptr[i+1]]-node_ptr[i]+temp.node_ptr[-i-2] for i in range(len(edge_ptr)-1)][::-1], 1)
    temp.face_distance = torch.cat([data.face_distance[edge_ptr[i]:edge_ptr[i+1]] for i in range(len(edge_ptr)-1)][::-1])
    temp.face_relative_distance = torch.cat([data.face_relative_distance[edge_ptr[i]:edge_ptr[i+1]] for i in range(len(edge_ptr)-1)][::-1])
    temp.edge_BC_length = torch.flip(data.edge_BC_length, [0])

    meshes = data.mesh.meshes[::-1]
    mesh = MultiscaleMesh()
    mesh.stack_meshes(meshes)
    temp.mesh = mesh
    temp.intra_mesh_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index)

    return temp

def create_dataset_folders(dataset_folder='datasets'):
    """Creates the folders for storing training and testing datasets
    
    Args:
        dataset_folder (str): path to the dataset folder
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    return None

def save_database(dataset, name, out_path='datasets'):
    '''This function saves the geometric database into a pickle file.
    
    Args:
        dataset: list of pytorch geometric Data objects
        name: str, name of the file
        out_path: str, path to the output folder
    '''
    path = f"{out_path}/{name}.pkl"
    
    if os.path.exists(path):
        os.remove(path)
    elif not os.path.exists(out_path):
        os.mkdir(out_path)
    
    pickle.dump(dataset, open(path, "wb" ))
        
    return None