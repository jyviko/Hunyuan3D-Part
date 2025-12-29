import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency fallback
    cKDTree = None

LOGGER = logging.getLogger(__name__)


def _align_cylinder_to_axis(
    radius: float,
    height: float,
    center: np.ndarray,
    axis: np.ndarray,
    sections: int = 32,
) -> trimesh.Trimesh:
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
    transform[:3, 3] = center
    cylinder.apply_transform(transform)
    return cylinder


def _select_pin_centers(
    points: np.ndarray,
    directions: np.ndarray,
    count: int,
    min_spacing: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or count <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    # Farthest point sampling with a minimum spacing constraint.
    idx = rng.integers(0, points.shape[0])
    chosen = [idx]
    dist = np.linalg.norm(points - points[idx], axis=1)
    for _ in range(1, count):
        next_idx = int(np.argmax(dist))
        if dist[next_idx] < min_spacing:
            break
        chosen.append(next_idx)
        dist = np.minimum(dist, np.linalg.norm(points - points[next_idx], axis=1))
    return points[chosen], directions[chosen]


def _apply_union(
    base: trimesh.Trimesh,
    additions: List[trimesh.Trimesh],
    engine: Optional[str] = None,
) -> trimesh.Trimesh:
    if not additions:
        return base
    try:
        return trimesh.boolean.union([base, *additions], engine=engine)
    except Exception as exc:
        LOGGER.warning("Boolean union failed, concatenating parts instead: %s", exc)
        return trimesh.util.concatenate([base, *additions])


def _apply_difference(
    base: trimesh.Trimesh,
    cutters: List[trimesh.Trimesh],
    engine: Optional[str] = None,
) -> trimesh.Trimesh:
    if not cutters:
        return base
    try:
        return trimesh.boolean.difference([base, *cutters], engine=engine)
    except Exception as exc:
        LOGGER.warning("Boolean difference failed, keeping original mesh: %s", exc)
        return base


def add_pin_sockets(
    scene: trimesh.Scene,
    pin_diameter: float = 3.0,
    pin_length: float = 6.0,
    clearance: float = 0.2,
    area_per_pin: float = 2000.0,
    interface_distance: float = 0.5,
    sample_count: int = 20000,
    min_pin_spacing: Optional[float] = None,
    edge_distance: Optional[float] = None,
    pin_embed: float = 1.0,
    max_pins_per_interface: Optional[int] = None,
    boolean_engine: Optional[str] = None,
    seed: int = 42,
) -> trimesh.Scene:
    """
    Add pin-and-socket connectors between touching parts in a scene.

    This method keeps part alignment by placing pins along the interface
    normal direction (from part A to part B). It uses surface sampling
    to estimate interface area and candidate pin locations.
    """
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("scene must be a trimesh.Scene")
    if len(scene.geometry) < 2:
        return scene

    pin_radius = pin_diameter * 0.5
    socket_radius = pin_radius + clearance
    min_pin_spacing = min_pin_spacing or (2.0 * pin_diameter)
    edge_distance = edge_distance if edge_distance is not None else (2.0 * pin_diameter)
    pin_embed = min(pin_embed, pin_length * 0.45)

    rng = np.random.default_rng(seed)
    names = list(scene.geometry.keys())
    meshes = {name: scene.geometry[name].copy() for name in names}
    pins: Dict[str, List[trimesh.Trimesh]] = {name: [] for name in names}
    sockets: Dict[str, List[trimesh.Trimesh]] = {name: [] for name in names}

    for i in range(len(names)):
        mesh_a = meshes[names[i]]
        if mesh_a.is_empty:
            continue
        for j in range(i + 1, len(names)):
            mesh_b = meshes[names[j]]
            if mesh_b.is_empty:
                continue

            sample_points, face_idx = trimesh.sample.sample_surface(
                mesh_a, sample_count, seed=seed + i + j
            )
            normals = mesh_a.face_normals[face_idx]
            proximity = trimesh.proximity.ProximityQuery(mesh_b)
            closest, distances, _ = proximity.on_surface(sample_points)
            interface_mask = distances <= interface_distance
            if not np.any(interface_mask):
                continue

            interface_points = sample_points[interface_mask]
            interface_normals = normals[interface_mask]
            interface_closest = closest[interface_mask]

            # Estimate interface area from sampling density.
            area_per_sample = mesh_a.area / float(sample_count)
            interface_area = float(interface_points.shape[0]) * area_per_sample
            if interface_area <= 0.0:
                continue

            # Keep candidates away from interface boundary if possible.
            if edge_distance > 0 and cKDTree is not None:
                non_interface = sample_points[~interface_mask]
                if non_interface.shape[0] > 0:
                    tree = cKDTree(non_interface)
                    distances_to_boundary, _ = tree.query(interface_points, k=1)
                    keep = distances_to_boundary >= edge_distance
                    interface_points = interface_points[keep]
                    interface_normals = interface_normals[keep]
                    interface_closest = interface_closest[keep]
            if interface_points.shape[0] == 0:
                continue

            # Direction from A to B.
            directions = interface_closest - interface_points
            dir_norm = np.linalg.norm(directions, axis=1)
            valid = dir_norm > 1e-6
            interface_points = interface_points[valid]
            interface_normals = interface_normals[valid]
            directions = directions[valid]
            dir_norm = dir_norm[valid]
            if interface_points.shape[0] == 0:
                continue
            directions = directions / dir_norm[:, None]

            # Align normals to the A->B direction.
            flip = (np.sum(interface_normals * directions, axis=1) < 0.0)
            interface_normals[flip] *= -1.0

            pin_count = int(np.ceil(interface_area / max(area_per_pin, 1e-6)))
            pin_count = max(1, pin_count)
            if max_pins_per_interface is not None:
                pin_count = min(pin_count, max_pins_per_interface)
            pin_count = min(pin_count, interface_points.shape[0])

            pin_centers, pin_dirs = _select_pin_centers(
                interface_points, directions, pin_count, min_pin_spacing, rng
            )
            if pin_centers.shape[0] == 0:
                continue

            for center, axis in zip(pin_centers, pin_dirs):
                # Pin on A: embed slightly to ensure union.
                pin_center = center + axis * (pin_length * 0.5 - pin_embed)
                pin_mesh = _align_cylinder_to_axis(
                    pin_radius, pin_length, pin_center, axis
                )
                pins[names[i]].append(pin_mesh)

                # Socket in B: subtract along the same axis with clearance.
                socket_center = center + axis * (pin_length * 0.5)
                socket_mesh = _align_cylinder_to_axis(
                    socket_radius, pin_length, socket_center, axis
                )
                sockets[names[j]].append(socket_mesh)

    # Apply pins and sockets per part.
    out_scene = trimesh.Scene()
    for name in names:
        mesh = meshes[name]
        mesh = _apply_union(mesh, pins[name], engine=boolean_engine)
        mesh = _apply_difference(mesh, sockets[name], engine=boolean_engine)
        out_scene.add_geometry(mesh, node_name=name)
    return out_scene
