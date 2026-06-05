use i3_math::{nalgebra::Point3, AABB};
use i3_voxel::SdfScene;

use crate::{BRICK_SIZE, svo::SvoNode};

/// Screen-space projected size score for `aabb` from camera position.
/// Returns pixel-footprint / 1 (dimensionless ratio; > lod_threshold → split).
pub fn screen_score(aabb: &AABB, cam: &Point3<f32>, vp: &nalgebra::Matrix4<f32>) -> f32 {
    if !aabb.is_in_frustum(vp) { return 0.0; }
    let dist = (cam - aabb.clamp(cam)).norm().max(0.001);
    aabb.diagonal_length() / dist
}

/// Non-linearity of the SDF over `aabb`, in world units: how far the centre value
/// deviates from the average of the corner values. A linear SDF (flat plane,
/// half-space) gives ~0 — trilinear interpolation already reproduces it exactly,
/// so the brick needs no further subdivision. Curved surfaces (sphere, torus,
/// edges) give a positive value that grows with sub-brick detail.
pub fn sdf_curvature(scene: &SdfScene, aabb: &AABB) -> f32 {
    let nodes = scene.get_nodes(aabb);
    if nodes.is_empty() { return 0.0; }

    let mid = aabb.center();
    let c   = SdfScene::sample(&nodes, &mid).value;

    let min = aabb.min;
    let max = aabb.max;
    let corners = [
        Point3::new(min.x, min.y, min.z), Point3::new(max.x, min.y, min.z),
        Point3::new(min.x, max.y, min.z), Point3::new(max.x, max.y, min.z),
        Point3::new(min.x, min.y, max.z), Point3::new(max.x, min.y, max.z),
        Point3::new(min.x, max.y, max.z), Point3::new(max.x, max.y, max.z),
    ];
    let mut mean = 0.0;
    for p in &corners { mean += SdfScene::sample(&nodes, p).value; }
    mean /= 8.0;

    (c - mean).abs()
}

/// Surface-proximity SDF error for `node`: fraction of sample points where
/// the analytical SDF is within 2 voxels of the surface at this node's resolution.
/// Returns 0..=1. Cheap approximation: samples 9 points (corners + centre).
pub fn sdf_approximation_error(node: &SvoNode, sdf: &SdfScene) -> f32 {
    let voxel_size = (node.aabb.max.x - node.aabb.min.x) / BRICK_SIZE as f32;
    let threshold  = voxel_size * 2.0;

    let nodes = sdf.get_nodes(&node.aabb);
    if nodes.is_empty() { return 0.0; }

    let min = node.aabb.min;
    let max = node.aabb.max;
    let mid = node.aabb.center();

    let pts = [
        mid,
        Point3::new(min.x, min.y, min.z),
        Point3::new(max.x, min.y, min.z),
        Point3::new(min.x, max.y, min.z),
        Point3::new(max.x, max.y, min.z),
        Point3::new(min.x, min.y, max.z),
        Point3::new(max.x, min.y, max.z),
        Point3::new(min.x, max.y, max.z),
        Point3::new(max.x, max.y, max.z),
    ];

    let near = pts.iter()
        .filter(|&&p| SdfScene::sample(&nodes, &p).value.abs() < threshold)
        .count();

    near as f32 / pts.len() as f32
}

/// Combined score: screen-space (gate) amplified by SDF surface proximity.
/// `sdf_weight = 0.0` → pure screen-space (fast).
/// `sdf_weight = 1.0` → surface areas get up to 2× higher priority.
pub fn compute_score(
    node:          &SvoNode,
    cam:           &Point3<f32>,
    vp:            &nalgebra::Matrix4<f32>,
    sdf:           Option<&SdfScene>,
    lod_threshold: f32,
    sdf_weight:    f32,
) -> f32 {
    let ss = screen_score(&node.aabb, cam, vp);
    if ss < 1e-4 { return -ss; }

    if sdf_weight > 0.0 {
        if let Some(scene) = sdf {
            if ss > lod_threshold * 0.5 {
                let err = sdf_approximation_error(node, scene);
                return ss * (1.0 + sdf_weight * err);
            }
        }
    }
    ss
}
