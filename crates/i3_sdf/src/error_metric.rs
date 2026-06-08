use i3_math::{nalgebra::{Point3, Vector3}, AABB};
use i3_voxel::SdfScene;

use crate::BRICK_SIZE;

/// Estimated tangent-plane reconstruction error of a brick over `aabb`, in WORLD
/// units — the basis of error-driven refinement.
///
/// A brick stores, per voxel, the distance and the analytic normal, and the
/// renderer reconstructs the surface by blending per-corner tangent planes
/// `d_i + n_i·(p − corner_i)`. That reconstruction is **exact for planar
/// surfaces**, so the only error is the surface's deviation from planar over one
/// voxel. We measure it directly: at several seed points across the node, project
/// onto the surface and sample the true SDF one voxel away along the surface
/// tangents — points the tangent plane predicts to be *on* the surface (value 0).
/// Their actual `|SDF|` is the residual; we take the max over all seeds.
///
/// Properties that make it the right refinement driver:
/// - **planar / half-space → 0** (tangent-exact ⇒ the floor never over-refines);
/// - **curved → grows with sub-voxel detail** (sphere/torus refine where needed);
/// - **sharp edge → bounded by ~`voxel_size`** (the field is 1-Lipschitz, the
///   probe moves by `h`, so `|SDF| ≤ h`). This is the key difference from a
///   curvature term, which diverges on edges and caused runaway subdivision.
///
/// **Multi-seed** (the 8 octant centres) so a node holding several surfaces — a
/// flat floor *and* a sphere — reports the worst, not just the nearest.
pub fn reconstruction_residual(scene: &SdfScene, aabb: &AABB) -> f32 {
    let nodes = scene.get_nodes(aabb);
    if nodes.is_empty() { return 0.0; }

    let h    = (aabb.max.x - aabb.min.x) / BRICK_SIZE as f32; // voxel size at this LOD
    let min  = aabb.min;
    let ext  = aabb.max - aabb.min;
    let mut resid = 0.0f32;

    // Seed at the 8 octant centres (¼ / ¾ along each axis).
    for &fz in &[0.25f32, 0.75] {
    for &fy in &[0.25f32, 0.75] {
    for &fx in &[0.25f32, 0.75] {
        let seed = Point3::new(min.x + ext.x * fx, min.y + ext.y * fy, min.z + ext.z * fz);
        let s0 = SdfScene::sample(&nodes, &seed);

        // Skip octants whose surface is too far to lie in this octant.
        if s0.value.abs() > 0.5 * ext.x { continue; }

        let n = s0.gradient.try_normalize(1e-6).unwrap_or_else(Vector3::y);
        let surf = seed - n * s0.value;           // one Newton step onto the surface
        if surf.coords.zip_map(&min.coords, |a, b| a - b).iter().any(|&d| d < -h)
            || (aabb.max - surf).iter().any(|&d| d < -h)
        {
            continue;                              // surface point not within the node
        }

        // Two orthogonal tangents; |true SDF| one voxel out = local non-planarity.
        let a  = if n.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
        let t1 = n.cross(&a).normalize();
        let t2 = n.cross(&t1);
        resid = resid.max(SdfScene::sample(&nodes, &(surf + t1 * h)).value.abs());
        resid = resid.max(SdfScene::sample(&nodes, &(surf + t2 * h)).value.abs());
    }}}
    resid
}
