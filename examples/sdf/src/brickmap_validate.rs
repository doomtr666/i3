/// Bake a small test brickmap at startup and log timings.
#[cfg(debug_assertions)]
pub fn run() {
    use i3_brickmap::{BrickmapBaker, BrickmapData};
    use i3_math::Transform;
    use i3_math::nalgebra::{UnitQuaternion, Vector3};
    use i3_voxel::{SdfPrimitive, SdfScene};
    use tracing::info;

    let mut scene = SdfScene::new();
    scene.add(
        &Transform::new(Vector3::new(0.0, 1.2, 0.0), UnitQuaternion::identity(), 1.0),
        &SdfPrimitive::Sphere { radius: 1.0 },
    );
    scene.add(
        &Transform::new(Vector3::new(0.0, -0.5, 0.0), UnitQuaternion::identity(), 1.0),
        &SdfPrimitive::Box { half_extents: Vector3::new(5.0, 0.5, 5.0) },
    );
    scene.build_bvh();

    let baker = BrickmapBaker {
        grid_dims:        [32, 32, 32],
        world_origin:     [-12.8, -12.8, -12.8],
        voxel_size:       0.1,
        max_atlas_bricks: 32 * 32 * 32,
    };

    let t0 = std::time::Instant::now();
    let data: BrickmapData = baker.bake_all(&scene);
    info!(
        "Brickmap bake: {} / {} bricks in {:.2?}",
        data.brick_count, 32u32 * 32 * 32, t0.elapsed(),
    );
}

#[cfg(not(debug_assertions))]
pub fn run() {}
