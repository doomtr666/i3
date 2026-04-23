use bytemuck::{Pod, Zeroable};

/// Per-frame uniform buffer content — uploaded once per frame, bound at set 1, binding 0.
///
/// Replaces the many duplicated push-constant fields (`view`, `inv_projection`,
/// `screen_size`, `frame_index`, …) that were scattered across ~12 pass structs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuCommonData {
    pub view:                 [[f32; 4]; 4],
    pub projection:           [[f32; 4]; 4],
    pub view_projection:      [[f32; 4]; 4],
    pub inv_projection:       [[f32; 4]; 4],
    pub inv_view_projection:  [[f32; 4]; 4],
    pub prev_view_projection: [[f32; 4]; 4],

    /// .xyz = camera_pos, .w = near_plane
    pub camera_pos:           [f32; 4],
    /// .xyz = sun_dir, .w = sun_intensity
    pub sun_dir:              [f32; 4],
    /// .xyz = sun_color, .w = far_plane
    pub sun_color:            [f32; 4],

    /// .x = screen_width, .y = screen_height, .z = light_count, .w = frame_index
    pub screen_size:          [u32; 4],
    /// .x = ibl_lut_index, .y = ibl_irr_index, .z = ibl_pref_index, .w = ibl_env_index
    pub ibl_indices:          [u32; 4],
    /// .x = blue_noise_index, .y = debug_channel, .z = unused, .w = unused
    pub extra_indices:        [u32; 4],

    /// .x = dt, .y = ibl_intensity, .z = unused, .w = unused
    pub time_params:          [f32; 4],
}

