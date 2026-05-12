use std::f32::consts::PI;

/// Decode hemi-octa UV [0,1]² → direction sur la sphère unité.
/// L'espace [0, 0.5] en V = hémisphère nord (y >= 0).
/// L'espace [0.5, 1] en V = hémisphère sud (y < 0).
pub fn hemi_octa_decode(u: f32, v: f32) -> [f32; 3] {
    // Remappe V dans [0,1] selon hémisphère
    let (vv, sign_y) = if v < 0.5 { (v * 2.0, 1.0f32) } else { ((v - 0.5) * 2.0, -1.0f32) };
    let fx = u * 2.0 - 1.0;
    let fz = vv * 2.0 - 1.0;
    let fy = 1.0 - fx.abs() - fz.abs();
    let len = (fx * fx + fy * fy + fz * fz).sqrt().max(1e-8);
    [fx / len, sign_y * fy.abs() / len, fz / len]
}

/// Encode direction sphère → hemi-octa UV [0,1]².
pub fn hemi_octa_encode(d: [f32; 3]) -> (f32, f32) {
    let [x, y, z] = d;
    let l = x.abs() + y.abs() + z.abs();
    let ox = x / l;
    let oz = z / l;
    let u = ox * 0.5 + 0.5;
    let v_local = oz * 0.5 + 0.5;
    // y==0 with z>0: use lower section (v→1.0) — upper would give v=0.5
    // (the seam where decode returns (0,0,-1) instead of (0,0,+1)).
    let lower = y < 0.0 || (y == 0.0 && z > 0.0);
    let v = if lower { 0.5 + v_local * 0.5 } else { v_local * 0.5 };
    (u, v)
}

/// Pixel equirect (ix, iy) → direction sphère unité.
pub fn equirect_to_dir(ix: u32, iy: u32, width: u32, height: u32) -> [f32; 3] {
    let u = (ix as f32 + 0.5) / width as f32;
    let v = (iy as f32 + 0.5) / height as f32;
    let phi = u * 2.0 * PI - PI;       // [-π, π]
    let theta = v * PI;                  // [0, π]
    let sin_theta = theta.sin();
    [sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin()]
}

/// Solid angle d'un pixel equirect (correction cos(lat)).
pub fn equirect_solid_angle(iy: u32, height: u32) -> f32 {
    let theta = (iy as f32 + 0.5) / height as f32 * PI;
    (2.0 * PI / (height as f32)) * (PI / height as f32) * theta.sin()
}

/// Luminance perceptuelle d'un pixel RGB linéaire.
pub fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hemi_octa() {
        // Zénith Nord (V=0.25 car c'est le milieu de [0, 0.5])
        let d = hemi_octa_decode(0.5, 0.25);
        assert!((d[0].abs() < 1e-6));
        assert!((d[1] - 1.0).abs() < 1e-6);
        assert!((d[2].abs() < 1e-6));

        // Zénith Sud (V=0.75 car c'est le milieu de [0.5, 1])
        let d = hemi_octa_decode(0.5, 0.75);
        assert!((d[0].abs() < 1e-6));
        assert!((d[1] + 1.0).abs() < 1e-6);
        assert!((d[2].abs() < 1e-6));
    }
}
