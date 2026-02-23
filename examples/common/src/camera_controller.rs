use i3_gfx::graph::backend::{Event, KeyCode};
use nalgebra_glm as glm;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Orbit,
    Free,
}

pub struct CameraController {
    pub mode: CameraMode,

    // Orbit parameters
    pub orbit_radius: f32,
    pub orbit_speed: f32,
    pub orbit_height: f32,
    pub orbit_time: f32,

    // Free mode parameters
    pub position: glm::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub move_speed: f32,
    pub look_sensitivity: f32,

    // Internal state
    pub keys_down: std::collections::HashSet<KeyCode>,
    pub last_mouse_pos: Option<(i32, i32)>,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            mode: CameraMode::Orbit,
            orbit_radius: 15.0,
            orbit_speed: 0.5,
            orbit_height: 5.0,
            orbit_time: 0.0,

            position: glm::vec3(0.0, 5.0, 15.0),
            yaw: -90.0f32.to_radians(),
            pitch: 0.0,
            move_speed: 10.0,
            look_sensitivity: 0.005,

            keys_down: std::collections::HashSet::new(),
            last_mouse_pos: None,
        }
    }

    pub fn handle_event(&mut self, event: &Event) {
        match event {
            Event::KeyDown { key } => {
                self.keys_down.insert(*key);
                if matches!(key, KeyCode::Tab) {
                    self.mode = match self.mode {
                        CameraMode::Orbit => CameraMode::Free,
                        CameraMode::Free => CameraMode::Orbit,
                    };
                    if self.mode == CameraMode::Free {
                        self.last_mouse_pos = None;
                    }
                }
            }
            Event::KeyUp { key } => {
                self.keys_down.remove(key);
            }
            Event::MouseMove { x, y } => {
                if self.mode == CameraMode::Free {
                    if let Some((old_x, old_y)) = self.last_mouse_pos {
                        let dx = x - old_x;
                        let dy = y - old_y;
                        self.yaw += dx as f32 * self.look_sensitivity;
                        self.pitch -= dy as f32 * self.look_sensitivity;
                        self.pitch = self.pitch.clamp(-1.5, 1.5);
                    }
                    self.last_mouse_pos = Some((*x, *y));
                }
            }
            _ => {}
        }
    }

    pub fn update(&mut self, dt: Duration) {
        let dt_sec = dt.as_secs_f32();

        match self.mode {
            CameraMode::Orbit => {
                self.orbit_time += dt_sec * self.orbit_speed;
            }
            CameraMode::Free => {
                let forward = glm::vec3(
                    self.yaw.cos() * self.pitch.cos(),
                    self.pitch.sin(),
                    self.yaw.sin() * self.pitch.cos(),
                );
                let right = glm::normalize(&glm::cross(&forward, &glm::vec3(0.0, 1.0, 0.0)));
                let up = glm::vec3(0.0, 1.0, 0.0);

                let mut velocity = glm::vec3(0.0, 0.0, 0.0);
                if self.keys_down.contains(&KeyCode::W) || self.keys_down.contains(&KeyCode::Z) {
                    velocity += forward;
                }
                if self.keys_down.contains(&KeyCode::S) {
                    velocity -= forward;
                }
                if self.keys_down.contains(&KeyCode::A) || self.keys_down.contains(&KeyCode::Q) {
                    velocity -= right;
                }
                if self.keys_down.contains(&KeyCode::D) {
                    velocity += right;
                }
                if self.keys_down.contains(&KeyCode::Space) {
                    velocity += up;
                }
                if self.keys_down.contains(&KeyCode::LShift) {
                    velocity -= up;
                }

                if glm::length(&velocity) > 0.0 {
                    self.position += glm::normalize(&velocity) * self.move_speed * dt_sec;
                }
            }
        }
    }

    pub fn view_matrix(&self) -> glm::Mat4 {
        match self.mode {
            CameraMode::Orbit => {
                let x = self.orbit_time.cos() * self.orbit_radius;
                let z = self.orbit_time.sin() * self.orbit_radius;
                let pos = glm::vec3(x, self.orbit_height, z);
                glm::look_at(&pos, &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 1.0, 0.0))
            }
            CameraMode::Free => {
                let forward = glm::vec3(
                    self.yaw.cos() * self.pitch.cos(),
                    self.pitch.sin(),
                    self.yaw.sin() * self.pitch.cos(),
                );
                glm::look_at(
                    &self.position,
                    &(self.position + forward),
                    &glm::vec3(0.0, 1.0, 0.0),
                )
            }
        }
    }
}
