use egui::{Event as EguiEvent, PointerButton, Pos2, Vec2, Key};
use i3_gfx::graph::backend::{Event as EngineEvent, KeyCode};

pub fn map_event(event: &EngineEvent) -> Option<EguiEvent> {
    match event {
        EngineEvent::MouseDown { button, x, y } => {
            Some(EguiEvent::PointerButton {
                pos: Pos2::new(*x as f32, *y as f32),
                button: match button {
                    1 => PointerButton::Primary,
                    2 => PointerButton::Secondary,
                    3 => PointerButton::Middle,
                    _ => PointerButton::Primary,
                },
                pressed: true,
                modifiers: Default::default(),
            })
        }
        EngineEvent::MouseUp { button, x, y } => {
            Some(EguiEvent::PointerButton {
                pos: Pos2::new(*x as f32, *y as f32),
                button: match button {
                    1 => PointerButton::Primary,
                    2 => PointerButton::Secondary,
                    3 => PointerButton::Middle,
                    _ => PointerButton::Primary,
                },
                pressed: false,
                modifiers: Default::default(),
            })
        }
        EngineEvent::MouseMove { x, y } => {
            Some(EguiEvent::PointerMoved(Pos2::new(*x as f32, *y as f32)))
        }
        EngineEvent::MouseWheel { x: dx, y: dy } => {
            Some(EguiEvent::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: Vec2::new(*dx as f32, *dy as f32),
                modifiers: Default::default(),
            })
        }
        EngineEvent::KeyDown { key } => {
            if let Some(egui_key) = map_key(*key) {
                Some(EguiEvent::Key {
                    key: egui_key,
                    physical_key: Some(egui_key),
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                })
            } else {
                None
            }
        }
        EngineEvent::KeyUp { key } => {
            if let Some(egui_key) = map_key(*key) {
                Some(EguiEvent::Key {
                    key: egui_key,
                    physical_key: Some(egui_key),
                    pressed: false,
                    repeat: false,
                    modifiers: Default::default(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

fn map_key(key: KeyCode) -> Option<Key> {
    match key {
        KeyCode::Escape => Some(Key::Escape),
        KeyCode::Tab => Some(Key::Tab),
        KeyCode::Space => Some(Key::Space),
        KeyCode::W => Some(Key::W),
        KeyCode::A => Some(Key::A),
        KeyCode::S => Some(Key::S),
        KeyCode::D => Some(Key::D),
        KeyCode::Z => Some(Key::Z),
        KeyCode::Q => Some(Key::Q),
        _ => None,
    }
}
