use std::collections::VecDeque;

use i3_egui::prelude::*;
use i3_renderer::passes::debug_viz::DebugChannel;
use i3_renderer::prelude::*;
use i3_renderer::render_graph::AoMode;

use crate::camera_controller::CameraController;

// ─── RendererDebugGui ─────────────────────────────────────────────────────────

/// Shared "Engine Debug" panel usable by any sample.
///
/// In your app's `render()`:
/// ```ignore
/// self.debug_gui.show(&egui_ctx, &mut self.render_graph, &self.camera, self.smoothed_dt, |ui| {
///     // sample-specific controls here
/// });
/// ```
pub struct RendererDebugGui {
    // Perf graph
    frame_time_history: VecDeque<f32>,
    show_perf_graph:    bool,
    sample_accum_time:  f32,
    sample_max_dt:      f32,

    // Culling debug (accessed by caller for debug draw)
    pub show_culling_debug: bool,
    pub culling_show_ids:   bool,
}

impl Default for RendererDebugGui {
    fn default() -> Self {
        Self {
            frame_time_history: VecDeque::with_capacity(256),
            show_perf_graph:    false,
            sample_accum_time:  0.0,
            sample_max_dt:      0.0,
            show_culling_debug: false,
            culling_show_ids:   false,
        }
    }
}

impl RendererDebugGui {
    pub fn new() -> Self {
        Self::default()
    }

    /// Call once per frame from `update()` with the raw delta in seconds.
    pub fn update(&mut self, dt: f32) {
        self.sample_accum_time += dt;
        self.sample_max_dt = self.sample_max_dt.max(dt);

        // Sample at ~50 Hz → 256 samples ≈ 5 s of history
        if self.sample_accum_time >= 0.020 {
            self.frame_time_history.push_back(self.sample_max_dt);
            if self.frame_time_history.len() > 256 {
                self.frame_time_history.pop_front();
            }
            self.sample_accum_time = 0.0;
            self.sample_max_dt = 0.0;
        }
    }

    /// Draw the "Engine Debug" window (and the optional perf graph window).
    ///
    /// `extra` is called at the bottom of the window — use it for sample-specific controls.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        render_graph: &mut DefaultRenderGraph,
        camera: &CameraController,
        smoothed_dt: f32,
        extra: impl FnOnce(&mut egui::Ui),
    ) {
        // Pull mutable booleans into locals so we can still access self later.
        let mut show_perf       = self.show_perf_graph;
        let mut show_culling    = self.show_culling_debug;
        let mut culling_ids     = self.culling_show_ids;

        egui::Window::new("Engine Debug").show(ctx, |ui| {
            ui.heading("Renderer");
            let fps = if smoothed_dt > 0.0 { 1.0 / smoothed_dt } else { 0.0 };
            ui.label(format!("Frame time: {:.2}ms ({:.1} FPS)", smoothed_dt * 1000.0, fps));
            ui.checkbox(&mut show_perf, "Show Performance Graph");
            ui.separator();

            if camera.camera_locked {
                ui.label("Camera: LOCKED (Tab to unlock)");
            } else {
                ui.label("Camera: FREE (Tab to lock)");
            }
            ui.separator();

            // ── FXAA ──────────────────────────────────────────────────────────
            ui.checkbox(&mut render_graph.fxaa_enabled, "FXAA");

            // ── AO ────────────────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label("AO:");
                ui.radio_value(&mut render_graph.ao_mode, AoMode::None, "None");
                ui.radio_value(&mut render_graph.ao_mode, AoMode::Gtao, "GTAO");
                ui.radio_value(&mut render_graph.ao_mode, AoMode::Rtao, "RTAO");
            });
            match render_graph.ao_mode {
                AoMode::Gtao => {
                    let gtao = &mut render_graph.gtao_group.gtao_pass;
                    ui.add(egui::Slider::new(&mut gtao.radius,      0.1_f32..=2.0).text("AO Radius"));
                    ui.add(egui::Slider::new(&mut gtao.final_power, 0.5_f32..=4.0).text("AO Power"));
                    ui.add(egui::Slider::new(&mut gtao.slice_count, 1_u32..=4).text("AO Slices"));
                    ui.add(egui::Slider::new(&mut gtao.step_count,  2_u32..=8).text("AO Steps"));
                    let alpha = &mut render_graph.gtao_group.gtao_temporal_pass.alpha;
                    ui.add(egui::Slider::new(alpha, 0.01_f32..=1.0).text("AO Temporal α").logarithmic(true));
                }
                AoMode::Rtao => {
                    let rtao = &mut render_graph.rtao_group.rtao_pass;
                    ui.add(egui::Slider::new(&mut rtao.radius, 0.1_f32..=5.0).text("RTAO Radius"));
                    let t = &mut render_graph.rtao_group.rtao_temporal_pass;
                    ui.add(egui::Slider::new(&mut t.alpha, 0.01_f32..=0.5).text("RTAO Temporal α"));
                }
                AoMode::None => {}
            }

            // ── SSR ───────────────────────────────────────────────────────────
            ui.separator();
            {
                let ssr = &mut render_graph.sssr_sample_pass;
                ui.checkbox(&mut ssr.enabled, "SSR");
                if ssr.enabled {
                    ui.add(egui::Slider::new(&mut ssr.max_mip_level, 1_u32..=5).text("SSR Max Mip"));
                    ui.add(egui::Slider::new(&mut ssr.thickness, 0.01_f32..=1.0).text("SSR Thickness"));
                    let intensity = &mut render_graph.sssr_composite_pass.intensity;
                    ui.add(egui::Slider::new(intensity, 0.0_f32..=2.0).text("SSR Intensity"));
                    let dbg = &mut render_graph.sssr_sample_pass.debug_mode;
                    ui.horizontal(|ui| {
                        ui.label("SSR debug:");
                        ui.selectable_value(dbg, 0, "Off");
                        ui.selectable_value(dbg, 1, "hit/iter/thick");
                        ui.selectable_value(dbg, 2, "hit_uv/hit");
                    });
                    let mut factor = render_graph.sssr_sample_pass.downsample_factor;
                    if ui.add(egui::Slider::new(&mut factor, 1_u32..=4).text("SSR Downsample")).changed() {
                        render_graph.sssr_sample_pass.downsample_factor = factor;
                        render_graph.sssr_bilateral_pass.downsample_factor = factor;
                    }
                }
            }

            // ── Bloom ─────────────────────────────────────────────────────────
            ui.separator();
            {
                let bloom = &mut render_graph.bloom_pass;
                ui.checkbox(&mut bloom.enabled, "Bloom");
                if bloom.enabled {
                    ui.add(egui::Slider::new(&mut bloom.threshold, 0.5_f32..=4.0).text("Bloom Threshold"));
                    ui.add(egui::Slider::new(&mut bloom.knee,      0.0_f32..=1.0).text("Bloom Knee"));
                    ui.add(egui::Slider::new(&mut bloom.intensity, 0.0_f32..=1.0).logarithmic(true).text("Bloom Intensity"));
                }
            }

            // ── Culling debug ─────────────────────────────────────────────────
            ui.separator();
            ui.label("Culling Debug:");
            ui.checkbox(&mut show_culling, "Show bounding boxes");
            if show_culling {
                ui.checkbox(&mut culling_ids, "  Instance IDs");
            }

            // ── Debug channel ─────────────────────────────────────────────────
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Debug Channel:");
                let label = debug_channel_label(render_graph.debug_channel);
                egui::ComboBox::from_id_salt("debug_channel")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        for (ch, name) in DEBUG_CHANNELS {
                            ui.selectable_value(&mut render_graph.debug_channel, *ch, *name);
                        }
                    });
            });

            // ── Sample-specific controls ───────────────────────────────────────
            extra(ui);
        });

        // Write booleans back
        self.show_perf_graph    = show_perf;
        self.show_culling_debug = show_culling;
        self.culling_show_ids   = culling_ids;

        // Performance graph (separate floating window)
        if self.show_perf_graph {
            let history = &self.frame_time_history;
            egui::Window::new("Performance Graph").show(ctx, |ui| {
                draw_perf_graph(history, ui);
            });
        }
    }
}

// ─── Debug channel table ──────────────────────────────────────────────────────

static DEBUG_CHANNELS: &[(DebugChannel, &str)] = &[
    (DebugChannel::Lit,          "Lit (Final)"),
    (DebugChannel::Albedo,       "Albedo"),
    (DebugChannel::Normal,       "Normals"),
    (DebugChannel::Roughness,    "Roughness"),
    (DebugChannel::Metallic,     "Metallic"),
    (DebugChannel::Emissive,     "Emissive"),
    (DebugChannel::Depth,        "Depth"),
    (DebugChannel::AO,           "AO (accumulated)"),
    (DebugChannel::SsrRaw,       "SSR (raw stochastic)"),
    (DebugChannel::SsrUpsampled, "SSR (upsampled)"),
    (DebugChannel::BloomBuffer,  "Bloom buffer"),
    (DebugChannel::LightDensity, "Light density"),
    (DebugChannel::ClusterGrid,  "Cluster grid"),
];

fn debug_channel_label(ch: DebugChannel) -> &'static str {
    DEBUG_CHANNELS
        .iter()
        .find(|(c, _)| *c == ch)
        .map(|(_, n)| *n)
        .unwrap_or("?")
}

// ─── Performance graph ────────────────────────────────────────────────────────

fn draw_perf_graph(history: &VecDeque<f32>, ui: &mut egui::Ui) {
    use egui::{Color32, Pos2, RichText, Stroke};

    const GRAPH_W: f32 = 320.0;
    const GRAPH_H: f32 = 80.0;
    const MAX_SAMPLES: usize = 256;

    if history.is_empty() {
        return;
    }

    // ── Frame time (ms) ───────────────────────────────────────────────────────
    ui.label(RichText::new("Frame Time (ms)").strong().color(Color32::LIGHT_BLUE));
    let (rect, _) = ui.allocate_at_least(egui::vec2(GRAPH_W, GRAPH_H), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, Color32::from_black_alpha(180));

    let mut actual_max_ms = 0.1_f32;
    for &dt in history {
        actual_max_ms = actual_max_ms.max(dt * 1000.0);
    }
    let max_ms = nice_ms_ceil(actual_max_ms);

    hline_ms(&painter, rect, GRAPH_H, max_ms, 16.6, Color32::GREEN,            "16.6ms");
    hline_ms(&painter, rect, GRAPH_H, max_ms, 33.3, Color32::GOLD,             "33.3ms");
    if max_ms < 5.0 {
        hline_ms(&painter, rect, GRAPH_H, max_ms, 1.0, Color32::from_rgb(100,100,255), "1.0ms");
        hline_ms(&painter, rect, GRAPH_H, max_ms, 0.5, Color32::from_rgb(150,150,255), "0.5ms");
    }

    let pts: Vec<Pos2> = history.iter().enumerate().map(|(i, &dt)| {
        let x = rect.left() + (i as f32 / (MAX_SAMPLES - 1) as f32) * GRAPH_W;
        let y = rect.bottom() - (dt * 1000.0 / max_ms) * GRAPH_H;
        Pos2::new(x, y.clamp(rect.top(), rect.bottom()))
    }).collect();
    if pts.len() > 1 {
        painter.add(egui::Shape::line(pts, Stroke::new(1.0, Color32::LIGHT_BLUE)));
    }
    painter.text(Pos2::new(rect.left()+5.0, rect.top()+5.0), egui::Align2::LEFT_TOP,
        format!("max: {:.2}ms", actual_max_ms), egui::FontId::monospace(10.0), Color32::WHITE);

    ui.add_space(8.0);

    // ── FPS ───────────────────────────────────────────────────────────────────
    ui.label(RichText::new("FPS").strong().color(Color32::LIGHT_YELLOW));
    let (rect, _) = ui.allocate_at_least(egui::vec2(GRAPH_W, GRAPH_H), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, Color32::from_black_alpha(180));

    let mut actual_max_fps = 60.0_f32;
    for &dt in history {
        if dt > 0.0 { actual_max_fps = actual_max_fps.max(1.0 / dt); }
    }
    let max_fps = nice_fps_ceil(actual_max_fps);

    hline_fps(&painter, rect, GRAPH_H, max_fps,  60.0, Color32::GREEN,              "60");
    hline_fps(&painter, rect, GRAPH_H, max_fps, 144.0, Color32::from_rgb(0,200,200), "144");
    if max_fps > 1000.0 { hline_fps(&painter, rect, GRAPH_H, max_fps, 1000.0, Color32::from_rgb(200,200,0), "1k"); }
    if max_fps > 2500.0 { hline_fps(&painter, rect, GRAPH_H, max_fps, 2500.0, Color32::from_rgb(200,100,0), "2.5k"); }

    let pts: Vec<Pos2> = history.iter().enumerate().map(|(i, &dt)| {
        let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        let x = rect.left() + (i as f32 / (MAX_SAMPLES - 1) as f32) * GRAPH_W;
        let y = rect.bottom() - (fps / max_fps) * GRAPH_H;
        Pos2::new(x, y.clamp(rect.top(), rect.bottom()))
    }).collect();
    if pts.len() > 1 {
        painter.add(egui::Shape::line(pts, Stroke::new(1.0, Color32::LIGHT_YELLOW)));
    }
    painter.text(Pos2::new(rect.left()+5.0, rect.top()+5.0), egui::Align2::LEFT_TOP,
        format!("max: {:.0} FPS", actual_max_fps), egui::FontId::monospace(10.0), Color32::WHITE);

    ui.add_space(4.0);
    if let Some(&last) = history.back() {
        let fps = if last > 0.0 { 1.0 / last } else { 0.0 };
        ui.label(format!("current: {:.2}ms ({:.0} FPS)", last * 1000.0, fps));
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn hline_ms(painter: &egui::Painter, rect: egui::Rect, h: f32, max: f32, val: f32, color: egui::Color32, label: &str) {
    let y = rect.bottom() - (val / max) * h;
    if y > rect.top() && y <= rect.bottom() {
        painter.line_segment([egui::Pos2::new(rect.left(), y), egui::Pos2::new(rect.right(), y)],
            egui::Stroke::new(1.0, color.linear_multiply(0.5)));
        painter.text(egui::Pos2::new(rect.right()-5.0, y-2.0), egui::Align2::RIGHT_BOTTOM,
            label, egui::FontId::monospace(9.0), color);
    }
}

fn hline_fps(painter: &egui::Painter, rect: egui::Rect, h: f32, max: f32, val: f32, color: egui::Color32, label: &str) {
    let y = rect.bottom() - (val / max) * h;
    if y > rect.top() && y <= rect.bottom() {
        painter.line_segment([egui::Pos2::new(rect.left(), y), egui::Pos2::new(rect.right(), y)],
            egui::Stroke::new(1.0, color.linear_multiply(0.5)));
        painter.text(egui::Pos2::new(rect.right()-5.0, y-2.0), egui::Align2::RIGHT_BOTTOM,
            label, egui::FontId::monospace(9.0), color);
    }
}

fn nice_ms_ceil(v: f32) -> f32 {
    if v < 1.0 { 1.0 } else if v < 2.0 { 2.0 } else if v < 5.0 { 5.0 }
    else if v < 10.0 { 10.0 } else if v < 20.0 { 20.0 } else if v < 50.0 { 50.0 }
    else { (v / 10.0).ceil() * 10.0 }
}

fn nice_fps_ceil(v: f32) -> f32 {
    if v < 150.0 { 150.0 } else if v < 300.0 { 300.0 } else if v < 1000.0 { 1000.0 }
    else if v < 2000.0 { 2000.0 } else if v < 5000.0 { 5000.0 }
    else { (v / 1000.0).ceil() * 1000.0 }
}
