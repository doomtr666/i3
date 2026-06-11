use crate::clipmap::{ClipmapState, NUM_LEVELS, LEVEL_VOXEL_SIZES};

// ─── ClipmapParams ────────────────────────────────────────────────────────────────
// Clipmap has no per-frame LOD knobs (the cascade is fixed); only the debug flags
// remain. Kept as a small struct so the app's UI plumbing is unchanged.

#[derive(Default)]
pub struct ClipmapParams {
    pub debug_flags: u32,
}

// ─── ClipmapDebugUi ───────────────────────────────────────────────────────────────

pub struct ClipmapDebugUi;

impl ClipmapDebugUi {
    pub fn show(ui: &mut egui::Ui, clip: &ClipmapState, params: &mut ClipmapParams) {
        let s = clip.stats();

        ui.collapsing("Clipmap", |ui| {
            ui.label(format!(
                "frame: {} bakes  {} cells re-evaluated",
                s.bakes, s.cells_changed,
            ));
            if s.level_overflow {
                ui.colored_label(egui::Color32::RED, "⚠ a level ran out of brick slots (holes)");
            }

            // Per-level slot occupancy (red as it approaches the cap = potential holes).
            ui.collapsing("per-level bricks", |ui| {
                for l in 0..NUM_LEVELS {
                    let used = s.bricks_used[l];
                    let pct = used as f32 / s.bricks_cap.max(1) as f32;
                    let col = if pct > 0.9 { egui::Color32::RED }
                              else if pct > 0.7 { egui::Color32::YELLOW }
                              else { egui::Color32::GRAY };
                    ui.colored_label(col, egui::RichText::new(format!(
                        "  L{l} vs {:>5.2}m  {:>5}/{} ({:.0}%)",
                        LEVEL_VOXEL_SIZES[l], used, s.bricks_cap, pct * 100.0
                    )).monospace());
                }
            });

            ui.separator();

            let mut flag = |ui: &mut egui::Ui, bit: u32, label: &str| {
                let mut v = params.debug_flags & (1 << bit) != 0;
                ui.checkbox(&mut v, label);
                if v {
                    params.debug_flags |= 1 << bit;
                } else {
                    params.debug_flags &= !(1 << bit);
                }
            };
            flag(ui, 0, "Debug: level/voxel-size colors");
            flag(ui, 1, "Debug: brick grid");
            flag(ui, 2, "Debug: coarseness heatmap");
            flag(ui, 3, "Debug: step count heat");
            ui.separator();
            flag(ui, 7, "Perf: normal map OFF");
            flag(ui, 8, "Perf: terrain texturing OFF (flat)");
        });
    }
}
