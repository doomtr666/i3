use crate::svo::SvoTree;

// ─── SvoParams ────────────────────────────────────────────────────────────────

pub struct SvoParams {
    pub lod_threshold: f32,
    pub sdf_weight: f32,
    pub split_budget: u32,
    pub merge_budget: u32,
    pub debug_flags: u32,
}

impl Default for SvoParams {
    fn default() -> Self {
        Self {
            lod_threshold: 0.18,
            sdf_weight: 1.0,
            split_budget: 256,
            merge_budget: 128,
            debug_flags: 0,
        }
    }
}

// ─── SvoDebugUi ───────────────────────────────────────────────────────────────

pub struct SvoDebugUi;

impl SvoDebugUi {
    pub fn show(ui: &mut egui::Ui, tree: &SvoTree, params: &mut SvoParams) {
        let s = tree.stats();

        ui.collapsing("SVO", |ui| {
            // ── Occupancy (red when near cap = starvation) ────────────────────
            let node_pct  = s.nodes_live  as f32 / s.nodes_cap.max(1)  as f32;
            let brick_pct = s.bricks_used as f32 / s.bricks_cap.max(1) as f32;
            let col = |p: f32| if p > 0.9 { egui::Color32::RED }
                               else if p > 0.7 { egui::Color32::YELLOW }
                               else { egui::Color32::GRAY };
            ui.colored_label(col(node_pct),
                format!("Nodes:  {}/{} ({:.0}%)", s.nodes_live, s.nodes_cap, node_pct * 100.0));
            ui.colored_label(col(brick_pct),
                format!("Bricks: {}/{} ({:.0}%)", s.bricks_used, s.bricks_cap, brick_pct * 100.0));

            // ── This-frame mutation + starvation ──────────────────────────────
            ui.label(format!(
                "frame: splits {}/{} (wanted {})  merges {}  bakes {}  culls {}",
                s.splits, s.split_budget, s.split_wanted, s.merges, s.bakes, s.culls,
            ));
            if s.split_wanted > s.split_budget {
                ui.colored_label(egui::Color32::RED,
                    format!("⚠ split STARVED: {} wanted, {} budget", s.split_wanted, s.split_budget));
            }
            if s.node_cap_hit  { ui.colored_label(egui::Color32::RED, "⚠ NODE pool full"); }
            if s.brick_cap_hit { ui.colored_label(egui::Color32::RED, "⚠ BRICK atlas full"); }

            // ── Per-depth distribution (nodes / bricked) ──────────────────────
            ui.collapsing("per-depth", |ui| {
                for d in 0..16usize {
                    if s.per_depth[d] == 0 { continue; }
                    ui.label(egui::RichText::new(format!(
                        "  d{d:<2} nodes {:<5} bricks {}", s.per_depth[d], s.per_depth_bricked[d]
                    )).monospace());
                }
            });

            ui.add(egui::Slider::new(&mut params.lod_threshold, 0.05..=10.0).text("LOD threshold"));
            ui.add(egui::Slider::new(&mut params.sdf_weight, 0.0..=8.0).text("Curve detail (↑ = finer)"));
            ui.add(egui::Slider::new(&mut params.split_budget, 1..=1024).text("Split budget/frame"));
            ui.add(egui::Slider::new(&mut params.merge_budget, 1..=256).text("Merge budget/frame"));

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
            flag(ui, 0, "Debug: SVO depth colors");
            flag(ui, 1, "Debug: node AABBs");
            flag(ui, 2, "Debug: error heatmap");
            flag(ui, 3, "Debug: step count heat");
            ui.separator();
            flag(ui, 4, "★ GROUND TRUTH (analytic, no tree/atlas)");
            flag(ui, 5, "★ TRAVERSAL+analytic (tree, no atlas)");
            flag(ui, 6, "★ BRICK ERROR heat (red = wrong brick)");
        });
    }
}
