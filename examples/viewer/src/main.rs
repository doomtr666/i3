extern crate nalgebra_glm;

use examples_common::basic_scene::BasicScene;
use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;
use i3_io::mesh::MeshAsset;
use i3_io::scene_asset::SceneAsset;
use i3_io::vfs::{BundleBackend, Vfs};
use i3_renderer::render_graph::{DefaultRenderGraph, RenderConfig};
use i3_vulkan_backend::VulkanBackend;
use nalgebra_glm as glm;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

struct DeferredGltfApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    scene: BasicScene,
    time: f32,
    dt: f32,
    camera: examples_common::camera_controller::CameraController,
}

impl ExampleApp for DeferredGltfApp {
    fn update(&mut self, delta: Duration) {
        self.dt = delta.as_secs_f32();
        self.time += self.dt;
        self.camera.update(delta);
    }

    fn render(&mut self) {
        let view = self.camera.view_matrix();
        let (width, height) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let scene_diag = self.scene.bounds().diagonal_length();
        let far = (scene_diag * 2.0).max(100.0);
        let near = 1.0; //(scene_diag * 0.001).max(0.01).min(0.1);

        let projection = glm::perspective_rh_zo(
            width as f32 / height as f32,
            std::f32::consts::FRAC_PI_4,
            near,
            far,
        );

        self.render_graph.sync(&mut self.backend, &self.scene);

        let mut graph = FrameGraph::new();
        self.render_graph.record(
            &mut graph,
            self.window,
            &self.scene,
            view,
            projection,
            near,
            far,
            width,
            height,
            self.dt,
        );

        let compiler = graph.compile();
        if let Err(e) = compiler.execute(
            &mut self.backend,
            Some(&mut self.render_graph.temporal_registry),
        ) {
            warn!("Graph execution failed: {}", e);
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        let events = self.backend.poll_events();
        for event in &events {
            self.camera.handle_event(event);
        }
        events
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing("viewer.log");
    info!("Starting Deferred glTF Demo (IO Backend)");

    // 1. Initialize Backend
    let mut backend = VulkanBackend::new()?;
    backend.initialize(0)?;

    // 2. Create Window
    let window = backend.create_window(WindowDesc {
        title: "Deferred glTF (Baked)".to_string(),
        width: 1280,
        height: 720,
    })?;

    // 3. Setup IO and VFS
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let assets_dir = Path::new(&manifest_dir).join("assets");
    let blob_path = assets_dir.join("viewer_scenes.i3b");
    let catalog_path = assets_dir.join("viewer_scenes.i3c");

    info!("Mounting bundle from {:?}", assets_dir);
    let bundle = BundleBackend::mount(&catalog_path, &blob_path)?;
    let mut vfs = Vfs::new();
    vfs.mount(Box::new(bundle));
    let vfs = Arc::new(vfs);
    let loader = AssetLoader::new(vfs);

    // 4. Load baked assets
    // Try to load Sponza, fallback to Helmet
    let scene_name = std::env::var("I3_SCENE").unwrap_or_else(|_| "Sponza_scene".to_string());
    info!("Loading SceneAsset '{}'...", scene_name);

    let scene_handle = loader.load::<SceneAsset>(&scene_name);
    let final_handle = if scene_handle.wait_loaded().is_ok() {
        scene_handle
    } else {
        warn!(
            "Failed to load {}, falling back to DamagedHelmet_scene",
            scene_name
        );
        loader.load::<SceneAsset>("DamagedHelmet_scene")
    };
    let scene_asset = final_handle.wait_loaded()?;

    let mut scene = BasicScene::new();

    // Load all meshes referenced by the scene
    for mesh_uuid in &scene_asset.mesh_refs {
        let mesh_handle = loader.load_by_uuid::<MeshAsset>(mesh_uuid)?;
        let mesh_asset = mesh_handle.wait_loaded()?;
        scene.add_baked_mesh(&mut backend, &mesh_asset, *mesh_uuid);
    }

    // Populate scene objects/lights
    scene.load_baked_scene(&scene_asset);

    if scene_asset.mesh_refs.is_empty() {
        warn!("Scene contains no meshes!");
    }

    scene.add_default_lights();

    // 5. Create Render Graph
    let config = RenderConfig {
        width: 1280,
        height: 720,
    };
    let render_graph = DefaultRenderGraph::new(&mut backend, &config);

    // 6. Run
    let mut camera = examples_common::camera_controller::CameraController::new();

    // Adjust camera for scene scale
    let scene_diag = scene.bounds().diagonal_length();
    camera.move_speed = (scene_diag * 0.2).max(1.0);

    if scene_name.contains("Sponza") {
        camera.position = glm::vec3(0.0, 2.0, 0.0);
    } else {
        camera.position = glm::vec3(0.0, 0.0, (scene_diag * 0.8).max(3.0));
    }

    let app = DeferredGltfApp {
        backend,
        window,
        render_graph,
        scene,
        time: 0.0,
        dt: 0.016,
        camera,
    };
    main_loop(app);

    Ok(())
}
