#[path = "../../third_party/build-support/mod.rs"]
mod third_party_build_support;

fn main() {
    #[cfg(target_os = "windows")]
    {
        // Use the existing third_party build support to copy SDL2.dll
        let _ = third_party_build_support::setup_native_lib("sdl2", &["SDL2"]);
    }
}
