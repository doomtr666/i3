use std::sync::Once;

static INIT: Once = Once::new();

pub fn init_test_tracing() {
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("info")
            .with_test_writer()
            .with_thread_names(true) // Identify which test produced the log
            .try_init();
    });
}
