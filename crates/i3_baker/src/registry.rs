use crate::Result;
use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};
use std::path::Path;

pub struct Registry {
    conn: Connection,
}

impl Registry {
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(db_path)?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                source_path TEXT,
                source_hash TEXT,
                baker_version INTEGER,
                last_bake_time INTEGER
            )",
            [],
        )?;

        Ok(Self { conn })
    }

    pub fn should_rebake(&self, asset_id: &str, source_path: &Path) -> bool {
        let mut stmt = match self
            .conn
            .prepare("SELECT source_hash FROM assets WHERE id = ?")
        {
            Ok(s) => s,
            Err(_) => return true,
        };

        let current_hash = match self.calculate_hash(source_path) {
            Ok(h) => h,
            Err(_) => return true,
        };

        let stored_hash: Option<String> = stmt.query_row(params![asset_id], |row| row.get(0)).ok();

        match stored_hash {
            Some(h) => h != current_hash,
            None => true,
        }
    }

    pub fn update_asset(&self, asset_id: &str, source_path: &Path) -> Result<()> {
        let hash = self.calculate_hash(source_path)?;
        self.conn.execute(
            "INSERT OR REPLACE INTO assets (id, source_path, source_hash, baker_version, last_bake_time)
             VALUES (?, ?, ?, ?, ?)",
            params![
                asset_id,
                source_path.to_string_lossy(),
                hash,
                1, // Version 1
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
            ],
        )?;
        Ok(())
    }

    fn calculate_hash(&self, path: &Path) -> Result<String> {
        let mut file = std::fs::File::open(path).map_err(|e| crate::error::BakerError::Os {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher).map_err(|e| crate::error::BakerError::Os {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(format!("{:x}", hasher.finalize()))
    }
}
