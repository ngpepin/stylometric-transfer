//! GUID-addressed fingerprint store.
//!
//! Standalone equivalent of `save_fingerprint_to_store` /
//! `load_fingerprint_from_store` in `common.py`. Artifacts live in a
//! `fingerprint_store/` directory resolved at runtime (never at compile time),
//! so the built binary is relocatable and has no build-machine paths baked in.

use std::path::{Path, PathBuf};

/// Default store directory name.
pub const STORE_DIR: &str = "fingerprint_store";

/// Resolve the fingerprint store directory, creating it when missing.
///
/// Resolution order is runtime-only, so no build path is embedded in the
/// binary: an explicit `base` wins, then `./fingerprint_store`, then
/// `<exe-dir>/fingerprint_store`, then the system temp directory.
pub fn store_dir(base: Option<&Path>) -> PathBuf {
    let dir = match base {
        Some(path) => path.to_path_buf(),
        None => {
            let cwd = std::env::current_dir().unwrap_or_default().join(STORE_DIR);
            if std::fs::create_dir_all(&cwd).is_ok() {
                cwd
            } else {
                let beside_exe = std::env::current_exe()
                    .ok()
                    .and_then(|exe| exe.parent().map(|dir| dir.join(STORE_DIR)))
                    .unwrap_or_else(|| std::env::temp_dir().join(STORE_DIR));
                let _ = std::fs::create_dir_all(&beside_exe);
                beside_exe
            }
        }
    };
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Reject GUIDs that could escape the store directory.
pub fn is_valid_guid(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && value.len() <= 64
}

/// Path of the fingerprint JSON for `guid`.
pub fn fingerprint_path(dir: &Path, guid: &str) -> PathBuf {
    dir.join(format!("{guid}.fingerprint.json"))
}

/// Path of the metadata JSON for `guid`.
pub fn meta_path(dir: &Path, guid: &str) -> PathBuf {
    dir.join(format!("{guid}.meta.json"))
}

/// Persist a fingerprint plus metadata; returns the GUID.
pub fn save(
    dir: &Path,
    fingerprint: &serde_json::Value,
    meta: &serde_json::Value,
) -> Result<String, String> {
    let guid = uuid::Uuid::new_v4().to_string();
    let fp_path = fingerprint_path(dir, &guid);
    let meta_path = meta_path(dir, &guid);
    std::fs::write(
        &fp_path,
        serde_json::to_string_pretty(fingerprint).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("cannot write {}: {e}", fp_path.display()))?;
    std::fs::write(
        &meta_path,
        serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("cannot write {}: {e}", meta_path.display()))?;
    Ok(guid)
}

/// Load a stored fingerprint by GUID.
pub fn load(dir: &Path, guid: &str) -> Result<serde_json::Value, String> {
    if !is_valid_guid(guid) {
        return Err("invalid GUID".to_string());
    }
    let path = fingerprint_path(dir, guid);
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("invalid fingerprint JSON: {e}"))
}
