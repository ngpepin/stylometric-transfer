//! Style compliance scoring and fingerprint similarity.
//!
//! Ports `compute_style_compliance` / `l1_distance` / `relative_diff` from
//! `apply_fingerprint.py` and the calibrated-probability plus similarity
//! helpers from `common.py`. Everything is deterministic and interpretable:
//! the outputs are plain JSON with the same shape the Python version emits.

use crate::measurements::compute_measurements;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

/// Clamp into `[0, 1]`.
pub fn clamp01(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// Numerically stable logistic function.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Sum of absolute differences; 0.0 on length mismatch (Python behaviour).
pub fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Relative difference with a floor of 1.0 on the denominator.
pub fn relative_diff(a: f64, b: f64) -> f64 {
    let denom = b.abs().max(1.0);
    (a - b).abs() / denom
}

/// Fetch a nested numeric value by key path.
fn num_at(value: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = value;
    for key in path {
        cur = cur.get(*key)?;
    }
    cur.as_f64()
}

/// Fetch a nested numeric array by key path.
fn nums_at(value: &Value, path: &[&str]) -> Option<Vec<f64>> {
    let mut cur = value;
    for key in path {
        cur = cur.get(*key)?;
    }
    let arr = cur.as_array()?;
    Some(arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
}

/// Collect accumulated per-section scores plus the deviation list.
struct Scores {
    sections: BTreeMap<String, Vec<f64>>,
    deltas: Vec<Value>,
}

impl Scores {
    fn new() -> Self {
        Self {
            sections: BTreeMap::new(),
            deltas: Vec::new(),
        }
    }

    fn add(&mut self, section: &str, value: f64) {
        self.sections
            .entry(section.to_string())
            .or_default()
            .push(value);
    }

    fn delta(&mut self, metric: &str, diff: f64) {
        self.deltas.push(json!({"metric": metric, "diff": diff}));
    }

    /// Score `1 - min(1, avg(relative_diff))` over shared numeric keys.
    fn numeric_section(
        &mut self,
        fp: &Value,
        out: &Value,
        section: &str,
        prefix: &str,
        delta_threshold: f64,
    ) {
        let fp_obj = match fp.as_object() {
            Some(map) => map,
            None => return,
        };
        let out_obj = match out.as_object() {
            Some(map) => map,
            None => return,
        };
        let mut diffs: Vec<f64> = Vec::new();
        for (key, target) in fp_obj {
            let target = match target.as_f64() {
                Some(v) => v,
                None => continue,
            };
            let actual = match out_obj.get(key).and_then(Value::as_f64) {
                Some(v) => v,
                None => continue,
            };
            let diff = relative_diff(actual, target);
            diffs.push(diff);
            if diff > delta_threshold {
                self.delta(&format!("{prefix}.{key}"), diff);
            }
        }
        if !diffs.is_empty() {
            let avg = diffs.iter().sum::<f64>() / diffs.len() as f64;
            self.add(section, (1.0 - avg.min(1.0)).max(0.0));
        }
    }
}
/// Score a vector-histogram section (sentence / paragraph length).
fn score_histogram(
    scores: &mut Scores,
    fp: &Value,
    out: &Value,
    path: &[&str],
    section: &str,
    metric: &str,
) {
    let (a, b) = match (nums_at(fp, path), nums_at(out, path)) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };
    if a.is_empty() || a.len() != b.len() {
        return;
    }
    let diff = l1_distance(&a, &b) / 2.0;
    scores.add(section, (1.0 - diff.min(1.0)).max(0.0));
    if diff > 0.15 {
        scores.delta(metric, diff);
    }
}

/// Fetch a nested object by key path, falling back when absent.
fn obj_at<'a>(value: &'a Value, path: &[&str], fallback: &'a Value) -> &'a Value {
    let mut cur = value;
    for key in path {
        match cur.get(*key) {
            Some(next) => cur = next,
            None => return fallback,
        }
    }
    cur
}

/// Compute style compliance for `output_text` against a fingerprint.
///
/// Mirrors `compute_style_compliance` in `apply_fingerprint.py`: per-section
/// scores are averaged, then combined using `validators.weights`, with any
/// deviation above the Python thresholds recorded in `deltas`.
pub fn compute_style_compliance(fingerprint: &Value, output_text: &str) -> Value {
    let empty = json!({});
    let fp_meas = fingerprint.get("measurements").unwrap_or(&empty);
    let weights = fingerprint
        .get("validators")
        .and_then(|v| v.get("weights"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let out_meas = compute_measurements(output_text);
    let mut scores = Scores::new();

    score_histogram(
        &mut scores,
        fp_meas,
        &out_meas,
        &["sentence", "length_words", "histogram_p"],
        "sentence",
        "sentence_length_histogram",
    );
    score_histogram(
        &mut scores,
        fp_meas,
        &out_meas,
        &["paragraph", "length_sentences_histogram_p"],
        "paragraph",
        "paragraph_length_histogram",
    );

    if let (Some(target), Some(actual)) = (
        num_at(fp_meas, &["paragraph", "one_sentence_paragraph_rate"]),
        num_at(&out_meas, &["paragraph", "one_sentence_paragraph_rate"]),
    ) {
        let diff = (target - actual).abs();
        scores.add("paragraph", (1.0 - diff.min(1.0)).max(0.0));
        if diff > 0.1 {
            scores.delta("one_sentence_paragraph_rate", diff);
        }
    }

    let dict_sections: [(&[&str], &str, &str, f64); 8] = [
        (&["punctuation", "rates_per_1000w"], "punctuation", "punctuation", 0.5),
        (&["stance_signals"], "stance", "stance", 0.6),
        (&["rhetoric_moves"], "rhetoric_moves", "rhetoric_moves", 0.6),
        (&["epistemic_profile"], "epistemic_profile", "epistemic_profile", 0.6),
        (&["paragraph_cadence"], "paragraph_cadence", "paragraph_cadence", 0.6),
        (&["syntax_texture"], "syntax_texture", "syntax_texture", 0.6),
        (
            &["lexical_avoidance", "category_rates_per_1000w"],
            "lexical_avoidance",
            "lexical_avoidance",
            0.6,
        ),
        (
            &["templates_signals", "transition_marker_positions"],
            "discourse_markers",
            "discourse_markers",
            0.6,
        ),
    ];
    for (path, section, prefix, threshold) in dict_sections {
        let fp_val = obj_at(fp_meas, path, &empty);
        let out_val = obj_at(&out_meas, path, &empty);
        if fp_val.as_object().is_some_and(|m| !m.is_empty()) {
            scores.numeric_section(fp_val, out_val, section, prefix, threshold);
        }
    }


    // Orthography signals (fixed key set).
    for key in ["contractions_rate", "oxford_comma_signal"] {
        if let (Some(target), Some(actual)) = (
            num_at(fp_meas, &["orthography_signals", key]),
            num_at(&out_meas, &["orthography_signals", key]),
        ) {
            let diff = relative_diff(actual, target);
            scores.add("orthography", (1.0 - diff.min(1.0)).max(0.0));
            if diff > 0.5 {
                scores.delta(&format!("orthography.{key}"), diff);
            }
        }
    }

    // Self-echo repetition rates (fixed key set).
    let mut rep_diffs: Vec<f64> = Vec::new();
    for key in ["bigram_repeat_rate", "trigram_repeat_rate"] {
        if let (Some(target), Some(actual)) = (
            num_at(fp_meas, &["repetition", key]),
            num_at(&out_meas, &["repetition", key]),
        ) {
            let diff = relative_diff(actual, target);
            rep_diffs.push(diff);
            if diff > 0.6 {
                scores.delta(&format!("repetition.{key}"), diff);
            }
        }
    }
    if !rep_diffs.is_empty() {
        let avg = rep_diffs.iter().sum::<f64>() / rep_diffs.len() as f64;
        scores.add("repetition", (1.0 - avg.min(1.0)).max(0.0));
    }

    // Aggregate per-section averages, weighted when `validators.weights` exists.
    let mut section_avgs: BTreeMap<String, f64> = BTreeMap::new();
    for (section, vals) in &scores.sections {
        if !vals.is_empty() {
            section_avgs.insert(
                section.clone(),
                vals.iter().sum::<f64>() / vals.len() as f64,
            );
        }
    }
    let weight_map = weights.as_object();
    let score = if weight_map.is_some_and(|m| !m.is_empty()) && !section_avgs.is_empty() {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        for (section, value) in &section_avgs {
            let w = weight_map
                .and_then(|m| m.get(section))
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            if w <= 0.0 {
                continue;
            }
            weighted_sum += value * w;
            total_weight += w;
        }
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            1.0
        }
    } else if section_avgs.is_empty() {
        1.0
    } else {
        section_avgs.values().sum::<f64>() / section_avgs.len() as f64
    };

    json!({
        "score": score,
        "deltas": scores.deltas,
        "output_measurements": out_meas
    })
}

/// Normalize a vector into a probability distribution (uniform when empty/zero).
fn normalize_distribution(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let cleaned: Vec<f64> = values.iter().map(|v| v.max(0.0)).collect();
    let total: f64 = cleaned.iter().sum();
    if total <= 0.0 {
        let uniform = 1.0 / cleaned.len() as f64;
        return vec![uniform; cleaned.len()];
    }
    cleaned.iter().map(|v| v / total).collect()
}

/// Jensen-Shannon divergence in base 2, bounded to `[0, 1]`.
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    if p.is_empty() || q.is_empty() || p.len() != q.len() {
        return 1.0;
    }
    let pp = normalize_distribution(p);
    let qq = normalize_distribution(q);
    let m: Vec<f64> = pp.iter().zip(qq.iter()).map(|(a, b)| (a + b) / 2.0).collect();
    let eps = 1e-12;
    let kl = |a: &[f64], b: &[f64]| -> f64 {
        let mut out = 0.0;
        for (ai, bi) in a.iter().zip(b.iter()) {
            if *ai <= 0.0 {
                continue;
            }
            out += ai * ((ai + eps) / (bi + eps)).log2();
        }
        out
    };
    let jsd = 0.5 * kl(&pp, &m) + 0.5 * kl(&qq, &m);
    if !jsd.is_finite() {
        return 1.0;
    }
    jsd.clamp(0.0, 1.0)
}

/// Cosine similarity between two equal-length vectors, clamped to `[0, 1]`.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(0.0, 1.0)
}

/// Map compliance into a calibrated style-match probability.
///
/// Mirrors `calibrated_style_match_probability` in `common.py`: logistic
/// calibration with length-reliability shrinkage toward 0.5.
pub fn calibrated_style_match_probability(
    compliance_score: f64,
    token_count: u64,
    threshold: f64,
    slope: f64,
    evidence_tokens: u64,
) -> Value {
    let comp = clamp01(compliance_score);
    let safe_slope = slope.max(1e-6);
    let threshold = clamp01(threshold);
    let unshrunk = sigmoid((comp - threshold) / safe_slope);
    let n = token_count as f64;
    let denom = evidence_tokens.max(1) as f64;
    let reliability = 1.0 - (-n / denom).exp();
    let probability = 0.5 + reliability * (unshrunk - 0.5);
    let half_width = (1.0 - reliability) * 0.5;
    let ci_low = clamp01(probability - half_width);
    let ci_high = clamp01(probability + half_width);
    let probability = clamp01(probability);
    json!({
        "probability": probability,
        "probability_percent": (probability * 100.0 * 100.0).round() / 100.0,
        "confidence_interval_90": [
            (ci_low * 10000.0).round() / 10000.0,
            (ci_high * 10000.0).round() / 10000.0
        ],
        "calibration": {
            "compliance_score": comp,
            "threshold": threshold,
            "slope": safe_slope,
            "unshrunk_probability": unshrunk,
            "token_count": token_count,
            "reliability": reliability,
            "evidence_tokens": denom,
            "method": "logistic_calibration_with_length_reliability_shrinkage"
        }
    })
}

/// Jaccard similarity between two string sets.
pub fn jaccard_similarity(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    inter as f64 / union.max(1) as f64
}

/// Collect the numeric entries of a JSON object into a sorted map.
pub fn to_numeric_dict(value: &Value) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    if let Some(map) = value.as_object() {
        for (k, v) in map {
            if let Some(f) = v.as_f64() {
                if f.is_finite() {
                    out.insert(k.clone(), f);
                }
            }
        }
    }
    out
}

/// Symmetric relative similarity: `1 / (1 + rel)` (0 → 1, large → 0).
fn relative_similarity(a: f64, b: f64) -> f64 {
    let denom = ((a.abs() + b.abs()) / 2.0).max(1e-9);
    let rel = (a - b).abs() / denom;
    1.0 / (1.0 + rel)
}

/// Compare two numeric dictionaries; `None` when no keys are shared.
pub fn compare_numeric_dicts(
    a: &BTreeMap<String, f64>,
    b: &BTreeMap<String, f64>,
) -> Option<Value> {
    let keys: BTreeSet<&String> = a.keys().filter(|k| b.contains_key(*k)).collect();
    if keys.is_empty() {
        return None;
    }
    let mut rows: Vec<Value> = Vec::new();
    let mut sims: Vec<f64> = Vec::new();
    for key in keys {
        let av = a[key];
        let bv = b[key];
        let sim = relative_similarity(av, bv);
        sims.push(sim);
        rows.push(json!({
            "key": key,
            "a": av,
            "b": bv,
            "similarity": sim,
            "abs_diff": (av - bv).abs()
        }));
    }
    rows.sort_by(|x, y| {
        let xs = x.get("similarity").and_then(Value::as_f64).unwrap_or(0.0);
        let ys = y.get("similarity").and_then(Value::as_f64).unwrap_or(0.0);
        xs.partial_cmp(&ys).unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(5);
    Some(json!({
        "similarity": sims.iter().sum::<f64>() / sims.len() as f64,
        "keys_compared": sims.len(),
        "worst_keys": rows
    }))
}

/// Compare two equal-length distribution vectors (JSD plus L1/2 diagnostics).
pub fn compare_distribution_vectors(a: &[f64], b: &[f64]) -> Option<Value> {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return None;
    }
    let jsd = jensen_shannon_divergence(a, b);
    let pa = normalize_distribution(a);
    let pb = normalize_distribution(b);
    let l1 = l1_distance(&pa, &pb) / 2.0;
    Some(json!({
        "similarity": (1.0 - jsd).max(0.0),
        "js_divergence": jsd,
        "l1_distance_half": l1,
        "bins": a.len()
    }))
}

/// Compare two sparse feature maps as distributions (union of keys).
pub fn compare_distribution_dicts(
    a: &BTreeMap<String, f64>,
    b: &BTreeMap<String, f64>,
) -> Option<Value> {
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    if keys.is_empty() {
        return None;
    }
    let va: Vec<f64> = keys.iter().map(|k| *a.get(*k).unwrap_or(&0.0)).collect();
    let vb: Vec<f64> = keys.iter().map(|k| *b.get(*k).unwrap_or(&0.0)).collect();
    let jsd = jensen_shannon_divergence(&va, &vb);
    let cos = cosine_similarity(&va, &vb);
    Some(json!({
        "similarity": (1.0 - jsd).max(0.0),
        "js_divergence": jsd,
        "cosine_similarity": cos,
        "dimensions": keys.len()
    }))
}


/// Extract an estimated corpus word count from a fingerprint.
fn fingerprint_word_estimate(fp: &Value) -> u64 {
    for path in [
        &["metadata", "corpus", "size", "words_est"][..],
        &["measurements", "totals", "total_words_est"][..],
    ] {
        let mut cur = fp;
        let mut found = true;
        for key in path {
            match cur.get(*key) {
                Some(next) => cur = next,
                None => {
                    found = false;
                    break;
                }
            }
        }
        if found {
            if let Some(v) = cur.as_u64() {
                return v;
            }
            if let Some(v) = cur.as_f64() {
                if v >= 0.0 {
                    return v as u64;
                }
            }
        }
    }
    0
}

/// Normalize one or more lexicon lists into a lowercase string set.
fn lexicon_set(lex: &Value, keys: &[&str]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for key in keys {
        if let Some(arr) = lex.get(*key).and_then(Value::as_array) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    let s = s.trim().to_lowercase();
                    if !s.is_empty() {
                        out.insert(s);
                    }
                }
            }
        }
    }
    out
}

/// Default component weights used by fingerprint similarity.
pub fn default_similarity_weights() -> BTreeMap<String, f64> {
    [
        ("function_words_distribution", 0.20),
        ("sentence_histogram", 0.15),
        ("paragraph_histogram", 0.10),
        ("punctuation_rates", 0.10),
        ("stance_signals", 0.08),
        ("rhetoric_moves", 0.08),
        ("syntax_texture", 0.08),
        ("paragraph_cadence", 0.08),
        ("repetition", 0.05),
        ("lexicon_preferred_overlap", 0.04),
        ("lexicon_avoid_overlap", 0.04),
    ]
    .into_iter()
    .map(|(name, weight)| (name.to_string(), weight))
    .collect()
}



/// Compute interpretable similarity between two fingerprints.
///
/// Mirrors `compute_fingerprint_similarity` in `common.py`: every available
/// component contributes a clamped similarity, weighted by `component_weights`
/// (defaults in [`default_similarity_weights`]).
pub fn compute_fingerprint_similarity(
    fingerprint_a: &Value,
    fingerprint_b: &Value,
    component_weights: Option<&BTreeMap<String, f64>>,
) -> Result<Value, String> {
    if !fingerprint_a.is_object() || !fingerprint_b.is_object() {
        return Err("Both fingerprints must be JSON objects.".to_string());
    }

    let mut weights = default_similarity_weights();
    if let Some(overrides) = component_weights {
        for (key, value) in overrides {
            if weights.contains_key(key) {
                weights.insert(key.clone(), value.max(0.0));
            }
        }
    }

    let empty = json!({});
    let meas_a = fingerprint_a.get("measurements").unwrap_or(&empty);
    let meas_b = fingerprint_b.get("measurements").unwrap_or(&empty);
    let mut components: BTreeMap<String, Value> = BTreeMap::new();

    if let (Some(a), Some(b)) = (
        nums_at(meas_a, &["sentence", "length_words", "histogram_p"]),
        nums_at(meas_b, &["sentence", "length_words", "histogram_p"]),
    ) {
        if let Some(v) = compare_distribution_vectors(&a, &b) {
            components.insert("sentence_histogram".to_string(), v);
        }
    }
    if let (Some(a), Some(b)) = (
        nums_at(meas_a, &["paragraph", "length_sentences_histogram_p"]),
        nums_at(meas_b, &["paragraph", "length_sentences_histogram_p"]),
    ) {
        if let Some(v) = compare_distribution_vectors(&a, &b) {
            components.insert("paragraph_histogram".to_string(), v);
        }
    }

    let func_a = to_numeric_dict(obj_at(
        meas_a,
        &["function_words", "rates_per_1000w"],
        &empty,
    ));
    let func_b = to_numeric_dict(obj_at(
        meas_b,
        &["function_words", "rates_per_1000w"],
        &empty,
    ));
    if let Some(v) = compare_distribution_dicts(&func_a, &func_b) {
        components.insert("function_words_distribution".to_string(), v);
    }

    let numeric_paths: [(&str, &[&str]); 5] = [
        ("punctuation_rates", &["punctuation", "rates_per_1000w"][..]),
        ("stance_signals", &["stance_signals"][..]),
        ("rhetoric_moves", &["rhetoric_moves"][..]),
        ("syntax_texture", &["syntax_texture"][..]),
        ("paragraph_cadence", &["paragraph_cadence"][..]),
    ];
    for (name, path) in numeric_paths {
        let a = to_numeric_dict(obj_at(meas_a, path, &empty));
        let b = to_numeric_dict(obj_at(meas_b, path, &empty));
        if let Some(v) = compare_numeric_dicts(&a, &b) {
            components.insert(name.to_string(), v);
        }
    }

    let mut rep_a = to_numeric_dict(obj_at(meas_a, &["repetition"], &empty));
    let mut rep_b = to_numeric_dict(obj_at(meas_b, &["repetition"], &empty));
    rep_a.retain(|k, _| k == "bigram_repeat_rate" || k == "trigram_repeat_rate");
    rep_b.retain(|k, _| k == "bigram_repeat_rate" || k == "trigram_repeat_rate");
    if let Some(v) = compare_numeric_dicts(&rep_a, &rep_b) {
        components.insert("repetition".to_string(), v);
    }


    // Lexical overlap components (always present, possibly 0.0 similarity).
    let lex_a = fingerprint_a.get("lexicon").unwrap_or(&empty);
    let lex_b = fingerprint_b.get("lexicon").unwrap_or(&empty);
    let pref_a = lexicon_set(lex_a, &["preferred_words", "preferred_phrases"]);
    let pref_b = lexicon_set(lex_b, &["preferred_words", "preferred_phrases"]);
    let avoid_a = lexicon_set(lex_a, &["avoid_words"]);
    let avoid_b = lexicon_set(lex_b, &["avoid_words"]);
    components.insert(
        "lexicon_preferred_overlap".to_string(),
        json!({
            "similarity": jaccard_similarity(&pref_a, &pref_b),
            "size_a": pref_a.len(),
            "size_b": pref_b.len(),
            "intersection_size": pref_a.intersection(&pref_b).count()
        }),
    );
    components.insert(
        "lexicon_avoid_overlap".to_string(),
        json!({
            "similarity": jaccard_similarity(&avoid_a, &avoid_b),
            "size_a": avoid_a.len(),
            "size_b": avoid_b.len(),
            "intersection_size": avoid_a.intersection(&avoid_b).count()
        }),
    );

    let total_weight: f64 = weights.values().sum();
    let mut used_weight = 0.0;
    let mut weighted_sum = 0.0;
    let mut missing: Vec<Value> = Vec::new();
    let mut used_components: Vec<Value> = Vec::new();
    for (name, weight) in &weights {
        let comp = match components.get(name) {
            Some(v) => v,
            None => {
                missing.push(json!(name));
                continue;
            }
        };
        let sim = match comp.get("similarity").and_then(Value::as_f64) {
            Some(v) => clamp01(v),
            None => {
                missing.push(json!(name));
                continue;
            }
        };
        used_weight += weight;
        weighted_sum += sim * weight;
        let mut spec = comp.clone();
        if let Some(map) = spec.as_object_mut() {
            map.insert("name".to_string(), json!(name));
            map.insert("weight".to_string(), json!(weight));
        }
        used_components.push(spec);
    }

    let overall = clamp01(if used_weight > 0.0 {
        weighted_sum / used_weight
    } else {
        0.0
    });
    let words_a = fingerprint_word_estimate(fingerprint_a);
    let words_b = fingerprint_word_estimate(fingerprint_b);
    let min_words = if words_a > 0 && words_b > 0 {
        words_a.min(words_b)
    } else {
        0
    };
    let evidence_factor = 1.0 - (-(min_words as f64) / 20000.0).exp();
    let coverage = if total_weight > 0.0 {
        used_weight / total_weight
    } else {
        0.0
    };
    let confidence_hint = clamp01(0.6 * coverage + 0.4 * evidence_factor);

    used_components.sort_by(|x, y| {
        let xs = x.get("similarity").and_then(Value::as_f64).unwrap_or(0.0);
        let ys = y.get("similarity").and_then(Value::as_f64).unwrap_or(0.0);
        xs.partial_cmp(&ys).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_differences: Vec<Value> = used_components.iter().take(5).cloned().collect();
    let components_expected = weights.len();

    Ok(json!({
        "similarity_score": overall,
        "distance_score": 1.0 - overall,
        "coverage": {
            "components_available": used_components.len(),
            "components_expected": components_expected,
            "missing_components": missing,
            "used_weight": used_weight,
            "total_weight": total_weight,
            "coverage_ratio": coverage
        },
        "confidence_hint": confidence_hint,
        "evidence": {
            "words_est_a": words_a,
            "words_est_b": words_b,
            "min_words_est": min_words,
            "evidence_factor": evidence_factor
        },
        "components": used_components,
        "top_differences": top_differences,
        "weights": weights
    }))
}

