//! Experimental, fixed-reference speaker presence. All work belongs off the audio callback.
use crate::voice_coefficients::{ENROLL_FIR, HANN, LIVE_FIR, MEL};
use ort::{session::Session, value::TensorRef};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{
    collections::VecDeque,
    fs,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub const ENROLL_SAMPLES: usize = 20 * 48_000;
pub const HOLD_SAMPLES: usize = 9 * crate::HOP;
const MAGIC: &[u8; 8] = b"HMVRST01";

fn session(path: &Path) -> Result<Session, String> {
    crate::runtime::ensure_runtime().map_err(|e| e.to_string())?;
    Session::builder()
        .map_err(|e| e.to_string())?
        .with_intra_threads(1)
        .map_err(|e| e.to_string())?
        .with_inter_threads(1)
        .map_err(|e| e.to_string())?
        .commit_from_file(path)
        .map_err(|e| e.to_string())
}
fn validate_reference(reference: &[f32; 40]) -> Result<(), String> {
    if reference.iter().any(|v| !v.is_finite()) || reference.iter().all(|v| *v == 0.0) {
        return Err("Invalid voice reference; learn your voice again".into());
    }
    Ok(())
}
pub fn read_reference(path: &Path) -> Result<[f32; 40], String> {
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() != 168 {
        return Err("Invalid voice reference format".into());
    }
    let mut bytes = [0; 168];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    if &bytes[..8] != MAGIC {
        return Err("Voice reference uses a different detector; learn again".into());
    }
    let reference = std::array::from_fn(|i| {
        f32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap())
    });
    validate_reference(&reference)?;
    Ok(reference)
}
pub fn write_reference(path: &Path, reference: &[f32; 40]) -> Result<(), String> {
    validate_reference(reference)?;
    let parent = path
        .parent()
        .ok_or("Voice reference needs a parent directory")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let temp = parent.join(format!(
        ".voice-reference-{}-{}.tmp",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        file.write_all(MAGIC).map_err(|e| e.to_string())?;
        for value in reference {
            file.write_all(&value.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        file.sync_all().map_err(|e| e.to_string())?;
        fs::rename(&temp, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

struct Mel {
    fft: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex<f32>>,
    buffer: [Complex<f32>; 400],
}
impl Mel {
    fn new() -> Self {
        let fft = FftPlanner::new().plan_fft_forward(400);
        let scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
        Self {
            fft,
            scratch,
            buffer: [Complex::default(); 400],
        }
    }
    fn frame(&mut self, mut sample: impl FnMut(isize) -> f32, center: isize) -> [f32; 24] {
        for (i, slot) in self.buffer.iter_mut().enumerate() {
            *slot = Complex::new(sample(center + i as isize - 200) * HANN[i], 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        std::array::from_fn(|m| {
            let power: f32 = (0..201)
                .map(|k| self.buffer[k].norm_sqr() * MEL[m * 201 + k])
                .sum();
            (power + 1e-6).log10()
        })
    }
}
fn reflect(index: isize, len: usize) -> usize {
    if index < 0 {
        (-index) as usize
    } else if index as usize >= len {
        2 * len - 2 - index as usize
    } else {
        index as usize
    }
}

/// Exactly twenty seconds of clean, physical-microphone audio, at 48 kHz mono.
/// The caller owns and discards its capture buffer after this returns.
pub fn enroll(raw: &[f32], encoder_path: &Path) -> Result<[f32; 40], String> {
    if raw.len() != ENROLL_SAMPLES {
        return Err("Record a complete twenty-second voice sample".into());
    }
    if raw.iter().any(|v| !v.is_finite()) {
        return Err("Recording contains invalid audio; retry".into());
    }
    if raw.iter().filter(|v| v.abs() >= 0.999).count() > raw.len() / 1000 {
        return Err("Recording is clipped; lower microphone gain and retry".into());
    }
    let voiced = raw
        .as_chunks::<480>()
        .0
        .iter()
        .filter(|hop| hop.iter().map(|x| x * x).sum::<f32>() / 480.0 > 0.00001)
        .count();
    if voiced < 500 {
        return Err(
            "Too little audible speech; speak clearly for twenty seconds in a quiet room".into(),
        );
    }
    let samples: Vec<f32> = (0..320_000)
        .map(|j| {
            ENROLL_FIR
                .iter()
                .enumerate()
                .map(|(k, c)| {
                    let i = j as isize * 3 + k as isize - 19;
                    c * if (0..raw.len() as isize).contains(&i) {
                        raw[i as usize]
                    } else {
                        0.0
                    }
                })
                .sum()
        })
        .collect();
    let mut frontend = Mel::new();
    let mut features = vec![0.0; 24 * 2001];
    for j in 0..2001 {
        let frame = frontend.frame(|i| samples[reflect(i, samples.len())], j as isize * 160);
        for m in 0..24 {
            features[m * 2001 + j] = frame[m];
        }
    }
    let mut encoder = session(encoder_path)?;
    let tensor = TensorRef::from_array_view(([1usize, 24, 2001], features.as_slice()))
        .map_err(|e| e.to_string())?;
    let outputs = encoder
        .run(ort::inputs! {"mel"=>tensor})
        .map_err(|e| e.to_string())?;
    let (_, values) = outputs
        .get("reference")
        .ok_or("Encoder has no reference output")?
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    let reference: [f32; 40] = values
        .try_into()
        .map_err(|_| "Invalid encoder reference shape")?;
    validate_reference(&reference)?;
    Ok(reference)
}

/// Streaming probabilities with their original 10 ms source-frame index.
/// Reference is immutable; reset clears only audio and temporal history.
pub struct Presence {
    session: Session,
    reference: [f32; 40],
    fir: [f32; 61],
    fir_position: usize,
    raw_count: usize,
    decimated_count: usize,
    samples: VecDeque<f32>,
    sample_start: usize,
    features: VecDeque<[f32; 24]>,
    frame_count: usize,
    frontend: Mel,
}
impl Presence {
    pub fn new(model: &Path, reference: [f32; 40]) -> Result<Self, String> {
        validate_reference(&reference)?;
        Ok(Self {
            session: session(model)?,
            reference,
            fir: [0.0; 61],
            fir_position: 0,
            raw_count: 0,
            decimated_count: 0,
            samples: VecDeque::with_capacity(9600),
            sample_start: 0,
            features: VecDeque::with_capacity(50),
            frame_count: 0,
            frontend: Mel::new(),
        })
    }
    pub fn reset(&mut self) {
        self.fir.fill(0.0);
        self.fir_position = 0;
        self.raw_count = 0;
        self.decimated_count = 0;
        self.samples.clear();
        self.sample_start = 0;
        self.features.clear();
        self.frame_count = 0;
    }
    pub fn process_hop(&mut self, raw: &[f32; crate::HOP]) -> Result<Option<(usize, f32)>, String> {
        if raw.iter().any(|x| !x.is_finite()) {
            return Err("Detector received non-finite audio".into());
        }
        for &value in raw {
            self.fir[self.fir_position] = value;
            if self.raw_count.is_multiple_of(3) {
                let value: f64 = LIVE_FIR
                    .iter()
                    .enumerate()
                    .map(|(k, c)| c * self.fir[(self.fir_position + 61 - k) % 61] as f64)
                    .sum();
                if self.decimated_count >= 10 {
                    if self.samples.len() == 9600 {
                        self.samples.pop_front();
                        self.sample_start += 1;
                    }
                    self.samples.push_back(value as f32);
                }
                self.decimated_count += 1;
            }
            self.fir_position = (self.fir_position + 1) % 61;
            self.raw_count += 1;
        }
        let available = self.sample_start + self.samples.len();
        if self.frame_count * 160 + 200 > available {
            return Ok(None);
        }
        let samples = &self.samples;
        let start = self.sample_start;
        let frame = self.frontend.frame(
            |i| samples[reflect(i, available) - start],
            self.frame_count as isize * 160,
        );
        if self.features.len() == 50 {
            self.features.pop_front();
        }
        self.features.push_back(frame);
        self.frame_count += 1;
        if self.features.len() < 6 {
            return Ok(None);
        }
        let time = self.features.len();
        let mut features = [0.0; 24 * 50];
        for (j, frame) in self.features.iter().enumerate() {
            for m in 0..24 {
                features[m * time + j] = frame[m];
            }
        }
        let mel = TensorRef::from_array_view(([1usize, 24, time], &features[..24 * time]))
            .map_err(|e| e.to_string())?;
        let reference = TensorRef::from_array_view(([1usize, 40], self.reference.as_slice()))
            .map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs! {"mel"=>mel,"reference"=>reference})
            .map_err(|e| e.to_string())?;
        let (_, values) = outputs
            .get("probability")
            .ok_or("Detector has no probability output")?
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if values.len() != 1 || !values[0].is_finite() || !(0.0..=1.0).contains(&values[0]) {
            return Err("Detector returned invalid probability".into());
        }
        Ok(Some((self.frame_count - 6, values[0])))
    }
}

#[cfg(all(test, feature = "load-dynamic"))]
mod tests {
    use super::*;
    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }
    fn raw() -> Vec<f32> {
        let mut seed = 0x12345678u32;
        (0..ENROLL_SAMPLES)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                ((seed >> 8) as i32 - 8388608) as f32 / 104857600.0
            })
            .collect()
    }
    #[test]
    fn native_enrollment_and_stream_match_frozen_candidate() {
        let root = root();
        let runtime = root.join("assets/lib/libonnxruntime.so");
        if !runtime.exists() && std::env::var("HUSHMIC_ASSERT_ASSETS").as_deref() != Ok("1") {
            return;
        }
        crate::init_runtime(runtime).expect("runtime");
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../assets/models/voice-frontend-parity.json"
        ))
        .unwrap();
        let raw = raw();
        let reference = enroll(
            &raw,
            &root.join("assets/models/voice-reference-encoder.onnx"),
        )
        .unwrap();
        let reference_error = reference
            .iter()
            .zip(fixture["reference"].as_array().unwrap())
            .map(|(a, e)| (a - e.as_f64().unwrap() as f32).abs())
            .fold(0.0f32, f32::max);
        eprintln!("native reference maximum error {reference_error}");
        let mut feature_error = 0.0f32;
        let mut probability_error = 0.0f32;
        for (actual, expected) in reference
            .iter()
            .zip(fixture["reference"].as_array().unwrap())
        {
            assert!(
                (actual - expected.as_f64().unwrap() as f32).abs() < 0.0001,
                "reference parity: {actual} vs {expected}"
            );
        }
        let mut detector = Presence::new(
            &root.join("assets/models/voice-presence-int8.onnx"),
            reference,
        )
        .unwrap();
        for hop in raw.as_chunks::<{ crate::HOP }>().0.iter().take(153) {
            let result = detector.process_hop(hop).unwrap();
            if detector.frame_count > 0 {
                let frame = detector.frame_count - 1;
                if let Some(i) = fixture["live_frames"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .position(|v| v.as_u64() == Some(frame as u64))
                {
                    for (actual, expected) in detector
                        .features
                        .back()
                        .unwrap()
                        .iter()
                        .zip(fixture["live_features"][i].as_array().unwrap())
                    {
                        feature_error =
                            feature_error.max((actual - expected.as_f64().unwrap() as f32).abs());
                        assert!(
                            (actual - expected.as_f64().unwrap() as f32).abs() < 0.0001,
                            "live frontend frame {frame}: {actual} vs {expected}"
                        );
                    }
                }
            }
            if let Some((j, p)) = result {
                for row in fixture["probabilities"].as_array().unwrap() {
                    if row["source_hop"].as_u64() == Some(j as u64) {
                        probability_error = probability_error
                            .max((p - row["probability"].as_f64().unwrap() as f32).abs());
                        assert!(
                            (p - row["probability"].as_f64().unwrap() as f32).abs() < 0.005,
                            "INT8 probability frame{j}: {p} vs {}",
                            row["probability"]
                        );
                    }
                }
            }
        }
        eprintln!("native live frontend maximum error {feature_error}; INT8 probability maximum error {probability_error}");
        detector.reset();
        for hop in raw.as_chunks::<{ crate::HOP }>().0.iter().take(7) {
            let result = detector.process_hop(hop).unwrap();
            if let Some((j, _)) = result {
                assert_eq!(j, 0);
            }
        }
    }
    #[test]
    fn reference_replace_is_private_and_invalid_replacement_preserves_old() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("hushmic-voice-{}", std::process::id()));
        let path = dir.join("reference");
        let reference = [1.0; 40];
        write_reference(&path, &reference).unwrap();
        assert_eq!(read_reference(&path).unwrap(), reference);
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(write_reference(&path, &[f32::NAN; 40]).is_err());
        assert_eq!(read_reference(&path).unwrap(), reference);
        fs::remove_dir_all(dir).unwrap();
        assert!(enroll(&[0.0; 480], Path::new("missing")).is_err());
        assert!(enroll(&vec![0.0; ENROLL_SAMPLES], Path::new("missing")).is_err());
    }
}
