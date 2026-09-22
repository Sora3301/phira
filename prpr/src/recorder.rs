//! Framebuffer recording
//!
//! Captures rendered frames (via `glReadPixels` on the default framebuffer),
//! pipes them to a system `ffmpeg` process as raw RGBA video, and writes a
//! sidecar CSV with the audio playback position of every frame so that the
//! recorded video can be re-synced with the song later.

use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{ChildStdin, Command, Stdio},
    sync::mpsc::{channel, sync_channel, Receiver, SyncSender, TrySendError},
    thread::JoinHandle,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};

pub const RECORD_FPS: f64 = 60.0;

static RECORD_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Set the directory where recordings are written (e.g. the app cache dir).
pub fn set_record_dir(dir: impl Into<PathBuf>) {
    let _ = RECORD_DIR.set(dir.into());
}

/// The root directory recordings are saved under.
pub fn record_dir() -> PathBuf {
    RECORD_DIR.get().cloned().unwrap_or_else(|| std::env::temp_dir().join("phira-recordings"))
}

/// Clean a string so it is safe to use as part of a file path.
pub fn sanitize_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    while out.ends_with(['.', ' ']) {
        out.pop();
    }
    out.chars().take(120).collect()
}

enum Msg {
    Frame { data: Vec<u8>, audio_time: f64, index: u64 },
    End,
}

pub struct Recording {
    tx: SyncSender<Msg>,
    return_rx: Receiver<Vec<u8>>,
    worker: Option<JoinHandle<()>>,

    timestamp: u128,
    pub buffer: Vec<u8>,
    first_capture_real: Option<f64>,
    last_data: Vec<u8>,
    index: u64,
}

impl Recording {
    pub fn start(dir: impl Into<PathBuf>, width: u32, height: u32) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).with_context(|| format!("failed to create recording dir {}", dir.display()))?;
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|it| it.as_millis()).unwrap_or_default();
        let base = dir.join(format!("recording-{ts}"));
        let video_path = base.with_extension("mp4");
        let sidecar_path = base.with_extension("csv");

        let mut child = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-video_size",
                &format!("{width}x{height}"),
                "-framerate",
                &format!("{RECORD_FPS:.0}"),
                "-i",
                "pipe:0",
                "-vf",
                "vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&video_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn ffmpeg (make sure ffmpeg is installed and on PATH)")?;
        let stdin = child.stdin.take().context("failed to open ffmpeg stdin")?;

        let (tx, rx) = sync_channel::<Msg>(2);
        let (return_tx, return_rx) = channel::<Vec<u8>>();
        let worker = std::thread::spawn(move || {
            let result = run_worker(stdin, rx, return_tx, &sidecar_path);
            if let Err(err) = result {
                tracing::warn!("recording worker error: {err}");
            }
            let _ = child.wait();
        });

        Ok(Self {
            tx,
            return_rx,
            worker: Some(worker),
            timestamp: ts,
            buffer: Vec::new(),
            first_capture_real: None,
            last_data: Vec::new(),
            index: 0,
        })
    }

    /// Timestamp (milliseconds) shared by this recording's files. Sidecar files
    /// can use it to pair with the recorded video.
    pub fn timestamp(&self) -> u128 {
        self.timestamp
    }

    fn send_frame(&mut self, data: Vec<u8>, audio_time: f64) -> Option<u64> {
        match self.tx.try_send(Msg::Frame {
            data,
            audio_time,
            index: self.index,
        }) {
            Ok(()) => {
                self.index += 1;
                Some(self.index - 1)
            }
            Err(TrySendError::Full(Msg::Frame { data, .. })) => {
                // Encoder is lagging behind; drop the frame and keep the buffer for reuse.
                self.buffer = data;
                None
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => None,
        }
    }

    /// Send the buffer filled by `read_framebuffer` to the encoder, recording
    /// the audio playback position as the timestamp of this frame.
    /// Returns the video frame index if the frame was written.
    ///
    /// Frames are aligned to a constant `{RECORD_FPS}` timeline in real time:
    /// when the game renders slower than the target rate, repeats of the
    /// previous frame fill the gaps, and when it renders faster, frames are
    /// skipped. The video duration therefore always matches real time.
    pub fn capture_frame(&mut self, audio_time: f64, real_time: f64) -> Option<u64> {
        if self.first_capture_real.is_none() {
            self.first_capture_real = Some(real_time);
        }
        let elapsed = real_time - self.first_capture_real.unwrap_or(real_time);
        let target = ((elapsed * RECORD_FPS).round() as u64).max(1);
        if self.index >= target {
            if let Ok(buf) = self.return_rx.try_recv() {
                self.buffer = buf;
            }
            return None;
        }

        let data = std::mem::take(&mut self.buffer);
        self.last_data.clear();
        self.last_data.extend_from_slice(&data);
        let result = self.send_frame(data, audio_time);
        if let Ok(buf) = self.return_rx.try_recv() {
            self.buffer = buf;
        }
        while self.index < target {
            if self.send_frame(self.last_data.clone(), audio_time).is_none() {
                break;
            }
        }
        result
    }
}

fn run_worker(mut stdin: ChildStdin, rx: Receiver<Msg>, return_tx: std::sync::mpsc::Sender<Vec<u8>>, sidecar_path: &Path) -> Result<()> {
    let mut sidecar = std::fs::File::create(sidecar_path).with_context(|| format!("failed to create {}", sidecar_path.display()))?;
    writeln!(sidecar, "frame,audio_position").context("failed to write sidecar header")?;
    for msg in rx {
        match msg {
            Msg::Frame { data, audio_time, index } => {
                if stdin.write_all(&data).is_err() {
                    break;
                }
                writeln!(sidecar, "{index},{audio_time}").context("failed to write sidecar")?;
                sidecar.flush().ok();
                let _ = return_tx.send(data);
            }
            Msg::End => break,
        }
    }
    drop(stdin);
    Ok(())
}

impl Drop for Recording {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::End);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Read the currently rendered framebuffer region into `buf` as RGBA bytes.
///
/// `viewport` uses the same coordinate space as the game camera
/// (`(x, y, w, h)` with a top-left origin), which is what `glViewport` was
/// called with while rendering. Pixels are bottom-up, which the `vflip`
/// filter later compensates for.
pub unsafe fn read_framebuffer(viewport: (i32, i32, i32, i32), buf: &mut Vec<u8>) {
    use miniquad::gl::*;
    let (x, y, w, h) = viewport;
    let framebuffer_height = macroquad::window::screen_height() as i32;
    let gl_y = (framebuffer_height - y - h).max(0);
    buf.resize((w * h * 4) as usize, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glReadPixels(x, gl_y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf.as_mut_ptr() as *mut _);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_video_and_sidecar() {
        let dir = std::env::temp_dir().join(format!("phira-record-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        set_record_dir(&dir);
        let mut rec = Recording::start(record_dir().join("test chart"), 16, 16).unwrap();
        // Simulate a slow game (10 captures over ~1s of real time): the
        // recorder should pad the video to a constant 60 fps timeline.
        for i in 0..10 {
            rec.buffer = vec![(i * 20) as u8; 16 * 16 * 4];
            rec.capture_frame(i as f64 / 10.0, i as f64 / 10.0);
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        drop(rec);

        let out_dir = dir.join("test chart");
        let entries: Vec<_> = std::fs::read_dir(&out_dir).unwrap().filter_map(Result::ok).collect();
        assert!(entries.iter().any(|e| e.path().extension().map(|s| s == "mp4").unwrap_or(false)));
        let csv_path = entries
            .iter()
            .find(|e| e.path().extension().map(|s| s == "csv").unwrap_or(false))
            .map(|e| e.path())
            .unwrap();
        let csv = std::fs::read_to_string(&csv_path).unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next(), Some("frame,audio_position"));
        let mut count = 0;
        for (frame, line) in lines.enumerate() {
            let mut it = line.split(',');
            assert_eq!(it.next().unwrap(), frame.to_string());
            let audio: f64 = it.next().unwrap().parse().unwrap();
            assert!((0.0..=1.0).contains(&audio));
            count += 1;
        }
        // 10 real captures + repeats filling the 60 fps timeline.
        assert!(count >= 10, "expected padded frames, got {count}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
