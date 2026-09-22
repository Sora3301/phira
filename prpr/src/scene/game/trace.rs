//! Theoretical ("ghost") finger trace recording.
//!
//! While a gameplay video is being recorded, this writes a JSONL sidecar that
//! contains, for every recorded frame, the virtual fingers a perfect player
//! would need for that frame: their positions (normalised screen coordinates,
//! the same space `Judge` uses for touches and `export_events` stores) plus
//! every finger-down / finger-up event.
//!
//! The trace is derived from the chart itself, not from real input:
//! * a note puts a finger down at `note.time`;
//! * a hold keeps that finger down until `end_time` and, because hold notes can
//!   slide left/right, the finger x follows the note's animated x on every
//!   frame (never just the position it had when it was pressed);
//! * other note kinds are tapped briefly.
//!
//! Each note gets its own finger id (`line << 32 | note`), so simultaneous
//! notes produce simultaneous fingers. One JSONL line is written per video
//! frame, with `frame` matching the video / sidecar frame index.

use crate::core::{Chart, NoteKind, Point, Resource};
use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// How long a non-hold note keeps its virtual finger down (seconds).
const TAP_DURATION: f64 = 0.06;

struct NoteSpan {
    start: f64,
    end: f64,
    line: usize,
    note: usize,
    id: u64,
}

#[derive(Serialize)]
struct Finger {
    id: u64,
    x: f32,
    y: f32,
}

#[derive(Serialize)]
struct FingerEvent {
    id: u64,
    phase: &'static str,
    x: f32,
    y: f32,
}

#[derive(Serialize)]
struct Frame {
    t: f64,
    frame: i64,
    fingers: Vec<Finger>,
    events: Vec<FingerEvent>,
}

pub struct TraceRecorder {
    file: BufWriter<File>,
    down: BTreeMap<u64, (f32, f32)>,
    spans: Vec<NoteSpan>,
    next: usize,
    active: Vec<usize>,
}

impl TraceRecorder {
    pub fn start(path: &Path, chart: &Chart) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("failed to create finger trace {}", path.display()))?;
        Ok(Self {
            file: BufWriter::new(file),
            down: BTreeMap::new(),
            spans: build_spans(chart),
            next: 0,
            active: Vec::new(),
        })
    }

    /// Record one video frame.
    ///
    /// `audio_t` is the audio playback position (the same base as the video
    /// sidecar CSV), `chart_t` is the chart time used for note lookup and
    /// `frame` is the video frame index.
    pub fn record_frame(&mut self, audio_t: f64, chart_t: f64, frame: i64, chart: &mut Chart, res: &Resource) -> Result<()> {
        // Advance over newly started notes. Frames are recorded in increasing
        // chart time, so a single forward cursor is enough.
        while self.next < self.spans.len() && self.spans[self.next].start <= chart_t {
            self.active.push(self.next);
            self.next += 1;
        }
        {
            let spans = &self.spans;
            self.active.retain(|&i| chart_t <= spans[i].end);
        }

        for line in chart.lines.iter_mut() {
            line.object.set_time(chart_t);
        }
        let mut current: BTreeMap<u64, (f32, f32)> = BTreeMap::new();
        for &i in &self.active {
            let span = &self.spans[i];
            let (x, y) = note_screen_pos(chart, res, span.line, span.note, chart_t);
            if x.is_finite() && y.is_finite() {
                current.insert(span.id, (x, y));
            }
        }

        let mut events = Vec::new();
        for (&id, &(x, y)) in &self.down {
            if !current.contains_key(&id) {
                events.push(FingerEvent { id, phase: "up", x, y });
            }
        }
        for (&id, &(x, y)) in &current {
            if !self.down.contains_key(&id) {
                events.push(FingerEvent { id, phase: "down", x, y });
            }
        }

        let fingers = current.iter().map(|(&id, &(x, y))| Finger { id, x, y }).collect();
        self.down = current;

        let data = Frame {
            t: audio_t,
            frame,
            fingers,
            events,
        };
        serde_json::to_writer(&mut self.file, &data).context("failed to serialize finger trace")?;
        self.file.write_all(b"\n").context("failed to write finger trace")?;
        self.file.flush().ok();
        Ok(())
    }
}

fn build_spans(chart: &Chart) -> Vec<NoteSpan> {
    let mut spans = Vec::new();
    for (line_id, line) in chart.lines.iter().enumerate() {
        for (note_id, note) in line.notes.iter().enumerate() {
            let start = note.time;
            let end = match &note.kind {
                NoteKind::Hold { end_time, .. } => *end_time,
                _ => start + TAP_DURATION,
            };
            spans.push(NoteSpan {
                start,
                end,
                line: line_id,
                note: note_id,
                id: ((line_id as u64) << 32) | note_id as u64,
            });
        }
    }
    spans.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    spans
}

/// Normalised screen position of a note at chart time `t`.
///
/// This matches the coordinate space `Judge` uses for touches (and the one
/// `export_events` stores), including the animated x of sliding hold notes.
fn note_screen_pos(chart: &mut Chart, res: &Resource, line_id: usize, note_id: usize, t: f64) -> (f32, f32) {
    chart.lines[line_id].notes[note_id].object.set_time(t);
    let local = chart.lines[line_id].notes[note_id].object.now_translation(res);
    let world = chart.lines[line_id]
        .now_transform(res, &chart.lines)
        .transform_point(&Point::new(local.x, local.y));
    (world.x, -world.y)
}
