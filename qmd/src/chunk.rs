//! Smart document chunking for embedding pipelines.
//!
//! Splits documents into overlapping chunks using scored break points
//! and code fence protection.

/// Default chunk size in characters (~200 tokens).
pub const DEFAULT_CHUNK_CHARS: usize = 3200;

/// Default overlap in characters (~15%).
pub const DEFAULT_OVERLAP_CHARS: usize = DEFAULT_CHUNK_CHARS * 15 / 100;

/// A text chunk produced by [`Chunker`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Chunk {
    /// Chunk text.
    pub text: String,
    /// Byte offset in the source document.
    pub pos: usize,
}

/// Configurable document chunker.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Chunker {
    /// Max characters per chunk.
    pub max_chars: usize,
    /// Overlap in characters.
    pub overlap_chars: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_CHUNK_CHARS,
            overlap_chars: DEFAULT_OVERLAP_CHARS,
        }
    }
}

impl Chunker {
    /// Create a chunker with custom limits.
    #[must_use]
    pub const fn new(max_chars: usize, overlap_chars: usize) -> Self {
        Self {
            max_chars,
            overlap_chars,
        }
    }

    /// Split a document into chunks using scored break points.
    #[must_use]
    pub fn split(&self, content: &str) -> Vec<Chunk> {
        if content.len() <= self.max_chars {
            return vec![Chunk {
                text: content.to_string(),
                pos: 0,
            }];
        }

        let breaks = scan_break_points(content);
        let fences = scan_code_fences(content);
        let window = self.max_chars / 4;

        let mut chunks = Vec::new();
        let mut pos = 0;

        while pos < content.len() {
            // Snap to a char boundary: break points are boundary-safe, but the
            // fallback `pos + max_chars` offset can land inside a multi-byte char,
            // which would panic when slicing `content[pos..end]` below.
            let target_end = floor_char_boundary(content, (pos + self.max_chars).min(content.len()));
            let mut end = target_end;

            if end < content.len()
                && let Some(bp) = find_best_cutoff(&breaks, target_end, window, &fences)
                && bp > pos
                && bp <= target_end
            {
                end = bp;
            }

            if end <= pos {
                end = target_end;
            }

            chunks.push(Chunk {
                text: content[pos..end].to_string(),
                pos,
            });

            if end >= content.len() {
                break;
            }

            let next = floor_char_boundary(content, end.saturating_sub(self.overlap_chars));
            pos = if chunks.last().is_some_and(|c| next <= c.pos) {
                end
            } else {
                next
            };
        }

        chunks
    }
}

/// A scored break point position.
#[derive(Debug, Clone, Copy)]
struct BreakPoint {
    /// Byte offset in the document.
    pos: usize,
    /// Base score — higher means better cut point.
    score: u32,
}

/// Ranked break-point patterns.
const PATTERNS: &[(&str, u32)] = &[
    ("\n# ", 100),
    ("\n## ", 90),
    ("\n### ", 85),
    ("\n#### ", 80),
    ("\n```", 75),
    ("\n---", 70),
    ("\n\n", 60),
    ("\n- ", 45),
    ("\n* ", 45),
    ("\n1. ", 45),
    (". ", 30),
    (".\n", 30),
    ("\n", 10),
];

/// Scan the document for break points, sorted by position.
fn scan_break_points(content: &str) -> Vec<BreakPoint> {
    let mut points = Vec::new();
    for &(pat, score) in PATTERNS {
        let mut start = 0;
        while let Some(idx) = content[start..].find(pat) {
            let abs = start + idx;
            let cut = if pat.starts_with('\n') {
                abs + 1
            } else {
                abs + pat.len()
            };
            if cut > 0 && cut < content.len() {
                points.push(BreakPoint { pos: cut, score });
            }
            start = abs + pat.len().max(1);
        }
    }
    points.sort_by_key(|bp| bp.pos);
    points.dedup_by_key(|bp| bp.pos);
    points
}

/// A byte range `[start, end)` inside a fenced code block.
#[derive(Debug, Clone, Copy)]
struct FenceRange {
    /// Start of the opening fence line.
    start: usize,
    /// End of the closing fence line (exclusive).
    end: usize,
}

/// Scan for fenced code blocks.
fn scan_code_fences(content: &str) -> Vec<FenceRange> {
    let mut ranges = Vec::new();
    let mut open: Option<usize> = None;

    for (idx, line) in content.split('\n').scan(0usize, |pos, line| {
        let start = *pos;
        *pos += line.len() + 1;
        Some((start, line))
    }) {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if let Some(start) = open {
                ranges.push(FenceRange {
                    start,
                    end: idx + line.len(),
                });
                open = None;
            } else {
                open = Some(idx);
            }
        }
    }
    ranges
}

/// Check if a position falls inside a code fence.
fn inside_fence(pos: usize, fences: &[FenceRange]) -> bool {
    fences.iter().any(|f| pos > f.start && pos < f.end)
}

/// Find the best break point near `target` within `±window`.
fn find_best_cutoff(
    breaks: &[BreakPoint],
    target: usize,
    window: usize,
    fences: &[FenceRange],
) -> Option<usize> {
    let lo = target.saturating_sub(window);
    let hi = target + window / 4;

    breaks
        .iter()
        .filter(|bp| bp.pos >= lo && bp.pos <= hi)
        .filter(|bp| !inside_fence(bp.pos, fences))
        .max_by(|a, b| {
            let sa = weighted_score(a, target, window);
            let sb = weighted_score(b, target, window);
            sa.total_cmp(&sb)
        })
        .map(|bp| bp.pos)
}

/// Score a break point with distance decay: `score × (1 - (d/w)²)`.
#[allow(clippy::cast_precision_loss)]
fn weighted_score(bp: &BreakPoint, target: usize, window: usize) -> f64 {
    let dist = bp.pos.abs_diff(target) as f64;
    let w = window as f64;
    let ratio = dist / w;
    let proximity = ratio.mul_add(-ratio, 1.0).max(0.0);
    f64::from(bp.score) * proximity
}

/// Largest index `<= i` that lies on a UTF-8 char boundary of `s`.
///
/// Offsets computed from `str::len()` can land inside a multi-byte character;
/// slicing there panics. Snapping down to the nearest boundary keeps slicing
/// safe while staying within the intended region.
fn floor_char_boundary(s: &str, i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    let mut idx = i;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_char_boundary_snaps_into_multibyte_char() {
        let s = "a…b"; // '…' occupies bytes 1..4
        assert_eq!(floor_char_boundary(s, 1), 1);
        assert_eq!(floor_char_boundary(s, 2), 1);
        assert_eq!(floor_char_boundary(s, 3), 1);
        assert_eq!(floor_char_boundary(s, 4), 4);
        assert_eq!(floor_char_boundary(s, 999), s.len());
    }

    #[test]
    fn split_multibyte_without_break_points_does_not_panic() {
        // A long run of multi-byte chars with no good break points forces the
        // `pos + max_chars` fallback to land inside a char. This used to panic
        // with "byte index N is not a char boundary".
        let content = "…".repeat(2000); // 3 bytes each, no newlines/sentences
        let chunks = Chunker::new(200, 30).split(&content);
        assert!(chunks.len() > 1);
        // Reassembling the chunk texts must reproduce valid UTF-8 covering the doc.
        assert!(chunks.iter().all(|c| !c.text.is_empty()));
    }

    #[test]
    fn split_ascii_unchanged() {
        let content = "a".repeat(500);
        let chunks = Chunker::new(100, 10).split(&content);
        assert!(chunks.len() > 1);
    }
}
