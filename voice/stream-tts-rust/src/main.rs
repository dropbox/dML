use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use tokio::sync::mpsc;

// mod translation_ffi; // Not used in this version

/// Represents a content block in Claude's stream-json output
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
}

/// Represents a Claude message
#[derive(Debug, Deserialize, Serialize)]
struct Message {
    role: Option<String>,
    content: Option<serde_json::Value>,
}

/// Delta structure for incremental text updates
#[derive(Debug, Deserialize, Serialize)]
struct Delta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
}

/// Top-level stream-json message structure
#[derive(Debug, Deserialize, Serialize)]
struct StreamMessage {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    message: Option<Message>,
    role: Option<String>,
    content: Option<serde_json::Value>,
    delta: Option<Delta>,
    stats: Option<serde_json::Value>,
}

/// A sentence ready for TTS processing
#[derive(Debug, Clone)]
struct TextSegment {
    text: String,
}

/// Parser for Claude's stream-json output
struct ClaudeStreamParser {
    markdown_cleaner: Regex,
    url_cleaner: Regex,
    path_cleaner: Regex,
    code_block_pattern: Regex,
}

impl ClaudeStreamParser {
    fn new() -> Result<Self> {
        Ok(Self {
            markdown_cleaner: Regex::new(r"[*_`]")?,
            url_cleaner: Regex::new(r"https?://\S+")?,
            path_cleaner: Regex::new(r"/[/\w\-\.]+")?,
            code_block_pattern: Regex::new(r"```[\s\S]*?```")?,
        })
    }

    /// Extract text content from a message
    fn extract_text_from_message(&self, msg: &StreamMessage) -> Vec<String> {
        let mut texts = Vec::new();

        // Handle delta events (content_block_delta)
        if let Some(ref delta) = msg.delta {
            if let Some(ref text) = delta.text {
                if !text.trim().is_empty() {
                    texts.push(text.clone());
                }
            }
            return texts;
        }

        // Get content from either nested message or top-level
        let content = if let Some(ref inner_msg) = msg.message {
            inner_msg.content.as_ref()
        } else {
            msg.content.as_ref()
        };

        if let Some(content_val) = content {
            // Handle both array and single object
            let blocks: Vec<ContentBlock> = match content_val {
                serde_json::Value::Array(arr) => arr
                    .iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect(),
                serde_json::Value::Object(_) => {
                    if let Ok(block) = serde_json::from_value(content_val.clone()) {
                        vec![block]
                    } else {
                        vec![]
                    }
                }
                serde_json::Value::String(s) => {
                    texts.push(s.clone());
                    return texts;
                }
                _ => vec![],
            };

            // Extract text from text blocks only (ignore tool use/results)
            for block in blocks {
                if let ContentBlock::Text { text } = block {
                    if !text.trim().is_empty() {
                        texts.push(text);
                    }
                }
            }
        }

        texts
    }

    /// Clean text for TTS: remove markdown, code blocks, URLs, paths
    fn clean_text_for_speech(&self, text: &str) -> String {
        // Remove code blocks first
        let text = self.code_block_pattern.replace_all(text, "");

        // Remove markdown formatting
        let text = self.markdown_cleaner.replace_all(&text, "");

        // Remove URLs
        let text = self.url_cleaner.replace_all(&text, "URL");

        // Remove file paths
        let text = self.path_cleaner.replace_all(&text, "");

        // Clean up whitespace
        text.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Filter out system noise
    fn should_skip_text(&self, text: &str) -> bool {
        text.contains("Co-Authored-By:")
            || text.contains("🤖 Generated with")
            || text.contains("<system-reminder>")
            || text.contains("</system-reminder>")
            || text.contains("you should consider whether it would be considered malware")
    }

    /// Determine if text should be spoken (length check)
    fn should_speak(&self, text: &str) -> bool {
        let clean = text.trim();
        clean.len() >= 10 // Minimum length for speech
    }

    /// Segment text into sentences for streaming TTS
    fn segment_into_sentences(&self, text: &str) -> Vec<String> {
        // Split on common sentence boundaries
        let sentences: Vec<String> = text
            .split(|c| c == '.' || c == '!' || c == '?' || c == '。')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.len() >= 10 {
                    Some(trimmed.to_string())
                } else {
                    None
                }
            })
            .collect();

        if sentences.is_empty() && text.trim().len() >= 10 {
            // If no sentence boundaries found but text is long enough, return as-is
            vec![text.trim().to_string()]
        } else {
            sentences
        }
    }

    /// Process a single line of JSON input
    fn process_line(&self, line: &str) -> Result<Vec<TextSegment>> {
        let msg: StreamMessage = serde_json::from_str(line)
            .context("Failed to parse JSON")?;

        let mut segments = Vec::new();

        // Extract text from message
        let texts = self.extract_text_from_message(&msg);

        for text in texts {
            // Skip system noise
            if self.should_skip_text(&text) {
                continue;
            }

            // Clean text
            let cleaned = self.clean_text_for_speech(&text);

            if cleaned.is_empty() {
                continue;
            }

            // Check if should speak
            let is_speakable = self.should_speak(&cleaned);

            if !is_speakable {
                continue;
            }

            // Segment into sentences
            let sentences = self.segment_into_sentences(&cleaned);

            for sentence in sentences {
                segments.push(TextSegment {
                    text: sentence,
                });
            }
        }

        Ok(segments)
    }
}

/// Worker process manager for translation and TTS
struct WorkerManager {
    translation_process: Child,
    tts_process: Child,
}

impl WorkerManager {
    fn spawn_workers() -> Result<Self> {
        eprintln!("🔧 Starting Python workers...");

        // Get the python directory path - try multiple locations
        let python_dir = if let Ok(cwd) = std::env::current_dir() {
            // Try: stream-tts-rust/python relative to cwd
            let cwd_python = cwd.join("stream-tts-rust/python");
            if cwd_python.exists() {
                cwd_python
            } else {
                // Try: python relative to cwd (if we're inside stream-tts-rust)
                let local_python = cwd.join("python");
                if local_python.exists() {
                    local_python
                } else {
                    // Try: ../python (if we're in target/release)
                    std::env::current_exe()
                        .ok()
                        .and_then(|p| p.parent().map(|p| p.join("../../python")))
                        .and_then(|p| p.canonicalize().ok())
                        .unwrap_or_else(|| std::path::PathBuf::from("python"))
                }
            }
        } else {
            std::path::PathBuf::from("python")
        };

        // PRODUCTION: NLLB-200 + Google TTS (340ms total)
        // Translation: NLLB-200-600M on Metal GPU (154ms)
        // TTS: Google TTS cloud API (185ms)
        // Total: ~340ms (exceeds industry standard 300-500ms)
        let translation_script = python_dir.join("translation_worker_optimized.py");
        let tts_script = python_dir.join("tts_worker_gtts.py");

        eprintln!("   Translation script: {:?}", translation_script);
        eprintln!("   TTS script: {:?}", tts_script);

        // Use venv Python if available, otherwise system python3
        let python_exe = if std::path::Path::new("venv/bin/python").exists() {
            "venv/bin/python"
        } else if std::path::Path::new("/Users/ayates/voice/venv/bin/python").exists() {
            "/Users/ayates/voice/venv/bin/python"
        } else {
            "python3"
        };
        eprintln!("   Using Python: {}", python_exe);

        // Spawn translation worker
        let translation_process = Command::new(python_exe)
            .arg(&translation_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn translation worker")?;

        eprintln!("✅ Translation worker spawned (PID: {})", translation_process.id());

        // Spawn TTS worker
        let tts_process = Command::new(python_exe)
            .arg(&tts_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn TTS worker")?;

        eprintln!("✅ TTS worker spawned (PID: {})", tts_process.id());

        // Give workers time to initialize
        std::thread::sleep(std::time::Duration::from_secs(2));

        Ok(Self {
            translation_process,
            tts_process,
        })
    }

    fn translate(&mut self, text: &str) -> Result<String> {
        // Send text to translation worker
        let stdin = self.translation_process.stdin.as_mut()
            .context("Failed to get translation worker stdin")?;
        writeln!(stdin, "{}", text)?;
        stdin.flush()?;

        // Read response
        let stdout = self.translation_process.stdout.as_mut()
            .context("Failed to get translation worker stdout")?;
        let mut reader = BufReader::new(stdout);
        let mut response = String::new();
        reader.read_line(&mut response)?;

        Ok(response.trim().to_string())
    }

    fn synthesize(&mut self, text: &str) -> Result<String> {
        // Send text to TTS worker
        let stdin = self.tts_process.stdin.as_mut()
            .context("Failed to get TTS worker stdin")?;
        writeln!(stdin, "{}", text)?;
        stdin.flush()?;

        // Read audio file path
        let stdout = self.tts_process.stdout.as_mut()
            .context("Failed to get TTS worker stdout")?;
        let mut reader = BufReader::new(stdout);
        let mut audio_path = String::new();
        reader.read_line(&mut audio_path)?;

        Ok(audio_path.trim().to_string())
    }
}

impl Drop for WorkerManager {
    fn drop(&mut self) {
        eprintln!("🛑 Shutting down workers...");
        let _ = self.translation_process.kill();
        let _ = self.tts_process.kill();
        let _ = self.translation_process.wait();
        let _ = self.tts_process.wait();
        eprintln!("✅ Workers stopped");
    }
}

/// Play audio file using afplay (macOS built-in)
fn play_audio_file(path: &str) -> Result<()> {
    std::process::Command::new("afplay")
        .arg(path)
        .output()
        .context("Failed to play audio with afplay")?;
    Ok(())
}

/// Main async TTS pipeline coordinator
async fn run_tts_pipeline(mut rx: mpsc::Receiver<TextSegment>) -> Result<()> {
    // Spawn workers
    let workers = WorkerManager::spawn_workers()?;

    // Wrap in Arc<Mutex> for sharing across tasks
    let workers = std::sync::Arc::new(std::sync::Mutex::new(workers));

    eprintln!("🔊 Audio output ready (using afplay)");
    eprintln!("📡 Ready to process text segments\n");

    while let Some(segment) = rx.recv().await {
        eprintln!("[Pipeline] Processing: {}", &segment.text[..segment.text.len().min(50)]);

        // Clone Arc for this task
        let workers_clone = workers.clone();
        let text = segment.text.clone();

        // Run translation, TTS, and audio playback in blocking task
        let result = tokio::task::spawn_blocking(move || -> Result<String> {
            // Use lock().unwrap_or_else to handle poisoned mutex
            let mut workers = workers_clone.lock().unwrap_or_else(|poisoned| {
                eprintln!("[Pipeline] ⚠ Mutex was poisoned, recovering...");
                poisoned.into_inner()
            });

            // Translate
            let translated = workers.translate(&text)?;
            if translated.is_empty() {
                return Err(anyhow::anyhow!("Empty translation response"));
            }

            // UTF-8 safe string truncation for logging
            let preview: String = translated.chars().take(30).collect();
            eprintln!("[Pipeline] ✓ Translated: {}...", preview);

            // Synthesize
            let audio_path = workers.synthesize(&translated)?;
            if audio_path.is_empty() {
                return Err(anyhow::anyhow!("Empty TTS response"));
            }

            eprintln!("[Pipeline] ✓ Audio generated: {}", audio_path);

            // Play audio
            if std::path::Path::new(&audio_path).exists() {
                play_audio_file(&audio_path)?;
                eprintln!("[Pipeline] ✓ Played audio");
            } else {
                eprintln!("[Pipeline] ⚠ Audio file not found: {}", audio_path);
            }

            Ok(translated)
        })
        .await;

        match result {
            Ok(Ok(translated)) => {
                let preview: String = translated.chars().take(30).collect();
                eprintln!("[Pipeline] ✅ Complete: {}...\n", preview);
            }
            Ok(Err(e)) => {
                eprintln!("[Pipeline] ✗ Error: {}\n", e);
            }
            Err(e) => {
                eprintln!("[Pipeline] ✗ Task error: {}\n", e);
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    eprintln!("🚀 Stream TTS Rust - Starting up...");

    let parser = ClaudeStreamParser::new()?;

    // Create channel for text segments
    let (tx, rx) = mpsc::channel::<TextSegment>(100);

    // Spawn TTS pipeline task
    let tts_task = tokio::spawn(async move {
        run_tts_pipeline(rx).await
    });

    // Read from stdin
    let stdin = io::stdin();
    let reader = stdin.lock();

    eprintln!("📥 Listening for Claude JSON output on stdin...\n");

    for line in reader.lines() {
        let line = line.context("Failed to read line from stdin")?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as JSON
        match parser.process_line(&line) {
            Ok(segments) => {
                for segment in segments {
                    // Send to TTS pipeline
                    tx.send(segment).await?;
                }
            }
            Err(e) => {
                // Not JSON or parse error - this is expected for non-JSON lines
                eprintln!("[SKIP] Not JSON: {}", e);
            }
        }
    }

    // Close channel and wait for TTS to finish
    drop(tx);
    tts_task.await??;

    eprintln!("\n✅ Stream TTS completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text_for_speech() {
        let parser = ClaudeStreamParser::new().unwrap();

        let text = "This is **bold** and `code` with a https://example.com URL";
        let cleaned = parser.clean_text_for_speech(text);
        assert_eq!(cleaned, "This is bold and code with a URL URL");
    }

    #[test]
    fn test_segment_sentences() {
        let parser = ClaudeStreamParser::new().unwrap();

        let text = "First sentence. Second sentence. Third sentence!";
        let sentences = parser.segment_into_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence");
    }

    #[test]
    fn test_should_speak() {
        let parser = ClaudeStreamParser::new().unwrap();

        assert!(!parser.should_speak("Short"));
        assert!(parser.should_speak("This is a longer sentence that should be spoken"));
    }
}
