# Video & Audio Extraction System - Technical Specification
## AI Implementation Reference

**Purpose**: Complete technical specification for AI-driven implementation
**Scope**: Architecture, APIs, modules, tools, testing
**Audience**: AI coding agents

---

## PERFORMANCE NOTE

All performance metrics are reference values. Actual targets established through benchmarking.
**Core principle**: System must be faster than alternatives.

---

## 1. ARCHITECTURE DESIGN

### 1.1 Multi-Tier Processing Architecture

**Rationale**: Separate CPU and GPU workloads for optimal resource utilization.

```
Input → Ingestion → Orchestrator
                         ↓
         ┌───────────────┴───────────────┐
         ↓                               ↓
    CPU Tier                         GPU Tier
    (Fast I/O operations)            (ML inference)
         ↓                               ↓
         └───────────────┬───────────────┘
                         ↓
                  Fusion Layer
                         ↓
              Storage & Indexing
```

**CPU Tier**: FFmpeg operations, classical CV, format conversion
**GPU Tier**: Neural network inference (transcription, detection, embeddings)
**Why separated**: CPU operations don't block GPU, GPU operations batched efficiently

### 1.2 Three API Modes

**Design Decision**: Different use cases require different optimizations

#### Mode 1: Real-Time API
**Goal**: Minimum latency for single file
**Optimization**: Parallel CPU+GPU, no queuing, streaming results
**Tradeoff**: Lower CPU/GPU efficiency (60-70%) for faster response
**Use case**: User uploads file, wants immediate results

#### Mode 2: Bulk Processing API
**Goal**: Maximum throughput per CPU/GPU
**Optimization**: Staged processing, ML batching, queue depth = 2x cores
**Tradeoff**: Higher latency per file for better efficiency (90%+ CPU, 85%+ GPU)
**Use case**: Process thousands of files overnight

#### Mode 3: CPU-Only vs GPU+CPU
**CPU-Only**: Classical algorithms, no ML models
**GPU+CPU**: Full ML pipeline with neural networks
**Tradeoff**: Speed vs feature completeness

**Why three modes**: No single optimization works for all scenarios (Amdahl's law)

### 1.3 Language Choice: Rust + C++

**Primary**: Rust
- Memory safety without GC
- Zero-cost abstractions
- Excellent concurrency (tokio, rayon)
- Growing ML ecosystem (ort, ndarray)

**Secondary**: C++
- FFmpeg bindings (libav*)
- OpenCV operations
- CUDA kernels (if needed)

**Python Bridge**: PyO3 for ML models not yet in ONNX

**Why not pure Python**: Too slow for CPU-bound operations
**Why not pure C++**: Memory safety critical for production

### 1.4 Storage Architecture

**Object Storage** (S3/MinIO): Raw files, extracted assets
**Vector DB** (Qdrant): Semantic embeddings
**Search Index** (Tantivy): Full-text transcript search
**Metadata DB** (PostgreSQL): Structured data, timelines

**Why not single database**: Different access patterns require specialized storage

---

## 2. API SPECIFICATIONS

### 2.1 Real-Time API

**Endpoint**: `POST /api/v1/process/realtime`

**Request**:
```json
{
  "source": {
    "type": "url|upload|s3",
    "location": "string"
  },
  "processing": {
    "priority": "realtime",
    "required_features": ["transcription", "keyframes", "scenes"],
    "optional_features": ["objects", "faces", "ocr"],
    "quality_mode": "fast|balanced|accurate"
  },
  "streaming": {
    "enabled": true,
    "protocol": "sse|websocket"
  }
}
```

**Response** (streaming):
```
event: metadata
data: {"duration": 600, "resolution": "1920x1080"}

event: keyframe
data: {"timestamp": 0.0, "url": "s3://..."}

event: transcript_segment
data: {"start": 0.0, "end": 5.5, "text": "...", "confidence": 0.94}

event: complete
data: {"job_id": "...", "timeline_url": "..."}
```

**Architecture**:
- Immediate worker assignment (reject if resources unavailable)
- Parallel CPU + GPU tasks (concurrent, not sequential)
- Stream partial results as available
- Preloaded ML models (no cold start)
- Resource reservation (4-8 CPU cores, 4-8GB VRAM per request)

### 2.2 Bulk Processing API

**Endpoint**: `POST /api/v1/process/bulk`

**Request**:
```json
{
  "batch_id": "string",
  "files": [
    {
      "id": "file_1",
      "source": {"type": "s3", "location": "s3://..."},
      "processing": {"transcription": true, "embeddings": true}
    }
  ],
  "batch_config": {
    "priority": "bulk",
    "optimize_for": "throughput",
    "callback_url": "https://..."
  }
}
```

**Response** (webhook on completion):
```json
{
  "batch_id": "...",
  "status": "completed",
  "results": [
    {"file_id": "file_1", "status": "success", "timeline_url": "..."}
  ],
  "batch_stats": {
    "throughput_multiplier": 2.0,
    "cpu_efficiency": 0.92,
    "gpu_efficiency": 0.87
  }
}
```

**Architecture**:
- Staged processing: All CPU work → All GPU work
- ML model batching (8-16 files for transcription, 32-64 images for detection)
- Queue depth = 2x CPU cores
- Single model instance shared across batch

**Batching Configuration**:
```rust
const VIDEO_DECODE_BATCH: usize = 8;
const TRANSCRIPTION_BATCH: usize = 8;
const IMAGE_DETECTION_BATCH: usize = 32;
const EMBEDDING_BATCH: usize = 128;
const CPU_QUEUE_DEPTH: usize = NUM_CORES * 2;
const GPU_QUEUE_DEPTH: usize = 16;
```

### 2.3 CPU-Only vs GPU+CPU Configuration

**CPU-Only Request**:
```json
{
  "processing": {
    "mode": "cpu_only",
    "features": {
      "metadata": true,
      "keyframes": true,
      "scenes": {"method": "classical"},
      "transcription": {"enabled": false}
    }
  }
}
```

**GPU+CPU Request**:
```json
{
  "processing": {
    "mode": "gpu_accelerated",
    "hardware_decode": true,
    "features": {
      "transcription": {"model": "whisper-large-v3", "language": "auto"},
      "diarization": {"enabled": true},
      "visual_intelligence": {"objects": true, "faces": true, "ocr": true},
      "audio_intelligence": {"events": true, "separation": true}
    }
  }
}
```

---

## 3. CORE PROCESSING MODULES

### 3.1 Ingestion Module

**Function Signature**:
```rust
fn ingest_media(path: &Path) -> Result<MediaInfo, Error>

struct MediaInfo {
    format: String,           // "mp4", "mov", "avi"
    duration: f64,            // seconds
    streams: Vec<StreamInfo>,
    metadata: HashMap<String, String>,
}

struct StreamInfo {
    stream_type: StreamType,  // Video, Audio, Subtitle
    codec: String,
    bitrate: u64,
    // Video-specific
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<f64>,
    // Audio-specific
    sample_rate: Option<u32>,
    channels: Option<u8>,
}
```

**Implementation**: FFprobe wrapper
**Hardware Acceleration**: Not applicable (metadata only)

### 3.2 Video Decoder Module

**Function Signature**:
```rust
fn decode_video(
    input_path: &Path,
    output_format: PixelFormat,
    frame_filter: Option<FrameFilter>,
    hardware_accel: bool,
) -> Result<Vec<Frame>, Error>

enum PixelFormat {
    YUV420P,  // Most common
    RGB24,    // For image processing
}

enum FrameFilter {
    EveryNth(u32),
    IFramesOnly,
    Timestamps(Vec<f64>),
}
```

**Hardware Acceleration Priority**:
```rust
const HW_DECODERS: &[&str] = &[
    "h264_nvdec", "hevc_nvdec",         // NVIDIA
    "h264_vaapi", "hevc_vaapi",         // Intel/AMD Linux
    "h264_videotoolbox", "hevc_videotoolbox",  // Apple
    "h264_qsv", "hevc_qsv",             // Intel Quick Sync
];
```

**Implementation**: FFmpeg libavcodec + hardware decoder detection

### 3.3 Keyframe Extractor Module

**Algorithm**:
1. Detect I-frames (instant decode)
2. Calculate perceptual hash (dHash)
3. Filter duplicates (Hamming distance < 10)
4. Detect scene boundaries (histogram difference in HSV)
5. Select best frame per scene (Laplacian variance for sharpness)
6. Generate multi-resolution thumbnails

**Function Signature**:
```rust
struct KeyframeExtractor {
    interval: f64,
    max_keyframes: usize,
    similarity_threshold: u32,
    thumbnail_sizes: Vec<(u32, u32)>,
}

fn extract_keyframes(
    video_path: &Path,
    config: KeyframeExtractor,
) -> Result<Vec<Keyframe>, Error>

struct Keyframe {
    timestamp: f64,
    frame_number: u64,
    hash: u64,
    sharpness: f64,
    thumbnail_paths: HashMap<String, PathBuf>,
}
```

### 3.4 Scene Detector Module

**Classical Method**:
```rust
struct SceneDetector {
    threshold: f64,       // 0.3 default
    min_scene_length: f64,
    method: SceneDetectionMethod,
}

enum SceneDetectionMethod {
    ContentAware,  // Histogram difference in HSV
    Threshold,     // Fade detection
    Adaptive,      // Dynamic threshold
}

fn detect_scenes(
    video_path: &Path,
    config: SceneDetector,
) -> Result<Vec<Scene>, Error>

struct Scene {
    start_time: f64,
    end_time: f64,
    start_frame: u64,
    end_frame: u64,
    confidence: f64,
}
```

**Implementation**: PySceneDetect or custom Rust implementation

### 3.5 Audio Extractor Module

**Function Signature**:
```rust
fn extract_audio(
    input_path: &Path,
    output_config: AudioConfig,
) -> Result<PathBuf, Error>

struct AudioConfig {
    sample_rate: u32,     // 16000 for ML, 48000 for storage
    channels: u8,         // 1 (mono) for ML, 2 (stereo) for storage
    format: AudioFormat,  // PCM, FLAC, M4A, MP3
    normalize: bool,      // EBU R128 normalization
}

enum AudioFormat {
    PCM,   // Uncompressed, for ML
    FLAC,  // Lossless, for storage
    M4A,   // Compressed, for storage
    MP3,   // Compatibility
}
```

**Implementation**: FFmpeg audio filter chain

### 3.6 Transcription Module (Whisper)

**Function Signature**:
```rust
struct TranscriptionConfig {
    model_size: WhisperModel,
    language: Option<String>,  // "en", "es", or None for auto
    word_timestamps: bool,
    beam_size: u8,
    temperature: f32,
    compute_type: ComputeType,
}

enum WhisperModel {
    Tiny,      // 39M params
    Base,      // 74M params
    Small,     // 244M params
    Medium,    // 769M params
    LargeV3,   // 1.5B params
}

enum ComputeType {
    Float32,
    Float16,
    Int8,
}

fn transcribe_audio(
    audio_path: &Path,
    config: TranscriptionConfig,
) -> Result<Transcript, Error>

struct Transcript {
    text: String,
    language: String,
    language_probability: f32,
    segments: Vec<TranscriptSegment>,
}

struct TranscriptSegment {
    start: f64,
    end: f64,
    text: String,
    words: Vec<WordTiming>,
    no_speech_prob: f32,
}

struct WordTiming {
    word: String,
    start: f64,
    end: f64,
    probability: f32,
}
```

**Implementation**: Faster-Whisper via PyO3 or Whisper.cpp
**Languages**: 99 supported

### 3.7 Speaker Diarization Module

**Function Signature**:
```rust
struct DiarizationConfig {
    min_speakers: Option<u8>,
    max_speakers: Option<u8>,
    embedding_model: String,
    segmentation_model: String,
}

fn diarize_audio(
    audio_path: &Path,
    config: DiarizationConfig,
) -> Result<Diarization, Error>

struct Diarization {
    speakers: Vec<Speaker>,
    timeline: Vec<SpeakerSegment>,
}

struct Speaker {
    id: String,  // "SPEAKER_00", "SPEAKER_01"
    total_speaking_time: f64,
}

struct SpeakerSegment {
    start: f64,
    end: f64,
    speaker: String,
    confidence: f32,
}
```

**Implementation**: PyAnnote.audio via PyO3

### 3.8 Object Detection Module (YOLO)

**Function Signature**:
```rust
struct ObjectDetectionConfig {
    model_size: YOLOModel,
    confidence_threshold: f32,  // 0.25 default
    iou_threshold: f32,         // 0.45 default
    classes: Option<Vec<u8>>,   // Filter to specific COCO classes
    max_detections: usize,      // 300 default
}

enum YOLOModel {
    Nano,    // 6MB
    Small,   // 22MB
    Medium,  // 52MB
    Large,   // 87MB
    XLarge,  // 136MB
}

fn detect_objects(
    images: &[Image],
    config: ObjectDetectionConfig,
) -> Result<Vec<Vec<Detection>>, Error>

struct Detection {
    class_id: u8,
    class_name: String,
    confidence: f32,
    bbox: BoundingBox,
}

struct BoundingBox {
    x: f32,      // Normalized 0-1
    y: f32,
    width: f32,
    height: f32,
}
```

**Implementation**: YOLOv8 via ONNX Runtime
**Classes**: 80 COCO classes (person, car, laptop, etc.)

### 3.9 Face Detection Module

**Function Signature**:
```rust
struct FaceDetectionConfig {
    confidence_threshold: f32,
    nms_threshold: f32,
    detect_landmarks: bool,  // 5-point landmarks
}

fn detect_faces(
    images: &[Image],
    config: FaceDetectionConfig,
) -> Result<Vec<Vec<Face>>, Error>

struct Face {
    confidence: f32,
    bbox: BoundingBox,
    landmarks: Option<FacialLandmarks>,
}

struct FacialLandmarks {
    left_eye: (f32, f32),
    right_eye: (f32, f32),
    nose: (f32, f32),
    left_mouth: (f32, f32),
    right_mouth: (f32, f32),
}
```

**Implementation**: RetinaFace via ONNX Runtime

### 3.10 OCR Module

**Function Signature**:
```rust
struct OCRConfig {
    languages: Vec<String>,
    detect_direction: bool,
    detection_threshold: f32,
    recognition_threshold: f32,
}

fn detect_text(
    images: &[Image],
    config: OCRConfig,
) -> Result<Vec<Vec<TextRegion>>, Error>

struct TextRegion {
    text: String,
    confidence: f32,
    bbox: Polygon,  // 4-point polygon for rotated text
    direction: TextDirection,
}

enum TextDirection {
    Horizontal,
    Vertical,
    Rotated(f32),
}
```

**Implementation**: PaddleOCR via ONNX Runtime
**Languages**: 80+ supported

### 3.11 Embedding Modules

**Vision Embeddings**:
```rust
struct VisionEmbeddingConfig {
    model: CLIPModel,
    normalize: bool,
}

enum CLIPModel {
    VitB32,  // 512-dim, 149M params
    VitL14,  // 768-dim, 428M params
}

fn extract_vision_embeddings(
    images: &[Image],
    config: VisionEmbeddingConfig,
) -> Result<Vec<Vec<f32>>, Error>
```

**Text Embeddings**:
```rust
struct TextEmbeddingConfig {
    model: String,  // "all-MiniLM-L6-v2", "all-mpnet-base-v2"
    normalize: bool,
}

fn extract_text_embeddings(
    texts: &[String],
    config: TextEmbeddingConfig,
) -> Result<Vec<Vec<f32>>, Error>
```

**Popular Models**:
- `all-MiniLM-L6-v2`: 384-dim, fast
- `all-mpnet-base-v2`: 768-dim, accurate

**Audio Embeddings**:
```rust
struct AudioEmbeddingConfig {
    model: String,  // "laion/clap-htsat-fused"
    normalize: bool,
}

fn extract_audio_embeddings(
    audio_clips: &[AudioClip],
    config: AudioEmbeddingConfig,
) -> Result<Vec<Vec<f32>>, Error>
```

### 3.12 Fusion Module

**Function Signature**:
```rust
fn fuse_results(
    metadata: MediaInfo,
    transcript: Option<Transcript>,
    diarization: Option<Diarization>,
    scenes: Vec<Scene>,
    objects: Vec<Vec<Detection>>,
    faces: Vec<Vec<Face>>,
    text_regions: Vec<Vec<TextRegion>>,
) -> Result<Timeline, Error>

struct Timeline {
    duration: f64,
    events: Vec<Event>,
    entities: Vec<Entity>,
    relationships: Vec<Relationship>,
    quality_scores: QualityScores,
}

struct Event {
    id: String,
    event_type: EventType,
    start_time: f64,
    end_time: f64,
    confidence: f32,
    data: serde_json::Value,
}

enum EventType {
    TranscriptSegment,
    SpeakerChange,
    SceneBoundary,
    ObjectDetection,
    FaceDetection,
    TextDetection,
    AudioEvent,
}
```

**Cross-Modal Linking**:
1. Match detected persons (visual) with speakers (audio) by timestamp overlap
2. Match detected text (visual) with spoken words (audio) by content similarity
3. Correlate audio events with visual scenes by timestamp containment

---

## 4. TECHNOLOGY STACK

### 4.1 Core Dependencies

**Rust Crates**:
```toml
[dependencies]
tokio = { version = "1.35", features = ["full"] }
ffmpeg-next = "6.1"
ort = "1.16"  # ONNX Runtime
pyo3 = { version = "0.20", features = ["auto-initialize"] }
image = "0.24"
ndarray = "0.15"
aws-sdk-s3 = "1.8"
qdrant-client = "1.7"
tokio-postgres = "0.7"
redis = { version = "0.24", features = ["tokio-comp"] }
async-nats = "0.33"
tantivy = "0.21"
serde = { version = "1.0", features = ["derive"] }
rayon = "1.8"
```

**Python Dependencies** (ML models):
```txt
faster-whisper==1.0.0
torch==2.1.0
pyannote.audio==3.1.0
ultralytics==8.0.200
paddleocr==2.7.0
sentence-transformers==2.2.2
demucs==4.0.1
```

### 4.2 ML Model Selection

**Speech-to-Text**: Faster-Whisper (primary) or Whisper.cpp (CPU-only alternative)
- **Why**: 4x faster than OpenAI Whisper, same accuracy
- **Alternative**: Whisper.cpp for pure C++ implementation

**Speaker Diarization**: PyAnnote.audio
- **Why**: State-of-the-art DER (8-12%)
- **Alternative**: Speechbrain

**Object Detection**: YOLOv8
- **Why**: Best speed/accuracy tradeoff, ONNX export
- **Alternative**: DETR for better small object detection

**Face Detection**: RetinaFace
- **Why**: Best accuracy on WIDER FACE benchmark
- **Alternative**: MTCNN for lightweight

**OCR**: PaddleOCR
- **Why**: 80+ languages, high accuracy, ONNX export
- **Alternative**: EasyOCR for simpler API

**Embeddings**:
- Vision: CLIP (multi-modal) or DINOv2 (vision-only)
- Text: Sentence-Transformers (all-MiniLM-L6-v2 for speed, all-mpnet-base-v2 for accuracy)
- Audio: CLAP

**Scene Detection**: TransNetV2 (ML) + PySceneDetect (classical)
- **Why**: Hybrid approach combines speed + accuracy

**Audio Separation**: Demucs
- **Why**: State-of-the-art SDR (7-10 dB)

**Audio Events**: PANNs
- **Why**: 527 AudioSet classes, fast inference

### 4.3 Storage & Infrastructure

**Object Storage**: MinIO (self-hosted) or S3 (cloud)
- **Why**: S3-compatible, proven at scale

**Vector Database**: Qdrant (recommended) or Milvus
- **Why Qdrant**: Rust-native, fast, advanced filtering
- **Alternative Milvus**: Better for >1B vectors

**Search Index**: Tantivy (embedded) or Meilisearch (service)
- **Why Tantivy**: Rust-native, embeddable, Lucene-like
- **Alternative Meilisearch**: Better for typo-tolerance, instant search

**Metadata DB**: PostgreSQL 16+
- **Why**: JSONB, reliability, extensions (pgvector)

**Cache**: Redis 7+
- **Why**: Industry standard, fast

**Message Queue**: NATS (recommended) or Kafka
- **Why NATS**: Lightweight, excellent Rust client
- **Alternative Kafka**: Better for multi-consumer patterns

---

## 5. PROCESSING TASK GRAPHS

### 5.1 Real-Time API Task Graph

```rust
fn build_realtime_graph(job_id: String) -> TaskGraph {
    let mut graph = TaskGraph::new(job_id);

    // Root
    graph.add_task("ingestion", Task::Ingestion, vec![]);

    // CPU tier (parallel)
    graph.add_task("metadata", Task::MetadataExtraction, vec!["ingestion"]);
    graph.add_task("audio_extract", Task::AudioExtraction, vec!["ingestion"]);
    graph.add_task("keyframes", Task::KeyframeExtraction, vec!["ingestion"]);
    graph.add_task("scenes_classical", Task::SceneDetection, vec!["ingestion"]);

    // GPU tier (parallel)
    graph.add_task("transcription", Task::SpeechToText, vec!["audio_extract"]);
    graph.add_task("diarization", Task::SpeakerDiarization, vec!["audio_extract"]);
    graph.add_task("objects", Task::ObjectDetection, vec!["keyframes"]);
    graph.add_task("faces", Task::FaceDetection, vec!["keyframes"]);
    graph.add_task("ocr", Task::OCR, vec!["keyframes"]);
    graph.add_task("frame_embeddings", Task::FrameEmbeddings, vec!["keyframes"]);
    graph.add_task("text_embeddings", Task::TextEmbeddings, vec!["transcription"]);

    // Fusion (depends on all)
    graph.add_task("fusion", Task::Fusion, vec![
        "metadata", "transcription", "diarization", "objects",
        "faces", "ocr", "frame_embeddings", "text_embeddings", "scenes_classical"
    ]);

    // Storage
    graph.add_task("storage", Task::Storage, vec!["fusion"]);

    graph
}
```

**Execution**: Topological sort, execute all tasks with satisfied dependencies in parallel

### 5.2 Bulk API Task Graph

```rust
fn build_bulk_graph(batch_id: String, files: Vec<FileConfig>) -> TaskGraph {
    let mut graph = TaskGraph::new(batch_id);

    // Stage 1: Ingestion (all files parallel)
    for (i, _) in files.iter().enumerate() {
        graph.add_task(&format!("ingestion_{}", i), Task::Ingestion, vec![]);
    }

    // Stage 2: CPU processing (all files parallel)
    for i in 0..files.len() {
        graph.add_task(&format!("cpu_{}", i), Task::CPUProcessing,
            vec![format!("ingestion_{}", i)]);
    }

    // Stage 3: GPU processing (batched)
    let transcription_batches = files.chunks(8);
    for (batch_idx, batch) in transcription_batches.enumerate() {
        let deps: Vec<String> = batch.iter().enumerate()
            .map(|(i, _)| format!("cpu_{}", i)).collect();
        graph.add_task(&format!("transcription_batch_{}", batch_idx),
            Task::BatchTranscription, deps);
    }

    let detection_batches = files.chunks(32);
    for (batch_idx, batch) in detection_batches.enumerate() {
        let deps: Vec<String> = batch.iter().enumerate()
            .map(|(i, _)| format!("cpu_{}", i)).collect();
        graph.add_task(&format!("detection_batch_{}", batch_idx),
            Task::BatchObjectDetection, deps);
    }

    // Stage 4: Fusion (per file)
    for i in 0..files.len() {
        let deps = vec![
            format!("transcription_batch_{}", i / 8),
            format!("detection_batch_{}", i / 32),
        ];
        graph.add_task(&format!("fusion_{}", i), Task::Fusion, deps);
    }

    // Stage 5: Storage (per file)
    for i in 0..files.len() {
        graph.add_task(&format!("storage_{}", i), Task::Storage,
            vec![format!("fusion_{}", i)]);
    }

    graph
}
```

**Execution**: Execute stages sequentially, maximize parallelism within stages

---

## 6. OPTIMIZATION TECHNIQUES

### 6.1 Hardware Acceleration

**Video Decoding**:
```rust
fn open_hw_decoder(codec_name: &str) -> Result<Decoder> {
    let hw_types = vec!["nvdec", "vaapi", "videotoolbox", "qsv"];

    for hw_type in hw_types {
        let hw_codec = format!("{}_{}", codec_name, hw_type);
        if let Ok(decoder) = avcodec_find_decoder_by_name(&hw_codec) {
            return Ok(decoder);
        }
    }

    // Fallback to software
    avcodec_find_decoder_by_name(codec_name)
}
```

### 6.2 GPU Memory Management

```rust
struct GPUMemoryPool {
    transcription_pool: CUDAMemoryPool,  // 10GB
    detection_pool: CUDAMemoryPool,      // 8GB
    embedding_pool: CUDAMemoryPool,      // 4GB
}

impl GPUMemoryPool {
    fn allocate_for_transcription(&mut self, size: usize) -> *mut c_void {
        self.transcription_pool.allocate_or_reuse(size)
    }

    fn release(&mut self, ptr: *mut c_void) {
        // Don't free, return to pool for reuse
        self.transcription_pool.return_to_pool(ptr);
    }
}
```

### 6.3 Model Batching

```rust
struct BatchedInference {
    max_batch_size: usize,
    max_wait_time: Duration,
    pending_requests: Vec<InferenceRequest>,
}

impl BatchedInference {
    async fn add_request(&mut self, request: InferenceRequest) {
        self.pending_requests.push(request);

        if self.pending_requests.len() >= self.max_batch_size {
            self.flush_batch().await;
        }
    }

    async fn flush_batch(&mut self) {
        let batch = std::mem::take(&mut self.pending_requests);
        let results = self.model.infer_batch(&batch).await;

        for (request, result) in batch.into_iter().zip(results) {
            request.respond(result);
        }
    }
}
```

### 6.4 Quality Presets

```rust
enum QualityPreset {
    Fast,      // Min quality, max speed
    Balanced,  // Good quality, reasonable speed
    Accurate,  // Max quality
}

impl QualityPreset {
    fn to_config(&self) -> ProcessingConfig {
        match self {
            Self::Fast => ProcessingConfig {
                transcription: TranscriptionConfig {
                    model_size: WhisperModel::Small,
                    compute_type: ComputeType::Int8,
                    beam_size: 1,
                },
                object_detection: ObjectDetectionConfig {
                    model_size: YOLOModel::Nano,
                    confidence_threshold: 0.4,
                },
                keyframes: KeyframeConfig {
                    interval: 2.0,
                    max_keyframes: 300,
                },
            },
            Self::Balanced => ProcessingConfig {
                transcription: TranscriptionConfig {
                    model_size: WhisperModel::Medium,
                    compute_type: ComputeType::Float16,
                    beam_size: 5,
                },
                object_detection: ObjectDetectionConfig {
                    model_size: YOLOModel::Medium,
                    confidence_threshold: 0.3,
                },
                keyframes: KeyframeConfig {
                    interval: 1.0,
                    max_keyframes: 500,
                },
            },
            Self::Accurate => ProcessingConfig {
                transcription: TranscriptionConfig {
                    model_size: WhisperModel::LargeV3,
                    compute_type: ComputeType::Float16,
                    beam_size: 10,
                },
                object_detection: ObjectDetectionConfig {
                    model_size: YOLOModel::Large,
                    confidence_threshold: 0.25,
                },
                keyframes: KeyframeConfig {
                    interval: 0.5,
                    max_keyframes: 1000,
                },
            },
        }
    }
}
```

---

## 7. QUALITY ASSURANCE

### 7.1 Error Types

```rust
#[derive(Debug, thiserror::Error)]
enum ProcessingError {
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("File too large: {size} bytes (max: {max})")]
    FileTooLarge { size: u64, max: u64 },

    #[error("Duration exceeds limit: {duration}s (max: {max}s)")]
    DurationTooLong { duration: f64, max: f64 },

    #[error("No audio stream found")]
    NoAudioStream,

    #[error("No video stream found")]
    NoVideoStream,

    #[error("Corrupted file: {0}")]
    CorruptedFile(String),

    #[error("GPU out of memory")]
    GPUOutOfMemory,

    #[error("Transcription failed: WER > {threshold}")]
    TranscriptionQualityTooLow { threshold: f32 },

    #[error("Processing timeout after {0}s")]
    Timeout(u64),
}
```

### 7.2 Quality Scoring

```rust
fn calculate_transcription_quality(transcript: &Transcript) -> f32 {
    let mut quality = 0.0;
    let mut total_duration = 0.0;

    for segment in &transcript.segments {
        let duration = segment.end - segment.start;

        // Confidence score
        let confidence = segment.words.iter()
            .map(|w| w.probability)
            .sum::<f32>() / segment.words.len() as f32;

        // No-speech probability (lower is better)
        let speech_quality = 1.0 - segment.no_speech_prob;

        // Word density (2-4 words/sec is typical)
        let word_density = segment.words.len() as f32 / duration as f32;
        let density_quality = if word_density >= 2.0 && word_density <= 4.0 {
            1.0
        } else {
            0.5
        };

        let segment_quality = (confidence + speech_quality + density_quality) / 3.0;
        quality += segment_quality * duration as f32;
        total_duration += duration;
    }

    quality / total_duration as f32
}
```

---

## 8. TEST SUITE CHARACTERISTICS

### 8.1 Test File Inventory

**Total Available**: 47,860 audio/video files

**Categories**:

1. **Docling Test Clips** (16 files, 12 MB)
   - Purpose-built 10-second samples
   - All major formats: MP4, MOV, AVI, MP3, WAV, FLAC, AAC, OGG
   - Location: `/Users/ayates/docling/tests/data/audio/`
   - Use for: Format validation, quick unit tests

2. **Production Videos** (18 files, 4.5 GB)
   - Zoom meetings, screen recordings, demos
   - Size range: 32 MB - 1.8 GB
   - Location: `/Users/ayates/Desktop/stuff/`
   - Use for: Integration tests, realistic workloads

3. **Kinetics-600 Dataset** (18,288 files, ~14.6 GB)
   - 600 action categories, ~30 videos per category
   - Uniform 10-second clips, ~800 KB per file
   - Location: `/Users/ayates/Library/CloudStorage/Dropbox-BrandcraftSolutions/a.test/Kinetics dataset (5%)/`
   - Use for: Throughput benchmarks, action detection validation

### 8.2 Test Configuration

**Quick Validation** (5 min):
```bash
# Process 5 Docling clips
for file in sample_10s_video-mp4.mp4 sample_10s_audio-mp3.mp3 \
            sample_10s_audio-wav.wav sample_10s_video-quicktime.mov \
            silent_1s.wav; do
    ./video-processor process --input "$file" --mode cpu_only
done
```

**Integration Test** (30 min):
```json
{
  "files": [
    "editing-relevance-rubrics kg may 16 2025.mov",
    "review existing benchmarks/gonzolo meeting aug 14/video1171640589.mp4",
    "audio1171640589.m4a"
  ],
  "mode": "gpu_accelerated",
  "features": ["transcription", "diarization", "objects", "ocr"]
}
```

**Throughput Benchmark** (4 hours):
```json
{
  "batch_id": "kinetics_1000",
  "files": [/* 1000 random Kinetics videos */],
  "batch_config": {
    "optimize_for": "throughput"
  }
}
```

### 8.3 Metrics to Capture

**Per-File**:
- Processing time (ms)
- CPU usage (%)
- GPU usage (%)
- RAM usage (MB)
- VRAM usage (MB)
- Success/failure status

**Per-Feature**:
- Transcription confidence
- Speaker count detected
- Object detection count
- Face detection count
- OCR character count
- Quality scores (0.0-1.0)

**Aggregate**:
- Total processing time
- Throughput (files/hour)
- Success rate (%)
- CPU efficiency (%)
- GPU efficiency (%)
- Peak memory usage

---

## 9. DEPLOYMENT CONFIGURATION

### 9.1 Kubernetes Deployment

**CPU Worker**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpu-worker
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: worker
        image: video-processor:cpu
        resources:
          requests:
            cpu: "4"
            memory: "8Gi"
          limits:
            cpu: "8"
            memory: "16Gi"
        env:
        - name: WORKER_MODE
          value: "cpu"
        - name: NATS_URL
          value: "nats://nats:4222"
```

**GPU Worker**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-worker
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: worker
        image: video-processor:gpu
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "64Gi"
        env:
        - name: WORKER_MODE
          value: "gpu"
```

**HPA (Horizontal Pod Autoscaler)**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cpu-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cpu-worker
  minReplicas: 10
  maxReplicas: 100
  metrics:
  - type: External
    external:
      metric:
        name: nats_queue_depth
      target:
        type: AverageValue
        averageValue: "5"
```

---

## 10. IMPLEMENTATION PRIORITIES

### Critical Path (Implement First):
1. Ingestion module (FFmpeg integration)
2. Video decoder (hardware-accelerated)
3. Audio extractor
4. Keyframe extractor
5. Orchestrator (task graph)
6. Transcription module (Faster-Whisper)
7. Object detection (YOLOv8)
8. Storage layer (S3, Qdrant, PostgreSQL)

### Secondary Features:
9. Speaker diarization
10. OCR, face detection
11. Scene detection (ML)
12. Embeddings
13. Audio event detection
14. Fusion layer
15. Search indexing

### Advanced Features (Optional):
16. Audio source separation
17. Cross-modal linking
18. LLM summarization
19. Action item extraction

---

## END OF SPECIFICATION

This document provides complete technical specifications for AI-driven implementation with:
- Architecture rationale (WHY decisions were made)
- Complete API specifications (3 modes)
- Module interfaces and implementations
- Technology selection with criteria
- Test suite characteristics
- Deployment configurations
- Implementation priorities

All performance metrics are reference values. Actual targets established through benchmarking.
