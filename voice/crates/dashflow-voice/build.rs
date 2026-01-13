// Build script for dashflow-voice
// Compiles voice.proto to Rust

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to the voice.proto file in stream-tts-cpp
    let proto_path = "../../../stream-tts-cpp/proto/voice.proto";

    // Check if proto file exists
    if std::path::Path::new(proto_path).exists() {
        tonic_build::configure()
            .build_server(false) // Client only
            .build_client(true)
            .out_dir("src/generated")
            .compile_protos(&[proto_path], &["../../../stream-tts-cpp/proto"])?;
    } else {
        // Fallback: look for proto in voice repo root
        let alt_path = "../../stream-tts-cpp/proto/voice.proto";
        if std::path::Path::new(alt_path).exists() {
            tonic_build::configure()
                .build_server(false)
                .build_client(true)
                .out_dir("src/generated")
                .compile_protos(&[alt_path], &["../../stream-tts-cpp/proto"])?;
        } else {
            println!("cargo:warning=voice.proto not found, using pre-generated stubs");
        }
    }

    Ok(())
}
