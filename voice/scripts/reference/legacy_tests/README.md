Legacy test scripts (archived)
================================

These files were previously located under `tests/` and were discovered by pytest despite being ad-hoc scripts. They are preserved here for historical reference and manual experimentation only; they are **not** part of the automated test suite.

Archived scripts:
- test_claude_integration.sh
- test_cpp_system.sh
- test_integrated_pipeline.sh
- test_mecab_complete.sh
- test_phase2_integration.sh
- test_pipeline.sh
- test_production_quick.sh
- test_prototype.sh
- test_semantic_equivalence.py
- test_sentences.sh
- test_stream_tts.sh
- test_tts_comprehensive.py
- test_ultimate_quality.sh
- test_with_claude.sh

If any of these need to become real tests, convert them to pytest cases under `tests/` with assertions and markers.
