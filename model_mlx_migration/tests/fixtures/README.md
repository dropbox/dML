# Test Fixtures

Test fixtures for integration testing are not tracked in git (`.pt` files are ignored).

## Creating Test Fixtures

Run the following to create required test models:

```bash
python tests/fixtures/create_test_model.py
```

This creates:
- `simple_linear.pt` (~650KB) - Simple linear model for basic testing
- `transformer_block.pt` (~few MB) - Transformer model for attention testing

## Audio Fixtures

For Whisper tests, you may need audio files in `tests/fixtures/audio/`.
Tests will skip if audio fixtures are not present.

## Notes

- Integration tests will **skip** if fixtures are not created
- Unit tests do not require these fixtures
- Fixtures are machine-generated and can be recreated at any time
