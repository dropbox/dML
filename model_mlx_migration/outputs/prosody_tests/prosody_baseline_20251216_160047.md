# Kokoro Prosody Baseline Test Report

**Date**: 2025-12-16 16:00:47
**Voice**: af_bella
**Model**: prince-canuma/Kokoro-82M (MLX)

---

## Summary

| Category | Responds | Partial | No Response | Total |
|----------|----------|---------|-------------|-------|
| Capitalization | 0 | 0 | 3 | 3 |
| Context | 5 | 0 | 0 | 5 |
| Ipa | 0 | 0 | 3 | 3 |
| Position | 2 | 0 | 0 | 2 |
| Punctuation | 0 | 1 | 5 | 6 |
| Question | 0 | 0 | 0 | 5 |
| Stress | 0 | 0 | 3 | 3 |
| **Total** | **7** | **1** | **14** | **27** |

---

## Detailed Results

### Capitalization

#### Test 12: All-caps emphasis ❌

**Variants**: 'This is IMPORTANT', 'This is important'
**Expected**: All-caps should be emphasized
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| caps | 'This is IMPORTANT' | 1.900 | 206.2 | 29.4 | 104.5 |
| lowercase | 'This is important' | 1.900 | 206.1 | 29.5 | 104.5 |

**Comparisons**:
- caps vs lowercase: Similar (F0: 0.0%, Duration: 0.0%)

---

#### Test 13: Caps with punctuation ❌

**Variants**: 'NO!', 'No.'
**Expected**: Different emphasis level
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| caps_exclaim | 'NO!' | 1.325 | 219.4 | 22.1 | 76.3 |
| lower_period | 'No.' | 1.325 | 220.9 | 21.3 | 73.2 |

**Comparisons**:
- caps_exclaim vs lower_period: Similar (F0: 0.7%, Duration: 0.0%)

---

#### Test 14: Caps question ❌

**Variants**: 'WHY?', 'Why?'
**Expected**: Caps should add intensity
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| caps | 'WHY?' | 1.250 | 220.6 | 20.2 | 62.8 |
| lowercase | 'Why?' | 1.200 | 222.9 | 17.4 | 55.7 |

**Comparisons**:
- caps vs lowercase: Similar (F0: 1.1%, Duration: 4.0%)

---

### Context

#### Test 7: Word function: exclamation vs noun ✅

**Variants**: 'Alert!', 'The alert was triggered'
**Expected**: 'Alert!' should be more emphatic
**Response**: YES

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| exclaim | 'Alert!' | 1.425 | 226.1 | 21.9 | 89.8 |
| noun | 'The alert was triggered' | 1.950 | 223.8 | 18.6 | 74.7 |

**Comparisons**:
- exclaim vs noun: Different (F0: 1.0%, Duration: 26.9%)

---

#### Test 8: Urgency in context ✅

**Variants**: 'Help!', 'I need help with this'
**Expected**: 'Help!' should be urgent
**Response**: YES

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| urgent | 'Help!' | 1.325 | 231.2 | 27.7 | 87.2 |
| casual | 'I need help with this' | 1.925 | 213.6 | 29.2 | 90.2 |

**Comparisons**:
- urgent vs casual: Different (F0: 7.6%, Duration: 31.2%)

---

#### Test 9: Command vs verb ✅

**Variants**: 'Fire!', 'Fire the employee'
**Expected**: Different stress and urgency
**Response**: YES

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| command | 'Fire!' | 1.375 | 220.8 | 29.7 | 88.8 |
| verb | 'Fire the employee' | 1.875 | 211.6 | 26.3 | 89.2 |

**Comparisons**:
- command vs verb: Different (F0: 4.2%, Duration: 26.7%)

---

#### Test 10: Command vs noun ✅

**Variants**: 'Run!', 'Go for a run'
**Expected**: Different stress patterns
**Response**: YES

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| command | 'Run!' | 1.300 | 220.8 | 23.4 | 68.4 |
| noun | 'Go for a run' | 1.675 | 210.3 | 26.5 | 98.6 |

**Comparisons**:
- command vs noun: Different (F0: 4.8%, Duration: 22.4%)

---

#### Test 11: Standalone vs greeting ✅

**Variants**: 'Hey!', 'Hey, how are you?'
**Expected**: Standalone more emphatic
**Response**: YES

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| standalone | 'Hey!' | 1.275 | 221.5 | 26.2 | 74.9 |
| greeting | 'Hey, how are you?' | 1.675 | 222.1 | 24.2 | 95.0 |

**Comparisons**:
- standalone vs greeting: Different (F0: 0.3%, Duration: 23.9%)

---

### Ipa

#### Test 25: Primary stress marker ❌

**Variants**: 'record', 'Record'
**Expected**: Case might affect stress
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| lowercase | 'record' | 1.450 | 215.0 | 30.3 | 94.9 |
| capitalized | 'Record' | 1.450 | 214.4 | 29.9 | 82.3 |

**Comparisons**:
- lowercase vs capitalized: Similar (F0: 0.3%, Duration: 0.0%)

---

#### Test 26: Vowel length ❌

**Variants**: 'beat', 'bit'
**Expected**: Different vowel length
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| long | 'beat' | 1.225 | 229.1 | 19.9 | 66.2 |
| short | 'bit' | 1.200 | 228.0 | 21.1 | 65.4 |

**Comparisons**:
- long vs short: Similar (F0: 0.4%, Duration: 2.0%)

---

#### Test 27: Break marker (pipe) ❌

**Variants**: 'hello world', 'hello | world'
**Expected**: Pipe should insert pause
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| no_break | 'hello world' | 1.575 | 218.1 | 22.9 | 86.7 |
| with_break | 'hello | world' | 1.650 | 217.0 | 23.3 | 85.3 |

**Comparisons**:
- no_break vs with_break: Similar (F0: 0.5%, Duration: 4.5%)

---

### Position

#### Test 18: Fronted emphasis ✅

**Variants**: 'Important: do this now', 'This is important'
**Expected**: Fronted word more emphasized
**Response**: YES

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| fronted | 'Important: do this now' | 2.400 | 208.8 | 32.9 | 113.5 |
| embedded | 'This is important' | 1.900 | 206.0 | 29.5 | 104.5 |

**Comparisons**:
- fronted vs embedded: Different (F0: 1.4%, Duration: 20.8%)

---

#### Test 19: Position effect ✅

**Variants**: 'Listen. This matters.', 'This matters, listen'
**Expected**: Different emphasis patterns
**Response**: YES

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| front | 'Listen. This matters.' | 2.300 | 204.0 | 25.1 | 91.4 |
| end | 'This matters, listen' | 2.000 | 201.7 | 28.4 | 91.5 |

**Comparisons**:
- front vs end: Different (F0: 1.1%, Duration: 13.0%)

---

### Punctuation

#### Test 1: Terminal punctuation: period vs exclamation vs question ❌

**Variants**: 'Hello.', 'Hello!', 'Hello?'
**Expected**: Different intonation patterns
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| period | 'Hello.' | 1.400 | 218.9 | 26.8 | 85.8 |
| exclamation | 'Hello!' | 1.375 | 218.3 | 25.8 | 83.8 |
| question | 'Hello?' | 1.350 | 221.4 | 25.1 | 80.7 |

**Comparisons**:
- period vs exclamation: Similar (F0: 0.3%, Duration: 1.8%)
- period vs question: Similar (F0: 1.1%, Duration: 3.6%)
- exclamation vs question: Similar (F0: 1.4%, Duration: 1.8%)

---

#### Test 2: Command vs exclaim vs question ❌

**Variants**: 'Stop.', 'Stop!', 'Stop?'
**Expected**: Different emphasis and intonation
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| command | 'Stop.' | 1.350 | 224.2 | 28.9 | 107.5 |
| exclaim | 'Stop!' | 1.350 | 226.7 | 31.4 | 130.3 |
| question | 'Stop?' | 1.350 | 223.9 | 28.6 | 106.5 |

**Comparisons**:
- command vs exclaim: Similar (F0: 1.1%, Duration: 0.0%)
- command vs question: Similar (F0: 0.1%, Duration: 0.0%)
- exclaim vs question: Similar (F0: 1.2%, Duration: 0.0%)

---

#### Test 3: Single word punctuation ❌

**Variants**: 'Really.', 'Really!', 'Really?'
**Expected**: Different prosody on single word
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| statement | 'Really.' | 1.425 | 217.6 | 29.0 | 95.8 |
| exclamation | 'Really!' | 1.400 | 218.6 | 29.0 | 96.5 |
| question | 'Really?' | 1.400 | 218.4 | 29.1 | 93.4 |

**Comparisons**:
- statement vs exclamation: Similar (F0: 0.4%, Duration: 1.8%)
- statement vs question: Similar (F0: 0.4%, Duration: 1.8%)
- exclamation vs question: Similar (F0: 0.1%, Duration: 0.0%)

---

#### Test 4: Ellipsis vs period vs exclamation ❌

**Variants**: 'I see...', 'I see.', 'I see!'
**Expected**: Ellipsis should have trailing/thoughtful tone
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| ellipsis | 'I see...' | 1.525 | 218.3 | 26.8 | 91.4 |
| period | 'I see.' | 1.475 | 216.4 | 25.0 | 101.2 |
| exclamation | 'I see!' | 1.475 | 217.7 | 25.3 | 93.2 |

**Comparisons**:
- ellipsis vs period: Similar (F0: 0.9%, Duration: 3.3%)
- ellipsis vs exclamation: Similar (F0: 0.2%, Duration: 3.3%)
- period vs exclamation: Similar (F0: 0.6%, Duration: 0.0%)

---

#### Test 5: Comma pause ❌

**Variants**: 'Wait, what?', 'Wait what?'
**Expected**: Comma should introduce pause
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| with_comma | 'Wait, what?' | 1.500 | 217.2 | 24.7 | 89.3 |
| without_comma | 'Wait what?' | 1.525 | 219.7 | 24.8 | 92.3 |

**Comparisons**:
- with_comma vs without_comma: Similar (F0: 1.2%, Duration: 1.6%)

---

#### Test 6: Em-dash vs comma vs no punctuation ⚠️

**Variants**: 'No—never', 'No, never', 'No never'
**Expected**: Different pause patterns
**Response**: PARTIAL

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| em_dash | 'No—never' | 1.700 | 215.5 | 17.9 | 51.0 |
| comma | 'No, never' | 1.850 | 209.7 | 21.2 | 68.6 |
| no_punct | 'No never' | 1.500 | 225.2 | 15.4 | 47.9 |

**Comparisons**:
- em_dash vs comma: Similar (F0: 2.7%, Duration: 8.1%)
- em_dash vs no_punct: Different (F0: 4.3%, Duration: 11.8%)
- comma vs no_punct: Different (F0: 6.9%, Duration: 18.9%)

---

### Question

#### Test 20: Yes/no question (rising) ℹ️

**Variants**: 'Is this correct?'
**Expected**: Rising intonation
**Response**: N/A

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| yes_no | 'Is this correct?' | 1.750 | 215.6 | 28.5 | 90.7 |

---

#### Test 21: Wh-question (falling) ℹ️

**Variants**: 'What is this?'
**Expected**: Falling intonation
**Response**: N/A

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| wh_question | 'What is this?' | 1.650 | 219.0 | 27.2 | 85.3 |

---

#### Test 22: Confirmation question (rising) ℹ️

**Variants**: "You're sure?"
**Expected**: Rising intonation
**Response**: N/A

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| confirmation | "You're sure?" | 1.500 | 219.9 | 26.7 | 103.4 |

---

#### Test 23: Incredulous question (stronger rising) ℹ️

**Variants**: 'This is what you want?'
**Expected**: Strong rising intonation
**Response**: N/A

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| incredulous | 'This is what you want?' | 1.925 | 209.7 | 31.2 | 118.5 |

---

#### Test 24: Echo question (rising on wh-word) ℹ️

**Variants**: 'You did what?'
**Expected**: Rising on 'what'
**Response**: N/A

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| echo | 'You did what?' | 1.600 | 217.7 | 26.8 | 90.7 |

---

### Stress

#### Test 15: Record: noun vs verb stress ❌

**Variants**: 'the record', 'to record'
**Expected**: REcord (noun) vs reCORD (verb)
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| noun | 'the record' | 1.500 | 212.5 | 29.4 | 102.6 |
| verb | 'to record' | 1.575 | 222.2 | 30.6 | 108.9 |

**Comparisons**:
- noun vs verb: Similar (F0: 4.4%, Duration: 4.8%)

---

#### Test 16: Present: noun vs verb stress ❌

**Variants**: 'a present', 'to present'
**Expected**: PREsent (noun) vs preSENT (verb)
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| noun | 'a present' | 1.525 | 217.4 | 31.1 | 90.7 |
| verb | 'to present' | 1.600 | 217.5 | 28.6 | 99.8 |

**Comparisons**:
- noun vs verb: Similar (F0: 0.0%, Duration: 4.7%)

---

#### Test 17: Contract: noun vs verb stress ❌

**Variants**: 'the contract', 'to contract'
**Expected**: CONtract (noun) vs conTRACT (verb)
**Response**: NO

| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |
|---------|------|--------------|--------------|-------------|---------------|
| noun | 'the contract' | 1.625 | 220.0 | 26.5 | 87.1 |
| verb | 'to contract' | 1.650 | 220.3 | 26.2 | 113.3 |

**Comparisons**:
- noun vs verb: Similar (F0: 0.1%, Duration: 1.5%)

---
