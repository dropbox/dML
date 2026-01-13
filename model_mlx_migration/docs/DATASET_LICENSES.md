# Dataset Licenses for CTC Training

## Summary Table

| Dataset | License | Commercial Use | Attribution Required |
|---------|---------|----------------|---------------------|
| **LibriSpeech** | CC BY 4.0 | Yes | Yes |
| **TED-LIUM 3** | CC BY-NC-ND 3.0 | **No** | Yes |
| **Common Voice** | CC0 (Public Domain) | Yes | No |
| **GigaSpeech** | Apache 2.0 | Yes | Yes |
| **People's Speech** | CC BY-SA 4.0 | Yes | Yes, ShareAlike |
| **MLS** | CC BY 4.0 | Yes | Yes |
| **VoxPopuli** | CC0 | Yes | No |
| **FLEURS** | CC BY 4.0 | Yes | Yes |

## Detailed License Information

### LibriSpeech (960 hours English)
- **License**: CC BY 4.0
- **Source**: LibriVox audiobooks (public domain)
- **Commercial**: Yes
- **URL**: https://www.openslr.org/12/

### TED-LIUM 3 (450 hours English) - DO NOT USE
- **License**: CC BY-NC-ND 3.0
- **Source**: TED Talks
- **Commercial**: **NO - Non-commercial only**
- **Derivatives**: **NO - No derivatives allowed**
- **Status**: EXCLUDED from training due to NC license

### Common Voice (1000+ hours, 100+ languages)
- **License**: CC0 (Public Domain)
- **Source**: Mozilla crowdsourced
- **Commercial**: Yes
- **URL**: https://commonvoice.mozilla.org/
- **Access**: Requires Mozilla account

### GigaSpeech (10,000 hours English)
- **License**: Apache 2.0
- **Source**: YouTube, podcasts, audiobooks
- **Commercial**: Yes
- **URL**: https://github.com/SpeechColab/GigaSpeech
- **Access**: Requires HuggingFace approval

### People's Speech (30,000 hours English)
- **License**: CC BY-SA 4.0
- **Source**: Various public domain sources
- **Commercial**: Yes (ShareAlike)
- **URL**: https://mlcommons.org/en/peoples-speech/
- **Access**: Requires approval

### Multilingual LibriSpeech (MLS)
- **License**: CC BY 4.0
- **Languages**: en, de, nl, fr, es, it, pt, pl
- **Commercial**: Yes
- **URL**: https://www.openslr.org/94/

### VoxPopuli
- **License**: CC0
- **Source**: European Parliament recordings
- **Languages**: 23 EU languages
- **Commercial**: Yes
- **URL**: https://github.com/facebookresearch/voxpopuli

### FLEURS
- **License**: CC BY 4.0
- **Languages**: 102 languages
- **Commercial**: Yes
- **URL**: https://huggingface.co/datasets/google/fleurs

## Downloaded Datasets (Commercial-Friendly)

### Chinese (111GB total)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| AISHELL | 29G | Apache 2.0 | SLR33 |
| AISHELL-3 | 43G | Apache 2.0 | SLR93 |
| Primewords | 19G | CC BY-SA 3.0 | SLR47 |
| ST-CMDS | 20G | CC BY-SA 4.0 | SLR38 |

### Korean (21GB total)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| Zeroth-Korean | 20G | CC BY 4.0 | SLR40 |
| Deeply Korean | 606M | CC BY-SA 4.0 | SLR97 |

### Russian (19GB)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| Russian LibriSpeech | 19G | CC BY 4.0 | SLR96 |

### German (3.5GB + 29GB downloading)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| Thorsten German | 3.5G | CC0 | SLR95 |
| MLS German | 29G | CC BY 4.0 | SLR94 (downloading) |

### French (2.2GB + 16GB downloading)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| African French | 2.2G | Apache 2.0 | SLR57 |
| MLS French | 16G | CC BY 4.0 | SLR94 (downloading) |

### Spanish (2.5GB + 14GB downloading)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| HEROICO Spanish | 2.5G | Apache 2.0 | SLR39 |
| MLS Spanish | 14G | CC BY 4.0 | SLR94 (downloading) |

### Dutch (23GB downloading)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| MLS Dutch | 23G | CC BY 4.0 | SLR94 (downloading) |

### Italian (3.8GB downloading)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| MLS Italian | 3.8G | CC BY 4.0 | SLR94 (downloading) |

### Portuguese (2.5GB downloading)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| MLS Portuguese | 2.5G | CC BY 4.0 | SLR94 (downloading) |

### Polish (1.6GB downloading)
| Dataset | Size | License | Source |
|---------|------|---------|--------|
| MLS Polish | 1.6G | CC BY 4.0 | SLR94 (downloading) |

### Japanese - NEEDS MANUAL ACCESS
**No easily accessible commercial-friendly datasets found**
- ReazonSpeech: CDLA-Sharing-1.0 with Article 30-4 restriction (data mining only, NOT suitable for commercial training)
- JVS/JSUT: Requires login to Google Sites
- Common Voice Japanese: CC0 but requires manual Mozilla download

**Recommended**: Download Common Voice Japanese manually from https://commonvoice.mozilla.org/

### Hindi - NEEDS MANUAL ACCESS
**No easily accessible commercial-friendly datasets found**
- SLR103 (MUCS): Requires permission for commercial use
- SLR118: Not commercial-friendly (academic only)
- Common Voice Hindi: CC0 but requires manual Mozilla download

**Recommended**: Download Common Voice Hindi manually from https://commonvoice.mozilla.org/

### Kashmiri
- **Current data**: 847 MB
- **Source**: FLEURS, OpenSLR
- **License**: Varies by source
- **OpenSLR datasets**:
  - SLR122: Kashmiri ASR (CC BY-SA 4.0)

## Recommendations

### For Commercial Use
Use only:
- LibriSpeech (CC BY 4.0)
- Common Voice (CC0)
- GigaSpeech (Apache 2.0)
- MLS (CC BY 4.0)
- VoxPopuli (CC0)
- FLEURS (CC BY 4.0)

### Avoid for Commercial
- **TED-LIUM**: NC (non-commercial) restriction

## How to Request Access

### GigaSpeech
1. Go to https://huggingface.co/datasets/speechcolab/gigaspeech
2. Click "Access repository"
3. Fill form and wait for approval (usually 1-2 days)

### People's Speech
1. Go to https://huggingface.co/datasets/MLCommons/peoples_speech
2. Request access
3. Wait for approval

### Common Voice
1. Go to https://commonvoice.mozilla.org/
2. Create Mozilla account
3. Download from website (not HuggingFace)
