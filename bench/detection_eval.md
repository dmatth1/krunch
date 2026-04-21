# Detection accuracy eval

- files evaluated: **1023**
- overall accuracy: **0.958**  (980/1023)
- clean accuracy: **0.959**  (936/976)  — PHASE_14 target: ≥0.95
- edge accuracy: **0.936**  (44/47)  — PHASE_14 target: ≥0.80

**RELEASE GATE: PASS** — both accuracy thresholds met.

## Confusion matrix (rows=expected, cols=predicted)

| expected \ predicted | prose | code | structured | logs | tabular | markup | fallback | support |
|---|---|---|---|---|---|---|---|---|
| **prose** | 149 | 1 | 5 | 0 | 1 | 0 | 0 | 156 |
| **code** | 2 | 145 | 5 | 0 | 1 | 8 | 9 | 170 |
| **structured** | 3 | 0 | 146 | 0 | 0 | 0 | 0 | 149 |
| **logs** | 0 | 0 | 0 | 137 | 8 | 0 | 0 | 145 |
| **tabular** | 0 | 0 | 0 | 0 | 140 | 0 | 0 | 140 |
| **markup** | 0 | 0 | 0 | 0 | 0 | 143 | 0 | 143 |
| **fallback** | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 120 |

## Per-class metrics

| class | support | precision | recall | accuracy |
|---|---:|---:|---:|---:|
| prose | 156 | 0.968 | 0.955 | 0.955 |
| code | 170 | 0.993 | 0.853 | 0.853 |
| structured | 149 | 0.936 | 0.980 | 0.980 |
| logs | 145 | 1.000 | 0.945 | 0.945 |
| tabular | 140 | 0.933 | 1.000 | 1.000 |
| markup | 143 | 0.947 | 1.000 | 1.000 |
| fallback | 120 | 0.930 | 1.000 | 1.000 |

## Per-class clean-only accuracy (PHASE_14 common-domain gate ≥0.95)

| class | correct | total | accuracy | gate |
|---|---:|---:|---:|---:|
| prose | 141 | 145 | 0.972 | PASS |
| code | 135 | 160 | 0.844 | FAIL |
| structured | 138 | 141 | 0.979 | PASS |
| logs | 132 | 140 | 0.943 | FAIL |
| tabular | 135 | 135 | 1.000 | PASS |
| markup | 135 | 135 | 1.000 | PASS |
| fallback | 120 | 120 | 1.000 | PASS |

## Misclassifications (43 total)

### By (expected → predicted) pair

| expected | predicted | count | category mix | example file | notes |
|---|---|---:|---|---|---|
| code | fallback | 9 | clean=9 | `code/code_py_044.py` | code_diverse_real Python (strict-filtered) |
| code | markup | 8 | clean=8 | `code/code_py_011.py` | code_diverse_real Python (strict-filtered) |
| logs | tabular | 8 | clean=8 | `logs/bgl_006.log` | BGL real logs (timestamp-filtered) |
| prose | structured | 5 | clean=2, edge=3 | `prose/prose_026.txt` | pile_raw_1gb sample (strict-filtered) |
| code | structured | 5 | clean=5 | `code/code_py_049.py` | code_diverse_real Python (strict-filtered) |
| structured | prose | 3 | clean=3 | `structured/yaml_real_002.yaml` | code_real YAML (shape-filtered) |
| code | prose | 2 | clean=2 | `code/code_py_009.py` | code_diverse_real Python (strict-filtered) |
| prose | code | 1 | clean=1 | `prose/prose_015.txt` | pile_raw_1gb sample (strict-filtered) |
| prose | tabular | 1 | clean=1 | `prose/prose_061.txt` | pile_raw_1gb sample (strict-filtered) |
| code | tabular | 1 | clean=1 | `code/code_py_015.py` | code_diverse_real Python (strict-filtered) |

### Individual misclassifications (first 60)

| file | expected | predicted | conf | category | notes |
|---|---|---|---:|---|---|
| `prose/prose_015.txt` | prose | code | 0.65 | clean | pile_raw_1gb sample (strict-filtered) |
| `prose/prose_026.txt` | prose | structured | 0.75 | clean | pile_raw_1gb sample (strict-filtered) |
| `prose/prose_056.txt` | prose | structured | 0.80 | clean | pile_raw_1gb sample (strict-filtered) |
| `prose/prose_061.txt` | prose | tabular | 0.70 | clean | pile_raw_1gb sample (strict-filtered) |
| `code/code_py_009.py` | code | prose | 0.55 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_011.py` | code | markup | 0.85 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_015.py` | code | tabular | 0.70 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_039.py` | code | markup | 0.70 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_044.py` | code | fallback | 0.40 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_048.py` | code | fallback | 0.40 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_049.py` | code | structured | 0.75 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_050.py` | code | markup | 0.85 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_051.py` | code | prose | 0.80 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_052.py` | code | markup | 0.70 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_056.py` | code | markup | 0.85 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_062.py` | code | structured | 0.90 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_065.py` | code | fallback | 0.40 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_068.py` | code | structured | 0.90 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_070.py` | code | markup | 0.85 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_074.py` | code | markup | 0.85 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_078.py` | code | structured | 0.70 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_079.py` | code | fallback | 0.40 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_080.py` | code | markup | 0.85 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_py_087.py` | code | structured | 0.90 | clean | code_diverse_real Python (strict-filtered) |
| `code/code_mk_000.mk` | code | fallback | 0.40 | clean | makefile |
| `code/code_mk_001.mk` | code | fallback | 0.40 | clean | makefile |
| `code/code_mk_002.mk` | code | fallback | 0.40 | clean | makefile |
| `code/code_mk_003.mk` | code | fallback | 0.40 | clean | makefile |
| `code/code_mk_004.mk` | code | fallback | 0.40 | clean | makefile |
| `structured/yaml_real_002.yaml` | structured | prose | 0.55 | clean | code_real YAML (shape-filtered) |
| `structured/yaml_extra_008.yaml` | structured | prose | 0.55 | clean | code_real_extra YAML (shape-filtered) |
| `structured/yaml_extra_017.yaml` | structured | prose | 0.55 | clean | code_real_extra YAML (shape-filtered) |
| `logs/bgl_006.log` | logs | tabular | 0.85 | clean | BGL real logs (timestamp-filtered) |
| `logs/bgl_022.log` | logs | tabular | 0.85 | clean | BGL real logs (timestamp-filtered) |
| `logs/bgl_026.log` | logs | tabular | 0.70 | clean | BGL real logs (timestamp-filtered) |
| `logs/bgl_035.log` | logs | tabular | 0.85 | clean | BGL real logs (timestamp-filtered) |
| `logs/bgl_049.log` | logs | tabular | 0.85 | clean | BGL real logs (timestamp-filtered) |
| `logs/bgl_050.log` | logs | tabular | 0.70 | clean | BGL real logs (timestamp-filtered) |
| `logs/bgl_052.log` | logs | tabular | 0.85 | clean | BGL real logs (timestamp-filtered) |
| `logs/bgl_054.log` | logs | tabular | 0.70 | clean | BGL real logs (timestamp-filtered) |
| `prose/book_header_000.txt` | prose | structured | 0.80 | edge | narrative with short YAML-like book header |
| `prose/book_header_001.txt` | prose | structured | 0.80 | edge | narrative with short YAML-like book header |
| `prose/book_header_002.txt` | prose | structured | 0.80 | edge | narrative with short YAML-like book header |
