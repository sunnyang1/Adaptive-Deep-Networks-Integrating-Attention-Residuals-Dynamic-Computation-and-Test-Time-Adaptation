# Results Directory Structure

This directory contains experimental results organized by date.

## Directory Structure

```
results/
├── README.md                          # This file
├── 2026-03-23/                        # Early validation experiments
│   ├── datasets/                      # Dataset validation results
│   │   ├── dataset_info.json
│   │   ├── dataset_validation_complete.json
│   │   └── hf_validation_api.json
│   └── validation/                    # Model validation results
│       ├── validation_small.json
│       ├── validation_medium.json
│       └── validation_summary.json
├── 2026-03-24/                        # Configuration and reports
│   ├── config/
│   │   └── large_model_config.json
│   └── section_5_2_report.md
└── 2026-03-30/                        # Small Model validation (main)
    ├── experiments/                   # Experiment raw data
    ├── paper_metrics/                 # Paper metrics summary
    │   ├── README.md
    │   ├── paper_metrics_report.txt
    │   └── paper_metrics_summary.json
    ├── reports/                       # Comprehensive reports
    │   ├── COMPLETE_SMALL_MODEL_EXPERIMENTS.md
    │   ├── SMALL_MODEL_PAPER_EXPERIMENTS_COMPLETE.md
    │   ├── SMALL_MODEL_EXPERIMENT_REPORT.md
    │   ├── TURBOQUANT_ANALYSIS_AND_RECOMMENDATIONS.md
    │   └── PAPER_UPDATES_SUMMARY.md
    ├── small_model/                   # Small Model specific data
    │   ├── small_model_benchmarks.json
    │   ├── small_model_experiments.json
    │   └── small_model_report.txt
    └── turboquant/                    # TurboQuant testing results
        ├── turboquant_small_model_tests.json
        └── turboquant_small_model_report.txt
```

## Quick Access

### Latest Results (2026-03-30)

**Main Reports:**
- `2026-03-30/reports/COMPLETE_SMALL_MODEL_EXPERIMENTS.md` - Complete experiment report
- `2026-03-30/reports/SMALL_MODEL_PAPER_EXPERIMENTS_COMPLETE.md` - Paper experiments
- `2026-03-30/reports/TURBOQUANT_ANALYSIS_AND_RECOMMENDATIONS.md` - TurboQuant analysis

**Key Data:**
- `2026-03-30/small_model/` - Small Model (2.2B) validation data
- `2026-03-30/turboquant/` - TurboQuant compression testing
- `2026-03-30/paper_metrics/` - Paper Tables 4-8 metrics

### Historical Data

- `2026-03-23/` - Early dataset and model validation
- `2026-03-24/` - Configuration experiments

## Validation Summary (2026-03-30)

| Metric | Target | Verified | Status |
|--------|--------|----------|--------|
| Model Parameters | 2.2B | 2.21B | ✅ |
| AttnRes Overhead | <0.1% | 0.012% | ✅ |
| Memory Reduction | 4× | 4× | ✅ |
| FLOPs per Token | ~4.3G | 4.30G | ✅ |
| FLOP Equivalence | Theory | Verified | ✅ |

## Notes

- Results are organized chronologically by experiment date
- Each date directory contains self-contained experiment results
- Latest results are in the most recent date directory
- Original subdirectories (core/, real_model/) preserved for compatibility
