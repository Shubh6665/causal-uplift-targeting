# Criteo Data Audit (Real Dataset)

## 1. File Metadata
- **Source**: Criteo AI Lab Official Unbiased Release (v2.1)
- **Local Path**: `data/raw/criteo-uplift-v2.1.csv.gz`
- **Total Rows**: 13,979,592
- **Total Columns**: 16

## 2. Schema
- **Features**: 12 columns (`f0` to `f11`)
- **Treatment**: `treatment` (binary)
- **Outcomes**: `conversion` (binary), `visit` (binary)
- **Leakage variable**: `exposure` (binary)

## 3. Global Statistics
- **Treatment Count**: 11,882,655
- **Control Count**: 2,096,937
- **Treatment Ratio**: 85.00%
- **Total Conversions**: 40,774 (0.2917%)
- **Total Visits**: 656,929 (4.70%)
- **Total Exposures**: 428,212 (3.06%)

## 4. Integrity Checks
- **Exposure in Control Group**: 0 (MUST be 0. Proves exposure requires treatment).
- **Leakage check**: `exposure` must NEVER be used as a feature, as it is post-treatment.

## 5. Baseline Difference in Means (ATE)
- **Treated Conversion Rate**: 0.003089
- **Control Conversion Rate**: 0.001938
- **Difference (Uplift)**: +0.001152

## 6. Feature Statistics
| Feature | Mean | Std Dev |
|---------|------|---------|
| f0 | 19.6203 | 5.3775 |
| f1 | 10.0700 | 0.1048 |
| f2 | 8.4466 | 0.2993 |
| f3 | 4.1789 | 1.3366 |
| f4 | 10.3388 | 0.3433 |
| f5 | 4.0285 | 0.4311 |
| f6 | -4.1554 | 4.5779 |
| f7 | 5.1018 | 1.2052 |
| f8 | 3.9336 | 0.0567 |
| f9 | 16.0276 | 7.0190 |
| f10 | 5.3334 | 0.1682 |
| f11 | -0.1710 | 0.0228 |
