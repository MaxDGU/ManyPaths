# K=1 vs K=10 Data Efficiency Comparison Report

## Basic Statistics

- K=1 trajectory files: 18
- K=10 trajectory files: 3
- Thresholds analyzed: [50, 60, 70, 80]

## Convergence Analysis

                   count  sum      mean
threshold k_value                      
50        K=1         18   18  1.000000
          K=10         3    3  1.000000
60        K=1         18    9  0.500000
          K=10         3    3  1.000000
70        K=1         18    4  0.222222
          K=10         3    2  0.666667
80        K=1         18    0  0.000000
          K=10         3    2  0.666667

## Statistical Tests (60% Threshold)

   features  depth complexity  k1_mean  k10_mean  efficiency_gain  mann_whitney_u   p_value  cohens_d  significant
0         8      3      F8_D3  30000.0   15000.0              2.0             3.5  0.414216  1.341641        False

### Summary:
- Significant results: 0/1
- Mean efficiency gain (K=1/K=10): 2.00
- Mean Cohen's d: 1.342