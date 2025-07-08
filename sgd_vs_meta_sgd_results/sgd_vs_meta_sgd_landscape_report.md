# SGD vs Meta-SGD Loss Landscape Analysis Report

## Summary Statistics

### Performance Comparison
                         final_accuracy         accuracy_roughness         loss_roughness        
                                   mean     std               mean     std           mean     std
complexity      method                                                                           
Complex (F32D3) Meta-SGD         0.5233  0.0174             0.0169  0.0004         0.0168  0.0252
                SGD              0.5000  0.0000             0.1699  0.0026         0.2638  0.0009
F16D5           Meta-SGD         0.5695  0.0332             0.0170  0.0030         0.0044  0.0001
F16D7           Meta-SGD         0.5745  0.0049             0.0177  0.0027         0.0050  0.0011
F32D5           Meta-SGD         0.5430  0.0014             0.0164  0.0022         0.0023  0.0003
F32D7           Meta-SGD         0.5380  0.0113             0.0178  0.0027         0.0022  0.0002
F8D7            Meta-SGD         0.6645  0.0403             0.0139  0.0012         0.0079  0.0000
Medium (F16D3)  Meta-SGD         0.5630  0.0212             0.0136  0.0020         0.0046  0.0000
Medium (F8D5)   Meta-SGD         0.7115  0.0488             0.0167  0.0039         0.0087  0.0012
                SGD              0.5167  0.0289             0.1665  0.0009         0.2019  0.0019
Simple (F8D3)   Meta-SGD         0.7588  0.0603             0.0148  0.0041         0.0122  0.0031
                SGD              0.7333  0.1607             0.1662  0.0015         0.2024  0.0008

## Key Findings

### Landscape Roughness Analysis

### Complex (F32D3)
- **SGD Performance**: 0.5000 (roughness: 0.1699)
- **Meta-SGD Performance**: 0.5233 (roughness: 0.0169)
- **Improvement**: +0.0233 (+4.7%)
- **Roughness Change**: -0.1531 (-90.1%)

### Simple (F8D3)
- **SGD Performance**: 0.7333 (roughness: 0.1662)
- **Meta-SGD Performance**: 0.7588 (roughness: 0.0148)
- **Improvement**: +0.0254 (+3.5%)
- **Roughness Change**: -0.1514 (-91.1%)

### Medium (F8D5)
- **SGD Performance**: 0.5167 (roughness: 0.1665)
- **Meta-SGD Performance**: 0.7115 (roughness: 0.0167)
- **Improvement**: +0.1948 (+37.7%)
- **Roughness Change**: -0.1499 (-90.0%)
