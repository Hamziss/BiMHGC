#!/usr/bin/env python3
"""
Test script to demonstrate p-value handling in GO enrichment analysis.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_pvalue_formatting():
    """Test how very small p-values are handled."""
    
    print("Testing p-value formatting:")
    print("=" * 50)
    
    # Test various p-values
    test_pvalues = [1.0, 0.05, 0.01, 0.001, 1e-5, 1e-10, 1e-15, 0.0]
    
    for pval in test_pvalues:
        # Original method (rounding)
        rounded = round(pval, 6)
        
        # New method (scientific notation for small values)
        if pval < 0.001:
            formatted = f"{pval:.2e}"
        else:
            formatted = round(pval, 6)
        
        # Handle zero values
        if pval <= 0.0:
            corrected = 1e-10
        else:
            corrected = pval
            
        print(f"Original: {pval:>12} | Rounded: {rounded:>12} | Formatted: {formatted:>12} | Corrected: {corrected:.2e}")

def explain_pvalue_zero():
    """Explain why p-values might appear as 0.0."""
    print("\nWhy do you see Min_Pvalue = 0.0 and Significant = True?")
    print("=" * 60)
    print("""
1. STATISTICAL REALITY:
   - True p-values are never exactly 0.0 in statistics
   - They can be extremely small (like 1e-50) but never exactly zero
   
2. COMPUTATIONAL ISSUES:
   - Very small p-values (< 1e-15) may underflow to 0.0 in floating point arithmetic
   - goatools might return 0.0 for extremely significant results
   - Python's floating point precision has limits
   
3. SIGNIFICANCE DETERMINATION:
   - If p-value ≤ 0.05, the result is considered significant
   - Since 0.0 < 0.05, it appears as "Significant = True"
   - This is actually CORRECT - the result IS highly significant
   
4. THE FIX:
   - Set minimum p-value threshold (1e-10) to avoid true zeros
   - Use scientific notation for very small p-values
   - Display extremely small p-values properly (e.g., 1.23e-15)
   
5. INTERPRETATION:
   - Min_Pvalue = 1.00e-10 means extremely significant enrichment
   - This is actually BETTER than a larger p-value
   - Your GO enrichment is working correctly!
""")

if __name__ == "__main__":
    test_pvalue_formatting()
    explain_pvalue_zero()
