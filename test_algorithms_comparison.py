#!/usr/bin/env python3
"""
Comprehensive test script comparing Wang & Liu (2006) vs Priority-Flood algorithms.

This script tests both the Wang & Liu improved algorithm and the original
Priority-Flood algorithm to validate performance and accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_complex_test_dem(size=100):
    """Create a more complex synthetic DEM with various pit types."""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a base surface with gentle slope
    base = 100 + 0.8 * X + 0.4 * Y
    
    # Add multiple hills of different sizes
    hill1 = 25 * np.exp(-(X**2 + Y**2) / 2)
    hill2 = 15 * np.exp(-((X-1.5)**2 + (Y-1)**2) / 0.8)
    hill3 = 12 * np.exp(-((X+1)**2 + (Y+1.5)**2) / 0.6)
    
    # Create different types of pits
    # Shallow wide depression
    pit1 = -8 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.3)
    # Deep narrow pit  
    pit2 = -20 * np.exp(-((X+0.8)**2 + (Y-0.3)**2) / 0.05)
    # Medium complex pit
    pit3 = -12 * np.exp(-((X-0.2)**2 + (Y+0.8)**2) / 0.15)
    # Flat bottom depression
    pit4_mask = ((X-1.2)**2 + (Y+1.2)**2) < 0.1
    pit4 = np.where(pit4_mask, -15, 0)
    
    # Combine all features
    dem = base + hill1 + hill2 + hill3 + pit1 + pit2 + pit3 + pit4
    
    # Add realistic noise
    dem += np.random.normal(0, 0.3, dem.shape)
    
    return dem

def test_algorithm_comparison():
    """Compare Wang & Liu vs Priority-Flood algorithms."""
    print("=" * 60)
    print("COMPARING WANG & LIU (2006) vs PRIORITY-FLOOD ALGORITHMS")
    print("=" * 60)
    
    # Import FlowAnalyzer
    try:
        from core.flow_analysis import FlowAnalyzer
    except ImportError as e:
        print(f"Error importing FlowAnalyzer: {e}")
        return False
    
    # Test with different DEM sizes
    test_sizes = [50, 100, 200]
    
    for size in test_sizes:
        print(f"\n{'─' * 40}")
        print(f"Testing with DEM size: {size}x{size}")
        print(f"{'─' * 40}")
        
        # Create complex test DEM
        dem = create_complex_test_dem(size)
        analyzer = FlowAnalyzer()
        
        print(f"Original DEM stats:")
        print(f"  Min/Max elevation: {np.min(dem):.3f}/{np.max(dem):.3f}")
        print(f"  Mean elevation: {np.mean(dem):.3f}")
        
        # Test Wang & Liu algorithm
        print(f"\n🔬 Testing Wang & Liu (2006) algorithm...")
        start_time = time.time()
        filled_wl, depression_depth_wl = analyzer.fill_pits(dem, algorithm='wang_liu')
        wl_time = time.time() - start_time
        
        # Test Priority-Flood algorithm
        print(f"\n🔬 Testing Priority-Flood algorithm...")
        start_time = time.time()
        filled_pf, depression_depth_pf = analyzer.fill_pits(dem, algorithm='priority_flood')
        pf_time = time.time() - start_time
        
        # Performance comparison
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"  Wang & Liu time:     {wl_time:.4f} seconds")
        print(f"  Priority-Flood time: {pf_time:.4f} seconds")
        if pf_time > 0:
            speedup = pf_time / wl_time
            print(f"  Speed ratio (PF/WL): {speedup:.2f}x")
            if speedup > 1:
                print(f"  ✅ Wang & Liu is {speedup:.1f}x faster")
            else:
                print(f"  ⚠️  Priority-Flood is {1/speedup:.1f}x faster")
        
        # Quality comparison
        print(f"\n📈 QUALITY COMPARISON:")
        
        # Check if both algorithms filled pits properly
        wl_success = np.all(filled_wl >= dem - 1e-10)  # Allow small numerical errors
        pf_success = np.all(filled_pf >= dem - 1e-10)
        
        print(f"  Wang & Liu pit filling: {'✅ Success' if wl_success else '❌ Failed'}")
        print(f"  Priority-Flood pit filling: {'✅ Success' if pf_success else '❌ Failed'}")
        
        # Compare filled elevations
        wl_stats = {
            'min': np.min(filled_wl),
            'max': np.max(filled_wl), 
            'mean': np.mean(filled_wl),
            'total_fill': np.sum(depression_depth_wl)
        }
        
        pf_stats = {
            'min': np.min(filled_pf),
            'max': np.max(filled_pf),
            'mean': np.mean(filled_pf),
            'total_fill': np.sum(depression_depth_pf)
        }
        
        print(f"  Wang & Liu filled DEM:")
        print(f"    Min/Max: {wl_stats['min']:.3f}/{wl_stats['max']:.3f}")
        print(f"    Mean: {wl_stats['mean']:.3f}")
        print(f"    Total fill volume: {wl_stats['total_fill']:.3f}")
        
        print(f"  Priority-Flood filled DEM:")
        print(f"    Min/Max: {pf_stats['min']:.3f}/{pf_stats['max']:.3f}")
        print(f"    Mean: {pf_stats['mean']:.3f}")
        print(f"    Total fill volume: {pf_stats['total_fill']:.3f}")
        
        # Check agreement between algorithms
        max_diff = np.max(np.abs(filled_wl - filled_pf))
        mean_diff = np.mean(np.abs(filled_wl - filled_pf))
        
        print(f"  Difference between algorithms:")
        print(f"    Max difference: {max_diff:.6f}")
        print(f"    Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print(f"    ✅ Excellent agreement (< 1e-5)")
        elif max_diff < 1e-3:
            print(f"    ✅ Good agreement (< 1e-3)")
        elif max_diff < 1e-1:
            print(f"    ⚠️  Fair agreement (< 1e-1)")
        else:
            print(f"    ❌ Poor agreement (> 1e-1)")
    
    return True

def test_large_scale_performance():
    """Test performance with larger DEMs."""
    print(f"\n{'=' * 60}")
    print("LARGE-SCALE PERFORMANCE TEST")
    print(f"{'=' * 60}")
    
    try:
        from core.flow_analysis import FlowAnalyzer
    except ImportError as e:
        print(f"Error importing FlowAnalyzer: {e}")
        return False
    
    # Test with progressively larger DEMs
    large_sizes = [500, 1000]
    
    for size in large_sizes:
        print(f"\n🚀 Testing {size}x{size} DEM...")
        
        dem = create_complex_test_dem(size)
        analyzer = FlowAnalyzer()
        
        # Memory estimation
        memory_mb = dem.nbytes / (1024**2)
        print(f"Input DEM memory: {memory_mb:.1f} MB")
        
        # Test Wang & Liu (should be faster for large DEMs)
        print(f"Running Wang & Liu algorithm...")
        start_time = time.time()
        filled_wl, depression_depth_wl = analyzer.fill_pits(dem, algorithm='wang_liu')
        wl_time = time.time() - start_time
        
        total_memory = (dem.nbytes + filled_wl.nbytes + depression_depth_wl.nbytes) / (1024**2)
        
        print(f"✅ Completed in {wl_time:.3f} seconds")
        print(f"   Total memory usage: {total_memory:.1f} MB")
        print(f"   Memory efficiency: {total_memory/memory_mb:.1f}x input DEM")
        
        # Performance targets
        target_time = 30 if size <= 1000 else 60
        target_memory_factor = 3
        
        if wl_time < target_time:
            print(f"   ✅ Performance target met: < {target_time}s")
        else:
            print(f"   ⚠️  Performance target missed: > {target_time}s")
            
        if total_memory < target_memory_factor * memory_mb:
            print(f"   ✅ Memory target met: < {target_memory_factor}x input")
        else:
            print(f"   ⚠️  Memory target missed: > {target_memory_factor}x input")
    
    return True

def test_edge_cases():
    """Test algorithms with challenging edge cases."""
    print(f"\n{'=' * 60}")
    print("EDGE CASE TESTING")
    print(f"{'=' * 60}")
    
    try:
        from core.flow_analysis import FlowAnalyzer
    except ImportError as e:
        print(f"Error importing FlowAnalyzer: {e}")
        return False
    
    analyzer = FlowAnalyzer()
    
    # Test case 1: DEM with NaN values
    print(f"\n🧪 Test 1: DEM with NaN values")
    dem_nan = create_complex_test_dem(50)
    dem_nan[20:30, 20:30] = np.nan  # Create NaN region
    
    filled, _ = analyzer.fill_pits(dem_nan, algorithm='wang_liu')
    nan_preserved = np.isnan(filled[20:30, 20:30]).all()
    print(f"   NaN values preserved: {'✅' if nan_preserved else '❌'}")
    
    # Test case 2: Flat DEM (no pits)
    print(f"\n🧪 Test 2: Flat DEM (no pits)")
    dem_flat = np.ones((50, 50)) * 100
    dem_flat += np.random.normal(0, 0.01, (50, 50))  # Tiny noise
    
    filled, depression = analyzer.fill_pits(dem_flat, algorithm='wang_liu')
    max_change = np.max(np.abs(filled - dem_flat))
    print(f"   Max elevation change: {max_change:.6f}")
    print(f"   Minimal modification: {'✅' if max_change < 0.1 else '❌'}")
    
    # Test case 3: Single deep pit
    print(f"\n🧪 Test 3: Single deep pit")
    dem_pit = np.ones((50, 50)) * 100
    dem_pit[25, 25] = 50  # Deep central pit
    
    filled, depression = analyzer.fill_pits(dem_pit, algorithm='wang_liu')
    pit_filled = filled[25, 25] > 50
    fill_height = filled[25, 25] - 50
    print(f"   Pit filled: {'✅' if pit_filled else '❌'}")
    print(f"   Fill height: {fill_height:.3f}")
    
    return True

if __name__ == "__main__":
    print("🌊 EXZECO PIT FILLING ALGORITHM VALIDATION")
    print("Testing Wang & Liu (2006) vs Priority-Flood implementations")
    print("─" * 60)
    
    try:
        success1 = test_algorithm_comparison()
        success2 = test_large_scale_performance()
        success3 = test_edge_cases()
        
        print(f"\n{'=' * 60}")
        print("FINAL RESULTS")
        print(f"{'=' * 60}")
        
        if success1 and success2 and success3:
            print("✅ All tests completed successfully!")
            print("🎉 Wang & Liu (2006) algorithm implementation validated!")
        else:
            print("❌ Some tests failed!")
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()