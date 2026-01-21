"""
Generate synthetic post-silicon test data for Reinforcement Learning
This script creates realistic test scenarios for dynamic test flow selection
"""

import numpy as np
import pandas as pd
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("GENERATING SYNTHETIC POST-SILICON TEST DATA FOR RL")
print("=" * 70)

# Configuration
NUM_CHIPS = 1000  # Number of chips to test
NUM_TEST_TYPES = 10  # Number of different test types available

# Define test types with characteristics
test_types = {
    'VOLTAGE_TEST': {'cost': 2, 'time': 5, 'defect_coverage': 0.15},
    'CURRENT_TEST': {'cost': 3, 'time': 7, 'defect_coverage': 0.12},
    'FREQUENCY_TEST': {'cost': 4, 'time': 10, 'defect_coverage': 0.18},
    'TEMPERATURE_TEST': {'cost': 5, 'time': 12, 'defect_coverage': 0.10},
    'POWER_TEST': {'cost': 3, 'time': 8, 'defect_coverage': 0.14},
    'TIMING_TEST': {'cost': 6, 'time': 15, 'defect_coverage': 0.20},
    'LEAKAGE_TEST': {'cost': 4, 'time': 9, 'defect_coverage': 0.11},
    'NOISE_TEST': {'cost': 2, 'time': 6, 'defect_coverage': 0.08},
    'STRESS_TEST': {'cost': 8, 'time': 20, 'defect_coverage': 0.25},
    'FUNCTIONAL_TEST': {'cost': 10, 'time': 30, 'defect_coverage': 0.35},
}

# Create test configuration file
os.makedirs('data', exist_ok=True)

with open('data/test_config.json', 'w') as f:
    json.dump(test_types, f, indent=2)

print(f"\n✓ Test configuration saved to data/test_config.json")
print(f"✓ Number of test types: {len(test_types)}")

# Generate chip data with hidden defect types
chip_data = []

defect_types = ['voltage_defect', 'timing_defect', 'power_defect', 
                'functional_defect', 'no_defect']

# Mapping: which tests are most effective for which defects
test_effectiveness = {
    'voltage_defect': ['VOLTAGE_TEST', 'POWER_TEST', 'LEAKAGE_TEST'],
    'timing_defect': ['TIMING_TEST', 'FREQUENCY_TEST', 'STRESS_TEST'],
    'power_defect': ['POWER_TEST', 'CURRENT_TEST', 'TEMPERATURE_TEST'],
    'functional_defect': ['FUNCTIONAL_TEST', 'STRESS_TEST', 'TIMING_TEST'],
    'no_defect': []  # No defect - all tests will pass
}

for i in range(NUM_CHIPS):
    chip_id = f"CHIP_{i:05d}"
    
    # Randomly assign defect type (70% have defects, 30% are good)
    if np.random.random() < 0.7:
        defect_type = np.random.choice(defect_types[:-1])  # Exclude 'no_defect'
    else:
        defect_type = 'no_defect'
    
    # Generate test results for each test type
    test_results = {}
    for test_name, test_info in test_types.items():
        # Test passes if chip has no defect OR test is not effective for this defect
        if defect_type == 'no_defect':
            # Good chip - all tests pass with high probability
            test_results[test_name] = 1 if np.random.random() > 0.05 else 0
        else:
            # Defective chip - effective tests will likely fail
            if test_name in test_effectiveness[defect_type]:
                # Effective test - high probability of detecting defect
                test_results[test_name] = 0 if np.random.random() > 0.2 else 1
            else:
                # Non-effective test - may or may not detect defect
                test_results[test_name] = 1 if np.random.random() > 0.3 else 0
    
    chip_data.append({
        'chip_id': chip_id,
        'defect_type': defect_type,
        **test_results
    })

# Create DataFrame
df = pd.DataFrame(chip_data)

# Save to CSV
df.to_csv('data/chip_test_data.csv', index=False)

print(f"\n✓ Chip test data generated")
print(f"  Total chips: {len(df)}")
print(f"  Defect distribution:")
for defect in defect_types:
    count = (df['defect_type'] == defect).sum()
    percentage = (count / len(df)) * 100
    print(f"    {defect}: {count} ({percentage:.1f}%)")

# Generate test history data (simulated past test sequences)
print(f"\n✓ Generating test history data...")

test_history = []
for i in range(500):  # 500 historical test sequences
    chip_idx = np.random.randint(0, NUM_CHIPS)
    chip = chip_data[chip_idx]
    
    # Simulate a test sequence (3-7 tests)
    num_tests = np.random.randint(3, 8)
    test_sequence = np.random.choice(list(test_types.keys()), num_tests, replace=False)
    
    total_cost = sum([test_types[t]['cost'] for t in test_sequence])
    total_time = sum([test_types[t]['time'] for t in test_sequence])
    
    # Check if defect was detected
    defect_detected = False
    if chip['defect_type'] != 'no_defect':
        for test in test_sequence:
            if chip[test] == 0:  # Test failed
                defect_detected = True
                break
    
    test_history.append({
        'sequence_id': f"SEQ_{i:04d}",
        'chip_id': chip['chip_id'],
        'test_sequence': ','.join(test_sequence),
        'num_tests': num_tests,
        'total_cost': total_cost,
        'total_time': total_time,
        'defect_detected': defect_detected,
        'defect_type': chip['defect_type']
    })

df_history = pd.DataFrame(test_history)
df_history.to_csv('data/test_history.csv', index=False)

print(f"✓ Test history saved")
print(f"  Total sequences: {len(df_history)}")
print(f"  Average tests per sequence: {df_history['num_tests'].mean():.2f}")
print(f"  Average cost per sequence: {df_history['total_cost'].mean():.2f}")
print(f"  Average time per sequence: {df_history['total_time'].mean():.2f} minutes")
print(f"  Detection rate: {(df_history['defect_detected'].sum() / len(df_history)) * 100:.1f}%")

# Create summary statistics
summary = {
    'total_chips': NUM_CHIPS,
    'total_test_types': len(test_types),
    'defect_rate': ((df['defect_type'] != 'no_defect').sum() / len(df)),
    'avg_test_cost': np.mean([t['cost'] for t in test_types.values()]),
    'avg_test_time': np.mean([t['time'] for t in test_types.values()]),
    'total_sequences': len(df_history)
}

with open('data/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Summary statistics saved to data/summary.json")
print("=" * 70)
print("DATA GENERATION COMPLETE!")
print("=" * 70)
print(f"\nGenerated files:")
print(f"  1. data/test_config.json - Test type configurations")
print(f"  2. data/chip_test_data.csv - Chip test results")
print(f"  3. data/test_history.csv - Historical test sequences")
print(f"  4. data/summary.json - Summary statistics")
print(f"\nTotal data size: {os.path.getsize('data/chip_test_data.csv') / 1024:.2f} KB")
