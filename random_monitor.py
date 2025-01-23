#!/usr/bin/env python3
"""
random_monitor.py

Continuously reads random data from /dev/hwrng on a Raspberry Pi, every second.
Performs three research-grade randomness checks:
  1. Shannon Entropy
  2. Frequency (Monobit) Test [NIST STS]
  3. Runs Test [NIST STS]

Logs results to a CSV file: random_monitor_results.csv

Author: Generated Script
Date:   Current Date

Usage:
  1. Ensure /dev/hwrng exists (Raspberry Pi).
  2. `chmod +x random_monitor.py`
  3. `./random_monitor.py`  (press Ctrl+C to stop)
"""

import os
import sys
import time
import math
import signal
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

HWRNG_PATH = "/dev/hwrng"            # Hardware RNG device file
BYTES_PER_READ = 4096                # How many bytes to read each second
CSV_FILENAME = "random_monitor_results.csv"
SLEEP_SECONDS = 1                    # Wait 1 second between reads
SKIP_RUNS_TEST_THRESHOLD = 0.01      # If proportion of 1s is too far from 0.5

# =============================================================================
# Global Flag for Graceful Shutdown
# =============================================================================

keep_running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C / kill signals for a graceful exit."""
    global keep_running
    print("\n[INFO] Shutdown signal received. Stopping monitor...")
    keep_running = False

# =============================================================================
# Randomness Tests
# =============================================================================

def compute_shannon_entropy(byte_data):
    """
    Computes Shannon entropy (bits/byte) for the given bytes object.
    """
    # Count frequency of each possible byte value
    counts = [0]*256
    for b in byte_data:
        counts[b] += 1
    total = len(byte_data)
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c/total
            entropy -= p * math.log2(p)
    return entropy

def bits_from_bytes(byte_data):
    """
    Converts a bytes object into a list of bits (0 or 1).
    """
    bit_list = []
    for b in byte_data:
        for i in range(8):
            bit = (b >> i) & 1
            bit_list.append(bit)
    return bit_list

def frequency_monobit_test(bit_list):
    """
    NIST Frequency (Monobit) Test
    Returns the p-value of the test.
    """
    n = len(bit_list)
    if n == 0:
        return 0.0

    s = 0
    for bit in bit_list:
        s += (1 if bit == 1 else -1)

    s_obs = abs(s) / math.sqrt(n)
    # p-value
    p_val = math.erfc(s_obs / math.sqrt(2))
    return p_val

def runs_test(bit_list):
    """
    NIST Runs Test
    Returns the p-value of the runs test.
    """
    n = len(bit_list)
    if n < 2:
        return 0.0

    n1 = sum(bit_list)
    n0 = n - n1
    pi = n1 / n

    if abs(pi - 0.5) > SKIP_RUNS_TEST_THRESHOLD:
        return 0.0

    runs = 1
    for i in range(1, n):
        if bit_list[i] != bit_list[i-1]:
            runs += 1

    expected_runs = 1.0 + (2.0 * n0 * n1) / n
    var_runs = (2.0 * n0 * n1 * (2.0 * n0 * n1 - n)) / (n**2 * (n-1))
    if var_runs <= 0:
        return 0.0

    z = abs(runs - expected_runs) / math.sqrt(var_runs)
    p_val = math.erfc(z / math.sqrt(2))
    return p_val

# =============================================================================
# Main Monitoring Function
# =============================================================================

def main():
    # Check if /dev/hwrng exists
    if not os.path.exists(HWRNG_PATH):
        print(f"[ERROR] {HWRNG_PATH} does not exist on this system.")
        sys.exit(1)

    # Initialize CSV if not present
    if not os.path.isfile(CSV_FILENAME):
        with open(CSV_FILENAME, "w") as f:
            header = "timestamp,entropy_bits_per_byte,monobit_p_value,runs_p_value
"
            f.write(header)

    print(f"[INFO] Monitoring {HWRNG_PATH} every {SLEEP_SECONDS} second(s).")
    print(f"[INFO] Logging results to {CSV_FILENAME}. Press Ctrl+C to stop.\n")

    while keep_running:
        try:
            # Read from HWRNG
            with open(HWRNG_PATH, "rb") as hwrng:
                chunk = hwrng.read(BYTES_PER_READ)

            # Basic stats
            entropy = compute_shannon_entropy(chunk)
            bit_list = bits_from_bytes(chunk)

            # Frequency (monobit) test
            p_val_monobit = frequency_monobit_test(bit_list)

            # Runs test
            p_val_runs = runs_test(bit_list)

            # Append results to CSV
            now_str = datetime.utcnow().isoformat()
            with open(CSV_FILENAME, "a") as f:
                line = f"{now_str},{entropy:.6f},{p_val_monobit:.6g},{p_val_runs:.6g}\n"
                f.write(line)

            # Print a short status to console
            print(f"[{now_str}] Entropy={entropy:.4f} | Monobit p={p_val_monobit:.4g} | Runs p={p_val_runs:.4g}")

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(SLEEP_SECONDS)

    print("[INFO] Monitor stopped.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
