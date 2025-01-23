
#!/usr/bin/env python3
"""
rpi_random_monitor.py

Continuously monitors the hardware RNG device (/dev/hwrng by default) on a Raspberry Pi,
collects randomness statistics, and logs system metrics (CPU temp, CPU usage, memory, load).
Stores results in CSV and writes rotating logs.

Requirements:
- Python 3
- psutil (optional but recommended: pip3 install psutil)
- vcgencmd (usually pre-installed on Raspberry Pi OS)
"""

import os
import sys
import time
import math
import signal
import logging
import csv
import subprocess
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import Counter, deque

# Attempt to import psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

# Adjust this path if your Pi OS uses /dev/hwrandom or another RNG device
HWRNG_PATH = "/dev/hwrng"
BYTES_PER_READ = 4096              # Bytes to read per iteration
ACCUMULATE_BITS = 1024 * 1024      # Minimum bits for large-sample tests
CSV_FILENAME = "random_monitor_results.csv"
SLEEP_SECONDS = 1                  # Wait time between reads
LOW_ENTROPY_THRESHOLD = 6.0        # Default alert threshold for low entropy
POOR_RANDOMNESS_THRESHOLD = 0.01   # Alert threshold for p-values

# If True, the monitor will gather a few samples on startup to auto-calibrate
# LOW_ENTROPY_THRESHOLD based on observed average entropy.
AUTO_CALIBRATE = True
CALIBRATION_SAMPLES = 5

# Global flag for graceful shutdown
keep_running = True


# =============================================================================
# Logging Setup
# =============================================================================

def init_logger():
    """
    Initialize a rotating file logger plus console output.
    """
    logger = logging.getLogger("random_monitor")
    logger.setLevel(logging.INFO)

    # Remove any old handlers to avoid duplication
    logger.handlers = []

    # Create a rotating file handler (5MB max, 5 backups)
    file_handler = RotatingFileHandler(
        "random_monitor_research.log",
        maxBytes=5_000_000,
        backupCount=5
    )
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# Randomness Test Functions
# =============================================================================

def compute_shannon_entropy(byte_data):
    """
    Shannon entropy: -sum(p_i log2(p_i))
    """
    counts = Counter(byte_data)
    total = len(byte_data)
    return -sum((count / total) * math.log2(count / total)
                for count in counts.values() if count > 0)


def compute_min_entropy(byte_data):
    """
    Min-entropy: -log2(maximum single-event probability)
    """
    counts = Counter(byte_data)
    total = len(byte_data)
    max_freq = max(counts.values()) / total
    return -math.log2(max_freq)


def compute_conditional_entropy(byte_data):
    """
    Negative sum of p * log2(p).
    Despite the name, this is effectively the same form as Shannon entropy.
    Adjust if you truly need "conditional" entropy logic across adjacent samples.
    """
    counts = Counter(byte_data)
    total = len(byte_data)
    probs = [count / total for count in counts.values() if count > 0]
    return -sum(p * math.log2(p) for p in probs)


def bits_from_bytes(byte_data):
    """
    Convert bytes into a list of bits (0 or 1).
    """
    return [(b >> i) & 1 for b in byte_data for i in range(8)]


def frequency_monobit_test(bit_list):
    """
    Frequency (Monobit) Test.
    Returns p-value based on the test statistic.
    """
    n = len(bit_list)
    s = sum(1 if bit == 1 else -1 for bit in bit_list)
    sobs = abs(s) / math.sqrt(n)
    return math.erfc(sobs / math.sqrt(2))


def runs_test(bit_list):
    """
    Runs test.
    Returns p-value based on the number of runs.
    """
    n = len(bit_list)
    pi = sum(bit_list) / n
    # If all bits are identical, runs test is degenerate
    if pi in (0, 1):
        return 0.0
    runs = 1 + sum(bit_list[i] != bit_list[i - 1] for i in range(1, n))
    expected_runs = 2 * n * pi * (1 - pi) + 0.5
    var_runs = 2 * n * pi * (1 - pi) * (2 * n - 2) / (n - 1)
    z = abs(runs - expected_runs) / math.sqrt(var_runs)
    return math.erfc(z / math.sqrt(2))


def block_frequency_test(bit_list, block_size=128):
    """
    Block Frequency Test (simplified version).
    Returns p-value.
    """
    n = len(bit_list)
    if n < block_size:
        return None  # Not enough bits

    num_blocks = n // block_size
    chi_squared = 0.0
    for i in range(num_blocks):
        block = bit_list[i * block_size : (i + 1) * block_size]
        proportion = sum(block) / block_size
        chi_squared += (proportion - 0.5)**2

    # Simple approach:
    test_stat = math.sqrt(num_blocks * chi_squared)
    return math.erfc(test_stat)


def safe_test_call(test_func, *args, logger=None, default=None):
    """
    Safely call a test function. If there's an error, log it and return a default.
    """
    try:
        return test_func(*args)
    except Exception as e:
        if logger:
            logger.error(f"Error in {test_func.__name__}: {e}")
        return default


# =============================================================================
# Raspberry Pi System Metrics
# =============================================================================

def get_rpi_cpu_temp_from_vcgencmd(logger=None):
    """
    Attempt to read CPU temp via 'vcgencmd measure_temp'.
    Returns float temperature in Celsius or None if unavailable.
    """
    try:
        result = subprocess.check_output(["vcgencmd", "measure_temp"], universal_newlines=True)
        # Typically returns something like: "temp=49.2'C
"
        if "temp=" in result:
            temp_str = result.split("=")[1].strip().replace("'C", "")
            return float(temp_str)
    except FileNotFoundError:
        if logger:
            logger.warning("vcgencmd not found on the system.")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to read CPU temperature from vcgencmd: {e}")

    return None


def get_rpi_cpu_temp_from_sysfs(logger=None):
    """
    Attempt to read CPU temp from /sys/class/thermal/thermal_zone0/temp.
    Returns float temperature in Celsius or None if unavailable.
    """
    path = "/sys/class/thermal/thermal_zone0/temp"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            content = f.read().strip()
        # On many Pi systems, this is an integer e.g. "49000" => 49.000°C
        temp_millideg = float(content)
        return temp_millideg / 1000.0
    except Exception as e:
        if logger:
            logger.warning(f"Failed to read CPU temperature from sysfs: {e}")
    return None


def get_system_metrics(logger=None):
    """
    Collect system metrics (CPU temp, CPU usage, memory usage, load avg).
    On Raspberry Pi, we attempt psutil, then vcgencmd, then sysfs for CPU temp.
    """
    metrics = {
        "cpu_temp": "N/A",
        "cpu_usage_percent": "N/A",
        "mem_usage_percent": "N/A",
        "load_avg_1m": "N/A",
        "load_avg_5m": "N/A",
        "load_avg_15m": "N/A",
    }

    # 1) Use psutil if available
    cpu_temp_read = None
    if PSUTIL_AVAILABLE:
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            metrics["cpu_usage_percent"] = round(cpu_usage, 1)

            # Memory usage
            mem = psutil.virtual_memory()
            metrics["mem_usage_percent"] = round(mem.percent, 1)

            # Load average (Unix only). On Windows, psutil.getloadavg() not supported
            if hasattr(psutil, "getloadavg"):
                la = psutil.getloadavg()  # returns (1min,5min,15min)
                metrics["load_avg_1m"] = round(la[0], 2)
                metrics["load_avg_5m"] = round(la[1], 2)
                metrics["load_avg_15m"] = round(la[2], 2)

            # Attempt CPU temp from psutil
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Common Pi keys: 'cpu_thermal', 'thermal_zone0', etc.
                    for key in ("cpu_thermal", "thermal_zone0", "coretemp", "cpu-thermal"):
                        if key in temps and len(temps[key]) > 0:
                            cpu_temp_read = temps[key][0].current
                            break
        except Exception as e:
            if logger:
                logger.warning(f"Failed to gather some psutil metrics: {e}")
    else:
        if logger:
            logger.warning("psutil not available; system metrics might be incomplete.")

    # 2) If psutil didn’t give us a CPU temp, try vcgencmd
    if cpu_temp_read is None:
        cpu_temp_read = get_rpi_cpu_temp_from_vcgencmd(logger=logger)

    # 3) If still None, try reading sysfs
    if cpu_temp_read is None:
        cpu_temp_read = get_rpi_cpu_temp_from_sysfs(logger=logger)

    # Finalize CPU temp in dictionary
    if cpu_temp_read is not None:
        metrics["cpu_temp"] = round(cpu_temp_read, 1)

    return metrics


# =============================================================================
# Signal Handling
# =============================================================================

def signal_handler(sig, frame):
    """
    Handle Ctrl+C / kill signals for graceful exit.
    """
    global keep_running
    logging.getLogger("random_monitor").info("Shutdown signal received. Stopping monitor...")
    keep_running = False


# =============================================================================
# CSV Utilities
# =============================================================================

def init_csv_file(filename, fieldnames, logger):
    """
    Initialize a CSV file with the given fieldnames if it doesn't already exist.
    Returns a tuple (file_handle, dict_writer).
    """
    file_exists = os.path.isfile(filename)
    csv_file = open(filename, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    if not file_exists:
        logger.info(f"Creating new CSV file with header: {filename}")
        writer.writeheader()

    return csv_file, writer


def write_csv_row(writer, row_data, csv_file):
    """
    Write a single row of data and flush to ensure durability.
    """
    writer.writerow(row_data)
    csv_file.flush()


# =============================================================================
# Main Monitoring Logic
# =============================================================================

def auto_calibrate_thresholds(logger):
    """
    Optionally calibrate LOW_ENTROPY_THRESHOLD by sampling hardware RNG.
    """
    global LOW_ENTROPY_THRESHOLD

    if not AUTO_CALIBRATE or CALIBRATION_SAMPLES <= 0:
        return

    logger.info("Auto-calibration of entropy threshold started...")
    total_entropy = 0.0
    samples_collected = 0

    for _ in range(CALIBRATION_SAMPLES):
        try:
            with open(HWRNG_PATH, "rb") as hwrng:
                chunk = hwrng.read(BYTES_PER_READ)
            if len(chunk) == 0:
                logger.warning("Calibration read returned 0 bytes.")
                continue
            e = compute_shannon_entropy(chunk)
            total_entropy += e
            samples_collected += 1
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            continue

    if samples_collected > 0:
        avg_entropy = total_entropy / samples_collected
        # Example strategy: set threshold to 80% of observed average
        new_threshold = 0.8 * avg_entropy
        logger.info(f"Auto-calibration complete. Old threshold={LOW_ENTROPY_THRESHOLD:.2f}, "
                    f"New threshold={new_threshold:.2f} (avg observed={avg_entropy:.2f})")
        LOW_ENTROPY_THRESHOLD = new_threshold
    else:
        logger.warning("No calibration samples collected. Threshold remains unchanged.")


def read_random_data(logger):
    """
    Safe read from /dev/hwrng. Returns bytes or None on failure.
    """
    if not os.path.exists(HWRNG_PATH):
        logger.error(f"{HWRNG_PATH} does not exist.")
        return None

    try:
        with open(HWRNG_PATH, "rb") as hwrng:
            chunk = hwrng.read(BYTES_PER_READ)
        if not chunk:
            logger.warning("Read 0 bytes from hardware RNG.")
            return None
        return chunk
    except Exception as e:
        logger.error(f"Error reading from {HWRNG_PATH}: {e}")
        return None


def run_monitor(logger):
    """
    Main loop that continuously reads from HWRNG, computes metrics, logs results, and writes CSV.
    Designed for Raspberry Pi environment (with psutil or vcgencmd or sysfs).
    """
    global keep_running

    # Auto-calibrate thresholds if requested
    auto_calibrate_thresholds(logger)

    # Prepare CSV logging
    fieldnames = [
        "timestamp",
        "entropy",
        "min_entropy",
        "cond_entropy",
        "monobit_p",
        "runs_p",
        "block_freq_p",
        "cpu_temp",
        "cpu_usage_percent",
        "mem_usage_percent",
        "load_avg_1m",
        "load_avg_5m",
        "load_avg_15m"
    ]
    csv_file, csv_writer = init_csv_file(CSV_FILENAME, fieldnames, logger)

    # Accumulated bits for occasional large-sample tests (if desired)
    accumulated_bits = deque()

    logger.info(f"Monitoring {HWRNG_PATH}... (Press Ctrl+C to stop)")

    try:
        while keep_running:
            chunk = read_random_data(logger)
            if not chunk:
                # Skip this iteration if read failed or returned nothing
                time.sleep(SLEEP_SECONDS)
                continue

            # Convert to bits (for tests that need bits)
            bit_list = bits_from_bytes(chunk)

            # Extend accumulated bits (avoid indefinite growth)
            accumulated_bits.extend(bit_list)
            if len(accumulated_bits) > ACCUMULATE_BITS:
                # If we exceed the threshold, clear or handle accordingly
                accumulated_bits.clear()

            # Perform randomness tests safely
            entropy = safe_test_call(compute_shannon_entropy, chunk, logger=logger, default=0.0)
            min_entropy = safe_test_call(compute_min_entropy, chunk, logger=logger, default=0.0)
            cond_entropy = safe_test_call(compute_conditional_entropy, chunk, logger=logger, default=0.0)

            p_val_monobit = safe_test_call(frequency_monobit_test, bit_list, logger=logger, default=None)
            p_val_runs = safe_test_call(runs_test, bit_list, logger=logger, default=None)
            p_val_blockfreq = safe_test_call(block_frequency_test, bit_list, 128, logger=logger, default=None)

            # Gather Raspberry Pi system metrics (including CPU temp)
            system_metrics = get_system_metrics(logger=logger)

            # Prepare CSV row
            now_str = datetime.utcnow().isoformat()
            row = {
                "timestamp": now_str,
                "entropy": round(entropy, 6),
                "min_entropy": round(min_entropy, 6),
                "cond_entropy": round(cond_entropy, 6),
                "monobit_p": f"{p_val_monobit:.6g}" if p_val_monobit is not None else "",
                "runs_p": f"{p_val_runs:.6g}" if p_val_runs is not None else "",
                "block_freq_p": f"{p_val_blockfreq:.6g}" if p_val_blockfreq is not None else "",
                "cpu_temp": system_metrics["cpu_temp"],
                "cpu_usage_percent": system_metrics["cpu_usage_percent"],
                "mem_usage_percent": system_metrics["mem_usage_percent"],
                "load_avg_1m": system_metrics["load_avg_1m"],
                "load_avg_5m": system_metrics["load_avg_5m"],
                "load_avg_15m": system_metrics["load_avg_15m"],
            }
            write_csv_row(csv_writer, row, csv_file)

            # Log a concise summary
            logger.info(
                f"[{now_str}] "
                f"E={entropy:.2f}, E_min={min_entropy:.2f}, "
                f"monobit_p={p_val_monobit}, runs_p={p_val_runs}, "
                f"CPU Temp={system_metrics['cpu_temp']}"
            )

            # Alert if below threshold
            if entropy < LOW_ENTROPY_THRESHOLD:
                logger.warning(f"[ALERT] Low entropy detected: {entropy:.2f} bits/byte "
                               f"(threshold={LOW_ENTROPY_THRESHOLD:.2f}).")

            # Sleep
            time.sleep(SLEEP_SECONDS)

    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        # Ensure everything is closed properly
        csv_file.close()
        logger.info("Monitor stopped.")


def main():
    # Initialize logger
    logger = init_logger()

    # Register signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run monitor
    run_monitor(logger)


if __name__ == "__main__":
    main()
