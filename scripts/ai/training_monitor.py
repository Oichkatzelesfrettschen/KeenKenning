#!/usr/bin/env python3
"""
Live Training Monitor - Detects anomalies and logs alerts.

Monitors:
1. Loss trajectory (should decrease)
2. Constraint loss (should increase then stabilize)
3. Valid grid rate (should increase)
4. Violations (should decrease)
5. Process health (still running?)
6. GPU utilization

Alerts on:
- Loss spike (>20% increase)
- Training stall (no output for 5 min)
- cst collapse (drops >50%)
- Valid rate regression
- Process crash
"""

import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Configuration
OUTPUT_FILE = "/tmp/claude/-home-eirikr-Github-KeenKenning/tasks/training.output"
ALERT_LOG = "/tmp/training_alerts.log"
CHECK_INTERVAL = 30  # seconds
STALL_THRESHOLD = 300  # 5 minutes

# State tracking
last_loss = None
last_cst = None
last_valid_rate = None
last_violations = None
last_line_count = 0
last_update_time = time.time()
epoch_losses = []


def log_alert(level, message):
    """Log alert to file and print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    print(entry)
    with open(ALERT_LOG, "a") as f:
        f.write(entry + "\n")


def parse_metrics(text):
    """Extract metrics from training output."""
    metrics = {}

    # Current batch loss/cst
    batch_match = re.findall(r'\[\s*\d+\]\s+loss=([\d.]+)\s+ce=[\d.]+\s+cst=([\d.]+)', text)
    if batch_match:
        metrics['batch_loss'] = float(batch_match[-1][0])
        metrics['batch_cst'] = float(batch_match[-1][1])

    # Validation metrics
    val_loss = re.search(r'Val Loss:\s+([\d.]+)', text)
    if val_loss:
        metrics['val_loss'] = float(val_loss.group(1))

    valid_rate = re.search(r'Valid Grid Rate:\s+([\d.]+)%', text)
    if valid_rate:
        metrics['valid_rate'] = float(valid_rate.group(1))

    violations = re.search(r'Avg Violations:\s+([\d.]+)', text)
    if violations:
        metrics['violations'] = float(violations.group(1))

    entropy = re.search(r'Gen Entropy:\s+([\d.]+)', text)
    if entropy:
        metrics['entropy'] = float(entropy.group(1))

    epoch = re.findall(r'Epoch (\d+)/(\d+)', text)
    if epoch:
        metrics['epoch'] = int(epoch[-1][0])
        metrics['total_epochs'] = int(epoch[-1][1])

    return metrics


def check_gpu():
    """Check GPU utilization."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(', ')
        return {
            'gpu_util': int(parts[0]),
            'gpu_mem': int(parts[1]),
            'gpu_temp': int(parts[2])
        }
    except:
        return None


def check_process():
    """Check if training process is still running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'train_autoregressive'],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except:
        return False


def monitor_cycle():
    """Run one monitoring cycle."""
    global last_loss, last_cst, last_valid_rate, last_violations
    global last_line_count, last_update_time, epoch_losses

    # Check if output file exists
    if not os.path.exists(OUTPUT_FILE):
        log_alert("ERROR", f"Output file not found: {OUTPUT_FILE}")
        return

    # Read output file
    with open(OUTPUT_FILE, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    current_line_count = len(lines)

    # Check for stall
    if current_line_count == last_line_count:
        stall_time = time.time() - last_update_time
        if stall_time > STALL_THRESHOLD:
            log_alert("WARN", f"Training stalled! No output for {stall_time:.0f}s")
    else:
        last_update_time = time.time()
        last_line_count = current_line_count

    # Parse metrics
    metrics = parse_metrics(content)

    if not metrics:
        return

    # Check process health
    if not check_process():
        log_alert("CRITICAL", "Training process not running!")

    # Check GPU
    gpu = check_gpu()
    if gpu:
        if gpu['gpu_util'] < 10:
            log_alert("WARN", f"Low GPU utilization: {gpu['gpu_util']}%")
        if gpu['gpu_temp'] > 85:
            log_alert("WARN", f"High GPU temperature: {gpu['gpu_temp']}C")

    # Analyze metrics
    if 'batch_loss' in metrics:
        loss = metrics['batch_loss']
        if last_loss is not None:
            # Check for loss spike (>20% increase)
            if loss > last_loss * 1.2 and last_loss > 0.1:
                log_alert("WARN", f"Loss spike: {last_loss:.4f} -> {loss:.4f} (+{(loss/last_loss-1)*100:.1f}%)")
        last_loss = loss

    if 'batch_cst' in metrics:
        cst = metrics['batch_cst']
        if last_cst is not None and last_cst > 0.001:
            # Check for cst collapse (>50% drop)
            if cst < last_cst * 0.5:
                log_alert("WARN", f"Constraint loss dropped: {last_cst:.4f} -> {cst:.4f}")
        last_cst = cst

    if 'valid_rate' in metrics:
        rate = metrics['valid_rate']
        if last_valid_rate is not None:
            # Check for valid rate regression (>5% drop)
            if rate < last_valid_rate - 5 and last_valid_rate > 1:
                log_alert("WARN", f"Valid rate regression: {last_valid_rate:.1f}% -> {rate:.1f}%")
        last_valid_rate = rate

    if 'violations' in metrics:
        viols = metrics['violations']
        if last_violations is not None:
            # Check for violation increase (>20% increase)
            if viols > last_violations * 1.2 and last_violations > 5:
                log_alert("WARN", f"Violations increased: {last_violations:.1f} -> {viols:.1f}")
        last_violations = viols

    # Status update every 10 cycles
    epoch = metrics.get('epoch', '?')
    loss = metrics.get('batch_loss', last_loss or 0)
    cst = metrics.get('batch_cst', last_cst or 0)
    valid = metrics.get('valid_rate', last_valid_rate or 0)
    viols = metrics.get('violations', last_violations or 0)

    status = f"Epoch {epoch} | loss={loss:.4f} | cst={cst:.4f} | valid={valid:.1f}% | viols={viols:.1f}"
    if gpu:
        status += f" | GPU {gpu['gpu_util']}% {gpu['gpu_temp']}C"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")


def main():
    log_alert("INFO", "Training monitor started")
    log_alert("INFO", f"Watching: {OUTPUT_FILE}")
    log_alert("INFO", f"Alerts logged to: {ALERT_LOG}")

    try:
        while True:
            monitor_cycle()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        log_alert("INFO", "Monitor stopped by user")


if __name__ == "__main__":
    main()
