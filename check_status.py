#!/usr/bin/env python3
"""Quick status check for training."""

import os
import time
from datetime import datetime

def quick_status():
    """Quick status check."""
    print("🚀 FULL TRAINING STATUS")
    print("="*40)
    
    # Check if process is running
    result = os.system("ps aux | grep run_fixed_training | grep -v grep > /dev/null")
    if result == 0:
        print("✅ Training is RUNNING!")
        
        # Get CPU usage
        cpu_check = os.popen("ps aux | grep run_fixed_training | grep -v grep | awk '{print $3}'").read().strip()
        if cpu_check:
            print(f"💻 CPU Usage: {cpu_check}%")
        
        # Get memory usage  
        mem_check = os.popen("ps aux | grep run_fixed_training | grep -v grep | awk '{print $4}'").read().strip()
        if mem_check:
            print(f"🧠 Memory Usage: {mem_check}%")
            
        # Get process start time
        time_check = os.popen("ps aux | grep run_fixed_training | grep -v grep | awk '{print $9}'").read().strip()
        if time_check:
            print(f"⏰ Started: {time_check}")
        
        print(f"\n📊 Expected Results:")
        print(f"  - Win rates will become non-zero (no more 0%)")
        print(f"  - Games will complete with actual winners")
        print(f"  - Episode lengths will be shorter and meaningful")
        print(f"  - Visualization will be generated automatically")
        
        print(f"\n📁 Results will be saved to:")
        print(f"  - visualizations/fixed_training_results_*.png")
        print(f"  - visualizations/fixed_training_data_*.json")
        
        print(f"\n⏳ Training Progress:")
        print(f"  - Target episodes: 1000")
        print(f"  - Progress updates every 50 episodes")
        print(f"  - Estimated time: 10-30 minutes")
        
    else:
        print("❌ Training not running")
        
        # Check for completed results
        if os.path.exists("visualizations"):
            fixed_files = [f for f in os.listdir("visualizations") if f.startswith('fixed_training')]
            if fixed_files:
                print(f"✅ Found {len(fixed_files)} result files - training may have completed!")
                latest = sorted(fixed_files)[-1]
                print(f"📊 Latest: {latest}")

if __name__ == "__main__":
    quick_status()