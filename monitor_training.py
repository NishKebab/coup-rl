#!/usr/bin/env python3
"""Monitor the training progress."""

import os
import time
import json
from datetime import datetime

def monitor_training():
    """Monitor training progress."""
    print("🔍 MONITORING TRAINING PROGRESS")
    print("="*50)
    
    # Check if training process is running
    result = os.system("ps aux | grep run_fixed_training | grep -v grep > /dev/null")
    if result == 0:
        print("✅ Training process is running!")
    else:
        print("❌ No training process found")
        return
    
    # Monitor for results files
    print("\n📊 Watching for results...")
    
    visualizations_dir = "visualizations"
    models_dir = "models"
    
    # Check directories exist
    if os.path.exists(visualizations_dir):
        print(f"  📁 {visualizations_dir}/ exists")
    else:
        print(f"  📁 Creating {visualizations_dir}/")
        os.makedirs(visualizations_dir, exist_ok=True)
        
    if os.path.exists(models_dir):
        print(f"  📁 {models_dir}/ exists")
    else:
        print(f"  📁 Creating {models_dir}/")
        os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n⏳ Training started at: {datetime.now().strftime('%H:%M:%S')}")
    print("Monitoring every 30 seconds...")
    
    start_time = time.time()
    check_count = 0
    
    while True:
        check_count += 1
        elapsed = int(time.time() - start_time)
        
        print(f"\n🕐 Check #{check_count} - Elapsed: {elapsed//60}m {elapsed%60}s")
        
        # Check if process is still running
        result = os.system("ps aux | grep run_fixed_training | grep -v grep > /dev/null")
        if result != 0:
            print("🏁 Training process completed!")
            break
            
        # Check for new files
        if os.path.exists(visualizations_dir):
            viz_files = [f for f in os.listdir(visualizations_dir) if f.startswith('fixed_training')]
            if viz_files:
                print(f"  📊 Found {len(viz_files)} result files:")
                for f in sorted(viz_files)[-3:]:  # Show last 3
                    print(f"    - {f}")
        
        # Check log file size
        if os.path.exists("training_output.log"):
            size = os.path.getsize("training_output.log")
            print(f"  📝 Log file size: {size} bytes")
        
        # Check CPU usage
        cpu_check = os.popen("ps aux | grep run_fixed_training | grep -v grep | awk '{print $3}'").read().strip()
        if cpu_check:
            cpu_lines = cpu_check.split('\n')
            total_cpu = sum(float(cpu) for cpu in cpu_lines if cpu)
            print(f"  💻 CPU usage: {total_cpu:.1f}%")
        
        # Break after reasonable time if something seems wrong
        if elapsed > 3600:  # 1 hour
            print("⚠️  Training taking longer than expected")
            response = input("Continue monitoring? (y/n): ")
            if response.lower() != 'y':
                break
        
        time.sleep(30)  # Check every 30 seconds
    
    # Final check for results
    print(f"\n🔍 Final Results Check:")
    
    if os.path.exists(visualizations_dir):
        viz_files = [f for f in os.listdir(visualizations_dir) if f.startswith('fixed_training')]
        if viz_files:
            latest_viz = sorted(viz_files)[-1]
            print(f"  📊 Latest visualization: {latest_viz}")
            
            # Try to read training data
            data_files = [f for f in viz_files if f.endswith('.json')]
            if data_files:
                latest_data = sorted(data_files)[-1]
                try:
                    with open(os.path.join(visualizations_dir, latest_data), 'r') as f:
                        data = json.load(f)
                    
                    if 'win_rates' in data:
                        print(f"  🏆 Training completed with results!")
                        
                        # Show win rates
                        for agent_id, wins in data['win_rates'].items():
                            if wins:
                                final_rate = sum(wins[-50:]) / min(50, len(wins)) if wins else 0
                                print(f"    {agent_id}: {final_rate:.3f} win rate")
                        
                        # Show completion rate
                        if 'completion_rates' in data and data['completion_rates']:
                            final_completion = data['completion_rates'][-1]
                            print(f"    Completion rate: {final_completion:.3f}")
                            
                except Exception as e:
                    print(f"  ❌ Error reading data: {e}")
        else:
            print(f"  📊 No visualization files found yet")
    
    print(f"\n✅ Monitoring complete!")

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")