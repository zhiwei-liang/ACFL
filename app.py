from flask import Flask, render_template, request, redirect, url_for, jsonify
import subprocess
import os
from tensorboard import program
import threading
import webbrowser
import glob
import psutil
import GPUtil
import time

app = Flask(__name__)

# Global variable to store tensorboard process
tb_process = None

def get_system_metrics():
    """Get system resource usage metrics"""
    # CPU信息
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # 内存信息
    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024.0 ** 3)  # GB
    memory_used = memory.used / (1024.0 ** 3)    # GB
    memory_percent = memory.percent
    
    # 磁盘信息
    disk = psutil.disk_usage('/')
    disk_total = disk.total / (1024.0 ** 3)      # GB
    disk_used = disk.used / (1024.0 ** 3)        # GB
    disk_percent = disk.percent
    
    # GPU信息
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'temperature': gpu.temperature
            })
    except:
        gpu_info = []
    
    return {
        'cpu': {
            'percent': cpu_percent,
            'count': cpu_count
        },
        'memory': {
            'total': round(memory_total, 2),
            'used': round(memory_used, 2),
            'percent': memory_percent
        },
        'disk': {
            'total': round(disk_total, 2),
            'used': round(disk_used, 2),
            'percent': disk_percent
        },
        'gpu': gpu_info,
        'timestamp': time.time()
    }

def get_log_dirs():
    """Get all tensorboard log directories"""
    log_base = "log"  # Base directory for logs
    if not os.path.exists(log_base):
        os.makedirs(log_base)
    
    # Get all subdirectories in log directory
    log_dirs = []
    for dir_path in glob.glob(os.path.join(log_base, "**/tensorboard"), recursive=True):
        log_dirs.append(dir_path)
    return log_dirs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        algorithm = request.form.get('algorithm')
        dataset = request.form.get('dataset')
        num_classes = request.form.get('num_classes')
        learning_rate = request.form.get('learning_rate')
        batch_size = request.form.get('batch_size')
        comm = request.form.get('comm')
        model = request.form.get('model')
        note = request.form.get('note')
        
        # Build command
        cmd = f"nohup python -u algorithms/{algorithm}/train_pacs{'_ACFL' if algorithm == 'ACFL' else ''}.py " \
              f"--dataset {dataset} --num_classes {num_classes} " \
              f"--lr {learning_rate} --batch_size {batch_size} --comm {comm} " \
              f"--model {model} --note {note} >{algorithm}_{dataset}.log &"
        
        # Execute command
        subprocess.Popen(cmd, shell=True)
        return "Task started! Please check the log file for progress."
    
    # Return page for GET request
    return render_template('index.html', active_page='train')

@app.route('/tensorboard', methods=['GET', 'POST'])
def tensorboard():
    global tb_process
    log_dirs = get_log_dirs()
    
    if request.method == 'POST':
        selected_dir = request.form.get('log_dir')
        
        # Kill existing tensorboard process if any
        if tb_process:
            tb_process.kill()
        
        # Start new tensorboard process
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', selected_dir, '--port', '6006'])
        url = tb.launch()
        
        # Store the process
        tb_process = tb
        
        # Open tensorboard in new tab
        webbrowser.open(url)
        return redirect(url_for('tensorboard'))
    
    return render_template('tensorboard.html', log_dirs=log_dirs, active_page='tensorboard')

@app.route('/stop_tensorboard')
def stop_tensorboard():
    global tb_process
    if tb_process:
        tb_process.kill()
        tb_process = None
    return redirect(url_for('tensorboard'))

@app.route('/monitor')
def monitor():
    return render_template('monitor.html', active_page='monitor')

@app.route('/api/system-metrics')
def system_metrics():
    return jsonify(get_system_metrics())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 