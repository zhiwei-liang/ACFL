from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
from tensorboard import program
import threading
import webbrowser
import glob

app = Flask(__name__)

# Global variable to store tensorboard process
tb_process = None

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 