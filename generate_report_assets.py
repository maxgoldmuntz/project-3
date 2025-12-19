import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = "./results"
PLOTS_DIR = "./report_plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Apply a professional style
plt.style.use('ggplot')

# Define experiment groups
EXP_QA_SIZE = {
    "30% Data": "qa_r8_sz0.3",
    "50% Data": "qa_r8_sz0.5",
    "100% Data": "qa_r8_sz1.0"
}

EXP_QA_RANK = {
    "Rank 8": "qa_r8_sz1.0",
    "Rank 16": "qa_r16_sz1.0",
    "Rank 32": "qa_r32_sz1.0"
}

EXP_CODE_RANK = {
    "Rank 8": "code_r8_sz1.0",
    "Rank 16": "code_r16_sz1.0",
    "Rank 32": "code_r32_sz1.0"
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_metric(folder_name, metric_key):
    path = os.path.join(RESULTS_DIR, folder_name, "eval_results.json")
    if not os.path.exists(path): return None
    try:
        with open(path, 'r') as f:
            return json.load(f).get(metric_key)
    except: return None

def load_loss_history(folder_name):
    path = os.path.join(RESULTS_DIR, folder_name, "trainer_state.json")
    if not os.path.exists(path): return ([], []), ([], [])
    
    with open(path, 'r') as f:
        history = json.load(f).get('log_history', [])
    
    steps_train = [e['step'] for e in history if 'loss' in e]
    loss_train = [e['loss'] for e in history if 'loss' in e]
    steps_val = [e['step'] for e in history if 'eval_loss' in e]
    loss_val = [e['eval_loss'] for e in history if 'eval_loss' in e]
    
    return (steps_train, loss_train), (steps_val, loss_val)

def smooth_curve(points, window_size=10):
    if len(points) < window_size: return points
    return np.convolve(points, np.ones(window_size)/window_size, mode='valid')

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================
def plot_bar_enhanced(title, xlabel, ylabel, data_dict, metric_key, filename):
    labels = []
    values = []
    colors = []
    patterns = []
    
    print(f"\n--- Processing: {title} ---")
    for label, folder in data_dict.items():
        val = load_metric(folder, metric_key)
        labels.append(label)
        
        if val is not None:
            if "accuracy" in metric_key: val = val * 100 
            values.append(val)
            colors.append('#3498db') # Professional Blue
            patterns.append(None)
        else:
            values.append(0.1) # Tiny bar for visual placeholder
            colors.append('#bdc3c7') # Gray
            patterns.append('//') # Hatched pattern for "Working on it"

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.9, width=0.6)
    
    # Apply patterns to missing data
    for bar, pattern in zip(bars, patterns):
        if pattern: bar.set_hatch(pattern)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add data labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        if colors[i] == '#bdc3c7':
            ax.text(bar.get_x() + bar.get_width()/2, 1, 
                    "RUNNING...", ha='center', va='bottom', fontsize=10, 
                    fontstyle='italic', color='#7f8c8d')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val + (max(values)*0.01), 
                    f"{val:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Dynamic scaling
    if max(values) > 1: ax.set_ylim(0, max(values) * 1.15)
    else: ax.set_ylim(0, 1) # Normalized metrics

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()
    print(f"✅ Saved High-Res Plot: {filename}")

def plot_learning_dynamics(tasks):
    """Generates a complex plot with smoothing for multiple tasks"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (title, folder) in enumerate(tasks):
        ax = axes[i]
        (t_steps, t_loss), (v_steps, v_loss) = load_loss_history(folder)
        
        if not t_steps:
            ax.text(0.5, 0.5, "AWAITING DATA\n(Job In Progress)", 
                    ha='center', va='center', fontsize=14, color='gray',
                    bbox=dict(facecolor='#ecf0f1', boxstyle='round,pad=1'))
            ax.set_title(f"{title} (Pending)", fontsize=14)
            continue

        # 1. Plot Raw Training Data (Transparent)
        ax.plot(t_steps, t_loss, alpha=0.15, color='blue', label='Raw Train Loss')
        
        # 2. Plot Smoothed Training Data (Solid)
        if len(t_loss) > 10:
            smooth_t = smooth_curve(t_loss, window_size=20)
            # Adjust steps to match smoothed length
            smooth_steps = t_steps[len(t_steps)-len(smooth_t):] 
            ax.plot(smooth_steps, smooth_t, color='blue', linewidth=2, label='Smoothed Train Trend')

        # 3. Plot Validation Data (Points + Line)
        if v_steps:
            ax.plot(v_steps, v_loss, color='#e74c3c', linewidth=2, marker='o', markersize=6, label='Validation Loss')

        ax.set_title(f"{title}: Convergence Analysis", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "detailed_learning_curves.png"), dpi=300)
    plt.close()
    print("✅ Saved High-Res Learning Curves")

def generate_summary_table():
    """Creates a PNG table of the results"""
    data = []
    # Collect QA Data
    for lbl, folder in EXP_QA_SIZE.items():
        val = load_metric(folder, "eval_accuracy")
        val_str = f"{val*100:.2f}%" if val else "Running"
        data.append(["QA Size Exp", lbl, val_str])
    for lbl, folder in EXP_QA_RANK.items():
        if lbl == "Rank 8": continue # Duplicate of above
        val = load_metric(folder, "eval_accuracy")
        val_str = f"{val*100:.2f}%" if val else "Running"
        data.append(["QA Rank Exp", lbl, val_str])
    # Collect Code Data
    for lbl, folder in EXP_CODE_RANK.items():
        val = load_metric(folder, "eval_loss")
        val_str = f"{val:.4f}" if val else "Running"
        data.append(["Code Rank Exp", lbl, val_str])

    fig, ax = plt.subplots(figsize=(10, len(data)*0.6 + 2))
    ax.axis('off')
    
    table_data = [["Experiment", "Condition", "Result"]] + data
    table = ax.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.3, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Style the header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            cell.set_edgecolor('#bdc3c7')
    
    plt.title("Summary of Experimental Results", fontsize=16, fontweight='bold', y=0.95)
    plt.savefig(os.path.join(PLOTS_DIR, "results_table.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved Results Table Image")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    plot_bar_enhanced("QA Accuracy vs. Data Size", "Data Fraction", "Exact Match (%)", 
                      EXP_QA_SIZE, "eval_accuracy", "exp_qa_size_v2.png")
    
    plot_bar_enhanced("QA Accuracy vs. LoRA Rank", "Rank (r)", "Exact Match (%)", 
                      EXP_QA_RANK, "eval_accuracy", "exp_qa_rank_v2.png")
    
    plot_bar_enhanced("Code Model Loss vs. LoRA Rank", "Rank (r)", "Validation Loss", 
                      EXP_CODE_RANK, "eval_loss", "exp_code_rank_v2.png")

    plot_learning_dynamics([
        ("QA Task (RoBERTa)", "qa_r8_sz1.0"),
        ("Code Task (Mellum)", "code_r8_sz1.0")
    ])
    
    generate_summary_table()
