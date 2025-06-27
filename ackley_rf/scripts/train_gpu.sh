#!/bin/bash

# Ackley function random forest regression GPU acceleration training script

# Set working directory to project root
cd "$(dirname "$0")/.."

# Create necessary directories
mkdir -p data/raw data/processed models visualizations

echo "=========================================="
echo "Starting Ackley function random forest GPU training"
echo "=========================================="

# Try to set CUDA environment variables
if [ -f "scripts/setup_cuda_env.sh" ]; then
    echo "Setting CUDA environment variables..."
    source scripts/setup_cuda_env.sh
fi

# Check dependencies
echo "Checking dependencies..."
MISSING_DEPS=false
for pkg in numpy pandas scikit-learn matplotlib statsmodels; do
    if ! python -c "import $pkg" &> /dev/null; then
        echo "❌ Missing dependency: $pkg"
        MISSING_DEPS=true
    fi
done

if [ "$MISSING_DEPS" = true ]; then
    echo "Installing missing dependencies..."
    pip install numpy pandas scikit-learn matplotlib statsmodels
fi

# Check GPU availability
echo "Checking GPU status..."
if nvidia-smi &> /dev/null; then
    nvidia-smi
    
    # Check CUDA libraries
    if [ -z "$LD_LIBRARY_PATH" ]; then
        echo "⚠️ LD_LIBRARY_PATH not set, may affect GPU library loading"
    else
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    fi
    
    # Simplified GPU detection logic
    python3 << 'END_PYTHON'
import sys
try:
    import cupy as cp
    # Simple CuPy test
    test_array = cp.array([1, 2, 3])
    test_result = cp.sum(test_array)
    cp.asnumpy(test_result)
    print('CuPy test successful')
    sys.exit(0)
except ImportError:
    print('CuPy library not installed')
    sys.exit(1)
except Exception as e:
    print(f'CuPy test failed: {e}')
    sys.exit(1)
END_PYTHON

    if [ $? -eq 0 ]; then
        echo "✅ CuPy library available, will use GPU acceleration"
        USE_GPU=true
    else
        echo "⚠️ CuPy library not available, will use CPU mode"
        echo "Attempting to install CuPy..."
        pip install cupy-cuda12x
        
        # Check CuPy again
        if python -c "import cupy" &> /dev/null; then
            echo "✅ CuPy installation successful, will use GPU acceleration"
            USE_GPU=true
        else
            echo "❌ CuPy installation failed, will use CPU mode"
            USE_GPU=false
        fi
    fi
else
    echo "⚠️ No NVIDIA GPU detected or drivers not properly installed"
    echo "Will run in CPU mode"
    USE_GPU=false
fi

# 1. Data preprocessing
echo "Step 1: Data preprocessing"
python3 << 'END_PYTHON'
import sys
sys.path.append('.')
from data.load_data import load_data, preprocess_data, save_data

# Load data
data = load_data()

# Preprocess data
processed_data = preprocess_data(data)

# Save processed data
save_data(processed_data, 'data/processed/ackley_data.npz')
print("Data preprocessing completed and saved")
END_PYTHON

# 2. Model training and tuning
echo "Step 2: Model training and tuning"
python3 << 'END_PYTHON'
import sys
import time
sys.path.append('.')
from data.load_data import load_processed_data
from models.model_tuning import bayesian_optimization, save_tuning_results

# Record start time
start_time = time.time()

# Load preprocessed data
data = load_processed_data('data/processed/ackley_data.npz')

# Use Bayesian optimization for parameter tuning
results = bayesian_optimization(
    data,
    n_trials=50,  # Increase number of trials
    timeout=None,
    use_gpu=True,  # Enable GPU acceleration
    random_state=42
)

# Save tuning results
save_tuning_results(results, 'models/tuning_results/bayesian_opt')

# Print training time
training_time = time.time() - start_time
print(f"Total training time: {training_time:.2f} seconds")
END_PYTHON

# 3. Generate evaluation plots
echo "Step 3: Generating evaluation plots"
python3 << 'END_PYTHON'
import sys
sys.path.append('.')
from evaluation.residual_plot import plot_residuals
from data.load_data import load_processed_data
import joblib

# Load data and model
data = load_processed_data('data/processed/ackley_data.npz')
model = joblib.load('models/tuning_results/bayesian_opt/best_model.pkl')

# Generate residual plot
plot_residuals(
    model, 
    data['X_test'], 
    data['y_test'],
    save_path='visualizations/residual_plot.png'
)
print("Evaluation plots generated")
END_PYTHON

echo "=========================================="
echo "Training process completed"
echo "==========================================" 