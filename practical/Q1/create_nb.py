import json
import os

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# CV HW5 - Question 1: Pet Segmentation Full Pipeline\n',
                '\n',
                'This notebook demonstrates the complete pipeline for the Oxford-IIIT Pet Segmentation task. It uses the modular scripts provided (`dataset.py`, `unet.py`, `train.py`, `utils.py`, `evaluate.py`, `sam_evaluation.py`, `section5_comparison.py`) to train a U-Net, evaluate it, perform error analysis, and compare it with the Segment Anything Model (SAM).'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '%matplotlib inline\n',
                'import torch\n',
                'import matplotlib.pyplot as plt\n',
                '\n',
                '# Load all functions and classes from the provided scripts\n',
                'from dataset import get_datasets, visualize_samples, DataLoader, denormalize\n',
                'from unet import UNet\n',
                'from utils import BCEDiceLoss, calculate_metrics\n',
                'from train import train_model\n',
                'from evaluate import evaluate_model, qualitative_evaluation, error_analysis\n',
                'from sam_evaluation import evaluate_sam_zero_shot\n',
                'from section5_comparison import compare_models\n',
                '\n',
                '# Set device\n',
                'device = \'cuda\' if torch.cuda.is_available() else \'cpu\'\n',
                'print(f"Using device: {device}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Task 1, 2, 3: Dataset Preparation and Visualization\n',
                'First, we load the dataset, apply transformations (resizing, normalizations, and augmentations for the training set), and visualize some samples.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'train_set, val_set, test_set = get_datasets(root_dir=\'./data\')\n',
                '\n',
                'print(f"Dataset sizes:\\n- Train: {len(train_set)}\\n- Validation: {len(val_set)}\\n- Test: {len(test_set)}\\n")\n',
                '\n',
                '# Visualize training samples (includes augmentations)\n',
                'visualize_samples(train_set, num_samples=3)\n',
                'plt.show()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Task 4: U-Net Architecture Verification\n',
                'We instantiate our U-Net architecture to ensure it correctly processes an input of the required shape.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'model = UNet(n_channels=3, n_classes=1).to(device)\n',
                'dummy_input = torch.randn(1, 3, 256, 256).to(device)\n',
                'output = model(dummy_input)\n',
                '\n',
                'print(f"Input shape:  {dummy_input.shape}")\n',
                'print(f"Output shape: {output.shape}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Task 5, 6, 7: Training the U-Net\n',
                'Train the U-Net model on the Oxford Pet dataset. The `train_model` function handles the training loop, validates after each epoch, saves the best weights (`best_unet_model.pth`), and generates loss/metric curves.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Note: Reduce num_epochs for a quick test run. \n',
                '# For actual training, you might want 10 or more.\n',
                'trained_model, history = train_model(num_epochs=10, batch_size=16, learning_rate=1e-3, device=device)\n',
                'plt.show()  # Display the training plots generated in the script'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Task 8, 9, 10: U-Net Evaluation & Error Analysis\n',
                'Load the best model checkpoint and evaluate it quantitatively (IoU, Dice Score) and qualitatively. We also perform an error analysis by finding the samples where the model struggled the most.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# 1. Quantitative Evaluation\n',
                'best_model, test_dataset = evaluate_model(\'best_unet_model.pth\', device=device)\n',
                '\n',
                '# 2. Qualitative Evaluation\n',
                'qualitative_evaluation(best_model, test_dataset, device=device, num_samples=3)\n',
                'plt.show()\n',
                '\n',
                '# 3. Error Analysis (Samples with low IoU)\n',
                'error_analysis(best_model, test_dataset, device=device, num_samples=2)\n',
                'plt.show()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Task 11, 12, 13: Segment Anything Model (SAM) Evaluation\n',
                'Evaluate the zero-shot capabilities of SAM on our test dataset. It downloads the SAM weights (if needed) and generates masks. Since SAM extracts all prominent objects, we calculate metrics against the ground truth to find the best matching mask.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'evaluate_sam_zero_shot(num_samples=3, device=device)\n',
                'plt.show()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Task 14, 15: Model Comparison (U-Net vs. SAM)\n',
                'Perform a direct comparison between the fully-supervised U-Net and the zero-shot SAM across a subset of the test data.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Running quantitative comparison on 20 samples to save time (SAM is computationally heavy without specialized optimizations).\n',
                '# It will also generate a qualitative side-by-side plot for 3 samples.\n',
                'compare_models(num_qualitative_samples=3, num_quantitative_samples=20, device=device)\n',
                'plt.show()'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.9.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open(os.path.join(os.path.dirname(__file__), 'Q1_Pipeline.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print('Notebook successfully created with all imported scripts.')
