# PARS: Partial-Label-Learning-Inspired Recommender Systems

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

PARS (Partial-Label-Learning-Inspired Recommender Systems) is a novel approach that combines partial label learning with transformer-based sequence modeling for improved recommendation performance. This method addresses the challenge of implicit feedback in recommender systems by treating user interactions as partial labels and progressively refining them during training.

## Key Features

- **Partial Label Learning**: Implements PRODEN method for handling ambiguous user feedback
- **Transformer-based Architecture**: Utilizes BERT-style transformers for sequence modeling
- **Masked Language Modeling (MLM)**: Self-supervised learning for better item representations
- **Global Representation Learning**: Generates user/session representations for efficient recommendation
- **Multi-task Learning**: Combines MLM and partial label learning objectives

## Architecture

```
Input Sequences → Item Embeddings + Position Embeddings
                            ↓
                    Transformer Encoder
                            ↓
                 ┌──────────┴──────────┐
                 ↓                     ↓
            MLM Head           Global Projection
                 ↓                     ↓
           MLM Loss            Item Scores → PLL Loss
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bjtu-lucas-nlp/PARS.git
cd PARS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The input data should be in CSV format with the following columns:
- `session_id`: Session/user identifier
- `item_id_sequence`: List of item IDs in the sequence (as string)
- `label_sequence`: Binary labels for each item (0 or 1)
- `unique_item_sequence`: Mask for unique items in sequence

Example:
```csv
session_id	item_id_sequence	label_sequence	unique_item_sequence
0	"[1, 5, 3, 2]"	"[0, 0, 1, 0]"	"[1, 1, 1, 1]"
```

## Usage

### Training

Basic training command:
```bash
python pars_model.py \
    --train_data_file datasets/Yoochoose/train.csv \
    --val_data_file datasets/Yoochoose/val.csv \
    --test_data_file datasets/Yoochoose/test.csv \
    --num_items 39300 \
    --num_sessions 417370 \
    --epochs 300 \
    --batch_size 256 \
    --data_name Yoochoose \
    --max_seq_len 50 \
    --lr 1e-4 \
    --mlm_weight 1.0 \
    --hidden_size 256 \
    --embedding_dim 128 \
    --save_dir trainedmodel/Yoochoose
```

### Full Parameter List

```bash
python pars_model.py \
    --train_data_file PATH         # Path to training data
    --val_data_file PATH           # Path to validation data
    --test_data_file PATH          # Path to test data
    --sep_sym SEPARATOR            # CSV separator (default: '\t')
    --num_items NUM                # Total number of items
    --num_sessions NUM             # Total number of sessions
    --batch_size SIZE              # Batch size (default: 256)
    --data_name NAME               # Dataset name for logging
    --max_seq_len LENGTH           # Maximum sequence length (default: 50)
    --epochs NUM                   # Number of epochs (default: 10)
    --lr RATE                      # Learning rate (default: 1e-4)
    --mlm_weight WEIGHT            # Weight for MLM loss (default: 1.0)
    --hidden_size SIZE             # Transformer hidden size (default: 256)
    --embedding_dim DIM            # Final embedding dimension (default: 128)
    --save_dir DIR                 # Checkpoint directory (default: 'checkpoints')
    --seed NUM                     # Random seed (default: 42)
```

### Evaluation

The model automatically evaluates on the test set during training and saves metrics including:
- **AUC**: Area Under the ROC Curve
- **HR@K**: Hit Rate at K
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **Precision@K**: Precision at K
- **Recall@K**: Recall at K
- **F1@K**: F1 Score at K

## Model Architecture Details

### Components

1. **Item Embeddings**: Learnable embeddings for each item
2. **Position Embeddings**: Position-aware representations
3. **Transformer Encoder**: Multi-layer self-attention mechanism
4. **Global Projection**: Projects sequence representations to user/session embeddings
5. **MLM Head**: Predicts masked items for self-supervised learning

### Training Objectives

1. **Partial Label Learning (PLL)**: 
   - Uses PRODEN method to iteratively refine ambiguous labels
   - Updates pseudo-labels based on model predictions
   
2. **Masked Language Modeling (MLM)**:
   - Randomly masks items in sequences
   - Predicts masked items to learn item relationships

## Output

The training process generates:
- `checkpoints/best_model.pt`: Best model based on validation NDCG@10
- `checkpoints/PARS_*_test_metrics.csv`: Test metrics for each epoch
- `checkpoints/PARS_*_full_log.json`: Complete training log
- `checkpoints/final_results.json`: Final evaluation results

## Citation

If you use PARS in your research, please cite:

```bibtex
@article{
  title={PARS: Partial-Label-Learning-Inspired Recommender Systems},
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- PRODEN method for partial label learning
- Hugging Face Transformers library
- PyTorch community
