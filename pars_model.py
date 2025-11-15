"""
PARS: Partial-Label-Learning-Inspired Recommender Systems
A novel approach combining partial label learning with transformer-based sequence modeling
for improved recommendation performance.
"""

import fire
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, AdamW
from sklearn.metrics import roc_auc_score, ndcg_score, precision_score, recall_score, f1_score
import os
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import ast
import logging
import torch.nn.functional as F
import json
import csv
import os
from datetime import datetime

# Device configuration
cur_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {cur_device}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PARSDataset(Dataset):
    """
    Dataset class for PARS model training.
    Implements partial label learning with masked language modeling for sequences.
    """

    def __init__(self, data_file, mask_ratio=0.2, num_items=50, num_sessions=50, max_seq_len=50, sep_sym='\t'):
        """
        Initialize PARS Dataset.

        Args:
            data_file: Path to the data file
            mask_ratio: Ratio of tokens to mask for MLM
            num_items: Total number of items in the dataset
            num_sessions: Total number of sessions/users
            max_seq_len: Maximum sequence length
            sep_sym: Separator symbol in the data file
        """
        self.data = pd.read_csv(data_file, sep=sep_sym)
        self.mask_ratio = mask_ratio
        self.mask_token = num_items  # Mask token ID (assuming num_items is max ID + 1)
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.num_sessions = num_sessions

        # Initialize global partial label sequences
        self.global_partial_labels = self._initialize_global_partial_labels()

        # Track update statistics
        self.update_count = 0
        self.alpha = 0.9  # Momentum parameter for label updates

    def _initialize_global_partial_labels(self):
        """Initialize global partial label sequences for all sessions."""
        # Create tensor of shape [num_sessions, num_items]
        global_partial_labels = torch.zeros(self.num_sessions, self.num_items)

        # Process each data record
        for idx, row in self.data.iterrows():
            session_idx = row['session_id']

            # Ensure session_idx is within valid range
            if session_idx >= self.num_sessions:
                continue

            # Get item sequences
            items_seq = ast.literal_eval(row['item_id_sequence'])

            # Initialize with uniform distribution over sequence items
            unique_items = list(set([item for item in items_seq if item < self.num_items]))
            if unique_items:
                for item in unique_items:
                    global_partial_labels[session_idx, item] = 1 / len(unique_items)

        return global_partial_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        session_idx = row["session_id"]

        # Ensure session_idx is within valid range
        if session_idx >= self.num_sessions:
            session_idx = session_idx % self.num_sessions

        # Parse sequences
        items_seq = ast.literal_eval(row["item_id_sequence"])
        label_seq = ast.literal_eval(row["label_sequence"])
        label_valid_sequence = ast.literal_eval(row["unique_item_sequence"])

        # Convert to tensors
        items_seq = torch.tensor(items_seq, dtype=torch.long)
        label_seq = torch.tensor(label_seq, dtype=torch.long)
        label_valid_sequence = torch.tensor(label_valid_sequence, dtype=torch.long)

        original_seq_len = len(items_seq)

        # Create attention mask (1 for actual items, 0 for padding)
        attention_mask = torch.ones(original_seq_len, dtype=torch.long)

        # Create masked sequence for MLM
        masked_items_seq = items_seq.clone()
        mask_indices = []
        original_tokens = []

        for i in range(original_seq_len):
            if random.random() < self.mask_ratio:
                mask_indices.append(i)
                original_tokens.append(items_seq[i].item())
                masked_items_seq[i] = self.mask_token

        # Convert to tensors
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)
        original_tokens = torch.tensor(original_tokens, dtype=torch.long)

        # Create label set for partial label learning
        label_set = items_seq[label_seq != 0].unique().tolist()

        # Get partial label distribution for this session
        session_partial_labels = self.global_partial_labels[session_idx]

        # Generate negative samples for contrastive learning
        neg_samples = []
        all_items = set(range(1, self.num_items))
        seen_items = set(items_seq.tolist())
        candidate_negs = list(all_items - seen_items)

        # Select 5 negative samples
        if candidate_negs:
            neg_samples = random.sample(candidate_negs, min(5, len(candidate_negs)))
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)

        # Handle sequence length (truncate or pad)
        if original_seq_len > self.max_seq_len:
            # Truncate
            items_seq = items_seq[:self.max_seq_len]
            label_seq = label_seq[:self.max_seq_len]
            label_valid_sequence = label_valid_sequence[:self.max_seq_len]
            masked_items_seq = masked_items_seq[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

            # Update mask indices to only include positions within max_seq_len
            valid_mask = mask_indices < self.max_seq_len
            mask_indices = mask_indices[valid_mask]
            original_tokens = original_tokens[valid_mask]

            seq_len = self.max_seq_len
        else:
            seq_len = original_seq_len

            # Pad sequences if needed
            if seq_len < self.max_seq_len:
                padding_len = self.max_seq_len - seq_len

                items_seq = torch.cat([items_seq, torch.zeros(padding_len, dtype=torch.long)])
                label_seq = torch.cat([label_seq, torch.zeros(padding_len, dtype=torch.long)])
                label_valid_sequence = torch.cat([label_valid_sequence, torch.zeros(padding_len, dtype=torch.long)])
                masked_items_seq = torch.cat([masked_items_seq, torch.zeros(padding_len, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_len, dtype=torch.long)])

        return {
            'user_id': torch.tensor(session_idx, dtype=torch.long),
            'sequences': items_seq,
            'label_seq': label_seq,
            'label_valid_sequence': label_valid_sequence,
            'masked_sequences': masked_items_seq,
            'attention_mask': attention_mask,
            'mask_indices': mask_indices,
            'original_tokens': original_tokens,
            'label_set': torch.tensor(label_set, dtype=torch.long) if label_set else torch.tensor([], dtype=torch.long),
            'partial_label_seq': session_partial_labels,
            'neg_samples': neg_samples,
            'seq_len': torch.tensor(seq_len, dtype=torch.long)
        }

    def update_partial_labels(self, session_indices, logits, true_labels=None, alpha=None):
        """
        Update global partial labels using PRODEN method.

        Args:
            session_indices: Session indices [batch_size]
            logits: Model output scores [batch_size, num_items]
            true_labels: Optional true labels [batch_size, num_items]
            alpha: Optional momentum coefficient

        Returns:
            loss_tensor: Loss value with gradient
            loss_value: Numerical loss value
        """
        if alpha is None:
            alpha = self.alpha

        batch_size = logits.size(0)
        device = logits.device

        # Create loss tensor connected to computation graph
        loss_tensor = torch.tensor(0.0, device=device, requires_grad=True)
        total_loss = 0.0

        for i in range(batch_size):
            session_idx = session_indices[i].item()

            # Skip invalid session indices
            if session_idx >= self.num_sessions:
                continue

            # Get current session's partial labels
            current_labels = self.global_partial_labels[session_idx].to(device)

            # Calculate loss using PRODEN method
            batch_logits = logits[i:i + 1]  # Keep dimensions [1, num_items]
            batch_labels = current_labels.unsqueeze(0)  # [1, num_items]

            # Apply softmax to get probability distribution
            output = F.softmax(batch_logits, dim=1)

            # Calculate cross-entropy loss
            l = batch_labels * torch.log(output + 1e-10)
            batch_loss = (-torch.sum(l)) / l.size(0)

            # Accumulate loss tensor
            if i == 0:
                loss_tensor = batch_loss
            else:
                loss_tensor = loss_tensor + batch_loss

            # Update pseudo-labels (PRODEN method)
            revisedY = batch_labels.clone()
            revisedY[revisedY > 0] = 1  # Binarize
            revisedY = revisedY * output  # Weight by current predictions
            row_sums = revisedY.sum(dim=1, keepdim=True)
            revisedY = revisedY / (row_sums + 1e-10)  # Normalize

            # Update global partial labels
            with torch.no_grad():
                new_labels = revisedY.squeeze(0).cpu()
                self.global_partial_labels[session_idx] = alpha * self.global_partial_labels[session_idx] + (
                            1 - alpha) * new_labels

            # Track loss value
            total_loss += batch_loss.item()

        # Average loss
        loss_tensor = loss_tensor / batch_size
        loss_value = total_loss / batch_size

        # Increment update counter
        self.update_count += 1

        # Log statistics periodically
        if self.update_count % 1000 == 0:
            num_nonzero = torch.sum(self.global_partial_labels > 0.01).item()
            avg_value = torch.mean(self.global_partial_labels[self.global_partial_labels > 0.01]).item()
            logger.info(f"Update #{self.update_count}, Avg Loss: {loss_value:.4f}, Non-zero entries: {num_nonzero}")

        return loss_tensor, loss_value

    def save_global_partial_labels(self, path):
        """Save global partial label state."""
        torch.save({
            'global_partial_labels': self.global_partial_labels,
            'update_count': self.update_count,
            'alpha': self.alpha
        }, path)
        logger.info(f"Saved global partial labels to {path}")

    def load_global_partial_labels(self, path):
        """Load global partial label state."""
        state = torch.load(path)
        self.global_partial_labels = state['global_partial_labels']
        self.update_count = state['update_count']
        self.alpha = state.get('alpha', 0.9)
        logger.info(f"Loaded global partial labels from {path}")


def pars_collate_fn(batch):
    """
    Collate function for PARS dataset to handle variable-length elements.
    """
    batch_size = len(batch)

    # Fixed-length elements
    user_ids = torch.stack([item['user_id'] for item in batch])
    sequences = torch.stack([item['sequences'] for item in batch])
    label_seq = torch.stack([item['label_seq'] for item in batch])
    label_valid_sequence = torch.stack([item['label_valid_sequence'] for item in batch])
    masked_sequences = torch.stack([item['masked_sequences'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    seq_len = torch.stack([item['seq_len'] for item in batch])

    # Handle variable-length elements
    # 1. mask_indices and original_tokens
    max_masks = max(len(item['mask_indices']) for item in batch)
    padded_mask_indices = []
    padded_original_tokens = []
    mask_indices_length = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        indices = item['mask_indices']
        tokens = item['original_tokens']
        mask_indices_length[i] = len(indices)

        if len(indices) > 0:
            padded_indices = torch.cat([
                indices,
                torch.zeros(max_masks - len(indices), dtype=torch.long)
            ])
            padded_tokens = torch.cat([
                tokens,
                torch.zeros(max_masks - len(tokens), dtype=torch.long)
            ])
        else:
            padded_indices = torch.zeros(max_masks, dtype=torch.long)
            padded_tokens = torch.zeros(max_masks, dtype=torch.long)

        padded_mask_indices.append(padded_indices)
        padded_original_tokens.append(padded_tokens)

    if padded_mask_indices:
        mask_indices = torch.stack(padded_mask_indices)
        original_tokens = torch.stack(padded_original_tokens)
    else:
        mask_indices = torch.zeros((batch_size, 0), dtype=torch.long)
        original_tokens = torch.zeros((batch_size, 0), dtype=torch.long)

    # 2. label_set
    max_labels = max(len(item['label_set']) for item in batch)
    padded_label_sets = []
    label_set_length = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        label_set = item['label_set']
        label_set_length[i] = len(label_set)

        if len(label_set) > 0:
            padded_set = torch.cat([
                label_set,
                torch.zeros(max_labels - len(label_set), dtype=torch.long)
            ])
        else:
            padded_set = torch.zeros(max_labels, dtype=torch.long)

        padded_label_sets.append(padded_set)

    if padded_label_sets:
        label_set = torch.stack(padded_label_sets)
    else:
        label_set = torch.zeros((batch_size, 0), dtype=torch.long)

    # 3. neg_samples
    max_negs = max(len(item['neg_samples']) for item in batch)
    padded_neg_samples = []

    for item in batch:
        neg_samples = item['neg_samples']
        if len(neg_samples) > 0:
            padded_negs = torch.cat([
                neg_samples,
                torch.zeros(max_negs - len(neg_samples), dtype=torch.long)
            ])
        else:
            padded_negs = torch.zeros(max_negs, dtype=torch.long)
        padded_neg_samples.append(padded_negs)

    if padded_neg_samples:
        neg_samples = torch.stack(padded_neg_samples)
    else:
        neg_samples = torch.zeros((batch_size, 0), dtype=torch.long)

    # 4. partial_label_seq
    partial_label_seqs = [item['partial_label_seq'] for item in batch]
    stacked_partial_labels = torch.stack(partial_label_seqs)

    return {
        'user_ids': user_ids,
        'sequences': sequences,
        'label_seq': label_seq,
        'label_valid_sequence': label_valid_sequence,
        'masked_sequences': masked_sequences,
        'attention_mask': attention_mask,
        'mask_indices': mask_indices,
        'mask_indices_length': mask_indices_length,
        'original_tokens': original_tokens,
        'label_set': label_set,
        'label_set_length': label_set_length,
        'partial_label_seq': stacked_partial_labels,
        'neg_samples': neg_samples,
        'seq_len': seq_len
    }


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


class PARSModel(nn.Module):
    """
    PARS: Partial-Label-Learning-Inspired Recommender System
    A transformer-based sequence recommendation model with partial label learning.
    """

    def __init__(self, num_items, num_sessions, hidden_size=256, embedding_dim=128,
                 num_layers=2, num_heads=4, dropout=0.1, max_seq_length=50):
        """
        Initialize PARS model.

        Args:
            num_items: Total number of items
            num_sessions: Total number of sessions
            hidden_size: Transformer hidden layer size
            embedding_dim: Final embedding dimension
            num_layers: Number of Transformer layers
            num_heads: Number of attention heads
            dropout: Dropout ratio
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.num_items = num_items
        self.num_sessions = num_sessions
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size)  # +1 for mask/padding token

        # Position embeddings
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)

        # Transformer configuration
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_seq_length
        )

        # Transformer encoder
        self.encoder = BertModel(config)

        # Global representation projection layer
        self.global_projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )

        # Item representation projection layer
        self.item_projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # MLM prediction head
        self.mlm_head = nn.Linear(hidden_size, num_items)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

        # Initialize projection layers
        for module in [self.global_projection, self.item_projection]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, user_ids, sequences, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            user_ids: User/session IDs [batch_size]
            sequences: Item sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            logits: Item preference scores [batch_size, num_items]
            sequence_output: Sequence representations [batch_size, seq_len, hidden_size]
            global_repr: Global representations [batch_size, embedding_dim]
        """
        batch_size, seq_len = sequences.shape

        # Ensure sequence length doesn't exceed model's maximum
        if seq_len > self.max_seq_length:
            sequences = sequences[:, :self.max_seq_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_length]
            seq_len = self.max_seq_length

        # Generate position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=sequences.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get position embeddings
        position_embeddings = self.position_embeddings(position_ids)

        # Get item embeddings
        item_embeddings = self.item_embeddings(sequences)

        # Combine embeddings
        embeddings = item_embeddings + position_embeddings

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (sequences != 0).long()

        # Pass through Transformer encoder
        encoder_outputs = self.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get sequence output
        sequence_output = encoder_outputs.last_hidden_state

        # Generate global representation using masked average pooling
        masked_output = sequence_output * attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(masked_output, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        global_repr = sum_embeddings / sum_mask

        # Project global representation to embedding space
        global_repr = self.global_projection(global_repr)

        # Get all item embeddings and project
        all_item_embeddings = self.item_embeddings.weight[:self.num_items]
        all_item_embeddings = self.item_projection(all_item_embeddings)

        # Calculate item preference scores (dot product)
        logits = torch.matmul(global_repr, all_item_embeddings.transpose(0, 1))

        return logits, sequence_output, global_repr

    def compute_mlm_loss(self, sequence_output, mask_indices, original_tokens, mask_indices_length=None):
        """
        Compute Masked Language Model loss.

        Args:
            sequence_output: Transformer encoder output [batch_size, seq_len, hidden_size]
            mask_indices: Masked positions [batch_size, max_masks]
            original_tokens: Original tokens at masked positions [batch_size, max_masks]
            mask_indices_length: Number of valid masks per sample [batch_size]

        Returns:
            mlm_loss: MLM task loss
        """
        batch_size = sequence_output.size(0)
        device = sequence_output.device

        # Check if there are masked positions
        if mask_indices.size(1) == 0:
            return torch.tensor(0.0, device=device)

        # Prepare MLM predictions
        all_preds = []
        all_labels = []

        for i in range(batch_size):
            # Determine valid masked positions
            if mask_indices_length is not None:
                valid_length = mask_indices_length[i].item()
                valid_positions = mask_indices[i, :valid_length]
                valid_tokens = original_tokens[i, :valid_length]
            else:
                valid_positions = mask_indices[i][mask_indices[i] > 0]
                valid_tokens = original_tokens[i][original_tokens[i] > 0]

            # Ensure length match
            min_length = min(len(valid_positions), len(valid_tokens))
            if min_length > 0:
                valid_positions = valid_positions[:min_length]
                valid_tokens = valid_tokens[:min_length]

                # Get embeddings at masked positions
                pos_embeds = sequence_output[i, valid_positions]
                # Use MLM head for predictions
                pos_preds = self.mlm_head(pos_embeds)

                all_preds.append(pos_preds)
                all_labels.append(valid_tokens)

        # If no valid masked positions
        if not all_preds:
            return torch.tensor(0.0, device=device)

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calculate cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()
        mlm_loss = loss_fn(all_preds, all_labels)

        return mlm_loss

    def save(self, path):
        """Save model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_items': self.num_items,
            'num_sessions': self.num_sessions,
            'hidden_size': self.hidden_size,
            'embedding_dim': self.embedding_dim,
            'max_seq_length': self.max_seq_length
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def train_model(model, train_loader, val_loader=None, test_loader=None, epochs=10, lr=1e-4,
                device='cuda', mlm_weight=1.0, data_name="Dataset", save_dir='./checkpoints'):
    """
    Train PARS model.

    Args:
        model: PARS model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Training device
        mlm_weight: Weight for MLM task
        data_name: Dataset name
        save_dir: Directory for saving checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)

    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"PARS_{data_name}_{timestamp}"

    # Create detailed log directory
    detail_log_dir = os.path.join(save_dir, f"{experiment_name}_detailed_logs")
    os.makedirs(detail_log_dir, exist_ok=True)

    # Initialize training log
    training_log = {
        "experiment_name": experiment_name,
        "config": {
            "epochs": epochs,
            "learning_rate": lr,
            "device": str(device),
            "model_name": model.__class__.__name__
        },
        "training_history": {
            "epoch": [],
            "train_loss": [],
            "val_metrics": [],
            "test_metrics": []
        }
    }

    # CSV file path for test metrics
    csv_file_path = os.path.join(save_dir, f"{experiment_name}_test_metrics.csv")
    csv_headers = None

    logger.info(f"Starting training, results will be saved to: {save_dir}")
    logger.info(f"Experiment name: {experiment_name}")

    # Get training dataset
    dataset = train_loader.dataset

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Track best validation performance
    best_val_metric = 0.0

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        mlm_losses = 0.0
        pll_losses = 0.0

        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            user_ids = batch['user_ids'].to(device)
            sequences = batch['sequences'].to(device)
            masked_sequences = batch['masked_sequences'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_indices = batch['mask_indices'].to(device)
            original_tokens = batch['original_tokens'].to(device)
            mask_indices_length = batch['mask_indices_length'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits, sequence_output, global_repr = model(user_ids, sequences, attention_mask)

            # MLM loss
            mlm_loss = 0.0
            if mlm_weight > 0:
                try:
                    # Forward pass on masked sequences
                    _, masked_sequence_output, _ = model(user_ids, masked_sequences, attention_mask)

                    # Compute MLM loss
                    mlm_loss = model.compute_mlm_loss(
                        masked_sequence_output,
                        mask_indices,
                        original_tokens,
                        mask_indices_length
                    )
                    mlm_losses += mlm_loss.item()
                except Exception as e:
                    logger.error(f"Error computing MLM loss: {e}")
                    mlm_loss = torch.tensor(0.0, device=device)

            # Partial label learning loss
            pll_loss_value = 0.0
            try:
                # Update global partial labels and get loss
                pll_loss_tensor, pll_loss_value = dataset.update_partial_labels(
                    session_indices=user_ids,
                    logits=logits
                )
                pll_losses += pll_loss_value
            except Exception as e:
                logger.error(f"Error updating partial labels: {e}")

            # Total loss
            loss = mlm_weight * mlm_loss + pll_loss_tensor

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Update total loss
            train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mlm': f"{mlm_loss.item():.4f}" if mlm_weight > 0 else "0.0",
                'pll': f"{pll_loss_value:.4f}"
            })

        # Calculate average losses
        avg_loss = train_loss / len(train_loader)
        avg_mlm = mlm_losses / len(train_loader) if mlm_weight > 0 else 0
        avg_pll = pll_losses / len(train_loader)

        logger.info(f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_loss:.4f} "
                    f"(MLM: {avg_mlm:.4f}, PLL: {avg_pll:.4f})")

        # Validation
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device)
            logger.info(f"Validation metrics: {val_metrics}")

            # Update learning rate based on NDCG@10
            val_metric = val_metrics.get('ndcg@10', 0)
            scheduler.step(val_metric)

            # Save best model
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                model_path = os.path.join(save_dir, 'best_model.pt')
                model.save(model_path)
                logger.info(f"Saving best model with NDCG@10 = {best_val_metric:.4f}")

        # Test evaluation
        if test_loader is not None:
            test_metrics = evaluate_model(model, test_loader, device)
            logger.info(f"Test metrics: {test_metrics}")

            # Record to training history
            training_log["training_history"]["epoch"].append(epoch + 1)
            training_log["training_history"]["train_loss"].append(avg_loss)
            if val_loader is not None:
                training_log["training_history"]["val_metrics"].append(val_metrics.copy())
            training_log["training_history"]["test_metrics"].append(test_metrics.copy())

            # Save to CSV file
            test_metrics_with_epoch = {"epoch": epoch + 1, "train_loss": avg_loss, **test_metrics}

            # First time writing creates headers
            if csv_headers is None:
                csv_headers = list(test_metrics_with_epoch.keys())
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                    writer.writeheader()
                    writer.writerow(test_metrics_with_epoch)
            else:
                # Append to existing file
                with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                    writer.writerow(test_metrics_with_epoch)

    # Final test evaluation with best model
    if test_loader is not None and os.path.exists(os.path.join(save_dir, 'best_model.pt')):
        model.load(os.path.join(save_dir, 'best_model.pt'))
        logger.info("Loaded best model for final test evaluation")

        test_metrics = evaluate_model(model, test_loader, device)
        logger.info(f"Final test metrics: {test_metrics}")

    # Save complete training log
    json_file_path = os.path.join(save_dir, f"{experiment_name}_full_log.json")
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    logger.info(f"Training complete!")
    logger.info(f"CSV results file: {csv_file_path}")
    logger.info(f"Complete log file: {json_file_path}")

    return model


def evaluate_model(model, data_loader, device, k_values=[1, 3, 5, 10, 20]):
    """
    Evaluate PARS model.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Evaluation device
        k_values: List of K values for top-K metrics

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    # Initialize metric containers
    results = {metric: {k: [] for k in k_values} for metric in ["HR", "HR-P", "NDCG", "Precision", "Recall", "F1"]}

    # Collect predictions for AUC
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            user_ids = batch['user_ids'].to(device)
            sequences = batch['sequences'].to(device)
            label_seq = batch['label_seq'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_valid_sequence = batch['label_valid_sequence'].to(device)

            # Forward pass
            logits, _, _ = model(user_ids, sequences, attention_mask)

            # Get sequence item scores
            scores = torch.softmax(logits, dim=1)
            sequence_scores = torch.gather(scores, dim=1, index=sequences)

            # Apply unique sequence mask
            sequence_scores = sequence_scores * label_valid_sequence
            labels = label_seq * label_valid_sequence

            # Convert to numpy for metrics calculation
            preds_np = sequence_scores.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.append(preds_np.flatten())
            all_labels.append(labels_np.flatten())

            # Calculate top-K metrics
            for k in k_values:
                _, topk_indices = torch.topk(sequence_scores, k, dim=1)
                topk_indices = topk_indices.cpu().numpy()
                true_indices = (labels_np == 1)

                for i in range(len(labels_np)):
                    relevant_items = np.where(true_indices[i])[0]
                    top_k = topk_indices[i]
                    hit_items = np.intersect1d(top_k, relevant_items)

                    # HR@K
                    results["HR"][k].append(1 if len(hit_items) > 0 else 0)

                    # HR-P@K
                    results["HR-P"][k].append(len(hit_items) / k)

                    # Precision@K
                    results["Precision"][k].append(len(hit_items) / k)

                    # Recall@K
                    results["Recall"][k].append(len(hit_items) / len(relevant_items) if len(relevant_items) > 0 else 0)

                    # F1@K
                    precision = results["Precision"][k][-1]
                    recall = results["Recall"][k][-1]
                    results["F1"][k].append(
                        2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)

                    # NDCG@K
                    dcg = sum([1.0 / np.log2(rank + 2) for rank, idx in enumerate(top_k) if idx in relevant_items])
                    idcg = sum([1.0 / np.log2(rank + 2) for rank in range(min(k, len(relevant_items)))])
                    results["NDCG"][k].append(dcg / idcg if idcg > 0 else 0)

    # Calculate final metrics
    final_results = {}

    # AUC
    all_preds = np.hstack(all_preds)
    all_labels = np.hstack(all_labels)
    auc_score = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    final_results["AUC"] = auc_score

    # Average top-K metrics
    for metric in results:
        for k in k_values:
            final_results[f"{metric}@{k}"] = np.mean(results[metric][k])

    # Print results
    print("\nEvaluation Results:")
    print(f"AUC: {final_results['AUC']:.4f}")
    for k in k_values:
        print(f"K={k} --> HR: {final_results[f'HR@{k}']:.4f}, HR-P: {final_results[f'HR-P@{k}']:.4f}, "
              f"NDCG: {final_results[f'NDCG@{k}']:.4f}, Precision: {final_results[f'Precision@{k}']:.4f}, "
              f"Recall: {final_results[f'Recall@{k}']:.4f}, F1: {final_results[f'F1@{k}']:.4f}")

    return final_results


def main(train_data_file: str = "data/train.csv",
         val_data_file: str = "data/val.csv",
         test_data_file: str = "data/test.csv",
         sep_sym: str = "\t",
         num_items: int = 50000,
         num_sessions: int = 100000,
         batch_size: int = 256,
         data_name: str = "Dataset",
         max_seq_len: int = 50,
         epochs: int = 10,
         lr: float = 1e-4,
         mlm_weight: float = 1.0,
         hidden_size: int = 256,
         embedding_dim: int = 128,
         save_dir: str = "checkpoints",
         seed: int = 42):
    """
    Main training function for PARS model.

    Args:
        train_data_file: Path to training data
        val_data_file: Path to validation data
        test_data_file: Path to test data
        sep_sym: Separator symbol in data files
        num_items: Total number of items
        num_sessions: Total number of sessions
        batch_size: Training batch size
        data_name: Name of the dataset
        max_seq_len: Maximum sequence length
        epochs: Number of training epochs
        lr: Learning rate
        mlm_weight: Weight for MLM loss
        hidden_size: Hidden size for transformer
        embedding_dim: Embedding dimension
        save_dir: Directory for saving checkpoints
        seed: Random seed
    """

    # Set random seed
    set_seed(seed)

    logger.info(f"Training parameters: train_data={train_data_file}, test_data={test_data_file}, "
                f"val_data={val_data_file}, sep_sym={sep_sym}, num_items={num_items}, "
                f"num_sessions={num_sessions}, batch_size={batch_size}, max_seq_len={max_seq_len}, "
                f"epochs={epochs}, lr={lr}, mlm_weight={mlm_weight}, hidden_size={hidden_size}, "
                f"embedding_dim={embedding_dim}, save_dir={save_dir}")

    # Prepare datasets
    train_dataset = PARSDataset(train_data_file, mask_ratio=0.3, num_items=num_items,
                                num_sessions=num_sessions, max_seq_len=max_seq_len, sep_sym=sep_sym)
    val_dataset = PARSDataset(val_data_file, mask_ratio=0.3, num_items=num_items,
                              num_sessions=num_sessions, max_seq_len=max_seq_len, sep_sym=sep_sym)
    test_dataset = PARSDataset(test_data_file, mask_ratio=0.3, num_items=num_items,
                               num_sessions=num_sessions, max_seq_len=max_seq_len, sep_sym=sep_sym)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pars_collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=pars_collate_fn,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=pars_collate_fn,
        num_workers=2
    )

    # Create model
    logger.info("Creating model...")
    model = PARSModel(
        num_items=num_items,
        num_sessions=num_sessions,
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        max_seq_length=max_seq_len
    )

    # Train model
    logger.info("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        device=cur_device,
        mlm_weight=mlm_weight,
        data_name=data_name,
        save_dir=save_dir
    )

    logger.info("Training complete!")

    # Final evaluation
    logger.info("Final evaluation on test set...")
    final_metrics = evaluate_model(model, test_loader, cur_device)

    # Save final results
    results_path = os.path.join(save_dir, "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"Final results saved to {results_path}")


if __name__ == "__main__":
    fire.Fire(main)