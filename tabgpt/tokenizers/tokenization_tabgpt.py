"""HuggingFace-compatible tokenizer for TabGPT."""
import json
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from .tabular_tokenizer import TabularTokenizer as BaseTabularTokenizer
from ..data.preprocessing import TabularPreprocessor
from .tabular_tokenizer import ColumnMetadata


class TabGPTTokenizer(PreTrainedTokenizer):
    """
    HuggingFace-compatible tokenizer for TabGPT that handles tabular data.
    
    This tokenizer converts tabular data (DataFrames) into token sequences that can be
    processed by the TabGPT model. It handles numerical, categorical, and missing values
    with appropriate tokenization strategies.
    """
    
    vocab_files_names = {"vocab_file": "vocab.json", "metadata_file": "tokenizer_metadata.json"}
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        metadata_file: Optional[str] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        missing_token: str = "[MISSING]",
        numerical_binning_strategy: str = "quantile",
        num_numerical_bins: int = 100,
        categorical_vocab_size: int = 10000,
        max_columns: int = 100,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        
        self.missing_token = missing_token
        self.numerical_binning_strategy = numerical_binning_strategy
        self.num_numerical_bins = num_numerical_bins
        self.categorical_vocab_size = categorical_vocab_size
        self.max_columns = max_columns
        
        # Initialize base tokenizer
        self.base_tokenizer = BaseTabularTokenizer(
            numerical_binning_strategy=numerical_binning_strategy,
            num_numerical_bins=num_numerical_bins,
            categorical_vocab_size=categorical_vocab_size,
            missing_value_token=missing_token
        )
        
        # Initialize vocabulary
        self._vocab = {}
        self._ids_to_tokens = {}
        
        # Load vocabulary if provided
        if vocab_file is not None and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        else:
            self._init_default_vocab()
        
        # Load metadata if provided
        self.column_metadata = {}
        if metadata_file is not None and os.path.exists(metadata_file):
            self._load_metadata(metadata_file)
        
        # Data preprocessor
        self.preprocessor = None
    
    def _init_default_vocab(self):
        """Initialize default vocabulary with special tokens."""
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
            self.missing_token
        ]
        
        for i, token in enumerate(special_tokens):
            self._vocab[token] = i
            self._ids_to_tokens[i] = token
    
    def _load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
    
    def _save_vocab(self, vocab_file: str):
        """Save vocabulary to file."""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
    
    def _load_metadata(self, metadata_file: str):
        """Load tokenizer metadata from file."""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            # Convert back to ColumnMetadata objects
            self.column_metadata = {}
            for col_name, col_data in metadata.get('column_metadata', {}).items():
                # Simplified - in practice you'd reconstruct ColumnMetadata objects
                self.column_metadata[col_name] = col_data
    
    def _save_metadata(self, metadata_file: str):
        """Save tokenizer metadata to file."""
        metadata = {
            'numerical_binning_strategy': self.numerical_binning_strategy,
            'num_numerical_bins': self.num_numerical_bins,
            'categorical_vocab_size': self.categorical_vocab_size,
            'max_columns': self.max_columns,
            'column_metadata': self.column_metadata
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self._vocab.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text (not used for tabular data, but required by interface).
        """
        # This is a placeholder - TabGPT primarily works with structured data
        return text.split()
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self._ids_to_tokens.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string."""
        return " ".join(tokens)
    
    def fit_on_dataframe(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> "TabGPTTokenizer":
        """
        Fit tokenizer on a DataFrame to learn vocabulary and statistics.
        
        Args:
            df: DataFrame to fit on
            target_column: Optional target column name
            
        Returns:
            Self for method chaining
        """
        # Initialize and fit preprocessor
        self.preprocessor = TabularPreprocessor()
        self.preprocessor.fit(df, target_column)
        
        # Fit base tokenizer
        self.base_tokenizer.fit(df)
        
        # Update vocabulary with learned tokens
        self._update_vocab_from_tokenizer()
        
        # Store column metadata
        if self.preprocessor.data_types:
            for col_name, data_type in self.preprocessor.data_types.items():
                self.column_metadata[col_name] = {
                    'column_type': data_type.value,
                    'cardinality': len(df[col_name].unique()) if col_name in df.columns else 0,
                    'missing_rate': df[col_name].isnull().sum() / len(df) if col_name in df.columns else 0,
                    'dtype': str(df[col_name].dtype) if col_name in df.columns else 'unknown'
                }
        
        return self
    
    def _update_vocab_from_tokenizer(self):
        """Update vocabulary with tokens from base tokenizer."""
        # Get vocabulary from base tokenizer
        base_vocab = self.base_tokenizer.get_vocab()
        
        # Add new tokens to vocabulary
        current_id = len(self._vocab)
        for token in base_vocab:
            if token not in self._vocab:
                self._vocab[token] = current_id
                self._ids_to_tokens[current_id] = token
                current_id += 1
    
    def tokenize_dataframe(
        self,
        df: pd.DataFrame,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize a DataFrame into model inputs.
        
        Args:
            df: DataFrame to tokenize
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Type of tensors to return ('pt' for PyTorch)
            return_attention_mask: Whether to return attention mask
            return_token_type_ids: Whether to return token type IDs
            
        Returns:
            BatchEncoding with tokenized inputs
        """
        if self.base_tokenizer is None or not self.base_tokenizer.is_fitted:
            raise ValueError("Tokenizer must be fitted on data before tokenizing")
        
        # Tokenize using base tokenizer
        tokenized_data = self.base_tokenizer.tokenize_dataframe(df)
        
        # Convert to IDs
        input_ids = []
        attention_masks = []
        
        for row_tokens in tokenized_data:
            # Convert tokens to IDs
            token_ids = [self._convert_token_to_id(token) for token in row_tokens]
            
            # Handle max_length
            if max_length is not None:
                if truncation and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                elif padding and len(token_ids) < max_length:
                    pad_length = max_length - len(token_ids)
                    token_ids.extend([self.pad_token_id] * pad_length)
            
            input_ids.append(token_ids)
            
            # Create attention mask
            if return_attention_mask:
                attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]
                attention_masks.append(attention_mask)
        
        # Create BatchEncoding
        encoding_dict = {"input_ids": input_ids}
        
        if return_attention_mask:
            encoding_dict["attention_mask"] = attention_masks
        
        if return_token_type_ids:
            # Simple token type IDs (all zeros for now)
            encoding_dict["token_type_ids"] = [[0] * len(ids) for ids in input_ids]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            for key, value in encoding_dict.items():
                encoding_dict[key] = torch.tensor(value, dtype=torch.long)
        elif return_tensors == "np":
            import numpy as np
            for key, value in encoding_dict.items():
                encoding_dict[key] = np.array(value)
        
        return BatchEncoding(encoding_dict)
    
    def __call__(
        self,
        text: Union[str, List[str], pd.DataFrame] = None,
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        Main tokenization method that handles both text and DataFrame inputs.
        """
        if isinstance(text, pd.DataFrame):
            return self.tokenize_dataframe(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                **kwargs
            )
        else:
            # Fall back to standard text tokenization
            return super().__call__(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                return_tensors=return_tensors,
                **kwargs
            )
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        """
        Save vocabulary and metadata to directory.
        
        Args:
            save_directory: Directory to save files
            filename_prefix: Optional prefix for filenames
            
        Returns:
            Tuple of (vocab_file_path, metadata_file_path)
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        prefix = filename_prefix + "-" if filename_prefix else ""
        
        vocab_file = os.path.join(save_directory, f"{prefix}vocab.json")
        metadata_file = os.path.join(save_directory, f"{prefix}tokenizer_metadata.json")
        
        self._save_vocab(vocab_file)
        self._save_metadata(metadata_file)
        
        return vocab_file, metadata_file
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *inputs,
        **kwargs
    ) -> "TabGPTTokenizer":
        """
        Load tokenizer from pretrained model or path.
        """
        # This would typically load from HuggingFace Hub or local directory
        # For now, we'll create a basic implementation
        
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
            metadata_file = os.path.join(pretrained_model_name_or_path, "tokenizer_metadata.json")
            
            return cls(
                vocab_file=vocab_file if os.path.exists(vocab_file) else None,
                metadata_file=metadata_file if os.path.exists(metadata_file) else None,
                **kwargs
            )
        else:
            # Would download from HuggingFace Hub
            raise NotImplementedError("Loading from HuggingFace Hub not implemented yet")
    
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Get special tokens mask.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True
            )
        
        # Create mask for special tokens
        special_token_ids = {
            self.pad_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
            self.unk_token_id
        }
        
        mask = [1 if token_id in special_token_ids else 0 for token_id in token_ids_0]
        
        if token_ids_1 is not None:
            mask.extend([1 if token_id in special_token_ids else 0 for token_id in token_ids_1])
        
        return mask
    
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs from sequences.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs with special tokens.
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        
        return cls + token_ids_0 + sep + token_ids_1 + sep