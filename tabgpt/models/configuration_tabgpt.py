"""TabGPT configuration following HuggingFace patterns."""
from transformers import PretrainedConfig
from typing import Dict, List, Optional, Union, Any
import json


class TabGPTConfig(PretrainedConfig):
    """
    Configuration class for TabGPT model.
    
    This is the configuration class to store the configuration of a TabGPT model.
    It is used to instantiate a TabGPT model according to the specified arguments,
    defining the model architecture.
    
    Args:
        vocab_size (int, optional): Vocabulary size of the TabGPT model. Defaults to 50000.
        hidden_size (int, optional): Dimensionality of the encoder layers and the pooler layer. Defaults to 768.
        num_hidden_layers (int, optional): Number of hidden layers in the Transformer encoder. Defaults to 12.
        num_attention_heads (int, optional): Number of attention heads for each attention layer. Defaults to 12.
        intermediate_size (int, optional): Dimensionality of the "intermediate" (feed-forward) layer. Defaults to 3072.
        hidden_act (str, optional): The non-linear activation function. Defaults to "gelu".
        hidden_dropout_prob (float, optional): The dropout probability for all fully connected layers. Defaults to 0.1.
        attention_probs_dropout_prob (float, optional): The dropout ratio for the attention probabilities. Defaults to 0.1.
        max_position_embeddings (int, optional): The maximum sequence length. Defaults to 512.
        type_vocab_size (int, optional): The vocabulary size of the token_type_ids. Defaults to 2.
        initializer_range (float, optional): The standard deviation of the truncated_normal_initializer. Defaults to 0.02.
        layer_norm_eps (float, optional): The epsilon used by the layer normalization layers. Defaults to 1e-12.
        use_cache (bool, optional): Whether or not the model should return the last key/values attentions. Defaults to True.
        
        # TabGPT-specific parameters
        max_columns (int, optional): Maximum number of columns in tabular data. Defaults to 100.
        max_rows (int, optional): Maximum number of rows in a sequence. Defaults to 512.
        column_embedding_dim (int, optional): Dimensionality of column embeddings. Defaults to 256.
        numerical_embedding_dim (int, optional): Dimensionality of numerical feature embeddings. Defaults to 128.
        categorical_embedding_dim (int, optional): Dimensionality of categorical feature embeddings. Defaults to 128.
        use_column_attention (bool, optional): Whether to use cross-attention between rows and columns. Defaults to True.
        use_positional_encoding (bool, optional): Whether to use positional encoding for sequences. Defaults to True.
        
        # Tokenization parameters
        numerical_binning_strategy (str, optional): Strategy for binning numerical values. Defaults to "quantile".
        num_numerical_bins (int, optional): Number of bins for numerical features. Defaults to 100.
        categorical_vocab_size (int, optional): Maximum vocabulary size for categorical features. Defaults to 10000.
        handle_missing_values (bool, optional): Whether to handle missing values with special tokens. Defaults to True.
        missing_value_token (str, optional): Special token for missing values. Defaults to "[MISSING]".
        
        # Pre-training objectives
        use_masked_cell_modeling (bool, optional): Whether to use masked cell modeling objective. Defaults to True.
        use_masked_column_modeling (bool, optional): Whether to use masked column modeling objective. Defaults to True.
        use_contrastive_row_learning (bool, optional): Whether to use contrastive row learning objective. Defaults to True.
        use_next_row_prediction (bool, optional): Whether to use next row prediction objective. Defaults to True.
        
        # Loss weights
        mcm_loss_weight (float, optional): Weight for masked cell modeling loss. Defaults to 1.0.
        mcol_loss_weight (float, optional): Weight for masked column modeling loss. Defaults to 0.5.
        crl_loss_weight (float, optional): Weight for contrastive row learning loss. Defaults to 0.3.
        nrp_loss_weight (float, optional): Weight for next row prediction loss. Defaults to 0.2.
    """
    
    model_type = "tabgpt"
    
    def __init__(
        self,
        # Standard transformer parameters
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        
        # TabGPT-specific parameters
        max_columns: int = 100,
        max_rows: int = 512,
        column_embedding_dim: int = 256,
        numerical_embedding_dim: int = 128,
        categorical_embedding_dim: int = 128,
        use_column_attention: bool = True,
        use_positional_encoding: bool = True,
        
        # Tokenization parameters
        numerical_binning_strategy: str = "quantile",
        num_numerical_bins: int = 100,
        categorical_vocab_size: int = 10000,
        handle_missing_values: bool = True,
        missing_value_token: str = "[MISSING]",
        
        # Pre-training objectives
        use_masked_cell_modeling: bool = True,
        use_masked_column_modeling: bool = True,
        use_contrastive_row_learning: bool = True,
        use_next_row_prediction: bool = True,
        
        # Loss weights
        mcm_loss_weight: float = 1.0,
        mcol_loss_weight: float = 0.5,
        crl_loss_weight: float = 0.3,
        nrp_loss_weight: float = 0.2,
        
        # Additional parameters
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Standard transformer parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        
        # TabGPT-specific parameters
        self.max_columns = max_columns
        self.max_rows = max_rows
        self.column_embedding_dim = column_embedding_dim
        self.numerical_embedding_dim = numerical_embedding_dim
        self.categorical_embedding_dim = categorical_embedding_dim
        self.use_column_attention = use_column_attention
        self.use_positional_encoding = use_positional_encoding
        
        # Tokenization parameters
        self.numerical_binning_strategy = numerical_binning_strategy
        self.num_numerical_bins = num_numerical_bins
        self.categorical_vocab_size = categorical_vocab_size
        self.handle_missing_values = handle_missing_values
        self.missing_value_token = missing_value_token
        
        # Pre-training objectives
        self.use_masked_cell_modeling = use_masked_cell_modeling
        self.use_masked_column_modeling = use_masked_column_modeling
        self.use_contrastive_row_learning = use_contrastive_row_learning
        self.use_next_row_prediction = use_next_row_prediction
        
        # Loss weights
        self.mcm_loss_weight = mcm_loss_weight
        self.mcol_loss_weight = mcol_loss_weight
        self.crl_loss_weight = crl_loss_weight
        self.nrp_loss_weight = nrp_loss_weight
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        if self.max_columns <= 0:
            raise ValueError(f"max_columns must be positive, got {self.max_columns}")
        
        if self.max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {self.max_rows}")
        
        if self.num_numerical_bins <= 0:
            raise ValueError(f"num_numerical_bins must be positive, got {self.num_numerical_bins}")
        
        if self.categorical_vocab_size <= 0:
            raise ValueError(f"categorical_vocab_size must be positive, got {self.categorical_vocab_size}")
        
        valid_binning_strategies = ["uniform", "quantile", "kmeans"]
        if self.numerical_binning_strategy not in valid_binning_strategies:
            raise ValueError(
                f"numerical_binning_strategy must be one of {valid_binning_strategies}, "
                f"got {self.numerical_binning_strategy}"
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "TabGPTConfig":
        """
        Instantiates a TabGPTConfig from a Python dictionary of parameters.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary that will be used to instantiate the configuration object.
            
        Returns:
            TabGPTConfig: The configuration object instantiated from those parameters.
        """
        return cls(**config_dict, **kwargs)
    
    @classmethod
    def from_json_file(cls, json_file: str) -> "TabGPTConfig":
        """
        Instantiates a TabGPTConfig from the path to a JSON file of parameters.
        
        Args:
            json_file (str): Path to the JSON file containing the parameters.
            
        Returns:
            TabGPTConfig: The configuration object instantiated from the JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        output = super().to_dict()
        return output
    
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.
        
        Args:
            use_diff (bool): If set to True, only the difference between the config instance
                           and the default PretrainedConfig() is serialized to JSON string.
                           
        Returns:
            str: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    
    def get_head_dim(self) -> int:
        """Get the dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    def get_total_embedding_dim(self) -> int:
        """Get the total embedding dimension for tabular features."""
        return self.numerical_embedding_dim + self.categorical_embedding_dim
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get dictionary of loss weights for multi-objective training."""
        return {
            "mcm": self.mcm_loss_weight,
            "mcol": self.mcol_loss_weight,
            "crl": self.crl_loss_weight,
            "nrp": self.nrp_loss_weight
        }
    
    def update_vocab_size(self, new_vocab_size: int):
        """Update vocabulary size and validate."""
        self.vocab_size = new_vocab_size
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
    
    def update_max_columns(self, new_max_columns: int):
        """Update maximum number of columns and validate."""
        self.max_columns = new_max_columns
        if self.max_columns <= 0:
            raise ValueError(f"max_columns must be positive, got {self.max_columns}")
    
    def enable_objective(self, objective: str):
        """Enable a specific pre-training objective."""
        objective_map = {
            "mcm": "use_masked_cell_modeling",
            "mcol": "use_masked_column_modeling", 
            "crl": "use_contrastive_row_learning",
            "nrp": "use_next_row_prediction"
        }
        
        if objective not in objective_map:
            raise ValueError(f"Unknown objective: {objective}. Valid objectives: {list(objective_map.keys())}")
        
        setattr(self, objective_map[objective], True)
    
    def disable_objective(self, objective: str):
        """Disable a specific pre-training objective."""
        objective_map = {
            "mcm": "use_masked_cell_modeling",
            "mcol": "use_masked_column_modeling",
            "crl": "use_contrastive_row_learning", 
            "nrp": "use_next_row_prediction"
        }
        
        if objective not in objective_map:
            raise ValueError(f"Unknown objective: {objective}. Valid objectives: {list(objective_map.keys())}")
        
        setattr(self, objective_map[objective], False)
    
    def set_loss_weight(self, objective: str, weight: float):
        """Set loss weight for a specific objective."""
        weight_map = {
            "mcm": "mcm_loss_weight",
            "mcol": "mcol_loss_weight",
            "crl": "crl_loss_weight",
            "nrp": "nrp_loss_weight"
        }
        
        if objective not in weight_map:
            raise ValueError(f"Unknown objective: {objective}. Valid objectives: {list(weight_map.keys())}")
        
        if weight < 0:
            raise ValueError(f"Loss weight must be non-negative, got {weight}")
        
        setattr(self, weight_map[objective], weight)
    
    def get_model_size_info(self) -> Dict[str, Any]:
        """Get information about model size and parameters."""
        # Rough parameter count estimation
        embedding_params = self.vocab_size * self.hidden_size
        transformer_params = (
            self.num_hidden_layers * (
                # Self-attention
                4 * self.hidden_size * self.hidden_size +
                # Feed-forward
                2 * self.hidden_size * self.intermediate_size +
                # Layer norms
                2 * self.hidden_size
            )
        )
        
        total_params = embedding_params + transformer_params
        
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.get_head_dim(),
            "intermediate_size": self.intermediate_size,
            "estimated_parameters": total_params,
            "estimated_parameters_millions": round(total_params / 1e6, 2)
        }