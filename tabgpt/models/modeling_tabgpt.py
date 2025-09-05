"""TabGPT model implementation following HuggingFace patterns."""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

from .configuration_tabgpt import TabGPTConfig
from .base import TabGPTModel as BaseTabGPTModel
from ..heads.base import TaskOutput, TaskType, MLPHead
from ..tokenizers.tabular_tokenizer import TabularTokenizer


class TabGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    config_class = TabGPTConfig
    base_model_prefix = "tabgpt"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = ["position_ids"]
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BaseTabGPTModel):
            module.gradient_checkpointing = value


class TabGPTModel(TabGPTPreTrainedModel):
    """
    The bare TabGPT Model transformer outputting raw hidden-states without any specific head on top.
    
    This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    
    This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the
    PyTorch documentation for all matter related to general usage and behavior.
    """
    
    def __init__(self, config: TabGPTConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize the base TabGPT model
        self.tabgpt = BaseTabGPTModel(config)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.tabgpt.embeddings
    
    def set_input_embeddings(self, value):
        """Set input embeddings."""
        self.tabgpt.embeddings = value
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # TabGPT-specific inputs
        numerical_features: Optional[torch.Tensor] = None,
        categorical_features: Optional[torch.Tensor] = None,
        column_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        """
        Forward pass through TabGPT model.
        
        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            attention_mask: Mask to avoid performing attention on padding token indices.
            token_type_ids: Segment token indices to indicate first and second portions of the inputs.
            position_ids: Indices of positions of each input sequence token in the position embeddings.
            head_mask: Mask to nullify selected heads of the self-attention modules.
            inputs_embeds: Optionally, instead of passing input_ids you can choose to directly pass an embedded representation.
            output_attentions: Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states: Whether or not to return the hidden states of all layers.
            return_dict: Whether or not to return a ModelOutput instead of a plain tuple.
            numerical_features: Numerical features tensor.
            categorical_features: Categorical features tensor.
            column_embeddings: Column embeddings tensor.
            
        Returns:
            BaseModelOutput or tuple containing the model outputs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Prepare inputs for base model
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "column_embeddings": column_embeddings,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict
        }
        
        # Remove None values
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        
        # Forward through base model
        outputs = self.tabgpt(**model_inputs)
        
        if not return_dict:
            return outputs
        
        return BaseModelOutput(
            last_hidden_state=outputs.get("last_hidden_state"),
            hidden_states=outputs.get("hidden_states"),
            attentions=outputs.get("attentions")
        )


class TabGPTForSequenceClassification(TabGPTPreTrainedModel):
    """
    TabGPT Model with a sequence classification/regression head on top (a linear layer on top of the pooled output).
    """
    
    def __init__(self, config: TabGPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, 'num_labels') else 2
        self.config = config
        
        self.tabgpt = TabGPTModel(config)
        
        # Classification head
        if self.num_labels == 1:
            # Regression
            self.classifier = MLPHead(
                input_dim=config.hidden_size,
                output_dim=1,
                task_type=TaskType.REGRESSION,
                hidden_dims=[config.hidden_size // 2],
                dropout=config.hidden_dropout_prob
            )
        elif self.num_labels == 2:
            # Binary classification
            self.classifier = MLPHead(
                input_dim=config.hidden_size,
                output_dim=1,
                task_type=TaskType.BINARY_CLASSIFICATION,
                hidden_dims=[config.hidden_size // 2],
                dropout=config.hidden_dropout_prob
            )
        else:
            # Multi-class classification
            self.classifier = MLPHead(
                input_dim=config.hidden_size,
                output_dim=self.num_labels,
                task_type=TaskType.MULTICLASS_CLASSIFICATION,
                hidden_dims=[config.hidden_size // 2],
                dropout=config.hidden_dropout_prob
            )
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # TabGPT-specific inputs
        numerical_features: Optional[torch.Tensor] = None,
        categorical_features: Optional[torch.Tensor] = None,
        column_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass for sequence classification.
        
        Args:
            labels: Labels for computing the sequence classification/regression loss.
                   Indices should be in [0, ..., config.num_labels - 1].
                   If config.num_labels == 1 a regression loss is computed (Mean-Square loss),
                   If config.num_labels > 1 a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.tabgpt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            column_embeddings=column_embeddings
        )
        
        # Get pooled output (mean pooling over sequence dimension)
        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]
        pooled_output = sequence_output.mean(dim=1)  # [batch_size, hidden_size]
        
        # Forward through classifier
        classifier_output = self.classifier(pooled_output, labels)
        
        loss = classifier_output.loss
        logits = classifier_output.logits if classifier_output.logits is not None else classifier_output.predictions
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TabGPTForRegression(TabGPTPreTrainedModel):
    """
    TabGPT Model with a regression head on top.
    """
    
    def __init__(self, config: TabGPTConfig):
        super().__init__(config)
        self.config = config
        self.output_dim = getattr(config, 'output_dim', 1)
        
        self.tabgpt = TabGPTModel(config)
        
        # Regression head
        self.regressor = MLPHead(
            input_dim=config.hidden_size,
            output_dim=self.output_dim,
            task_type=TaskType.REGRESSION,
            hidden_dims=[config.hidden_size // 2],
            dropout=config.hidden_dropout_prob
        )
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # TabGPT-specific inputs
        numerical_features: Optional[torch.Tensor] = None,
        categorical_features: Optional[torch.Tensor] = None,
        column_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass for regression.
        
        Args:
            labels: Labels for computing the regression loss (MSE).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.tabgpt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            column_embeddings=column_embeddings
        )
        
        # Get pooled output (mean pooling over sequence dimension)
        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]
        pooled_output = sequence_output.mean(dim=1)  # [batch_size, hidden_size]
        
        # Forward through regressor
        regressor_output = self.regressor(pooled_output, labels)
        
        loss = regressor_output.loss
        predictions = regressor_output.predictions
        
        if not return_dict:
            output = (predictions,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=predictions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TabGPTForPreTraining(TabGPTPreTrainedModel):
    """
    TabGPT Model with multiple pre-training heads on top for self-supervised learning.
    """
    
    def __init__(self, config: TabGPTConfig):
        super().__init__(config)
        self.config = config
        
        self.tabgpt = TabGPTModel(config)
        
        # Pre-training heads
        if config.use_masked_cell_modeling:
            from ..training.masked_cell_modeling import MaskedCellModelingObjective
            self.mcm_head = MaskedCellModelingObjective(
                input_dim=config.hidden_size,
                vocab_size=config.vocab_size
            )
        
        if config.use_masked_column_modeling:
            from ..training.masked_column_modeling import MaskedColumnModelingObjective
            self.mcol_head = MaskedColumnModelingObjective(
                input_dim=config.hidden_size,
                num_columns=config.max_columns
            )
        
        if config.use_contrastive_row_learning:
            from ..training.contrastive_row_learning import ContrastiveRowLearningObjective, CRLConfig
            crl_config = CRLConfig()
            self.crl_head = ContrastiveRowLearningObjective(crl_config)
        
        if config.use_next_row_prediction:
            from ..training.next_row_prediction import NextRowPredictionObjective, NRPConfig
            nrp_config = NRPConfig()
            # This would need column metadata - simplified for now
            self.nrp_head = None  # Will be initialized when needed
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # TabGPT-specific inputs
        numerical_features: Optional[torch.Tensor] = None,
        categorical_features: Optional[torch.Tensor] = None,
        column_embeddings: Optional[torch.Tensor] = None,
        # Pre-training targets
        mcm_targets: Optional[torch.Tensor] = None,
        mcol_targets: Optional[torch.Tensor] = None,
        crl_targets: Optional[torch.Tensor] = None,
        nrp_targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for pre-training with multiple objectives.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.tabgpt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            column_embeddings=column_embeddings
        )
        
        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]
        
        # Compute losses for enabled objectives
        losses = {}
        total_loss = 0.0
        
        if self.config.use_masked_cell_modeling and hasattr(self, 'mcm_head') and mcm_targets is not None:
            mcm_loss = self.mcm_head(sequence_output, mcm_targets)
            losses['mcm_loss'] = mcm_loss
            total_loss += self.config.mcm_loss_weight * mcm_loss
        
        if self.config.use_masked_column_modeling and hasattr(self, 'mcol_head') and mcol_targets is not None:
            mcol_loss = self.mcol_head(sequence_output, mcol_targets)
            losses['mcol_loss'] = mcol_loss
            total_loss += self.config.mcol_loss_weight * mcol_loss
        
        if self.config.use_contrastive_row_learning and hasattr(self, 'crl_head') and crl_targets is not None:
            # CRL typically doesn't need explicit targets
            anchor, positive = self.crl_head.create_augmented_pairs(sequence_output)
            anchor_emb = anchor.mean(dim=1)
            positive_emb = positive.mean(dim=1)
            crl_loss = self.crl_head.compute_contrastive_loss(anchor_emb, positive_emb)
            losses['crl_loss'] = crl_loss
            total_loss += self.config.crl_loss_weight * crl_loss
        
        losses['total_loss'] = total_loss
        
        if not return_dict:
            output = (total_loss, sequence_output) + outputs[1:]
            return output
        
        return {
            'loss': total_loss,
            'losses': losses,
            'last_hidden_state': sequence_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }