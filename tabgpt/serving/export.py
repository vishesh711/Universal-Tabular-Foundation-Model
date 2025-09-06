"""Model export utilities for production deployment."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings
import json

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""
    
    # Export format
    export_format: str = "onnx"  # onnx, torchscript, tensorrt
    
    # ONNX settings
    onnx_opset_version: int = 11
    onnx_dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    onnx_optimize: bool = True
    
    # TorchScript settings
    torchscript_method: str = "trace"  # trace, script
    torchscript_strict: bool = True
    
    # Optimization settings
    optimize_for_inference: bool = True
    enable_quantization: bool = False
    
    # Input/Output settings
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    
    # Validation settings
    validate_export: bool = True
    tolerance: float = 1e-5
    
    def __post_init__(self):
        """Validate configuration."""
        valid_formats = ["onnx", "torchscript", "tensorrt"]
        if self.export_format not in valid_formats:
            raise ValueError(f"export_format must be one of {valid_formats}")
        
        if self.onnx_dynamic_axes is None:
            self.onnx_dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }


class ModelExporter:
    """Base class for model exporters."""
    
    def __init__(self, config: ExportConfig = None):
        """
        Initialize model exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
    
    def export_model(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export model to specified format.
        
        Args:
            model: Model to export
            sample_input: Sample input for tracing
            output_path: Output file path
            metadata: Additional metadata to save
            
        Returns:
            Export results and statistics
        """
        logger.info(f"Exporting model to {self.config.export_format} format...")
        
        # Prepare model for export
        model = self._prepare_model_for_export(model)
        
        # Export based on format
        if self.config.export_format == "onnx":
            return self._export_onnx(model, sample_input, output_path, metadata)
        elif self.config.export_format == "torchscript":
            return self._export_torchscript(model, sample_input, output_path, metadata)
        elif self.config.export_format == "tensorrt":
            return self._export_tensorrt(model, sample_input, output_path, metadata)
        else:
            raise ValueError(f"Unsupported export format: {self.config.export_format}")
    
    def _prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Prepare model for export."""
        # Set to evaluation mode
        model.eval()
        
        # Apply optimizations if requested
        if self.config.optimize_for_inference:
            model = self._optimize_for_inference(model)
        
        return model
    
    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply inference optimizations."""
        # Fuse operations where possible
        try:
            # This is a simplified example - real fusion would be more complex
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    # Look for fusable patterns
                    pass
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
        
        return model
    
    def _export_onnx(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Export model to ONNX format."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX export requires 'onnx' and 'onnxruntime' packages")
        
        logger.info("Exporting to ONNX format...")
        
        # Prepare input
        input_tuple = tuple(sample_input.values())
        input_names = self.config.input_names or list(sample_input.keys())
        output_names = self.config.output_names or ['output']
        
        # Export to ONNX
        torch.onnx.export(
            model,
            input_tuple,
            output_path,
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=self.config.onnx_optimize,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=self.config.onnx_dynamic_axes,
            verbose=False
        )
        
        # Validate export
        validation_results = {}
        if self.config.validate_export:
            validation_results = self._validate_onnx_export(
                output_path, model, sample_input
            )
        
        # Get model info
        onnx_model = onnx.load(output_path)
        model_size = Path(output_path).stat().st_size
        
        # Save metadata
        if metadata:
            metadata_path = output_path.replace('.onnx', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        results = {
            'export_format': 'onnx',
            'output_path': output_path,
            'model_size_bytes': model_size,
            'model_size_mb': model_size / (1024 * 1024),
            'opset_version': self.config.onnx_opset_version,
            'input_names': input_names,
            'output_names': output_names,
            'validation_results': validation_results
        }
        
        logger.info(f"ONNX export completed: {model_size / (1024 * 1024):.2f} MB")
        return results
    
    def _validate_onnx_export(
        self,
        onnx_path: str,
        original_model: nn.Module,
        sample_input: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Validate ONNX export by comparing outputs."""
        try:
            import onnxruntime as ort
            
            # Get original model output
            original_model.eval()
            with torch.no_grad():
                original_output = original_model(**sample_input)
            
            if hasattr(original_output, 'logits'):
                original_tensor = original_output.logits
            elif hasattr(original_output, 'predictions'):
                original_tensor = original_output.predictions
            else:
                original_tensor = original_output
            
            original_numpy = original_tensor.cpu().numpy()
            
            # Get ONNX model output
            ort_session = ort.InferenceSession(onnx_path)
            
            # Prepare ONNX input
            onnx_input = {}
            for name, tensor in sample_input.items():
                onnx_input[name] = tensor.cpu().numpy()
            
            onnx_outputs = ort_session.run(None, onnx_input)
            onnx_numpy = onnx_outputs[0]
            
            # Compare outputs
            max_diff = np.max(np.abs(original_numpy - onnx_numpy))
            mean_diff = np.mean(np.abs(original_numpy - onnx_numpy))
            
            validation_passed = max_diff < self.config.tolerance
            
            return {
                'validation_passed': validation_passed,
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'tolerance': self.config.tolerance
            }
            
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return {
                'validation_passed': False,
                'error': str(e)
            }
    
    def _export_torchscript(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Export model to TorchScript format."""
        logger.info("Exporting to TorchScript format...")
        
        try:
            if self.config.torchscript_method == "trace":
                # Trace the model
                traced_model = torch.jit.trace(
                    model,
                    tuple(sample_input.values()),
                    strict=self.config.torchscript_strict
                )
            elif self.config.torchscript_method == "script":
                # Script the model
                traced_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown TorchScript method: {self.config.torchscript_method}")
            
            # Optimize for inference
            if self.config.optimize_for_inference:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save model
            traced_model.save(output_path)
            
            # Validate export
            validation_results = {}
            if self.config.validate_export:
                validation_results = self._validate_torchscript_export(
                    traced_model, model, sample_input
                )
            
            # Get model info
            model_size = Path(output_path).stat().st_size
            
            # Save metadata
            if metadata:
                metadata_path = output_path.replace('.pt', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            results = {
                'export_format': 'torchscript',
                'output_path': output_path,
                'model_size_bytes': model_size,
                'model_size_mb': model_size / (1024 * 1024),
                'method': self.config.torchscript_method,
                'validation_results': validation_results
            }
            
            logger.info(f"TorchScript export completed: {model_size / (1024 * 1024):.2f} MB")
            return results
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
    
    def _validate_torchscript_export(
        self,
        traced_model: torch.jit.ScriptModule,
        original_model: nn.Module,
        sample_input: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Validate TorchScript export by comparing outputs."""
        try:
            # Get original model output
            original_model.eval()
            with torch.no_grad():
                original_output = original_model(**sample_input)
            
            if hasattr(original_output, 'logits'):
                original_tensor = original_output.logits
            elif hasattr(original_output, 'predictions'):
                original_tensor = original_output.predictions
            else:
                original_tensor = original_output
            
            # Get TorchScript model output
            with torch.no_grad():
                traced_output = traced_model(*sample_input.values())
            
            if isinstance(traced_output, (list, tuple)):
                traced_tensor = traced_output[0]
            else:
                traced_tensor = traced_output
            
            # Compare outputs
            max_diff = torch.max(torch.abs(original_tensor - traced_tensor)).item()
            mean_diff = torch.mean(torch.abs(original_tensor - traced_tensor)).item()
            
            validation_passed = max_diff < self.config.tolerance
            
            return {
                'validation_passed': validation_passed,
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'tolerance': self.config.tolerance
            }
            
        except Exception as e:
            logger.error(f"TorchScript validation failed: {e}")
            return {
                'validation_passed': False,
                'error': str(e)
            }
    
    def _export_tensorrt(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Export model to TensorRT format."""
        logger.warning("TensorRT export is not fully implemented in this example")
        
        try:
            import tensorrt as trt
            import torch_tensorrt
        except ImportError:
            raise ImportError("TensorRT export requires 'tensorrt' and 'torch-tensorrt' packages")
        
        # This is a placeholder for TensorRT export
        # Real implementation would involve:
        # 1. Converting to TorchScript first
        # 2. Using torch_tensorrt.compile() or TensorRT Python API
        # 3. Optimizing for specific hardware
        
        logger.info("TensorRT export would be implemented here")
        
        return {
            'export_format': 'tensorrt',
            'output_path': output_path,
            'status': 'not_implemented'
        }


class ONNXExporter(ModelExporter):
    """Specialized ONNX exporter with additional optimizations."""
    
    def __init__(self, config: ExportConfig = None):
        if config is None:
            config = ExportConfig(export_format="onnx")
        elif config.export_format != "onnx":
            config.export_format = "onnx"
        
        super().__init__(config)
    
    def export_with_optimization(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: str,
        optimization_level: str = "basic"
    ) -> Dict[str, Any]:
        """
        Export ONNX model with additional optimizations.
        
        Args:
            model: Model to export
            sample_input: Sample input for tracing
            output_path: Output file path
            optimization_level: Level of optimization (basic, extended, all)
            
        Returns:
            Export results
        """
        # First, do standard export
        results = self.export_model(model, sample_input, output_path)
        
        # Apply ONNX-specific optimizations
        if optimization_level in ["extended", "all"]:
            try:
                optimized_path = output_path.replace('.onnx', '_optimized.onnx')
                self._optimize_onnx_model(output_path, optimized_path)
                results['optimized_path'] = optimized_path
            except Exception as e:
                logger.warning(f"ONNX optimization failed: {e}")
        
        return results
    
    def _optimize_onnx_model(self, input_path: str, output_path: str):
        """Apply ONNX-specific optimizations."""
        try:
            import onnx
            from onnxoptimizer import optimize
            
            # Load model
            model = onnx.load(input_path)
            
            # Apply optimizations
            optimized_model = optimize(model)
            
            # Save optimized model
            onnx.save(optimized_model, output_path)
            
            logger.info(f"ONNX model optimized and saved to {output_path}")
            
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping optimization")
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")


class TorchScriptExporter(ModelExporter):
    """Specialized TorchScript exporter with additional optimizations."""
    
    def __init__(self, config: ExportConfig = None):
        if config is None:
            config = ExportConfig(export_format="torchscript")
        elif config.export_format != "torchscript":
            config.export_format = "torchscript"
        
        super().__init__(config)
    
    def export_with_mobile_optimization(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: str
    ) -> Dict[str, Any]:
        """
        Export TorchScript model optimized for mobile deployment.
        
        Args:
            model: Model to export
            sample_input: Sample input for tracing
            output_path: Output file path
            
        Returns:
            Export results
        """
        # Export standard TorchScript
        results = self.export_model(model, sample_input, output_path)
        
        try:
            # Load the exported model
            traced_model = torch.jit.load(output_path)
            
            # Optimize for mobile
            mobile_optimized = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            
            # Save mobile-optimized version
            mobile_path = output_path.replace('.pt', '_mobile.ptl')
            mobile_optimized._save_for_lite_interpreter(mobile_path)
            
            results['mobile_optimized_path'] = mobile_path
            results['mobile_size_mb'] = Path(mobile_path).stat().st_size / (1024 * 1024)
            
            logger.info(f"Mobile-optimized model saved to {mobile_path}")
            
        except Exception as e:
            logger.warning(f"Mobile optimization failed: {e}")
        
        return results


def export_model_for_serving(
    model: nn.Module,
    tokenizer: Any,
    sample_data: pd.DataFrame,
    output_dir: str,
    formats: List[str] = None,
    optimization_level: str = "basic"
) -> Dict[str, Any]:
    """
    Export model in multiple formats for serving.
    
    Args:
        model: Model to export
        tokenizer: Tokenizer for preprocessing
        sample_data: Sample data for creating inputs
        output_dir: Output directory
        formats: Export formats to use
        optimization_level: Level of optimization
        
    Returns:
        Export results for all formats
    """
    if formats is None:
        formats = ["onnx", "torchscript"]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare sample input
    sample_input = tokenizer.encode_batch(sample_data.iloc[:1])
    
    # Move to CPU for export
    device = next(model.parameters()).device
    model = model.cpu()
    for key, value in sample_input.items():
        if isinstance(value, torch.Tensor):
            sample_input[key] = value.cpu()
    
    results = {}
    
    for format_name in formats:
        try:
            logger.info(f"Exporting model in {format_name} format...")
            
            # Create exporter
            config = ExportConfig(export_format=format_name)
            
            if format_name == "onnx":
                exporter = ONNXExporter(config)
                output_path = Path(output_dir) / "model.onnx"
                result = exporter.export_with_optimization(
                    model, sample_input, str(output_path), optimization_level
                )
            elif format_name == "torchscript":
                exporter = TorchScriptExporter(config)
                output_path = Path(output_dir) / "model.pt"
                result = exporter.export_with_mobile_optimization(
                    model, sample_input, str(output_path)
                )
            else:
                exporter = ModelExporter(config)
                output_path = Path(output_dir) / f"model.{format_name}"
                result = exporter.export_model(
                    model, sample_input, str(output_path)
                )
            
            results[format_name] = result
            
        except Exception as e:
            logger.error(f"Export failed for {format_name}: {e}")
            results[format_name] = {'error': str(e)}
    
    # Move model back to original device
    model = model.to(device)
    
    # Save export summary
    summary_path = Path(output_dir) / "export_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model export completed. Results saved to {output_dir}")
    
    return results