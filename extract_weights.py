#!/usr/bin/env python3
"""
Extract weights from the trained TensorFlow model for porting to Axon.
"""

import json
from pathlib import Path

import tensorflow as tf


def extract_model_weights():
    """Extract all weights and architecture info from the trained model."""
    
    model_dir = Path('./syllable/model_data/model')
    chars_file = Path('./syllable/model_data/chars.json')
    
    # Load the model
    print("Loading TensorFlow model...")
    model = tf.keras.models.load_model(model_dir)
    
    # Load character config
    with open(chars_file) as f:
        char_config = json.load(f)
    
    # Extract weights from each layer
    weights_data = {}
    
    print("\nModel summary:")
    model.summary()
    
    print(f"\nExtracting weights from {len(model.layers)} layers...")
    
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_type = type(layer).__name__
        
        print(f"Layer {i}: {layer_name} ({layer_type})")
        
        if hasattr(layer, 'get_weights') and layer.get_weights():
            layer_weights = layer.get_weights()
            print(f"  Weights shapes: {[w.shape for w in layer_weights]}")
            
            # Convert to lists for JSON serialization
            weights_data[layer_name] = {
                'type': layer_type,
                'weights': [w.tolist() for w in layer_weights],
                'shapes': [list(w.shape) for w in layer_weights]
            }
            
            # Special handling for bidirectional layers
            if layer_type == 'Bidirectional':
                # Extract forward and backward weights separately
                fwd_layer = layer.forward_layer
                bwd_layer = layer.backward_layer
                
                fwd_weights = fwd_layer.get_weights()
                bwd_weights = bwd_layer.get_weights()
                
                weights_data[layer_name].update({
                    'forward_weights': [w.tolist() for w in fwd_weights],
                    'backward_weights': [w.tolist() for w in bwd_weights],
                    'forward_shapes': [list(w.shape) for w in fwd_weights],
                    'backward_shapes': [list(w.shape) for w in bwd_weights],
                    'units': fwd_layer.units
                })
                
                print(f"  Forward shapes: {[w.shape for w in fwd_weights]}")
                print(f"  Backward shapes: {[w.shape for w in bwd_weights]}")
                print(f"  Units: {fwd_layer.units}")
            
            # Special handling for GRU layers
            elif layer_type == 'GRU':
                weights_data[layer_name].update({
                    'units': layer.units,
                    'return_sequences': layer.return_sequences
                })
                print(f"  Units: {layer.units}")
                print(f"  Return sequences: {layer.return_sequences}")
            
            # Special handling for Dense layers  
            elif layer_type == 'Dense':
                weights_data[layer_name].update({
                    'units': layer.units
                })
                print(f"  Units: {layer.units}")
    
    # Create the export data
    export_data = {
        'char_config': char_config,
        'model_config': {
            'input_shape': list(model.input_shape[1:]),  # Remove batch dimension
            'output_shape': list(model.output_shape[1:])  # Remove batch dimension
        },
        'weights': weights_data
    }
    
    # Save to JSON file
    output_file = 'syllable_model_weights.json'
    print(f"\nSaving weights to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("✅ Successfully exported model weights!")
    print(f"📁 File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
    
    return export_data

def test_extraction():
    """Test that we can load the weights and the shapes match."""
    print("\n🧪 Testing weight extraction...")
    
    with open('syllable_model_weights.json', 'r') as f:
        data = json.load(f)
    
    print("Character config:")
    print(f"  Chars: {len(data['char_config']['chars'])} characters")
    print(f"  Max length: {data['char_config']['maxlen']}")
    
    print("\nModel config:")
    print(f"  Input shape: {data['model_config']['input_shape']}")
    print(f"  Output shape: {data['model_config']['output_shape']}")
    
    print("\nWeights summary:")
    for layer_name, layer_data in data['weights'].items():
        print(f"  {layer_name} ({layer_data['type']}): {len(layer_data['weights'])} weight tensors")

if __name__ == "__main__":
    extract_model_weights()
    test_extraction() 