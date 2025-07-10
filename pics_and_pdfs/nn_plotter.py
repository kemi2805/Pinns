import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_neural_network(model, figsize=(12, 8)):
    """
    Create a custom visualization of the neural network architecture
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Network structure
    input_dim = model.input_dim
    hidden_dims = model.hidden_dims
    output_dim = 1
    
    # All layer sizes
    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    n_layers = len(layer_sizes)
    
    # Positioning
    layer_spacing = 2.0
    max_neurons = max(layer_sizes)
    neuron_spacing = 0.5
    
    # Colors for different layer types
    colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightgreen', 'lightcoral', 'orange']
    layer_names = ['Input\n(D, τ/D, S/D)', 'ResBlock 1\n(64)', 'ResBlock 2\n(64)', 
                   'ResBlock 3\n(64)', 'ResBlock 4\n(32)', 'Output\n(Z)']
    
    # Draw neurons and connections
    neuron_positions = []
    
    for layer_idx, (size, color, name) in enumerate(zip(layer_sizes, colors, layer_names)):
        x = layer_idx * layer_spacing
        y_start = -(size - 1) * neuron_spacing / 2
        
        layer_positions = []
        
        # Draw neurons
        for neuron_idx in range(size):
            y = y_start + neuron_idx * neuron_spacing
            
            # Different shapes for different layers
            if layer_idx == 0:  # Input layer
                circle = patches.Circle((x, y), 0.1, color=color, ec='black', linewidth=1.5)
            elif layer_idx == n_layers - 1:  # Output layer
                circle = patches.Circle((x, y), 0.15, color=color, ec='black', linewidth=2)
            else:  # Hidden layers
                circle = patches.Circle((x, y), 0.08, color=color, ec='gray', linewidth=1)
            
            ax.add_patch(circle)
            layer_positions.append((x, y))
        
        neuron_positions.append(layer_positions)
        
        # Add layer labels
        ax.text(x, y_start - 0.5, name, ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Draw connections between layers
    for layer_idx in range(n_layers - 1):
        current_layer = neuron_positions[layer_idx]
        next_layer = neuron_positions[layer_idx + 1]
        
        for x1, y1 in current_layer:
            for x2, y2 in next_layer:
                # Make connections lighter for clarity
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
    
    # Add skip connection for final layer
    if len(neuron_positions) > 2:
        input_layer = neuron_positions[0]
        final_hidden = neuron_positions[-2]
        output_layer = neuron_positions[-1]
        
        # Draw skip connection from input to final layer
        for x1, y1 in input_layer:
            for x2, y2 in output_layer:
                ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.7, linewidth=1.5, 
                       label='Skip Connection' if x1 == input_layer[0][0] and y1 == input_layer[0][1] else "")
    
    # Add activation function annotations
    activation_y = max([pos[0][1] for pos in neuron_positions]) + 0.8
    for i in range(1, n_layers - 1):
        x = (i - 0.5) * layer_spacing + 0.5
        ax.text(x, activation_y, 'SiLU', ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
               fontsize=8)
    
    # Add final activation
    x = (n_layers - 1.5) * layer_spacing + 0.5
    ax.text(x, activation_y, 'Softplus', ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
           fontsize=8)
    
    # Formatting
    ax.set_xlim(-0.5, (n_layers - 1) * layer_spacing + 0.5)
    ax.set_ylim(-max_neurons * neuron_spacing / 2 - 1, max_neurons * neuron_spacing / 2 + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('C2P-PINN Architecture\nPhysics-Informed Neural Network for Conservative-to-Primitive Transformation', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', label='Input Layer'),
        patches.Patch(color='lightgreen', label='Residual Blocks'),
        patches.Patch(color='orange', label='Output Layer'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Skip Connection'),
        patches.Patch(color='yellow', alpha=0.7, label='SiLU Activation'),
        patches.Patch(color='red', alpha=0.7, label='Softplus Activation')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig, ax

# Usage example (you would need to adapt this to your actual model)
class MockModel:
    def __init__(self):
        self.input_dim = 3
        self.hidden_dims = [64, 64, 64, 32]

# Create and display the plot
model = MockModel()  # Replace with your actual model
fig, ax = plot_neural_network(model)
plt.show()

# Save the plot
plt.savefig('c2p_pinn_architecture.png', dpi=300, bbox_inches='tight')