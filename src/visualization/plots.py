import matplotlib.pyplot as plt

def plot_training_history(history, save_path):
    """Plot training curves for analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training and validation loss
    axes[0, 0].semilogy(history['train_loss'], label='Training Loss')
    axes[0, 0].semilogy(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Physics loss
    #axes[0, 1].semilogy(history['physics_losses'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Physics Loss')
    axes[0, 1].set_title('Physics-Informed Loss')
    axes[0, 1].grid(True)
    
    # Loss components comparison
    axes[1, 0].semilogy(history['train_loss'], label='Total Loss')
    #axes[1, 0].semilogy(history['physics_losses'], label='Physics Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Convergence analysis
    min_loss = min(history['val_loss'])
    axes[1, 1].semilogy([min_loss] * len(history['val_loss']), 
                        '--', label=f'Best Val Loss: {min_loss:.2e}')
    axes[1, 1].semilogy(history['val_loss'], label='Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Convergence Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)


