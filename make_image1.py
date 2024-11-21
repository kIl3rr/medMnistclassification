import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_model(ax, layers):
    """
    Draw a sequential neural network model with various types of layers in a horizontal layout,
    with labels placed above each layer.
    
    Args:
    ax -- matplotlib axis to draw on
    layers -- list of tuples (layer_type, description)
    """
    # Initialize settings for horizontal layout
    layer_width = 0.15
    layer_height = 0.08
    horizontal_spacing = 0.25
    label_offset = 0.1  # Vertical offset for labels above each layer
    current_x = 0.1  # Starting position for the first layer

    # Function to draw individual layers and labels above
    def draw_layer(x, layer_type, desc):
        color = 'lightblue' if 'conv' in layer_type else ('lightgreen' if 'dense' in layer_type else 'lightgrey')
        rect = patches.Rectangle((x, 0.5 - layer_height / 2), layer_width, layer_height, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        # Place label above the layer
        ax.text(x + layer_width / 2, 0.5 + layer_height / 2 + label_offset, f'{desc}', fontsize=10, va='bottom', ha='center')

    # Draw layers and connect them
    for i, (layer_type, desc) in enumerate(layers):
        draw_layer(current_x, layer_type, desc)
        if i > 0:
            # Draw an arrow from the previous layer to this one
            ax.add_patch(patches.FancyArrowPatch((current_x - horizontal_spacing + layer_width, 0.5), (current_x, 0.5),
                                                 connectionstyle="arc3,rad=0", arrowstyle='-|>', mutation_scale=10, color='gray'))
        current_x += layer_width + horizontal_spacing

    # Set plot settings
    ax.set_xlim(0, current_x)
    ax.set_ylim(0, 1)
    ax.axis('off')

# Create figure
fig, ax = plt.subplots(figsize=(12, 3))  # Adjusted figure size for horizontal layout and label space
ax.axis('off')

# Define the layers of the network
layers_info = [
    ('conv', 'Conv2D 32\n3x3, ReLU'),
    ('pool', 'MaxPooling\n2x2'),
    ('conv', 'Conv2D 64\n3x3, ReLU'),
    ('pool', 'MaxPooling\n2x2'),
    ('conv', 'Conv2D 64\n3x3, ReLU'),
    ('flatten', 'Flatten'),
    ('dense', 'Dense 64\nReLU'),
    ('dense', 'Dense\nSoftmax')
]

# Draw the model
draw_model(ax, layers_info)
plt.show()
