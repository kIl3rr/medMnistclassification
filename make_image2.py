import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_network(ax, layers):
    """
    Draw a neural network with convolutional and fully connected layers.
    
    Args:
    ax -- matplotlib axis to draw on
    layers -- list of tuples (layer_type, output_channels or nodes, kernel/pooling size)
    """
    layer_width = 0.1
    layer_height = 0.2
    vertical_distance_between_layers = 0.8
    horizontal_distance_between_neurons = 0.1
    neuron_radius = 0.05
    horizontal_distance_between_layers = 0.3  # 正确定义层间水平距离

    n_layers = len(layers)
    current_x = 0.1  # x position of the current layer
    
    # Function to draw neurons
    def draw_neuron(x, y, layer_type):
        if layer_type in ['conv', 'fc']:
            color = 'skyblue' if layer_type == 'conv' else 'lightgreen'
            ax.add_patch(patches.Circle((x, y), neuron_radius, color=color, zorder=4))
    
    # Draw layers
    for i, (layer_type, num_neurons, info) in enumerate(layers):
        layer_top = 1 - layer_height * num_neurons / 2 - (num_neurons - 1) * horizontal_distance_between_neurons / 2
        for j in range(num_neurons):
            neuron_y = layer_top + j * (layer_height + horizontal_distance_between_neurons)
            draw_neuron(current_x, neuron_y, layer_type)
            if i > 0:
                for k in range(layers[i-1][1]):  # Connect to previous layer
                    prev_neuron_y = 1 - layer_height * layers[i-1][1] / 2 - (layers[i-1][1] - 1) * horizontal_distance_between_neurons / 2 + k * (layer_height + horizontal_distance_between_neurons)
                    ax.add_patch(patches.ConnectionPatch(
                        (current_x - layer_width, neuron_y), (current_x - horizontal_distance_between_layers, prev_neuron_y),
                        "data", "data", arrowstyle="->", shrinkB=10))
        current_x += horizontal_distance_between_layers + layer_width
        ax.text(current_x - horizontal_distance_between_layers/2, 1.25, f'{layer_type.upper()}\n{info}', ha='center')

# Setup figure
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 1.2)
ax.axis('off')

# Layers info: (type, number of neurons, additional info)
layers_info = [
    ('conv', 6, '3x3, stride=1'),
    ('conv', 16, '3x3, stride=1'),
    ('fc', 120, '16*54*54'),
    ('fc', 84, ''),
    ('fc', 20, ''),
    ('fc', 10, 'len(class_names)')  # Assuming len(class_names) = 10 for example
]

draw_neural_network(ax, layers_info)
plt.show()
