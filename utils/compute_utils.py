def calculate_output_length_1d(L_in, kernel_size, stride, padding=0):
    """Calculate the output length of a 1D convolutional layer."""
    return (L_in + 2 * padding - kernel_size) // stride + 1