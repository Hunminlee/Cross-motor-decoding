import matplotlib.pyplot as plt

def learning_plot(history):
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.grid(True, linestyle='--', c='grey', which='major', pad=0.1)
    plt.show()