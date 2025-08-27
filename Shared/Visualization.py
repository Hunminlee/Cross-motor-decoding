import matplotlib.pyplot as plt


def learning_plot(history):
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.grid(True, linestyle='--', color='grey', which='major', alpha=0.3)  # color로 수정
    plt.show()