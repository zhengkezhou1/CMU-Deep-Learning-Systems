import matplotlib.pyplot as plt
import numpy as np
from simple_ml import read_idx_images,read_idx_labels, softmax_loss, parse_mnist

def verification():
    # 加载数据
    images = read_idx_images('data/train-images-idx3-ubyte.gz')
    labels = read_idx_labels('data/train-labels-idx1-ubyte.gz')

    # 随机选 5 个索引
    indices = np.random.randint(0, len(images), size=5)

    plt.figure(figsize=(10, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Label: {labels[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def verification_softmax_loss():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    print(softmax_loss(X,y))
    
# 执行函数
if __name__ == "__main__":
    verification_softmax_loss()