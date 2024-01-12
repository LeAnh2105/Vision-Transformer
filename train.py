from vit.model import ViT
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
import tensorflow_addons as tfa
import numpy as np
from tensorflow.image import resize
from argparse import ArgumentParser



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model', default='custom', type=str,
                        help='Type of ViT model, valid option: custom, base, large, huge')
    parser.add_argument('--num-classes', default=10,
                        type=int, help='Number of classes')
    parser.add_argument('--patch-size', default=2,
                        type=int, help='Size of image patch')
    parser.add_argument('--num-heads', default=4,
                        type=int, help='Number of attention heads')
    parser.add_argument('--att-size', default=64,
                        type=int, help='Size of each attention head for value')
    parser.add_argument('--num-layer', default=2,
                        type=int, help='Number of attention layer')
    parser.add_argument('--mlp-size', default=128,
                        type=int, help='Size of hidden layer in MLP block')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=1e-4,
                        type=float, help='Weight decay')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of training epoch')
    parser.add_argument('--image-size', default=32, type=int, 
                        help='Size of input image')

    parser.add_argument('--image-channels', default=3,
                        type=int, help='Number channel of input image')

    parser.add_argument('--model-folder', default='.output/',
                        type=str, help='Folder to save trained model')
    
    
    args = parser.parse_args()
    print('Training Vit Transformer model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    

    print("Data: Use CIFAR 10 dataset")

    # Thiết lập số kênh ảnh và số lớp
    args.image_channels = 3
    args.num_classes = 10

    # Tải dữ liệu từ CIFAR-10 dataset
    (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()

    # Định hình lại kích thước ảnh và chuyển đổi sang kiểu float32
    x_train = (x_train.reshape(-1, args.image_size, args.image_size, args.image_channels)).astype(np.float32)
    x_val = (x_val.reshape(-1, args.image_size, args.image_size, args.image_channels)).astype(np.float32)

    # Resize ảnh về kích thước mong muốn
    x_train_resized = np.array([resize(image, (args.image_size, args.image_size)).numpy() for image in x_train])
    x_val_resized = np.array([resize(image, (args.image_size, args.image_size)).numpy() for image in x_val])

    # Chuyển đổi sang kiểu float32
    x_train_resized = x_train_resized.astype(np.float32)
    x_val_resized = x_val_resized.astype(np.float32)

    # Tạo dataset sử dụng TensorFlow Dataset
    train_ds = Dataset.from_tensor_slices((x_train_resized, y_train))
    train_ds = train_ds.batch(args.batch_size)

    val_ds = Dataset.from_tensor_slices((x_val_resized, y_val))
    val_ds = val_ds.batch(args.batch_size)

    # Khởi tạo mô hình ViT
    model = ViT(
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_heads=args.num_heads,
            D=args.att_size,
            mlp_dim=args.mlp_size,
            num_layers=args.num_layer
        )

    # Xây dựng mô hình
    model.build(input_shape=(None, args.image_size,
                             args.image_size, args.image_channels))

    # Tạo optimizer và hàm loss
    optimizer = tfa.optimizers.AdamW(
        learning_rate=args.lr, weight_decay=args.weight_decay)
    loss = SparseCategoricalCrossentropy()
    # Compile mô hình
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    # Huấn luyện mô hình
    model.fit(train_ds,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=val_ds)

    # Lưu mô hình
    model.save(args.model_folder)
