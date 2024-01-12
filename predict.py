from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow_addons.optimizers import AdamW
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Danh sách tên lớp tương ứng với các lớp CIFAR-10
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

def display_images_and_predictions(image_folder, model, class_names):
    correct_predictions = 0
    total_samples = 0
    true_labels = []
    predicted_labels = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg")or filename.endswith(".jpeg"):  # Chỉ xử lý các file hình ảnh
            image_path = os.path.join(image_folder, filename)

            # Tải và tiền xử lý ảnh kiểm tra
            image = preprocessing.image.load_img(image_path, target_size=(32, 32))
            input_arr = preprocessing.image.img_to_array(image)
            x = np.array([input_arr])

            # Đưa ra dự đoán trên ảnh kiểm tra
            predictions = model.predict(x)
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = class_names[predicted_class_index]

            true_label = os.path.basename(image_folder)
            true_labels.append(true_label)
            predicted_labels.append(predicted_class_name)

            # Kiểm tra xem dự đoán có đúng không
            if predicted_class_name == true_label:
                correct_predictions += 1

            total_samples += 1

            # Hiển thị kết quả
            # print(f'Image: {filename}, Predicted Class: {predicted_class_name}, True Class: {true_label}')

    accuracy = correct_predictions / total_samples
    return accuracy, true_labels, predicted_labels, correct_predictions, total_samples

def evaluate_all_classes(test_root_folder, model, class_names):
    class_accuracies = []
    all_true_labels = []
    all_predicted_labels = []
    correct_predictions_total = 0
    total_samples_total = 0

    for class_name in class_names:
        class_folder = os.path.join(test_root_folder, class_name)
        print(f"\nEvaluating class: {class_name}")
        class_accuracy, true_labels, predicted_labels, correct_predictions, total_samples = display_images_and_predictions(class_folder, model, class_names)
        class_accuracies.append(class_accuracy)
        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)
        correct_predictions_total += correct_predictions
        total_samples_total += total_samples
        print(f'Accuracy for {class_name}: {class_accuracy * 100:.2f}%')

    return class_accuracies, all_true_labels, all_predicted_labels, correct_predictions_total, total_samples_total

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    accuracy = accuracy_score(true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print(f'Accuracy: {accuracy * 100:.2f}%')

def print_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    accuracy = accuracy_score(true_labels, predicted_labels)

    print("\nConfusion Matrix:")
    print(tabulate(cm, headers=class_names, showindex=class_names, tablefmt="grid"))
    print(f'\nAccuracy: {accuracy * 100:.2f}%')
    
    
if __name__ == "__main__":
    # Gán cứng đường dẫn cho thư mục mô hình
    model_folder = 'C:\\Users\\admin\\Desktop\\vit\\.output'

    # Tải mô hình đã đào tạo sử dụng các đối tượng tùy chỉnh (bộ tối ưu AdamW)
    custom_objects = {'AdamW': AdamW}
    model = load_model(model_folder, custom_objects=custom_objects)

    # Gán cứng đường dẫn cho thư mục gốc chứa các thư mục lớp
    test_root_folder = 'C:\\Users\\admin\\Desktop\\vit\\test'

    # Đánh giá tất cả các lớp và tính độ chính xác từng lớp và độ chính xác tổng
    class_accuracies, all_true_labels, all_predicted_labels, correct_predictions_total, total_samples_total = evaluate_all_classes(test_root_folder, model, class_names)

    # Hiển thị độ chính xác của từng lớp
    for i, accuracy in enumerate(class_accuracies):
        print(f'Accuracy for {class_names[i]}: {accuracy * 100:.2f}%')

    # Hiển thị độ chính xác tổng
    overall_accuracy = correct_predictions_total / total_samples_total
    print(f'\nOverall Accuracy: {overall_accuracy * 100:.2f}%')

    # Hiển thị confusion matrix
    plot_confusion_matrix(all_true_labels, all_predicted_labels, class_names)
    
    # Hiển thị confusion matrix
    print_confusion_matrix(all_true_labels, all_predicted_labels, class_names)
