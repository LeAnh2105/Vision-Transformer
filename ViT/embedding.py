from tensorflow.keras.layers import Layer, Embedding, Dense
import tensorflow as tf

# Lớp Patches: Trích xuất các patch từ ảnh.
# patch_size (int): Kích thước của một patch (P).
class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    #Truyền qua ảnh để lấy các patch.
    # images (tensor): Ảnh từ tập dữ liệu. shape: (..., W, H, C). Ví dụ: (64, 32, 32, 3)
    def call(self, images):
        """
        Returns:
            patches (tensor): Các patch được trích xuất từ ảnh.
                Shape: (..., S, P^2 x C) với S = (HW)/(P^2) Ví dụ: (64, 64, 48)
        """
        # Lấy kích thước batch
        batch_size = tf.shape(images)[0]

        # Sử dụng tf.image.extract_patches để trích xuất các patch từ ảnh
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )

        # Lấy số chiều của patch
        dim = patches.shape[-1]

        # Reshape các patch thành hình dạng mong muốn
        patches = tf.reshape(patches, (batch_size, -1, dim))
        return patches


#Lớp PatchEmbedding: Nhúng thông tin vị trí vào các patch từ ảnh.
# patch_size (int): Kích thước của một patch (P).
# image_size (int): Kích thước của một ảnh (H hoặc W).
# projection_dim (int): Kích thước chiều chiếu trước khi đưa các patch qua transformer.
class PatchEmbedding(Layer):
    def __init__(self, patch_size, image_size, projection_dim):
        super(PatchEmbedding, self).__init__()

        # Số lượng patch: S = self.num_patches
        self.num_patches = (image_size // patch_size) ** 2

        # Token "cls" được sử dụng cho mạng mlp cuối cùng
        self.cls_token = self.add_weight(
            "cls_token",
            #projection_dim là kích thước của không gian chiều mới
            shape=[1, 1, projection_dim],
            #khởi tạo giá trị cho token "cls"
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )
        
        # Lớp Patches để trích xuất các patch từ ảnh
        self.patches = Patches(patch_size)

        # Lớp Dense để chiếu các patch vào không gian chiếu mới
        self.projection = Dense(units=projection_dim)

        # self.position_embedding shape: (..., S + 1, D)
        self.position_embedding = self.add_weight(
            "position_embeddings",
            #self.num_patches là số lượng patch (S) trong ảnh, thêm 1 đại diện cho token "CLS" (một token đặc biệt được thêm vào để biểu diễn toàn bộ ảnh). 
            #projection_dim là kích thước của không gian chiều mới
            shape=[self.num_patches + 1, projection_dim],
            # khởi tạo giá trị cho trọng số.
            #sử dụng phân phối ngẫu nhiên theo phân phối chuẩn (normal distribution).
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )

    #Truyền qua ảnh để nhúng thông tin vị trí.
    #images (tensor): Ảnh từ tập dữ liệu. Shape: (..., W, H, C). Ví dụ: (64, 32, 32, 3)
    def call(self, images):
        """
            encoded_patches (tensor): Nhúng các patch với thông tin vị trí và nối với token "cls".
                Shape: (..., S + 1, D) với S = (HW)/(P^2) Ví dụ: (64, 65, 768)
        """
        # Lấy các patch từ ảnh
        # Shape patch: (..., S, NEW_C)
        patch = self.patches(images)

        # Shape encoded_patches: (..., S, D)
        # Chiếu patch xuống không gian ẩn D
        encoded_patches = self.projection(patch)

        # Số ảnh trong mỗi batch
        batch_size = tf.shape(images)[0]

        # số chiều của vector ẩn tại mỗi vị trí của patch
        hidden_size = tf.shape(encoded_patches)[-1]

        # cls_broadcasted shape: (..., 1, D)
        # Tạo token [class] để học cho nhiệm vụ phân loại
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls_token, [batch_size, 1, hidden_size]),
            dtype=images.dtype,
        )

        # Shape encoded_patches: (..., S + 1, D)
        # Thêm token [class] vào đầu của encoder_patch
        encoded_patches = tf.concat([cls_broadcasted, encoded_patches], axis=1)

        # Shape encoded_patches: (..., S + 1, D)
        # Thêm position_embedding vào encoded_patches
        encoded_patches = encoded_patches + self.position_embedding

        return encoded_patches

