from tensorflow.python.keras.layers.core import Dropout
from vit.embedding import PatchEmbedding
from vit.encoder import TransformerEncoder
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Resizing, RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model

# ViT cho bài toán phân loại
class ViT(Model):
    def __init__(self, num_layers=12, num_heads=12, D=768, mlp_dim=3072, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        """
        Mô hình Vision Transformer (ViT).

        Parameters
        ----------
        num_layers: int,
            Số lớp transformer
            Ví dụ: 12
        num_heads: int,
            Số lượng heads của lớp multi-head attention
        D: int
            Kích thước của mỗi attention head cho value
        mlp_dim: int
            Kích thước hoặc chiều của hidden layer của mạng MLP
        num_classes: int
            Số lượng lớp (số lượng classes)
        patch_size: int
            Kích thước của một patch (P)
        image_size: int
            Kích thước của ảnh (H hoặc W)
        dropout: float,
            Tỷ lệ dropout của mạng MLP
        norm_eps: float,
            Epsilon của layer norm
        """
        super(ViT, self).__init__()

        # Tăng cường dữ liệu
        self.data_augmentation = Sequential([
            #Chuyển đổi giá trị pixel của ảnh về khoảng [0, 1]
            Rescaling(scale=1./255),
            #Thay đổi kích thước của ảnh đầu vào thành image_size x image_size
            Resizing(image_size, image_size),
            #Ngẫu nhiên lật ảnh theo trục ngang. 
            RandomFlip("horizontal"),
            #Ngẫu nhiên xoay ảnh một lượng nhỏ (2% của tổng số độ).
            RandomRotation(factor=0.02),
            #Ngẫu nhiên zoom ảnh theo chiều cao và chiều rộng với hệ số 0.2.
            RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ])

        # Nhúng các patch
        self.embedding = PatchEmbedding(patch_size, image_size, D)

        # Bộ mã hóa với transformer
        self.encoder = TransformerEncoder(
            #Số lượng đầu ra của lớp multi attention  trong mỗi khối Transformer. 
            num_heads=num_heads,
            #Số lượng các khối Transformer trong bộ mã hóa. 
            num_layers=num_layers,
            # Kích thước của mỗi vector đầu ra của các đầu attention, kích thước của các vector nhúng cho từng đầu vào patch
            D=D,
            #Kích thước của các đầu ra của mạng MLP bên trong mỗi khối Transformer. 
            mlp_dim=mlp_dim,
            #Tỷ lệ dropout 
            dropout=dropout,
            #Epsilon được sử dụng trong chuẩn hóa layer 
            norm_eps=norm_eps,
        )

        # Đầu MLP cho nhiệm vụ phân loại
        self.mlp_head = Sequential([
            #Chuẩn hóa layer
            LayerNormalization(epsilon=norm_eps),
            #Lớp Dense với kích thước đầu ra là mlp_dim
            Dense(mlp_dim),
            #ớp Dropout được sử dụng để giảm nguy cơ overfitting
            Dropout(dropout),
            #Lớp Dense cuối cùng với đầu ra có kích thước là num_classes đại diện cho số lớp phân loại. 
            Dense(num_classes, activation='softmax'),
        ])

        self.last_layer_norm = LayerNormalization(epsilon=norm_eps)

    def call(self, inputs):
        """
        Thực hiện quá trình truyền qua mô hình ViT.

        Parameters
        ----------
        inputs: tensor,
            Dữ liệu đầu vào
            shape: (..., H, W, C)

        Returns
        -------
        output: tensor,
            Dự đoán của mô hình
            Shape: (..., num_classes)
        """
        
        
        # Tạo dữ liệu được tăng cường (augmented data)
        # augmented shape: (..., image_size, image_size, c)
        #Ảnh đầu vào được tăng cường dữ liệu. 
        augmented = self.data_augmentation(inputs)

        # Tạo position embedding + CLS Token
        # embedded shape: (..., S + 1, D)
        # ảnh được đưa vào quá trình nhúng (embedding) với self.embedding, nơi position embedding và token "CLS" được thêm vào.
        embedded = self.embedding(augmented)

        # Mã hóa các patch (Encode patchs) với transformer
        # embedded shape: (..., S + 1, D)
        # Ảnh được đưa qua một chuỗi các TransformerBlocks để mã hóa thông tin của các patch. Kết quả là encoded, chứa biểu diễn của các patch.
        encoded = self.encoder(embedded)

        # Embedded CLS
        # embedded_cls shape: (..., D)
        #Từ encoded, chỉ giữ lại phần của token "CLS" để được embedded_cls, là biểu diễn của toàn bộ ảnh sau khi đã được mã hóa.
        embedded_cls = encoded[:, 0]
        
        # Chuẩn hóa lớp cuối cùng
        # Biểu diễn này sau đó được chuẩn hóa bởi last_layer_norm
        y = self.last_layer_norm(embedded_cls)
        
        # Feed MLP head
        # output shape: (..., num_classes)
        #biểu diễn đã được chuẩn hóa của token "CLS" được đưa vào mạng MLP cuối cùng (self.mlp_head) để dự đoán lớp của ảnh. Đầu ra của mô hình là một phân phối xác suất qua các lớp.
        output = self.mlp_head(y)

        return output


