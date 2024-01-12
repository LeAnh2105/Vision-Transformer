from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention, Dense, Dropout, Flatten
from tensorflow.keras import Sequential

#Khối MLP trong Bộ mã hóa Transformer
class MLPBlock(Layer):

    #hidden_layers: Danh sách các lớp cho khối mạng MLP
    #dropout: tỷ lệ dropout cho khối mạng MLP
    #activation: hàm kích hoạt của lớp MLP

    def __init__(self, hidden_layers, dropout=0.1, activation='gelu'):
        super(MLPBlock, self).__init__()

        # Tạo một chuỗi các lớp Dense và Dropout để tạo một mạng MLP
        layers = []
        for num_units in hidden_layers:
            layers.extend([
                Dense(num_units, activation=activation),
                Dropout(dropout)
            ])
        self.mlp = Sequential(layers)

    # Chuyển đầu ra của attention multi-head vào mạng MLP
    def call(self, inputs, *args, **kwargs):
        #inputs: Đầu ra của lớp multi-head attention. Shape: (..., S, D). Ví dụ: (64, 100, 768)
        #*args và **kwargs:ác tham số được chuyển vào mạng MLP
        outputs = self.mlp(inputs, *args, **kwargs)
        return outputs

# Các khối Transformer bao gồm lớp multi-head attention và khối mạng MLP
class TransformerBlock(Layer):
    #num_heads: Số lượng heads của lớp multi-head attention
    #D: Kích thước của mỗi head attention cho giá trị (value)
    #hidden_layers:  Danh sách các lớp cho khối mạng MLP
    #dropout: Số thực Tỷ lệ dropout cho khối mạng MLP
    #norm_eps: Giá trị epsilon cho layer normalization
    def __init__(self, num_heads, D, hidden_layers, dropout=0.1, norm_eps=1e-12):
        super(TransformerBlock, self).__init__()

        # Lớp attention multi-head
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=D, dropout=dropout
        )
        self.norm_attention = LayerNormalization(epsilon=norm_eps)

        # Mạng MLP
        self.mlp = MLPBlock(hidden_layers, dropout)
        self.norm_mlp = LayerNormalization(epsilon=norm_eps)

    # Truyền qua các lớp của một Transformer Block
    def call(self, inputs):
        #inputs: embedding patches
        
        # Feed attention
        # Chuẩn hóa đầu vào
        norm_attention = self.norm_attention(inputs)
        
        # Đưa vào Multi-head attention
        attention = self.attention(query=norm_attention, value=norm_attention)
        
        # Skip Connection:cộng với đầu vòa ban đầu
        attention += inputs  

        # Feed MLP
        #đầu ra của Multi-Head Attention và đầu vào ban đầu được sử dụng làm đầu vào cho một mạng MLP
        # Chuẩn hóa layer
        outputs = self.mlp(self.norm_mlp(attention))
        
        # Skip Connection: cộng với đầu vào 
        outputs += attention  

        return outputs


#Bộ mã hóa Transformer bao gồm nhiều lớp transformer
class TransformerEncoder(Layer):
    # num_layers: Số lớp transformer
    #num_heads: Số lượng heads của lớp multi-head attention
    #D: Kích thước của mỗi head attention cho giá trị (value)
    # mlp_dim:  Kích thước hoặc chiều của lớp ẩn trong khối mạng MLP
    # dropout: Tỷ lệ dropout cho khối mạng MLP
    # norm_eps: Giá trị epsilon cho layer normalization
    def __init__(self, num_layers, num_heads, D, mlp_dim, dropout=0.1, norm_eps=1e-12):
        super(TransformerEncoder, self).__init__()

        # Tạo một chuỗi các lớp TransformerBlock để tạo encoder
        self.encoder = Sequential(
            [
                TransformerBlock(num_heads=num_heads,
                                 D=D,
                                 hidden_layers=[mlp_dim, D],
                                 dropout=dropout,
                                 norm_eps=norm_eps)
                for _ in range(num_layers)
            ]
        )
    ## Chuyển đầu ra của mạng MLP thông qua tất cả các lớp TransformerBlock
    def call(self, inputs, *args, **kwargs):
        outputs = self.encoder(inputs, *args, **kwargs)
        return outputs
