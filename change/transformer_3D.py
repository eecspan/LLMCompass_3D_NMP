import math
from software_model.operators import Operator, Reshape, Concat, Transpose
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.communication_primitives import AllReduceMultiPCB
from software_model.utils import Tensor, DataType
from hardware_model.system import System

# 3Dstack 版本的 matmul
from change.matmul_HBM import Matmul, BatchedMatmul


class TransformerBlockInitComputationTP3D(Operator):
    def __init__(self, d_model, d_intermediate, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.n_heads = n_heads
        self.device_count = device_count

        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, d_intermediate // device_count], data_type)
        self.W2 = Tensor([d_intermediate // device_count, d], data_type)

        # operators
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, X: Tensor) -> Tensor:
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        Q = self.Q_proj(X, self.Wq)
        K = self.K_proj(X, self.Wk)
        V = self.V_proj(X, self.Wv)
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])
        K_T = self.K_transpose(K, [0, 2, 3, 1])
        V_T = self.V_transpose(V, [0, 2, 1, 3])
        A = self.Q_mul_K(Q_T, K_T)
        A_prob = self.A_softmax(A)
        H = self.A_mul_V(A_prob, V_T)
        H = self.H_transpose(H, [0, 2, 1, 3])
        H = self.H_reshape(H, [b, s, d // dev_cnt])
        H0 = self.H_matmul0(H, self.W0)
        H0 = self.layer_norm0(H0)
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)

        H1 = self.H_matmul1(H0, self.W1)
        H1 = self.H_gelu(H1)
        H2 = self.H_matmul2(H1, self.W2)
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)

        return H2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv = 3 * (self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul)
        qk = self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        av = self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        h0 = self.H_matmul0.roofline_model(device) + device.compute_module.overhead.matmul
        h1 = self.H_matmul1.roofline_model(device) + device.compute_module.overhead.matmul
        h2 = self.H_matmul2.roofline_model(device) + device.compute_module.overhead.matmul
        matmul_total = qkv + qk + av + h0 + h1 + h2

        softmax_lat = self.A_softmax.roofline_model(device) + device.compute_module.overhead.softmax
        ln_lat = self.layer_norm0.roofline_model(device) + device.compute_module.overhead.layernorm
        norm_total = softmax_lat + ln_lat * 2
        gelu_lat = self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu

        if self.device_count > 1:
            allreduce_lat = self.allreduce_mha.simulate(interconnect)
            allreduce_total = allreduce_lat * 2
        else:
            allreduce_total = 0

        self.roofline_latency = matmul_total + norm_total + gelu_lat + allreduce_total
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str = "3D_stacked"):
        device = system.device
        interconnect = system.interconnect

        # matmul 用 3D_stacked 详细模拟
        qkv = 3 * (self.Q_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul)
        qk = self.Q_mul_K.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        av = self.A_mul_V.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        h0 = self.H_matmul0.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        h1 = self.H_matmul1.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        h2 = self.H_matmul2.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        matmul_total = qkv + qk + av + h0 + h1 + h2

        # 其他算子先用 roofline
        softmax_lat = self.A_softmax.roofline_model(device) + device.compute_module.overhead.softmax
        ln_lat = self.layer_norm0.roofline_model(device) + device.compute_module.overhead.layernorm
        norm_total = softmax_lat + ln_lat * 2
        gelu_lat = self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu

        if self.device_count > 1:
            allreduce_lat = self.allreduce_mha.simulate(interconnect)
            allreduce_total = allreduce_lat * 2
        else:
            allreduce_total = 0

        self.latency = matmul_total + norm_total + gelu_lat + allreduce_total
        return self.latency


class TransformerBlockAutoRegressionTP3D(Operator):
    def __init__(self, d_model, d_intermediate, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.n_heads = n_heads
        self.device_count = device_count

        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, d_intermediate // device_count], data_type)
        self.W2 = Tensor([d_intermediate // device_count, d], data_type)

        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        b, _, d = x.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        s = seq_len

        K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)

        q = self.Q_proj(x, self.Wq)
        k = self.K_proj(x, self.Wk)
        v = self.V_proj(x, self.Wv)
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])
        q_T = self.Q_transpose(q, [0, 2, 1, 3])
        k_T = self.K_transpose(k, [0, 2, 3, 1])
        v_T = self.V_transpose(v, [0, 2, 1, 3])
        K_T = self.K_concat(K_cache, k_T, 3)
        V_T = self.V_concat(V_cache, v_T, 2)
        a = self.Q_mul_K(q_T, K_T)
        a_prob = self.A_softmax(a)
        h0 = self.A_mul_V(a_prob, V_T)
        h0 = self.H_transpose(h0, [0, 2, 1, 3])
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        h0 = self.H_matmul0(h0, self.W0)
        h0 = self.layer_norm0(h0)
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        h1 = self.H_matmul1(h0, self.W1)
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )
        return h2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv = 3 * (self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul)
        qk = self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        av = self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        h0 = self.H_matmul0.roofline_model(device) + device.compute_module.overhead.matmul
        h1 = self.H_matmul1.roofline_model(device) + device.compute_module.overhead.matmul
        h2 = self.H_matmul2.roofline_model(device) + device.compute_module.overhead.matmul
        matmul_total = qkv + qk + av + h0 + h1 + h2

        softmax_lat = self.A_softmax.roofline_model(device) + device.compute_module.overhead.softmax
        ln_lat = self.layer_norm0.roofline_model(device) + device.compute_module.overhead.layernorm
        norm_total = softmax_lat + ln_lat * 2
        gelu_lat = self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu

        if self.device_count > 1:
            allreduce_lat = self.allreduce_mha.simulate(interconnect)
            allreduce_total = allreduce_lat * 2
        else:
            allreduce_total = 0

        self.roofline_latency = matmul_total + norm_total + gelu_lat + allreduce_total
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str = "3D_stacked"):
        device = system.device
        interconnect = system.interconnect

        qkv = 3 * (self.Q_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul)
        qk = self.Q_mul_K.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        av = self.A_mul_V.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        h0 = self.H_matmul0.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        h1 = self.H_matmul1.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        h2 = self.H_matmul2.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul
        matmul_total = qkv + qk + av + h0 + h1 + h2

        softmax_lat = self.A_softmax.roofline_model(device) + device.compute_module.overhead.softmax
        ln_lat = self.layer_norm0.roofline_model(device) + device.compute_module.overhead.layernorm
        norm_total = softmax_lat + ln_lat * 2
        gelu_lat = self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu

        if self.device_count > 1:
            allreduce_lat = self.allreduce_mha.simulate(interconnect)
            allreduce_total = allreduce_lat * 2
        else:
            allreduce_total = 0

        self.latency = matmul_total + norm_total + gelu_lat + allreduce_total
        return self.latency