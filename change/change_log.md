# 修改日志

## commit at 2025.12.15

### 核心功能：实时channel映射机制、数据位移机制

#### 主要改动
1. **数据块实时channel位置跟踪**
   - 实现了从逻辑 channel 号到物理 channel 号的动态映射机制
   - 支持权重矩阵和输入矩阵的位置跟踪，增强了后续扩展性
   - 通过字典 `_weight_location_map: Dict[Tuple[int, int], int]` 维护数据块的实际物理位置

2. **新增核心函数**
   ```python
   _init_location_mapping_if_needed(self) -> None:
       """懒加载初始化权重位置映射字典"""
   
   _get_weight_channel_location(self, n_idx: int, k_idx: int, channel_count: int) -> int:
       """获取权重块(n_idx, k_idx)当前实际所在的物理channel"""
   
   _get_input_channel_location(self, m_idx: int, k_idx: int, channel_count: int) -> int:
       """获取输入块(m_idx, k_idx)当前实际所在的物理channel"""
   
   _update_data_location(self, m_idx: int, n_idx: int, k_idx: int, new_channel: int, is_weight: bool) -> None:
       """更新数据块移位后的新物理位置"""
   ```

3. **首尾相连蛇形物理拓扑映射**
   - 修改 `_get_channel_coords()` 函数，实现首尾相连的蛇形channel排列
   - 优化相邻channel的物理距离，减少平均跳数
   - 实现逻辑：
    第一列，第一个数为0，第二个数为总结点数n-1，后续n-2、n-3以此类推
    然后从第一行的第二个数开始，也就是以（1，0）、（dim_x，0）、（dim_x，dim_x），（1，dim_x）为顶点的矩阵就是普通的蛇形拓扑
   - 经过验证拓扑逻辑正确，对于偶数平方个2Dmesh节点可以正常生成拓扑图
    

4. **通信模拟优化**
   - 修改 `simulate_ring_communication_estimate()` 函数，集成动态映射查询
   - 数据源 channel 从固定的逻辑 channel 改为**通过函数查询当前实际channel号**
   - 每次通信后更新数据块位置，实现类似环形移位寄存器效果，缩短每次数据传输的跳数

#### 技术细节
- **映射key设计**：
  - 权重块：`(n_idx, k_idx)`
  - 输入块：`(-m_idx-1, k_idx)` （使用负数避免键冲突）
- **映射值**：当前实际所在的物理 channel 编号
- **生命周期**：跨批次持久化，支持环形移位优化

#### 测试配置对比表
测试参数：
BS=8
M=4096
N=4096
K=4096
**内存形式：bandwidth**
| src_channel和dst_channel的映射关系 | 逻辑channel号到物理拓扑的映射关系 | 测试结果 |
|:---------------------------------:|:-------------------------------:|:--------:|
| 原处理方式 | 蛇形映射 | 102.557ms |
| 移位映射（类T10） | 蛇形映射 | 102.554ms |
| 移位映射（类T10） | 首尾相连蛇形映射 | 102.547ms |

### **TODO**
 - 需要添加对于HBM空间是否充足的判断
 - 可能需要增加从core到channel的写入时间，或者认为写入channel和计算是并行的
 - 暂时还没有考虑读取计算并行的内容