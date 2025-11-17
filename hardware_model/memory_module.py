class MemoryModule:
    def __init__(self, memory_capacity, channel_count):
        self.memory_capacity = memory_capacity
        self.channel_count = channel_count  # Number of memory channels

memory_module_dict = {'A100_80GB': MemoryModule(80e9, channel_count=40),'TPUv3': MemoryModule(float('inf'),channel_count=1),'MI210': MemoryModule(64e9, channel_count=32)}