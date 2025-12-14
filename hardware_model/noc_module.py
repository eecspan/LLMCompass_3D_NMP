class NOCModule:
    def __init__(self, bandwidth, first_packet_latency, hop_latency):
        self.bandwidth = bandwidth
        self.first_packet_latency = first_packet_latency
        self.hop_latency = hop_latency