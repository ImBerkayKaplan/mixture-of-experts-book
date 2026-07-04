import pynvml

def get_gpu_power_watts(gpu_index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    # Power is returned in milliwatts, convert to Watts
    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
    pynvml.nvmlShutdown()
    return power_mw / 1000.0

print(f"Current Power Draw: {get_gpu_power_watts()} W")