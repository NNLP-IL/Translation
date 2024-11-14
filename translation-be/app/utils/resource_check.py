import psutil
import GPUtil

class ResourceChecker:
    @staticmethod
    def check_cpu_memory_usage(threshold: int):
        """ Check if current CPU usage is below a specified threshold."""
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        return memory_usage < threshold

    @staticmethod
    def check_gpu_memory_usage(threshold: int):
        """ Check if current GPU usage is below a specified threshold."""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return False 
        gpu = gpus[0]  
        gpu_memory_usage = gpu.memoryUtil * 100  
        return gpu_memory_usage < threshold