import numpy as np

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def calc_dis(coordList1,coordList2):
    y = [coordList2 for _ in coordList1]
    y = np.array(y)
    x = [coordList1 for _ in coordList2]
    x = np.array(x)
    x = x.transpose(1, 0, 2)
    a = np.linalg.norm(np.array(x) - np.array(y), axis=2)
    return a