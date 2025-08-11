import subprocess
import time
from dingtalkchatbot.chatbot import DingtalkChatbot
from datetime import datetime

# 钉钉发送函数
def dingtalk_robot(webhook, secret, used_mem):
    dogBOSS = DingtalkChatbot(webhook, secret)
    red_msg = '<font color="#dd0000">级别:危险</font>'
    now_time = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
    url = 'https://blog.csdn.net/qq_46158060?type=blog'
    dogBOSS.send_markdown(
        title='来自inspur的训练状态提醒',
        text=f'### **GPU状态告警**\n'
             f'**{red_msg}**\n\n'
             f'**发送时间:**  {now_time}\n\n'
             f'**GPU-1 显存占用:** {used_mem} MB\n\n'
             f'**相关网址:**[点击跳转]({url}) \n',
        is_at_all=True
    )

# 获取 GPU-1 显存使用情况
def get_gpu_memory_usage(gpu_index=1):
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_list = result.strip().split('\n')
        if gpu_index < len(memory_list):
            return int(memory_list[gpu_index])
        else:
            print(f"没有找到 GPU-{gpu_index}")
            return None
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return None

# 主循环
def monitor_gpu(interval_minutes=5, threshold_mb=1000):
    webhook = 'https://oapi.dingtalk.com/robot/send?access_token=225c9b06d3e7fd045cd7448b366d0c0ec5c51b8650eccb8a5461ac26b58cada7'
    secret = 'SECffb77f6d240298fff9637eaac088f4e1da962979ec64f76751a17aa7e594e6cc'

    while True:
        used_mem = get_gpu_memory_usage(gpu_index=5)
        if used_mem is not None and used_mem < threshold_mb:
            dingtalk_robot(webhook, secret, used_mem)
            break
        time.sleep(interval_minutes * 60)

if __name__ == '__main__':
    monitor_gpu(interval_minutes=10, threshold_mb=1000)
