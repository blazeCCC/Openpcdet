from dingtalkchatbot.chatbot import DingtalkChatbot
from datetime import  datetime
def dingtalk_robot(webhook,secret):
    dogBOSS = DingtalkChatbot(webhook, secret)
    red_msg = '<font color="#dd0000">级别:危险</font>'
    orange_msg = '<font color="#FFA500">级别:警告</font>'

    now_time = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
    url = 'https://blog.csdn.net/qq_46158060?type=blog'
    dogBOSS.send_markdown(
        title=f'来自梦无矶小仔的提醒',
        text=f'### **我是主内容的第一行**\n'
              f'**{red_msg}**\n\n'
              f'**{orange_msg}**\n\n'
              f'**发送时间:**  {now_time}\n\n'
              f'**相关网址:**[点击跳转]({url}) \n',
        is_at_all=True)

if __name__ == '__main__':
    # webhook = '刚你记录的webhook填这里'
    webhook = 'https://oapi.dingtalk.com/robot/send?access_token=225c9b06d3e7fd045cd7448b366d0c0ec5c51b8650eccb8a5461ac26b58cada7'
    # secrets = '刚你记录的秘钥填这里'
    secrets = 'SECffb77f6d240298fff9637eaac088f4e1da962979ec64f76751a17aa7e594e6cc'
    dingtalk_robot(webhook=webhook,
                   secret=secrets)
