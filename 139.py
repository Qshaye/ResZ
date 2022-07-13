# -*- encoding=utf-8 -*-
import time, os

import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


# mail('failease@126.com', 'KLRCXDEZGWETNICY', 'failease@126.com', 'qshaye', 'qshaye', mail_msg)
def mail(my_sender='failease@126.com', my_pass='KLRCXDEZGWETNICY', to_user='failease@126.com', my_nick='qshaye', to_nick='qshaye', mail_msg=None):

    mail_msg = """	
    <p> 139 命令都跑完了，服务器空了！</p>	
    <p>进程结束！ 请存入指令重启程序</p>	
    """

    msg = MIMEText(mail_msg, 'html', 'utf-8')
    msg['From'] = formataddr([my_nick, my_sender])
    msg['To'] = formataddr([to_nick, to_user])
    msg['Subject'] = "Exit!"
    # 配置Python与邮件的SMTP服务器的连接通道
    server = smtplib.SMTP()
    server.connect('smtp.126.com', 25)
    server.login(my_sender, my_pass)
    server.sendmail(my_sender, [
        to_user,
    ], msg.as_string())
    server.quit()


# 指定GPU内存是否可以运行进程
def is_GPU_available_do(content, capacity=12500, gpu_num=0):
    line = content[gpu_num]
    if line.find("MiB") >= 0:
        pos1 = line.find("|")
        pos1 = line.find("|", pos1 + 1)
        pos2 = line.find("MiB", pos1)
        usage = line[pos1 + 1:pos2].strip()
        usage = int(usage)
        # print("GPU usage %d MB"%(usage,))
        if usage <= capacity:
            return True
    return False


def is_GPU_available(capacity=15000, gpu_num=0):
    cmd = "nvidia-smi|grep MiB"
    with os.popen(cmd, "r") as f:
        content = f.readlines()
    return is_GPU_available_do(content, capacity, gpu_num)


def read_commands(file_path):
    f = open(file_path)
    contents = f.readlines()
    f.close()
    return contents


def write_cmd_back(cmds, file_path):
    f = open(file_path, "w")
    for cmd in cmds:
        f.write(cmd)
        if not cmd.endswith("\n"):
            f.write("\n")
    f.close()


if __name__ == "__main__":

    cmd_path = "/home/Qshaye/pymarl/src/command"

    maximal_usage = 12500
    gpu_ids = [0, 1, 2, 3]
    print("The command path is", cmd_path)
    command_file_path = cmd_path

    while True:
        cmds = read_commands(command_file_path)
        if len(cmds) == 0:
            print("No command, exit!")
            try:
                mail()
                exit(1)
            except smtplib.SMTPException as e:
                print('发不了邮件额！', e)
        for i in range(0, len(gpu_ids)):
            gpu = gpu_ids[i]
            if is_GPU_available(maximal_usage, gpu):
                cmds = read_commands(command_file_path)
                # 如果没有指令 就退出程序并发邮件
                if len(cmds) == 0:
                    print("No command, exit!")
                    try:
                        mail()
                        exit(1)
                    except smtplib.SMTPException as e:
                        print('发不了邮件额！', e)
                cmd = cmds.pop()
                write_cmd_back(cmds, command_file_path)
                cmd = "CUDA_VISIBLE_DEVICES=" + str(gpu) + " " + cmd
                print(cmd)
                os.system(cmd)
                time.sleep(200)
        time.sleep(600)
