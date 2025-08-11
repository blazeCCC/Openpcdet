#!/bin/bash

PROCESS_NAME=$1

# 使用 pgrep 精确匹配进程名
PIDS=$(pgrep -f $PROCESS_NAME)

if [ -z "$PIDS" ]; then
    echo "未找到名称包含 'perception_kl' 的进程。"
else
    echo "正在杀死以下进程:"
    echo "$PIDS"
    kill -9 $PIDS
fi

