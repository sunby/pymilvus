#!/bin/bash
# 清理 simple.py 可能残留的进程

echo "========================================="
echo "simple.py 进程清理脚本"
echo "========================================="
echo ""

# 检查是否有 simple.py 相关进程
echo "正在检查 simple.py 相关进程..."
PIDS=$(pgrep -f "simple.py")

if [ -z "$PIDS" ]; then
    echo "✅ 没有发现 simple.py 相关进程"
else
    echo "❌ 发现以下进程："
    ps aux | head -1
    ps aux | grep simple.py | grep -v grep
    echo ""
    echo "进程ID: $PIDS"
    echo ""
    
    # 询问是否清理
    read -p "是否要终止这些进程？[y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在温和地终止进程 (SIGTERM)..."
        pkill -15 -f simple.py
        
        # 等待3秒
        echo "等待进程退出..."
        sleep 3
        
        # 检查是否还有残留
        REMAINING=$(pgrep -f "simple.py")
        if [ -z "$REMAINING" ]; then
            echo "✅ 所有进程已成功终止"
        else
            echo "⚠️  仍有进程未退出，使用强制终止 (SIGKILL)..."
            pkill -9 -f simple.py
            sleep 1
            
            FINAL_CHECK=$(pgrep -f "simple.py")
            if [ -z "$FINAL_CHECK" ]; then
                echo "✅ 所有进程已强制终止"
            else
                echo "❌ 仍有进程残留，请手动检查："
                ps aux | grep simple.py | grep -v grep
            fi
        fi
    else
        echo "已取消清理"
    fi
fi

echo ""
echo "========================================="
echo "检查 Python 多进程残留"
echo "========================================="

# 检查所有 Python 进程
PYTHON_PROCS=$(ps aux | grep -E "python.*multiprocessing" | grep -v grep | wc -l)
if [ $PYTHON_PROCS -gt 0 ]; then
    echo "⚠️  发现 $PYTHON_PROCS 个 Python 多进程任务："
    ps aux | grep -E "python.*multiprocessing" | grep -v grep
else
    echo "✅ 没有发现异常的 Python 多进程"
fi

echo ""
echo "========================================="
echo "检查 Milvus 连接"
echo "========================================="

# 检查 Milvus 连接 (端口 19530)
MILVUS_CONNS=$(netstat -anp 2>/dev/null | grep 19530 | grep ESTABLISHED | wc -l)
if [ $MILVUS_CONNS -gt 0 ]; then
    echo "ℹ️  当前有 $MILVUS_CONNS 个活动的 Milvus 连接"
    netstat -anp 2>/dev/null | grep 19530 | grep ESTABLISHED
else
    echo "✅ 没有活动的 Milvus 连接"
fi

echo ""
echo "清理完成！"

