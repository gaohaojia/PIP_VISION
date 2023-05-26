# 设置自启动时使用，系统将会通过调用该文件实现自启动。
CURRENT_DIR="$(cd $(dirname $0); pwd)/main.py --mode release --color 0"
python3 $CURRENT_DIR