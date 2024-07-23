pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip uninstall setuptools
pip install setuptools==59.5.0
apt-get update && apt install libgl1-mesa-glx
sed -i '878s/self._sync_params()/pass/' /usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py
