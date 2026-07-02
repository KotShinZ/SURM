set -euxo pipefail

uv venv .venv2
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install packaging psutil ninja pytest einops wheel setuptools

# Install flash attention 3
if [ ! -d flash-attention ]; then
  git clone --recursive https://github.com/Dao-AILab/flash-attention.git
fi

# 途中生成物を掃除
rm -rf flash-attention/third_party/nvidia/backend/bin \
       flash-attention/third_party/nvidia/backend/nvvm \
       flash-attention/hopper/build \
       flash-attention/hopper/*.egg-info

cd flash-attention/hopper
export MAX_JOBS=8
export FLASH_ATTENTION_DISABLE_SM80=TRUE # only H100
python setup.py install #  | head -n 100
# export PYTHONPATH=$PWD
# pytest -q -s test_flash_attn.py # Run tests to verify installation

cd ../..

uv pip install -r requirements.txt
