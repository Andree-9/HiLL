pip install pip --upgrade
pip install uv
python -m uv pip install vllm==0.10.1
python -m uv pip install flash-attn==2.8.0.post2 --no-build-isolation
python -m uv pip install -r requirements.txt
python -m uv pip install -e .
python -m uv pip install transformers==4.57.6

wandb login