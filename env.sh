pip3 install numpy==2.2.4 pandas==2.2.3 scipy==1.15.2 scikit-learn==1.6.1 einops==0.8.1
pip3 install httpx==0.28.1 h11==0.16.0 fastapi uvicorn
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install safetensors==0.5.3
pip3 install datasets==2.21.0
pip3 install accelerate==1.6.0
pip3 install trl==0.15.2
git clone -b v4.51.3 https://github.com/huggingface/transformers.git
pip3 install -e ./transformers/
pip3 install -e ./TruthfulQA/
pip3 install huggingface_hub openai tqdm wandb==0.19.9 psutil==7.0.0 glances==4.0.8 matplotlib==3.10.1
pip3 install git+https://github.com/davidbau/baukit@9d51abd51ebf29769aecc38c4cbef459b731a36e
pip install -U sentence-transformers