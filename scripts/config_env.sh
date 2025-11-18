# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install transformers==4.57.1

# Install wheels
pip freeze > requirements_transfer.txt
mkdir wheels
pip wheel -r requirements_transfer.txt -w wheels/
