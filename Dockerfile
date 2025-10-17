FROM dustynv/l4t-pytorch:r36.2.0

WORKDIR /app



# # 2. Copy just the requirements file to leverage Docker cache
COPY requirements.txt .

# # 3. Now, install the Python packages
# # RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Example for JetPack 5 / L4T R35.x / Python 3.10
# Replace with the exact URLs from the NVIDIA forum for your version!
# RUN wget https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048/torch-2.3.0-cp310-cp310-linux_aarch64.whl -O torch.whl && \
#     pip install numpy torch.whl && \
#     rm torch.whl && \
#     pip install torchvision==0.16.0

# RUN apt-get update && apt-get install -y \
# build-essential \
# libpq-dev \
# libjpeg-dev \
# && rm -rf /var/lib/apt/lists/* # Clean up to keep image size small
