# DPVO (monocular, sin escala) y DROID-SLAM (RGB-D) de Princeton VL, para correr OFFLINE sobre los frames exportados.
# Imagen aparte con GPU: no toca las imágenes ni los contenedores del proyecto.
#   docker build -t visuomotor_learned_vo:v1 tools/xarm5/vio_gt/frameworks/learned
# TORCH_CUDA_ARCH_LIST=8.9 (RTX 4060, Ada): durante el build no hay GPU visible. MAX_JOBS limita la RAM de nvcc.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=4 FORCE_CUDA=1
RUN apt-get update && apt-get install -y --no-install-recommends git wget unzip ca-certificates libgl1 libglib2.0-0 \
      build-essential ninja-build && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy<2" opencv-python-headless==4.9.0.80 scipy tqdm einops numba yacs gdown \
      plyfile kornia pypose evo matplotlib \
    && pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
WORKDIR /opt
# ---- DPVO ----
RUN git clone --recursive https://github.com/princeton-vl/DPVO.git && cd DPVO \
    && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip -q eigen-3.4.0.zip -d thirdparty \
    && pip install --no-cache-dir --no-build-isolation .
# Enlace de download_models_and_data.sh de DPVO (el de Dropbox /s/... ya no sirve: devuelve HTML)
RUN cd /opt/DPVO && wget -q -O models.zip \
      "https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t&dl=1" \
    && unzip -q models.zip && rm models.zip && ls -l /opt/DPVO/dpvo.pth
# ---- DROID-SLAM ----
RUN git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git && cd DROID-SLAM \
    && (pip install --no-cache-dir --no-build-isolation thirdparty/lietorch || true) \
    && pip install --no-cache-dir --no-build-isolation .
# torch primero: las extensiones enlazan libc10.so de torch
RUN python -c "import torch, lietorch, droid_backends; print('droid OK')"
RUN cd /opt/DROID-SLAM && wget -q -O droid.pth \
      "https://drive.google.com/uc?export=download&id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh" && ls -l droid.pth
