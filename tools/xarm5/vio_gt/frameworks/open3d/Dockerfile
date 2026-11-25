# Open3D (odometría RGB-D densa, término híbrido fotométrico + geométrico) en una imagen aparte.
#   docker build -t visuomotor_open3d:0.18 tools/xarm5/vio_gt/frameworks/open3d
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libgomp1 libglib2.0-0 libx11-6 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir open3d==0.18.0 numpy==1.26.4
