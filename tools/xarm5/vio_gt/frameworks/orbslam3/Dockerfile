# ORB-SLAM3 (UZ-SLAMLab, v1.0) para correr OFFLINE sobre los frames exportados (export_bag_frames.py).
# Imagen aparte: no toca las imágenes ni los contenedores del proyecto.
#   docker build -t visuomotor_orbslam3:v1 tools/xarm5/vio_gt/frameworks/orbslam3
# Ubuntu 20.04: Pangolin v0.6 y ORB-SLAM3 compilan sin parches de C++17 (OpenCV 4.2 del sistema; el CMakeLists pide
# 4.4 y se baja a 4.2, la API usada es la misma).
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates pkg-config wget unzip \
      libeigen3-dev libopencv-dev libglew-dev libgl1-mesa-dev libboost-serialization-dev libboost-filesystem-dev \
      libssl-dev libpython3-dev python3 python3-numpy libepoxy-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git \
    && cmake -S Pangolin -B Pangolin/build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
    && cmake --build Pangolin/build -j6 && cmake --install Pangolin/build && rm -rf Pangolin
RUN git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git && cd ORB_SLAM3 && git checkout 4452a3c \
    && sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4.2)/' CMakeLists.txt \
    && sed -i 's/++11/++14/g' CMakeLists.txt \
    && cd Thirdparty/DBoW2 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j6 \
    && cd ../../g2o && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j6 \
    && cd ../../Sophus && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF && make -j6 \
    && cd /opt/ORB_SLAM3/Vocabulary && tar -xf ORBvoc.txt.tar.gz \
    && cd /opt/ORB_SLAM3 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j6 ORB_SLAM3
COPY vio_runner.cc /opt/runner/vio_runner.cc
COPY CMakeLists.runner.txt /opt/runner/CMakeLists.txt
RUN cmake -S /opt/runner -B /opt/runner/build -DCMAKE_BUILD_TYPE=Release && cmake --build /opt/runner/build -j6
ENV LD_LIBRARY_PATH=/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib:/usr/local/lib
