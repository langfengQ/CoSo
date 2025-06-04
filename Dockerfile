# FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

ENV HOME=/root

ENV ANDROID_HOME=$HOME/.android

ENV DEBIAN_FRONTEND=noninteractive
ENV SDK=$ANDROID_HOME
ENV ANDROID_SDK_ROOT=$ANDROID_HOME
ENV PATH=$SDK/emulator:$SDK/tools:$SDK/tools/bin:$SDK/platform-tools:$PATH
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    git \
    screen \
    qemu-kvm \
    libvirt-daemon-system \
    libvirt-clients \
    bridge-utils \
    openjdk-8-jdk \
    cpu-checker \
    fonts-dejavu \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# JDK 8
RUN wget https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u412-b08/openlogic-openjdk-8u412-b08-linux-x64-deb.deb \
    && apt-get install -y ./openlogic-openjdk-8u412-b08-linux-x64-deb.deb \
    && rm -rf openlogic-openjdk-8u412-b08-linux-x64-deb.deb
RUN update-alternatives --config java

# SDK Manager
RUN wget https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip -O /tmp/sdk-tools-linux.zip \
    && mkdir -p $ANDROID_HOME \
    && unzip /tmp/sdk-tools-linux.zip -d $ANDROID_HOME \
    && rm -rf /tmp/sdk-tools-linux.zip

# SDK Emulator
RUN yes | sdkmanager "platform-tools" "platforms;android-28" "emulator" \
    && yes | sdkmanager "system-images;android-28;google_apis;x86_64"\
    && yes | sdkmanager "build-tools;28.0.0"

# Conda and Dependencies
ENV PATH=/opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -rf /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y

RUN git config --global url."https://".insteadOf git://
RUN conda create -n coso python==3.10 -y
ENV PATH=/opt/conda/envs/coso/bin:$PATH
COPY ./requirements.txt /opt/requirements.txt
RUN /opt/conda/envs/coso/bin/pip install -r /opt/requirements.txt --verbose 
RUN /opt/conda/envs/coso/bin/pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121 --verbose
RUN /opt/conda/envs/coso/bin/pip cache purge

# AVD Initialization
RUN cd $ANDROID_HOME && mkdir avd
RUN /opt/conda/envs/coso/bin/pip install gdown \
    && /opt/conda/envs/coso/bin/gdown --folder https://drive.google.com/drive/folders/1ZGKrWiSoGqg8_NoIGT7rWmiZ8CXToaBF \
    && unzip digirl_device/test_Android.zip -d $ANDROID_HOME/avd \
    && rm -rf digirl_device/

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs

# Install Appium
RUN npm install -g appium \
    && appium driver install uiautomator2

RUN /opt/conda/envs/coso/bin/pip install Appium-Python-Client --verbose

# Download Checkpoints
RUN wget https://huggingface.co/cooelf/Auto-UI/resolve/main/Auto-UI-Base.zip \
    && unzip Auto-UI-Base.zip -d $HOME/Auto-UI-Base \
    && mv $HOME/Auto-UI-Base/Auto-UI-Base/* $HOME/Auto-UI-Base/


# Download Pre-Collected Trajectories
RUN /opt/conda/envs/coso/bin/gdown --folder https://drive.google.com/drive/folders/1ud1XyzCfh0257CixxdgLjjpX59jYbhfU \
    && mv digirl_data_release $HOME/data