FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

RUN apt-get update && apt-get full-upgrade -y && apt-get autoremove -y && apt-get autoclean -y

RUN wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.deb
RUN echo "826383cc7d10f6edd6b37acc499c306469ff42d113acb3c2411a422aeaddb2e8  jdk-21_linux-x64_bin.deb" | sha256sum -c -

RUN apt-get install ./jdk-21_linux-x64_bin.deb -y
RUN rm -f jdk-21_linux-x64_bin.deb

RUN JAVA_HOME_PATH=$(ls -d /usr/lib/jvm/jdk-*-oracle-x64) && \
    echo "export JAVA_HOME=$JAVA_HOME_PATH" >> /etc/profile.d/java_home.sh