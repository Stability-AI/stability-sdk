FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04
USER root
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    make build-essential libssl-dev zlib1g-dev mecab-ipadic-utf8 \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

USER vscode
WORKDIR /home/vscode
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="/home/vscode/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN pyenv install 3.8
RUN pyenv install 3.9
RUN pyenv install 3.10
RUN pyenv install 3.11
RUN pyenv global 3.8 3.9 3.10 3.11
RUN pip install --no-cache-dir --upgrade pip pipx
RUN pipx install --system-site-packages hatch