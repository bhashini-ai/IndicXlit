FROM nvcr.io/nvidia/tritonserver:22.12-py3

# Install IndicXlit dependencies
RUN pip install git+https://github.com/anoopkunchukuttan/indic_nlp_library
RUN apt update && apt install -y build-essential python3-dev
RUN pip install fairseq
RUN pip install tensorboardX

# Copy Triton Server models for IndicXlit
ENV MODEL_REPO=/model_repo
WORKDIR $MODEL_REPO/indic_xlit
COPY ./triton_deployment/indic_xlit .
COPY ./python/custom_interactive.py ./1/
COPY ./python/lang_list.txt ./1/

WORKDIR /
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
CMD tritonserver --model-repository=$MODEL_REPO
