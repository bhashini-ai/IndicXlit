#!/usr/bin/env bash
docker run -d --restart always --name "indicxlint-triron-server" --gpus all -p 9005:8001 -v /home/models:/models --shm-size=2g indicxlint-triton-server:22.12
