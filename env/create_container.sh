# Run container with mounted directories
sudo docker run -it --rm --name grouter-container \
     --gpus all \
     --ipc host \
     --workdir /workspace \
     -v $GROUTER_PATH:/workspace/Grouter \
     -e GROUTER_PATH=/workspace/Grouter \
     -p 1028:1028 \
     grouter:latest


