docker run \
    -it --rm \
    --runtime=nvidia \
    -p 9999:9999 \
    --shm-size=32G \
    --volume /home/philipp/data_local:/workspace/data_local \
    --volume /home/philipp/Projects/adversarial_ml.research:/workspace/Projects/adversarial_ml.research \
    phibenz/adversarial_ml.research
