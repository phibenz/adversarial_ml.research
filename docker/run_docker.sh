docker run \
    -it --rm \
    --runtime=nvidia \
    -p 8888:8888 \
    --shm-size=32G \
    --volume /home/philipp/data_local:/workspace/data_local \
    --volume /home/philipp/Projects/research-collections:/workspace/Projects/research-collections \
    --volume /media/philipp/ssd2_4tb:/workspace/ssd2_4tb \
    phibenz/research-collections
