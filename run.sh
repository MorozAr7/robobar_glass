sudo docker build -t glass_recognition:latest .

sudo docker run --rm -it --runtime nvidia \
    -v $(pwd):/app \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /etc/enctune.conf:/etc/enctune.conf \
    -v /etc/nv_tegra_release:/etc/nv_tegra_release \
    -v /tmp/nv_jetson_display:/tmp/nv_jetson_display \
    -v /dev/snd:/dev/snd \
    -e MALLOC_MMAP_THRESHOLD_=4096 \
    -e CUDNN_AUTO_TUNE=0 \
    glass_recognition:latest

