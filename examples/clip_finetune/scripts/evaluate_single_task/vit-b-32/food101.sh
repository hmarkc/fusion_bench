# evaluate pretrained
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    taskpool=clip-vit-classification_food101

# evaluate fine-tuned
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    modelpool.base_model=tanganke/clip-vit-base-patch32_food101 \
    taskpool=clip-vit-classification_food101
