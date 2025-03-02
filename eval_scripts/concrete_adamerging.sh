fusion_bench \
    method=concrete_subspace/clip_concrete_layer_wise_adamerging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8\
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=concrete_adamerging_TA8