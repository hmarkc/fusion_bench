export CUDA_VISIBLE_DEVICES=6
# fusion_bench \
#     method=simple_average \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=simple_average_TA4

# fusion_bench \
#     method=fisher_merging/clip_fisher_merging \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fisher_TA4

# fusion_bench \
#     method=regmean/clip_regmean \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=regmean_TA4

fusion_bench \
    method=surgery/adamerging_surgery \
        method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=surgery_TA4

fusion_bench \
    method=concrete_subspace/clip_concrete_layer_wise_adamerging.yaml \
        method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=concrete_TA4


