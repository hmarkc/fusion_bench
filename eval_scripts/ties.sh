# fusion_bench method=ties_merging method.scaling_factor=0.3 method.threshold=20 \
#   modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
#   taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8\
#   fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#   fabric.loggers.name=ties

# fusion_bench method=ties_merging method.scaling_factor=0.3 method.threshold=20 \
#   modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#   taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#   fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#   fabric.loggers.name=ties_TA4

# export CUDA_VISIBLE_DEVICES=1

# for modelpool in clip-vit-base-patch32_TALL10 clip-vit-base-patch32_TALL12 clip-vit-base-patch32_TALL16 clip-vit-base-patch32_TALL18 
# do
# echo "Running $modelpool"
# fusion_bench method=ties_merging method.scaling_factor=0.3 method.threshold=20 \
#     modelpool=CLIPVisionModelPool/$modelpool \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/scaling/ties \
#     fabric.loggers.name=$modelpool
# done

export CUDA_VISIBLE_DEVICES=1

for modelpool in clip-vit-base-patch32_TA4 clip-vit-base-patch32_TA6 clip-vit-base-patch32_TA8 clip-vit-base-patch32_TALL10 clip-vit-base-patch32_TALL12 clip-vit-base-patch32_TALL14 clip-vit-base-patch32_TALL16 clip-vit-base-patch32_TALL18 clip-vit-base-patch32_TALL20
do
echo "Running $modelpool"
fusion_bench method=ties_merging method.scaling_factor=0.3 method.threshold=20 \
    modelpool=CLIPVisionModelPool/$modelpool \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    fabric.loggers.root_dir=outputs/logs/scaling2/ties \
    fabric.loggers.name=$modelpool
done