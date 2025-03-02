# Taks Arithmetic
# fusion_bench \
#     method=task_arithmetic \
#         method.scaling_factor=0.3 \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8\
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=task_arithmetic

# fusion_bench \
#     method=task_arithmetic \
#         method.scaling_factor=0.3 \
#       modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=task_arithmetic_TA4

export CUDA_VISIBLE_DEVICES=1

# for modelpool in clip-vit-base-patch32_TALL10 clip-vit-base-patch32_TALL12 clip-vit-base-patch32_TALL16 clip-vit-base-patch32_TALL18 
# do
# echo "Running $modelpool"
# fusion_bench \
#     method=task_arithmetic \
#         method.scaling_factor=0.3 \
#     modelpool=CLIPVisionModelPool/$modelpool \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/scaling/task \
#     fabric.loggers.name=$modelpool
# done

for modelpool in clip-vit-base-patch32_TA4 clip-vit-base-patch32_TA6 clip-vit-base-patch32_TA8 clip-vit-base-patch32_TALL10 clip-vit-base-patch32_TALL12 clip-vit-base-patch32_TALL14 clip-vit-base-patch32_TALL16 clip-vit-base-patch32_TALL18 clip-vit-base-patch32_TALL20 
do
echo "Running $modelpool"
fusion_bench \
    method=task_arithmetic \
        method.scaling_factor=0.3 \
    modelpool=CLIPVisionModelPool/$modelpool \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    fabric.loggers.root_dir=outputs/logs/scaling2/task \
    fabric.loggers.name=$modelpool
done