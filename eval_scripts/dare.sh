# for modelpool in clip-vit-base-patch32_TA4 clip-vit-base-patch32_TA6 clip-vit-base-patch32_TA8 clip-vit-base-patch32_TALL10 clip-vit-base-patch32_TALL12 clip-vit-base-patch32_TALL14 clip-vit-base-patch32_TALL16 clip-vit-base-patch32_TALL18 clip-vit-base-patch32_TALL20
# do
# echo "Running $modelpool"
# fusion_bench \
#     method=dare/ties_merging \
#     modelpool=CLIPVisionModelPool/$modelpool \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/scaling/dare_ties \
#     fabric.loggers.name=$modelpool
# done

export CUDA_VISIBLE_DEVICES=1

for modelpool in clip-vit-base-patch32_TA4 clip-vit-base-patch32_TA6 clip-vit-base-patch32_TA8 clip-vit-base-patch32_TALL10 clip-vit-base-patch32_TALL12 clip-vit-base-patch32_TALL14 clip-vit-base-patch32_TALL16 clip-vit-base-patch32_TALL18 clip-vit-base-patch32_TALL20
do
echo "Running $modelpool"
fusion_bench \
    method=dare/ties_merging \
    modelpool=CLIPVisionModelPool/$modelpool \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    fabric.loggers.root_dir=outputs/logs/scaling2/dare_ties \
    fabric.loggers.name=$modelpool
done
