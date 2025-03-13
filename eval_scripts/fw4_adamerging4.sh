export CUDA_VISIBLE_DEVICES=0

for j in 15
do
for i in 500
do
for z in 1e-2
do
fusion_bench \
    method=fw_merging/adamerging \
        method.max_iters=$j method.ada_iters=$i method.ada_coeff=$z \
        method.init_weight=base  \
    method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=fw_merging_TA4/adamerging_cross_entropy/${i}_${j}_${z#1e-}
done
done 
done  
