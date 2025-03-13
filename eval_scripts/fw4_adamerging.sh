export CUDA_VISIBLE_DEVICES=2

for j in 1
do
for i in 500
do
for z in 1e-2
do
fusion_bench \
    method=fw_merging/adamerging \
        method.max_iters=$j method.ada_iters=$i method.ada_coeff=$z \
        method.init_weight=base method.granularity=layer  \
    method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=fw_merging_TA4/adamerging_cross_entropy/${i}_${j}_${z#1e-}
done
done 
done  


# for j in 10 15 20
# do
# for i in 100
# do
# for z in 1e-2 
# do
# fusion_bench \
#     method=fw_merging/adamerging \
#         method.max_iters=$j method.ada_iters=$i method.ada_coeff=$z \
#         method.init_weight=base  \
#     method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA4/adamerging_cross_entropy/${i}_${j}_${z#1e-}
# done
# done 
# done  

# for j in 1 2 3 5
# do
# for i in 500 200 100
# do
# for z in 3e-1 0 1e-8 1e-4 1e-2 1e-1
# do
# echo "Running fw_merging using adamerging with max iters $j and ada iters $i and ada_coeff $z"
# fusion_bench \
#     method=fw_merging/adamerging \
#         method.max_iters=$j method.ada_iters=$i method.ada_coeff=$z \
#         method.init_weight=base  \
#     method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA4/adamerging/${i}_${j}_${z#1e-}
# done
# done 
# done   


# for modelpool in clip-vit-base-patch32_TALL20
# # clip-vit-base-patch32_TA4 clip-vit-base-patch32_TA6 clip-vit-base-patch32_TA8 clip-vit-base-patch32_TALL10 clip-vit-base-patch32_TALL12 clip-vit-base-patch32_TALL14 clip-vit-base-patch32_TALL16 clip-vit-base-patch32_TALL18 
# do
# echo "Running $modelpool"
# fusion_bench \
#     method=fw_merging/adamerging \
#         method.max_iters=3 method.ada_iters=100 method.ada_coeff=1e-2 \
#         method.init_weight=base  \
#     method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
#     modelpool=CLIPVisionModelPool/$modelpool \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/scaling/fw_adamerging_3_100 \
#     fabric.loggers.name=$modelpool
# done

# for i in 4 6 8 10 12 14 16 18 20
# do
# echo "Running $i"
# fusion_bench \
#     method=fw_merging/adamerging \
#         method.max_iters=2 method.ada_iters=100 method.ada_coeff=1e-8 \
#         method.init_weight=base  \
#     method.tasks=[] \
#     method.max_num_models=$i \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
#     fabric.loggers.root_dir=outputs/logs/scaling2/fw_adamerging_2_100 \
#     fabric.loggers.name=model_$i
# done

# fusion_bench \
#     method=fw_merging/adamerging \
#         method.max_iters=5 method.ada_iters=500 method.ada_coeff=1e-8 \
#         method.init_weight=base  \
#     method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA4/single_dataset_adamerging    