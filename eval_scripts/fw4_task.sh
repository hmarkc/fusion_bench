export CUDA_VISIBLE_DEVICES=7

# for i in 0.1 0.05 0.01 0.005 0.001
# do
# for j in 10 7 5 3
# do
# echo "Running fw_merging using task_arithmetic with max iters $i and step size $j"
# fusion_bench \
#     method=fw_merging/task_arithmetic \
#         method.max_iters=$j \
#         method.step_size=$i \
#         method.init_weight=base \
#     method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA4/task/${i#0.}_$j
# done 
# done

# for j in 1 3 5 
# do
# for i in 0.007 0.006 0.004 0.003 0.002
# do
# echo "Running fw_merging using task_arithmetic with max iters $i and step size $j"
# fusion_bench \
#     method=fw_merging/task_arithmetic \
#         method.max_iters=$j \
#         method.step_size=$i \
#         method.init_weight=outputs/logs/ViT-B-32/adamerging_TA4_500/version_0/outputs/clip-vit-base-patch32_TA8_task_wise_adamerging_weights.pt  \
#     method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA4/adamerging_task/${i#0.}_$j
# done 
# done




# for j in 1 2 3 5 10
# do
# for i in 0.1 0.05 0.01 0.005 0.001
# do
# echo "Running fw_merging using task_arithmetic with max iters $j and step size $i"
# fusion_bench \
#     method=fw_merging/adamerging \
#         method.max_iters=$j method.step_size=$i \
#         method.init_weight=base method.merge_fn=task \
#     method.tasks=[gtsrb,sun397,stanford-cars,dtd] \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA4/task_multiple_model/${i#0.}_$j
# done 
# done   


fusion_bench \
    method=fw_merging/task_arithmetic \
        method.max_iters=3 \
        method.step_size=0.01 \
        method.init_weight=base \
        method.granularity=task \
    method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 

fusion_bench \
    method=fw_merging/adamerging \
        method.max_iters=5 method.step_size=0.1 \
        method.init_weight=base method.merge_fn=task method.granularity=task \
    method.tasks=[gtsrb,sun397,stanford-cars,dtd] \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 

for i in 4 6 8 10 12 14 16 18 20
do
for j in 0.1 0.05 0.01 0.005 0.001
do
for z in 10 7 5 3
do
echo "Running $i"
fusion_bench \
    method=fw_merging/adamerging \
        method.max_iters=$z method.step_size=$j \
        method.init_weight=base method.merge_fn=task \
    method.tasks=[] \
    method.max_num_models=$i \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    fabric.loggers.root_dir=outputs/logs/scaling2/fw_task \
    fabric.loggers.name=model_$i
done
done
done