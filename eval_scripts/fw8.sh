export CUDA_VISIBLE_DEVICES=6


# for i in 0.5 0.1 0.05 0.01 0.005 0.001
# do
# for j in 1 3 5 10
# do
# for z in 0.3
# do
# echo "Running fw_merging using ties with max iters $i and step size $j and scaling factor $z"
# fusion_bench \
#     method=fw_merging/adamerging_ties method.max_iters=$j method.step_size=$i method.scaling_factor=$z \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA8/adamerging_ties/${i#0.}_${j}_${z#0.}
# done
# done 
# done

# for i in 0.1 0.05 0.01 0.005 0.001
# do
# for j in 1 3 5 10
# do
# for z in 0.3
# do
# echo "Running fw_merging using ties with max iters $i and step size $j and scaling factor $z"
# fusion_bench \
#     method=fw_merging/adamerging_task method.max_iters=$j method.step_size=$i method.scaling_factor=$z \
#     method.init_weight=outputs/logs/ViT-B-32/adamerging_TA8_500/version_0/outputs/clip-vit-base-patch32_TA8_task_wise_adamerging_weights.pt \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA8/adamerging_task_500/${i#0.}_${j}_${z#0.}
# done
# done 
# done

# for i in 0.5 0.1 0.05 0.01 0.005 0.001
# do
# for j in 10 7 5 3
# do
# echo "Running fw_merging using task_arithmetic with max iters $i and step size $j"
# fusion_bench \
#     method=fw_merging/task_arithmetic method.max_iters=$j method.step_size=$i \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA8/task/${i#0.}_$j
# done 
# done

# for i in 0.5 0.1 0.05 0.01 0.005 0.001
# do
# for j in 10 7 5 3
# do
# echo "Running fw_merging using ties with max iters $i and step size $j"
# fusion_bench \
#     method=fw_merging/ties method.max_iters=$j method.step_size=$i \
#     modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
#     taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
#     fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#     fabric.loggers.name=fw_merging_TA8/ties/${i#0.}_$j
# done 
# done

for i in 0.1 0.05 0.01 0.005 0.001
do
for j in 10 7 5 3
do
echo "Running fw_merging using task_arithmetic with max iters $i and step size $j"
fusion_bench \
    method=fw_merging/task_arithmetic method.max_iters=$j method.step_size=$i \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=fw_merging_TA4/task/${i#0.}_$j
done 
done