export CUDA_VISIBLE_DEVICES=7

for i in 4 6 8 10 12 14 16 18 20
do
for j in 0.1 0.05 0.005
do
for z in 10 5
do
echo "Running $i"
fusion_bench \
    method=fw_merging/task_arithmetic \
        method.max_iters=$z method.step_size=$j \
        method.init_weight=base \
    method.tasks=[] \
    method.max_num_models=$i \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    fabric.loggers.root_dir=outputs/logs/scaling2/fw_task \
    fabric.loggers.name=model_$i
done
done
done