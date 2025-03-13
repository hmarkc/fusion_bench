export CUDA_VISIBLE_DEVICES=7

# fusion_bench \
#   method=adamerging \
#     method.name=clip_layer_wise_adamerging \
#     method.save_merging_weights=outputs/clip-vit-base-patch32_TA8_task_wise_adamerging_weights.pt \
#     method.max_steps=100 \
#     method.devices=[0,1,2,3] \
#   modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
#   taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
#   fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#   fabric.loggers.name=adamerging_TA8_100

# fusion_bench \
#   method=adamerging \
#     method.name=clip_layer_wise_adamerging \
#     method.save_merging_weights=outputs/clip-vit-base-patch32_TA8_task_wise_adamerging_weights.pt \
#     method.max_steps=500 \
#     method.devices=[4,5,6,7] \
#   modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
#   taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
#   fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#   fabric.loggers.name=adamerging_TA8_500

# fusion_bench \
#   method=adamerging \
#     method.name=clip_layer_wise_adamerging \
#     method.save_merging_weights=outputs/clip-vit-base-patch32_TA8_task_wise_adamerging_weights.pt \
#     method.max_steps=200 \
#     method.devices=[0,1,2,3] \
#     method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
#   modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20.yaml \
#   taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
#   fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
#   fabric.loggers.name=adamerging_TA4_200

fusion_bench \
  method=adamerging/layer_wise_roberta \
    method.name=flan_t5_layer_wise_adamerging \
    method.merging_weights_save_path=outputs/layer_wise_adamerging_weights.pt \
    method.max_steps=200 \
    method.tasks=[glue-qqp,glue-mnli,glue-qnli,glue-rte]\
  modelpool=SeqenceClassificationModelPool/roberta-base_glue.yaml


for modelpool in clip-vit-base-patch32_TA4 clip-vit-base-patch32_TA6 clip-vit-base-patch32_TA8 clip-vit-base-patch32_TALL10 clip-vit-base-patch32_TALL12 clip-vit-base-patch32_TALL14 clip-vit-base-patch32_TALL16 clip-vit-base-patch32_TALL18 clip-vit-base-patch32_TALL20 
do
echo "Running $modelpool"
fusion_bench \
  method=adamerging \
    method.name=clip_layer_wise_adamerging \
    method.save_merging_weights=outputs/clip-vit-base-patch32_TA8_task_wise_adamerging_weights.pt \
    method.max_steps=300 \
    method.devices=[0,1,2,3,4,5,6,7] \
    method.tasks=[sun397,stanford-cars,gtsrb,dtd] \
  modelpool=CLIPVisionModelPool/$modelpool \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA4 \
  fabric.loggers.root_dir=outputs/logs/scaling/adamerging_300 \
  fabric.loggers.name=$modelpool
done

# for i in 4 6 8 10 12 14 16 18 20
# do
# echo "Running $i"
# fusion_bench \
#   method=adamerging \
#     method.name=clip_layer_wise_adamerging \
#     method.save_merging_weights=outputs/clip-vit-base-patch32_TA8_task_wise_adamerging_weights.pt \
#     method.max_steps=200 \
#     method.devices=[1] \
#     method.tasks=[] \
#     method.max_num_models=$i \
#   modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20  \
#   taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
#   fabric.loggers.root_dir=outputs/logs/scaling2/adamerging \
#   fabric.loggers.name=model_$i
# done