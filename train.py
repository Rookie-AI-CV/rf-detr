from rfdetr import RFDETRLarge

model = RFDETRLarge(
    pretrain_weights="/root/autodl-tmp/model/rf-detr-large.pth",
)

model.train(
    dataset_dir="/root/autodl-tmp/chengdu_v2_dataset", 
    epochs=105, 
    batch_size=3, 
    grad_accum_steps=5, 
    lr=1e-4, 
    output_dir="/root/autodl-tmp/output/0608",
    resolution=1008,
    num_classes=23,
    tensorboard=True,
)
