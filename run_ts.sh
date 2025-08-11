source ~/.bashrc && source ~/anaconda3/bin/activate && conda activate py391
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
cd tools && python -m pcdet.datasets.truckscenes.truckscenes_dataset --cfg_file cfgs/dataset_configs/truck_scenes_dataset.yaml && cd ..
cd tools && CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python -u -m torch.distributed.launch --nproc_per_node=7 train.py --launcher pytorch --cfg_file=cfgs/truckscene_models/pointpillar.yaml --merge_all_iters_to_one_epoch  --use_amp &> vehicle.txt &
