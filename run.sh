source ~/.bashrc && source ~/anaconda3/bin/activate && conda activate py391
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
# cd tools && python -m pcdet.datasets.kl.kl_dataset --cfg_file cfgs/dataset_configs/kl_dataset.yaml && cd ..
# cd tools && CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python -u -m torch.distributed.launch --nproc_per_node=7 train.py --launcher pytorch --cfg_file=cfgs/kl_models/pointpillar.yaml --merge_all_iters_to_one_epoch  --use_amp &> vehicle.txt &
# cd tools && CUDA_VISIBLE_DEVICES=4,5,6,7  python -m torch.distributed.launch --nproc_per_node=4 test_pointpillar_kl.py --launcher pytorch --cfg_file=cfgs/kl_models/pointpillar.yaml --ckpt /home/chenxu/code/OpenPCDet/output/kl_models/pointpillar/default/ckpt/checkpoint_epoch_200.pth
# cd tools && CUDA_VISIBLE_DEVICES=4,5,6,7  python -m torch.distributed.launch --nproc_per_node=4 test_pointpillar_kl.py --launcher pytorch --cfg_file=/home/chenxu/code/OpenPCDet/output/kl_models/pointpillar/default_0425/pointpillar.yaml --ckpt /home/chenxu/code/OpenPCDet/output/kl_models/pointpillar/default_0425/ckpt/checkpoint_epoch_200.pth
cd tools && CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python -u -m torch.distributed.launch --nproc_per_node=7 train.py --launcher pytorch --cfg_file=cfgs/truckscene_models/pointpillar.yaml --merge_all_iters_to_one_epoch  --use_amp &> vehicle.txt &
# cd tools && CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5678 --wait-for-client -m torch.distributed.launch --nproc_per_node=1 train.py --launcher pytorch --cfg_file=cfgs/kl_models/pointpillar.yaml --merge_all_iters_to_one_epoch  --use_amp
