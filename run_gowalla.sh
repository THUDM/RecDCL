# baseline
## Gowalla
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 --embedding_size 32 > ./logs/Gowalla/lightgcn_32_1101.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --train_batch_size=1024 --embedding_size 32 > ./logs/Gowalla/lightgcn_32_1101.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --train_batch_size=1024 --embedding_size 64 > ./logs/Gowalla/lightgcn_64_1101.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 --embedding_size 128 > ./logs/Gowalla/lightgcn_128_1031.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --train_batch_size=1024 --embedding_size 128 > ./logs/Gowalla/lightgcn_128_1101.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --train_batch_size=1024 --embedding_size 256 > ./logs/Gowalla/lightgcn_256_1101.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --train_batch_size=1024 --embedding_size 512 > ./logs/Gowalla/lightgcn_512_1101.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --train_batch_size=1024 --embedding_size 1024 > ./logs/Gowalla/lightgcn_1024_1101.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --train_batch_size=1024 --embedding_size 2048 > ./logs/Gowalla/lightgcn_2048_1101.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --train_batch_size=1024 --embedding_size 4096 > ./logs/Gowalla/lightgcn_4096_1102.log 2>&1 &

nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 > ./logs/Gowalla/lightgcn_256_1031.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 > ./logs/Gowalla/lightgcn_512_1031.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 > ./logs/Gowalla/lightgcn_1024_1031.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 > ./logs/Gowalla/lightgcn_2048_1031.log 2>&1 &
nohup python run_recbole.py --model=LightGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 > ./logs/Gowalla/lightgcn_4096_1031.log 2>&1 &

nohup python run_recbole.py --model=LGCN --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 --embedding_size 64 > ./logs/Gowalla/directau_newlgcn_layer2_emb64_1101.log 2>&1 &

# DirectAU
## Gowalla
nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --gamma=5 --embedding_size 32 > ./logs/Gowalla/directau_lgcn_layer2_emb32_1101.log 2>&1 &
nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --gamma=5 --embedding_size 64 > ./logs/Gowalla/directau_lgcn_layer2_emb64_1101.log 2>&1 &
nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --gamma=5 --embedding_size 128 > ./logs/Gowalla/directau_lgcn_layer2_emb128_1101.log 2>&1 &
nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --gamma=5 --embedding_size 256 > ./logs/Gowalla/directau_lgcn_layer2_emb256_1101.log 2>&1 &
nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --gamma=5 --embedding_size 512 > ./logs/Gowalla/directau_lgcn_layer2_emb512_1101.log 2>&1 &
nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --gamma=5 --embedding_size 1024 > ./logs/Gowalla/directau_lgcn_layer2_emb1024_1101.log 2>&1 &
nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --gamma=5 --embedding_size 2048 > ./logs/Gowalla/directau_lgcn_layer2_emb2048_1101.log 2>&1 &


# BT4Rec
## Gowalla
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --train_batch_size=1024 --embedding_size 32 > ./logs/Gowalla/bt_lgcn_layer2_emb32_1101.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --embedding_size 32 > ./logs/Gowalla/bt_lgcn_layer2_emb32_1101.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --encoder=LightGCN --train_batch_size=1024 --embedding_size 64 > ./logs/Gowalla/bt_lgcn_layer2_emb64_1101.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 64 > ./logs/Gowalla/bt_lgcn_layer2_emb64_wd0_1101.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 32 > ./logs/Gowalla/bt_lgcn_layer2_emb32_wd0_1102.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 128 > ./logs/Gowalla/bt_lgcn_layer2_emb128_wd0_1102.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 256 > ./logs/Gowalla/bt_lgcn_layer2_emb256_wd0_1102.log 2>&1 &

nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.025 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.025_1103.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.05 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.05_1103.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.04 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.04_1103.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.075 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.075_1103.log 2>&1 &

nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.1 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.1_1103.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.001 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.001_1103.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.007 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.007_1104.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.06 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.06_1104.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.055 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.055_1105.log 2>&1 &

nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.05_1103.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.5 > ./logs/Gowalla/bt_lgcn_layer2_emb406_wd0_reg0.5_1103.log 2>&1 &

nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 2048 > ./logs/Gowalla/bt_lgcn_layer2_emb2048_wd0_reg0.05_1106.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 1024 > ./logs/Gowalla/bt_lgcn_layer2_emb1024_wd0_reg0.05_1106.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 512 > ./logs/Gowalla/bt_lgcn_layer2_emb512_wd0_reg0.05_1106.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 256 > ./logs/Gowalla/bt_lgcn_layer2_emb256_wd0_reg0.05_1106.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 128 > ./logs/Gowalla/bt_lgcn_layer2_emb128_wd0_reg0.05_1106.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 64 --gpu_id 4 > ./logs/Gowalla/bt_lgcn_layer2_emb64_wd0_reg0.05_1106.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 32 --gpu_id 8 > ./logs/Gowalla/bt_lgcn_layer2_emb32_wd0_reg0.05_1106.log 2>&1 &

nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.05 > ./logs/Gowalla/bt_lgcn_layer2_emb4096_wd0_reg0.05_uniform_1106.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 256 --reg_weight 0.05 > ./logs/Gowalla/bt_lgcn_layer2_emb256_wd0_reg0.05_gamma5_uniform_1106.log 2>&1 &

nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 64 --reg_weight 0.05 > ./logs/Gowalla/bt_lgcn_layer2_emb64_wd0_reg0.05_gamma5_uniform_1108.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 64 --reg_weight 0.05 > ./logs/Gowalla/bt_lgcn_layer2_emb64_wd0_reg0.05_gamma5_nobn_onlyon_uniform_1108.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 64 --reg_weight 0.05 > ./logs/Gowalla/bt_lgcn_layer2_emb64_wd0_reg0.05_gamma5_nobn_uniform_1108.log 2>&1 &

nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 64 --reg_weight 0.05 > ./logs/Gowalla/bt_lgcn_layer2_emb64_wd0_reg0.05_sample_1108.log 2>&1 &
nohup python run_recbole.py --model=BT4Rec --dataset=Gowalla --learning_rate=1e-3 --weight_decay=0 --encoder=LightGCN --train_batch_size=1024 --embedding_size 4096 --reg_weight 0.05 > ./logs/Gowalla/bt_lgcn_layer2_emb4096_wd0_reg0.05_sample_1108.log 2>&1 &
