data=DB15K
num_epoch=2000
hidden_dim=1024
lr=5e-4
dim=256
num_layer_dec=2
cuda_device=4

CUDA_VISIBLE_DEVICES=${cuda_device} python train_ers.py --data ${data} --num_epoch ${num_epoch}\
                                                        --hidden_dim ${hidden_dim} --lr ${lr}\
                                                        --dim ${dim} --num_layer_dec ${num_layer_dec} 
                                                        