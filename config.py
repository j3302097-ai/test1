from dataclasses import dataclass
import torch

# 以类的方式定义参数
@dataclass
class Args:
    # training config

    # model config 
    image_size_train = 256
    image_size_test_single = 256
    image_size_test_multiple = 256
    num_secret = 4

    # optimer config
    lr = 1e-3
    lr_min = 1e-6
    warm_up_epoch = 0
    warm_up_lr_init = 1e-6

    # dataset
    DIV2K_path = '/home/gsjsun24/StegFormer-master/StegFormer-master/Data'
    single_batch_size = 12
    multi_batch_szie = 8 
    multi_batch_iteration = (num_secret+1)*8
    test_multi_batch_size = num_secret+1
    
    epochs = 6000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_freq = 10
    save_freq = 200
    train_next = 0
    use_model = 'MambaStegFormer-B'
    input_dim = 3
    
    norm_train = 'clamp'
    output_act = None
    path='/home/gsjsun24/StegFormer-master/StegFormer-master'
    model_name='MambaStegFormer-B_1baseline'
 
