# arcface_training


## Installation

1. Run Pytorch container pytorch:23.01

2. Clone the repo: 

        git clone https://github.com/deepinsight/insightface.git

3. Run these 3 steps

    https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#requirements

## Prepare Data

1. Use face detector and create cropped images of multiple persons

2. Follow this to create train files.
    
    https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/docs/prepare_custom_dataset.md


## Training

1. Download pretrained model. 

    https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215577&cid=4A83B6B633B029CC

Example: 'ms1mv3_arcface_r100_fp16/backbone.pth'

2. Create a copy file of 'configs/3million.py' and name it 'configs/cdot.py'.

3. Open configs/cdot.py 
and add changes as mentioned 

        config.network="r100"
        config.rec = "/path/to/data/directory"
        config.num_classes = TOTAL_NUMBER_OF_PERSON_IDS
        config.num_image = TOTAL_NUMBER_OF_IMAGES

        config.pretrained='/path/to/ms1mv3_arcface_r100_fp16/backbone.pth'


4. Add these lines in train.py

   https://github.com/deepinsight/insightface/issues/2210
   
       if cfg.pretrained != None:
            pretrained_dict = torch.load(cfg.pretrained)
            pretrained = OrderedDict()
            for k, v in pretrained_dict.items():
                if k[:7] != 'module.':
                    name = 'module.' + k  # add `module.`
                else:
                    name = k
                pretrained[name] = v
            model_dict = backbone.state_dict()
            # 1. filter out unnecessary keys
            new_state_dict={k:v if v.size()==model_dict[k].size()  else  model_dict[k] for k,v in zip(model_dict.keys(), pretrained.values())}
            print(new_state_dict.keys())
            backbone.load_state_dict(new_state_dict,strict=False)
            print("----------------------") 
  
5. Train

### On 1 GPU
        python train.py configs/cdot


### On 8 GPUs (parallel computing)
        torchrun --nproc_per_node=8 train.py configs/cdot

