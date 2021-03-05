import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataroot = './test_dataset'
opt.name = 'enlightening'
opt.model = 'single'
opt.which_direction = 'AtoB'
opt.no_dropout = True
opt.dataset_mode = 'unaligned'
opt.which_model_netG = 'sid_unet_resize'
opt.skip = 1
opt.use_norm = 1
opt.use_wgan = 0
opt.self_attention = True
opt.times_residual = True
opt.instance_norm = 0
opt.resize_or_crop = 'no'
opt.which_epoch = 200

# for i in range(1):
# 		os.system("python predict.py \
# 			--dataroot ./test_dataset \
# 			--name enlightening \
# 			--model single \
# 			--which_direction AtoB \
# 			--no_dropout \
# 			--dataset_mode unaligned \
# 			--which_model_netG sid_unet_resize \
# 			--skip 1 \
# 			--use_norm 1 \
# 			--use_wgan 0 \
# 			--self_attention \
# 			--times_residual \
# 			--instance_norm 0 --resize_or_crop='no'\
# 			--which_epoch " + str(200 - i*5))

model = create_model(opt)

def get_enlightened_image():
    print('Creating DataLoader ...')
    data_loader = CreateDataLoader(opt)
    print('Creating DataSet ...')
    dataset = data_loader.load_data()
    print('Creating Visualiser ...')
    visualizer = Visualizer(opt)
    # create website
    # web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    print("Length of  Dataset "+str(len(dataset)))
    print(dataset)
    for i, data in enumerate(dataset):
        print("Setting Inputs")
        model.set_input(data)
        print("Predicting from Model")
        visuals = model.predict()
        print("Saving Visualiser")
        image = visualizer.save_images(visuals)
        
        return image
image = get_enlightened_image()
print(image)
image.save('new_image.png')
# webpage.save()
