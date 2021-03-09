import time
import os
from .options.test_options import TestOptions
from .data.data_loader import CreateDataLoader
from .models.models import create_model
from .util.visualizer import Visualizer
from pdb import set_trace as st
from .util import html
import uuid
import cv2

class Enlighten_GAN:
    def __init__(self):
        self.opt = TestOptions().parse()

        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.dataroot = './test_dataset'
        self.opt.name = 'enlightening'
        self.opt.model = 'single'
        self.opt.gpu_ids=-1
        self.opt.which_direction = 'AtoB'
        self.opt.no_dropout = True
        self.opt.dataset_mode = 'unaligned'
        self.opt.which_model_netG = 'sid_unet_resize'
        self.opt.skip = 1
        self.opt.use_norm = 1
        self.opt.use_wgan = 0
        self.opt.self_attention = True
        self.opt.times_residual = True
        self.opt.instance_norm = 0
        self.opt.resize_or_crop = 'no'
        self.opt.which_epoch = 200
        
        self.model = create_model(self.opt)


    def get_enlightened_image(self, image):
        img_name = str(uuid.uuid4())+'.png'
        img_name = os.path.join(self.opt.dataroot, 'testA', img_name)
        cv2.imwrite(img_name, image)

        print('Creating DataLoader ...')
        data_loader = CreateDataLoader(self.opt)
        print('Creating DataSet ...')
        dataset = data_loader.load_data()
        print('Creating Visualiser ...')
        visualizer = Visualizer(self.opt)
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
            os.remove(img_name)
            
            return image
