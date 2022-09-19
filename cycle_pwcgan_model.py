import torch
import math
import itertools
from util.image_pool import ImagePool
from .cycle_base_model import BaseModel
from . import cycle_networks
from .PWCNet import PWCDCNet_half


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, )

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = []
        visual_names_B = []


        self.netG_A = cycle_networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'pwcgan', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = cycle_networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'pwcgan', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.flow_decoder = PWCDCNet_half()

        self.netG_A.to(self.gpu_ids[0])
        self.netG_A = torch.nn.DataParallel(self.netG_A, self.gpu_ids) 
        self.netG_B.to(self.gpu_ids[0])
        self.netG_B = torch.nn.DataParallel(self.netG_B, self.gpu_ids)                
        self.flow_decoder.to(self.gpu_ids[0])
        self.flow_decoder = torch.nn.DataParallel(self.flow_decoder, self.gpu_ids)                  

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = cycle_networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = cycle_networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # initialize optimizers
            self.set_requires_grad([self.netG_A.module.encoder.conv4a,self.netG_A.module.encoder.conv4aa,self.netG_A.module.encoder.conv4b,\
                                    self.netG_A.module.encoder.conv5a,self.netG_A.module.encoder.conv5aa,self.netG_A.module.encoder.conv5b,\
                                    self.netG_A.module.encoder.conv6a,self.netG_A.module.encoder.conv6aa,self.netG_A.module.encoder.conv6b,\
                                    self.netG_B.module.encoder.conv4a,self.netG_B.module.encoder.conv4aa,self.netG_B.module.encoder.conv4b,\
                                    self.netG_B.module.encoder.conv5a,self.netG_B.module.encoder.conv5aa,self.netG_B.module.encoder.conv5b,\
                                    self.netG_B.module.encoder.conv6a,self.netG_B.module.encoder.conv6aa,self.netG_B.module.encoder.conv6b], requires_grad=False)      
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.netG_A.parameters()), \
                            filter(lambda p: p.requires_grad, self.netG_B.parameters())), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.set_requires_grad([self.netG_A, self.netG_B], requires_grad=True)               
            self.optimizer_pwc = torch.optim.Adam(itertools.chain(self.netG_A.module.encoder.parameters(), self.netG_B.module.encoder.parameters(), self.flow_decoder.parameters()),\
                                                lr=opt.lr, betas=(opt.beta1, 0.999))  

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),\
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_All = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.flow_decoder.parameters()),\
                                                lr=opt.lr, betas=(opt.beta1, 0.999))   
                                                          
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_All)
            self.optimizers.append(self.optimizer_pwc)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A1 = input['A1' if AtoB else 'B1']
        self.real_A2 = input['A2' if AtoB else 'B2']
        self.real_B1 = input['B1' if AtoB else 'A1']
        self.real_B2 = input['B2' if AtoB else 'A2']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B1, c2_real_A1, c3_real_A1, c4_real_A1, c5_real_A1, c6_real_A1 = self.netG_A(self.real_A1)
        self.fake_B2, c2_real_A2, c3_real_A2, c4_real_A2, c5_real_A2, c6_real_A2 = self.netG_A(self.real_A2)
        self.flow_real_A = self.flow_decoder(c2_real_A1, c2_real_A2, c3_real_A1, c3_real_A2,\
                    c4_real_A1, c4_real_A2, c5_real_A1, c5_real_A2, c6_real_A1, c6_real_A2)
        self.rec_A1, _, _, _, _, _ = self.netG_B(self.fake_B1)
        self.rec_A2, _, _, _, _, _ = self.netG_B(self.fake_B2)

        self.fake_A1, c2_real_B1, c3_real_B1, c4_real_B1, c5_real_B1, c6_real_B1 = self.netG_B(self.real_B1)
        self.fake_A2, c2_real_B2, c3_real_B2, c4_real_B2, c5_real_B2, c6_real_B2 = self.netG_B(self.real_B2)
        self.flow_real_B = self.flow_decoder(c2_real_B1, c2_real_B2, c3_real_B1, c3_real_B2,\
                    c4_real_B1, c4_real_B2, c5_real_B1, c5_real_B2, c6_real_B1, c6_real_B2)        
        self.rec_B1,  c2_fake_A1, c3_fake_A1, c4_fake_A1, c5_fake_A1, c6_fake_A1 = self.netG_A(self.fake_A1)
        self.rec_B2,  c2_fake_A2, c3_fake_A2, c4_fake_A2, c5_fake_A2, c6_fake_A2 = self.netG_A(self.fake_A2)
        self.flow_fake_A = self.flow_decoder(c2_fake_A1, c2_fake_A2, c3_fake_A1, c3_fake_A2,\
                    c4_fake_A1, c4_fake_A2, c5_fake_A1, c5_fake_A2, c6_fake_A1, c6_fake_A2) 

    def forward_flow_A(self,A1,A2):        
        _, c2_A1, c3_A1, c4_A1, c5_A1, c6_A1 = self.netG_A(A1)
        _, c2_A2, c3_A2, c4_A2, c5_A2, c6_A2 = self.netG_A(A2)
        flow_A = self.flow_decoder(c2_A1, c2_A2, c3_A1, c3_A2,\
                    c4_A1, c4_A2, c5_A1, c5_A2, c6_A1, c6_A2)
        return flow_A

    def forward_flow_B(self,B1,B2):        
        _, c2_B1, c3_B1, c4_B1, c5_B1, c6_B1 = self.netG_B(B1)
        _, c2_B2, c3_B2, c4_B2, c5_B2, c6_B2 = self.netG_B(B2)
        flow_B = self.flow_decoder(c2_B1, c2_B2, c3_B1, c3_B2,\
                    c4_B1, c4_B2, c5_B1, c5_B2, c6_B1, c6_B2)
        return flow_B
           
    def get_image(self, real_A, real_B):
        fake_B, _, _, _, _, _ = self.netG_A(real_A)
        fake_A, _, _, _, _, _ = self.netG_B(real_B)
        return fake_B, fake_A   

 
    def cuda(self, gpuid):
        self.netG_A = self.netG_A.cuda(gpuid)
        self.netG_B = self.netG_B.cuda(gpuid)
        self.netD_A = self.netD_A.cuda(gpuid)
        self.netD_B = self.netD_B.cuda(gpuid)
        self.criterionGAN = self.criterionGAN.cuda(gpuid)


def create_cycle_pwcgan(opt):
    instance = CycleGANModel()
    instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))                   
    return instance