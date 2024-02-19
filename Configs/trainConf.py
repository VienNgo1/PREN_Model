from easydict import EasyDict as edict

configs = edict()

# ---- training
configs['image_dir'] = "D:/BigDataset/TextRecognition/TestData_Irregullar/combine/train"
configs['train_list'] = "D:/BigDataset/TextRecognition/TestData_Irregullar/combine/train_label.txt"
configs['savedir'] = 'models/COCO_OCR'
configs['imgH'] = 64
configs['imgW'] = 256

# configs['alphabet'] = 'data/alphabet_en.txt'
configs['alphabet'] = "data/ascii.txt"

f = open(configs.alphabet, 'r')
l = f.readline().rstrip()
f.close()
configs['n_class'] = len(l) + 3  # pad, unk, eos

configs['device'] = 'cuda'
configs['random_seed'] = 1
configs['batchsize'] = 8 #32
configs['workers'] = 8

configs['n_epochs'] = 1
configs['lr'] = 0.05
configs['lr_milestones'] = [2, 5, 7]
configs['lr_gammas'] = [0.2, 0.1, 0.1]
configs['weight_decay'] = 0.

configs['aug_prob'] = 0.3
configs['continue_train'] = True
configs['continue_path'] = 'models/COCO_OCR/002_ascii_lr005/m_epoch1.pth'
configs['modify_model'] = False
configs['displayInterval'] = 100

# ---- model
configs['net'] = edict()

configs.net['n_class'] = configs.n_class
configs.net['max_len'] = 25
configs.net['n_r'] = 5  # number of primitive representations
configs.net['d_model'] = 384
configs.net['dropout'] = 0.1
