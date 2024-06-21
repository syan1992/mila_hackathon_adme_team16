import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import math
import warnings
import argparse
from copy import deepcopy
from typing import Set, Callable, Any

import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import to_dense_batch
import tensorboard_logger as tb_logger
from torch.nn.utils.rnn import pad_sequence

from models.deepgcn import SupConDeeperGCN
from models.smiles_bert import SMILESBert
from uti.evaluate import Evaluator
from uti.load_dataset import PygOurDataset
from uti.util import AverageMeter, adjust_learning_rate, set_optimizer, save_model, calmean
from loss.loss_scl_cls import SupConLossCls
from loss.loss_scl_reg import SupConLossReg
from loss.loss_cl import ConLossCls
from loss.rncloss import RnCLoss 
#from transformers.optimization import get_linear_schedule_with_warmup
from transformers import optimization
from unimol_tools import UniMolRepr

from mmvae import VAE

import pandas as pd
from torch_geometric.utils import to_dense_batch
import pdb
warnings.filterwarnings("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from gmoe import DeeperGCN, VirtualNodeGNN

def parse_option():
    """Parse arguments."""

    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--classification", action="store_true", help="classification task")

    parser.add_argument("--wscl", type=float, default=1, help="weight of scl")
    parser.add_argument("--wrecon", type=float, default=1, help="weight of recon")
    parser.add_argument("--wdiff", type=float, default=1, help="weight of recon")

    parser.add_argument("--global_feature", action="store_true", help="with global feature")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--lr_decay_epochs", type=str, default="1000", help="where to decay lr, can be a list"
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    parser.add_argument("--model", type=str, default="DeeperGCN")
    parser.add_argument("--dataset", type=str, default="freesolv", help="dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="path to custom dataset")
    parser.add_argument("--num_tasks", type=int, default=1, help="parameter for task number")

    parser.add_argument("--temp", type=float, default=0.07, help="temperature for loss function")
    parser.add_argument("--gamma1", type=float, default=2)
    parser.add_argument("--gamma2", type=float, default=2)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--num_gc_layers", type=int, default=3)

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--syncBN", action="store_true", help="using synchronized batch normalization"
    )
    parser.add_argument("--warm", action="store_true", help="warm-up for large batch training")
    parser.add_argument("--trial", type=str, default="0", help="id for recording multiple runs")

    opt = parser.parse_args()

    opt.model_path = "./save/SupCon/{}_models".format(opt.dataset)
    opt.tb_path = "./save/SupCon/{}_tensorboard".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.classification:
        opt.model_name = (
            "SupCon_{}_lr_{}_bsz_{}_trial_{}_mlp_{}_wscl_{}_wrecon_{}_wdiff_{}_decay_{}_rate_{}".format(
                opt.model,
                opt.learning_rate,
                opt.batch_size,
                opt.trial,
                opt.mlp_layers,
                opt.wscl,
                opt.wrecon,
                opt.wdiff,
                opt.lr_decay_epochs,
                opt.lr_decay_rate,
            )
        )
    else:
        opt.model_name = "SupCon_{}_lr_{}_bsz_{}_trial_{}_gamma1_{}_gamma2_{}_mlp_{}_wscl_{}_wrecon_{}_decay_{}_rate_{}".format(
            opt.model,
            opt.learning_rate,
            opt.batch_size,
            opt.trial,
            opt.gamma1,
            opt.gamma2,
            opt.mlp_layers,
            opt.wscl,
            opt.wrecon,
            opt.lr_decay_epochs,
            opt.lr_decay_rate,
        )

    if opt.cosine:
        opt.model_name = "{}_cosine".format(opt.model_name)

    if opt.batch_size > 1024:
        opt.warm = True
    if opt.warm:
        opt.model_name = "{}_warm".format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 100
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = (
                eta_min
                + (opt.learning_rate - eta_min)
                * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                / 2
            )
        else:
            opt.warmup_to = opt.learning_rate_gcn

    return opt


opt = parse_option()



def set_loader(opt: Any, dataname: str) -> Set[Data]:
    """Load dataset from opt.datas_dir.

    Args:
        opt (Any): Parsed arguments.
        dataname (str): The folder name of the dataset.

    Returns:
        Set[Data]: train/validation/test sets.
    """

    train_dataset = PygOurDataset(root=opt.data_dir, phase="train", dataname=dataname)
    test_dataset = PygOurDataset(root=opt.data_dir, phase="test", dataname=dataname)
    val_dataset = PygOurDataset(root=opt.data_dir, phase="valid", dataname=dataname)

    return train_dataset, test_dataset, val_dataset


class TeacherModel(torch.nn.Module):
    def __init__(self, model_2: Module, dim_feat: int):
        super(TeacherModel, self).__init__()

        self.model_smiles = model_2 
        self.bn_geom2d = torch.nn.BatchNorm1d(dim_feat)
        self.bn_smiles = torch.nn.BatchNorm1d(dim_feat)
        self.bn_geom3d = torch.nn.BatchNorm1d(dim_feat)
        self.bn_graph = torch.nn.BatchNorm1d(dim_feat)
        self.bn_fusion = torch.nn.BatchNorm1d(dim_feat)
        self.bn_fusion_1 = torch.nn.BatchNorm1d(dim_feat)
        self.bn_label = torch.nn.BatchNorm1d(dim_feat)
        self.bn_cat = torch.nn.BatchNorm1d(dim_feat * 3)

        self.dense1d = torch.nn.Linear(384, dim_feat)#768, dim_feat)
        self.dense2d = torch.nn.Linear(3400, dim_feat)#(2304, dim_feat)
        self.dense3d = torch.nn.Linear(512, dim_feat)#128*4, dim_feat)
        self.denseECFP = torch.nn.Linear(2048 + 167 + 512, dim_feat)
        self.denseMACCS = torch.nn.Linear(167, dim_feat)
        self.dropout = torch.nn.Dropout(0.5)

        self.dense_joint =  torch.nn.Sequential(
            torch.nn.Linear(dim_feat * 3, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        ) 
        self.bn_joint = torch.nn.BatchNorm1d(dim_feat)

        self.enc_1 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.enc_2 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.enc_3 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.enc_graph = torch.nn.Sequential(
                torch.nn.Linear(dim_feat, dim_feat),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(dim_feat, dim_feat),
        )

        self.enc_joint = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.enc_joint_1 = torch.nn.Sequential(
                torch.nn.Linear(dim_feat * 2, dim_feat),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(dim_feat, dim_feat),
        )

        self.enc_fusion = torch.nn.Sequential(
            torch.nn.Linear(dim_feat * 3, dim_feat),
        )

        self.enc_fusion_1 = torch.nn.Sequential(
                torch.nn.Linear(dim_feat * 4, dim_feat),
        )

        self.head_1 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_2 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_3 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_joint = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_4 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_5 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_6 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.head_fusion = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feat, dim_feat),
        )


        self.recon_4 = torch.nn.Linear(dim_feat, dim_feat)
        self.recon_5 = torch.nn.Linear(dim_feat, dim_feat)
        self.recon_6 = torch.nn.Linear(dim_feat, dim_feat)

        self.recon_1 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.recon_2 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
        )

        self.recon_3 = torch.nn.Sequential(
            torch.nn.Linear(dim_feat, dim_feat),
        )
        self.fusion_1 = torch.nn.Sequential()
        self.fusion_1.add_module(
            "fusion_layer_1", torch.nn.Linear(in_features=dim_feat, out_features=dim_feat)
        )
        self.fusion_1.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_1.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_1.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat, out_features=opt.num_tasks)
        )

        self.fusion_2 = torch.nn.Sequential()
        self.fusion_2.add_module(
            "fusion_layer_1", torch.nn.Linear(in_features=dim_feat, out_features=dim_feat)
        )
        self.fusion_2.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_2.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_2.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat, out_features=opt.num_tasks)
        )

        self.fusion_3 = torch.nn.Sequential()
        self.fusion_3.add_module(
            "fusion_layer_1", torch.nn.Linear(in_features=dim_feat, out_features=dim_feat)
        )
        self.fusion_3.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_3.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_3.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat, out_features=opt.num_tasks)
        )

        self.fusion_4 = torch.nn.Sequential()
        self.fusion_4.add_module(
            "fusion_layer_1", torch.nn.Linear(in_features=dim_feat, out_features=dim_feat)
        )
        self.fusion_4.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_4.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_4.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat, out_features=opt.num_tasks)
        )

        self.fusion_5 = torch.nn.Sequential()
        self.fusion_5.add_module(
            "fusion_layer_1", torch.nn.Linear(in_features=dim_feat, out_features=dim_feat)
        )
        self.fusion_5.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_5.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_5.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat, out_features=opt.num_tasks)
        )

        self.fusion_6 = torch.nn.Sequential()
        self.fusion_6.add_module(
            "fusion_layer_1", torch.nn.Linear(in_features=dim_feat, out_features=dim_feat)
        )
        self.fusion_6.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_6.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_6.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat, out_features=opt.num_tasks)
        )       

        self.fusion_7 = torch.nn.Sequential()
        self.fusion_7.add_module(
            "fusion_layer_1", torch.nn.Linear(in_features=dim_feat, out_features=dim_feat)
        )
        self.fusion_7.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_7.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_7.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat, out_features=opt.num_tasks)
        ) 

        self.fusion = torch.nn.Sequential()
        #self.fusion.add_module(
        #    "fusion_layer_1", torch.nn.Linear(in_features=dim_feat, out_features=dim_feat)
        #)
        self.fusion.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat, out_features=opt.num_tasks)
        )

        self.w1 = torch.nn.Linear(dim_feat, dim_feat, bias=False)
        self.w2 = torch.nn.Linear(dim_feat, dim_feat, bias=False)
        self.w3 = torch.nn.Linear(dim_feat, dim_feat, bias=False)

        self.vae = VAE(zsize=128)
        self.label = torch.nn.Linear(1, dim_feat, bias=False)

        self.gating = torch.nn.Linear(dim_feat, 3)
        self.deepgcn = DeeperGCN(dim_feat, 13)

        self.act = torch.nn.ReLU()
        self.norm = torch.nn.LayerNorm(dim_feat)
    def forward(self, input_molecule: Tensor):
        input_3 = input_molecule.geom3d_feature.view(input_molecule.y.shape[0], -1)
        input_2 = input_molecule.grover.view(input_molecule.y.shape[0], -1)
        input_1 = input_molecule.kpgt.view(input_molecule.y.shape[0], -1)
        input_ecfp = input_molecule.mgf.view(input_molecule.y.shape[0], -1)
        input_maccs = input_molecule.maccs.view(input_molecule.y.shape[0], -1)
        input_avalon = input_molecule.avalon.view(input_molecule.y.shape[0], -1)

        f3_raw = self.act(self.dense3d(self.dropout(F.normalize(input_3, dim=1))))
        f2_raw = self.act(self.dense2d(self.dropout(F.normalize(input_2, dim=1))))
        f1_raw = self.act(self.dense1d(self.dropout(F.normalize(input_1, dim=1))))
        fpfp = self.act(self.denseECFP(F.normalize(torch.cat((input_ecfp.float(), input_maccs.float(), input_avalon.float()), dim=1), dim=1)))
        #f_maccs = self.act(self.denseMACCS(input_maccs.float()))
        #x = self.w2(nn.functional.silu(self.w1(x1)) * self.w3(x2))
        experts = torch.cat((f1_raw.unsqueeze(1), f2_raw.unsqueeze(1), f3_raw.unsqueeze(1)), dim=1)

        f_graph = self.deepgcn(input_molecule.x, input_molecule.edge_index, input_molecule.edge_attr, input_molecule.batch) 

        gate_weights = F.softmax(self.gating(f_graph), dim=1)
        weighted_expert_outputs = gate_weights.unsqueeze(2) * experts
        f_moe = weighted_expert_outputs.sum(dim=1)

        f1_norm = f1_raw
        f2_norm = f2_raw
        f3_norm = f3_raw

        bn_f1 = self.bn_smiles(f_moe)
        bn_f2 = self.bn_geom2d(fpfp)
        bn_f3 = self.bn_geom3d(f3_raw)

        f1 = self.enc_1(bn_f1)  
        f2 = self.enc_2(bn_f2)
        f3 = self.enc_3(bn_f3)

        f1_joint = self.enc_joint(bn_f1)
        f2_joint = self.enc_joint(bn_f2)
        f3_joint = self.enc_joint(bn_f3)
        f_graph = self.enc_joint(f_graph)

        #f_joint = self.enc_fusion_1(torch.cat((bn_f1, bn_f2, bn_f3, f_graph), dim=1)) 
        #bn_fusion = torch.cat((f1_joint.unsqueeze(1), f2_joint.unsqueeze(1), f3_joint.unsqueeze(1)), dim=1)
        #f_joint, f_joint_y, loss_vae = self.vae(bn_fusion, bn_fusion, f1_raw, f2_raw, f3_raw, f1, f2, f3)
        loss_vae = 0

        f1_recon = self.recon_1(f1+f_graph)#F.normalize(f1,dim=1)+F.normalize(f1_joint,dim=1)))
        f2_recon = self.recon_2(f2+f_graph)#F.normalize(f2,dim=1)+F.normalize(f2_joint,dim=1)))
        f3_recon = self.recon_3(f3+f_graph)#F.normalize(f3,dim=1)+F.normalize(f3_joint,dim=1)))

        f1_joint_head = F.normalize(self.head_1(f1_joint), dim=1)
        f2_joint_head = F.normalize(self.head_2(f2_joint), dim=1)
        f3_joint_head = F.normalize(self.head_3(f3_joint), dim=1)

        f1_head = F.normalize(self.head_4(f1), dim=1)
        f2_head = F.normalize(self.head_5(f2), dim=1)
        f3_head = F.normalize(self.head_6(f3), dim=1)
        
        output_1 = self.fusion_1(f1_joint)
        output_2 = self.fusion_2(f2_joint)
        output_3 = self.fusion_3(f3_joint)
        output_joint_1 = self.fusion_4(f1_joint)
        output_joint_2 = self.fusion_5(f2_joint)
        output_joint_3 = self.fusion_6(f3_joint)
        output_graph = self.fusion_7(f_graph)

        f_fusion = self.enc_fusion(torch.concat((f_moe, fpfp, f_graph), dim=1))
        #f_fusion = self.w2(torch.nn.functional.silu(self.w1(f_fusion)) * self.w3(f_fusion))

        f_fusion_head = F.normalize(self.head_fusion(f_fusion), dim=1)
        output_final = self.fusion(f_fusion)

        return output_1, output_2, output_3, output_joint_1, output_joint_2, output_joint_3, output_final, \
        f1, f2, f3, f1_joint, f2_joint, f3_joint, f1_joint_head, f2_joint_head, f3_joint_head, f1_head, f2_head, f3_head, \
        f_fusion_head, f1_recon, f2_recon, f3_recon, f1_norm, f2_norm, f3_norm, loss_vae, f_fusion, output_graph

class BSCL(torch.nn.Sequential):
    """The Bimodal Supervised Contrastive Learning network."""

    def __init__(self, model_1: Module, model_2: Module, model_3: Module, opt: Any):
        """Initialization of the BSCL network.

        Args:
            model_1 (Module): The graph network
            model_2 (Module): The SMILES network
            opt (Any): Parsed arguments
        """
        super(BSCL, self).__init__()


        dim_feat = 128
        num_heads = 2 

        self.teacher = TeacherModel(model_2, dim_feat)

        self.fusion_global = torch.nn.Sequential()
        self.fusion_global.add_module(
            "fusion_layer_1",
            torch.nn.Linear(in_features=dim_feat * 4 + 2048 + 167, out_features=dim_feat * 2),
        )
        self.fusion_global.add_module("fusion_layer_1_dropout", torch.nn.Dropout(0.5))
        self.fusion_global.add_module("fusion_layer_1_activation", torch.nn.ReLU())
        self.fusion_global.add_module(
            "fusion_layer_3", torch.nn.Linear(in_features=dim_feat * 2, out_features=opt.num_tasks)
        )

    def forward(self, input_molecule: Tensor, opt: Any, phase: str = "train"):
        """The network of the BSCL.

        Args:
            input_molecule (Tensor): Input.
            opt (Any): Parsed arguments*4.
            phase (str, optional): Train phase or validation phase. Defaults to "train".

        Returns:
            Prediction results and representations learend by the model.
        """
        if opt.classification and opt.global_feature:
            global_feature = torch.cat(
                (
                    input_molecule.mgf.view(input_molecule.y.shape[0], -1),
                    input_molecule.maccs.view(input_molecule.y.shape[0], -1),
                ),
                dim=1,
            ).float()
        elif not opt.classification and opt.global_feature:
            global_feature = F.normalize(
                torch.cat(
                    (
                        input_molecule.mgf.view(input_molecule.y.shape[0], -1),
                        input_molecule.maccs.view(input_molecule.y.shape[0], -1),
                    ),
                    dim=1,
                ).float(),
                dim=1,
            )

        if opt.global_feature:
            if opt.classification:
                h_out = torch.cat((h_out[0], h_out[1], h_out[2], h_out[3], global_feature), dim=1)
                output = self.fusion_global(h_out)
            else:
                h_out = torch.cat((h_out[0], h_out[1], h_out[2], h_out[3]), dim=1)
                h_out = (h_out - torch.mean(h_out)) / torch.std(h_out)
                output = self.fusion_global(torch.cat((h_out, global_feature), dim=1))
        else:
            output_1, output_2, output_3, output_joint_1, output_joint_2, output_joint_3, output_final, f1, f2, f3, f1_joint, f2_joint, f3_joint, f1_joint_head, f2_joint_head, f3_joint_head, f1_head, f2_head, f3_head, f_fusion_head, f1_recon, f2_recon, f3_recon, f1_raw, f2_raw, f3_raw, loss_vae, f_joint_y, output_graph = self.teacher(input_molecule)

        if phase == "train":
            return (
                output_1,
                output_2,
                output_3,
                output_joint_1,
                output_joint_2,
                output_joint_3,
                output_final,
                f1,
                f2,
                f3,
                f1_joint,
                f2_joint,
                f3_joint,
                f1_joint_head,
                f2_joint_head,
                f3_joint_head,
                f1_head, 
                f2_head,
                f3_head,
                f_fusion_head,
                f1_recon,
                f2_recon,
                f3_recon,
                f1_raw,
                f2_raw,
                f3_raw,
                loss_vae,
                f_joint_y, 
                output_graph
            )
        else:
            return (
                output_1,
                output_2,
                output_3,
                output_joint_1,
                output_joint_2,
                output_joint_3,
                output_final,
                f1, 
                f2, 
                f3, 
                f1_joint, 
                f2_joint,
                f3_joint,
                f1_joint_head,
                f2_joint_head,
                f3_joint_head,

            )

class DiffLoss(torch.nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

def set_model(opt: Any):
    """Initialization of the model and loss functions.

    Args:
        opt (Any): Parsed arguments.

    Returns:
        Return the model and the loss functions.
    """
    model_1 = SupConDeeperGCN(opt)
    model_2 = SMILESBert()
    model_3 = UniMolRepr(data_type='molecule')

    model = BSCL(model_1, model_2, model_3, opt)
    
    for name, param in model.named_parameters():
        if "model_smiles.model.embeddings" in name or "model_smiles.model.encoder" in name:
            param.requires_grad = False
            print(name)

    if opt.classification:
        criterion_scl = SupConLossCls(temperature=opt.temp, base_temperature=opt.temp)
    else:
        criterion_scl = SupConLossReg(
            temperature=opt.temp,
            base_temperature=opt.temp,
            gamma1=opt.gamma1,
            gamma2=opt.gamma2,
            threshold=opt.threshold,
        )

    criterion_scl = ConLossCls(temperature=opt.temp, base_temperature=opt.temp)

    if opt.classification:
        criterion_task = torch.nn.BCEWithLogitsLoss()
    else:
        criterion_task = torch.nn.MSELoss()
    criterion_mse = torch.nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion_scl = criterion_scl.cuda()
        cudnn.benchmark = False
    return model, criterion_scl, criterion_mse, criterion_task


def train(
    train_loader: Any,
    model: torch.nn.Sequential,
    criterion_scl: Callable,
    criterion_mse: Callable,
    criterion_task: Callable,
    optimizer: Optimizer,
    scheduler: Any,
    opt: Any,
    mu: int = 0,
    std: int = 0,
    epoch: int = 0
):
    """One epoch training.

    Args:
        train_dataset (Set[Data]): Train set.
        model (torch.nn.Sequential): Model
        criterion_scl (Callable): Supervised contrastive loss function
        criterion_mse (Callable): Reconstruction loss function
        criterion_task (Callable): Task loss function
        optimizer (Optimizer): Optimizer
        opt (Any): Parsed arguments
        mu (int, optional): Mean value of the train set for the regression task. Defaults to 0.
        std (int, optional): Standard deviation of the train set for the regression task.
            Defaults to 0.

    Returns:
        Losses.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_task = AverageMeter()
    losses_recon = AverageMeter()
    losses_scl = AverageMeter()
    losses_kl = AverageMeter()
    losses_diff = AverageMeter()
    losses = AverageMeter()
    #train_dataset_shuffle = train_dataset.shuffle()
    #train_loader = DataLoader(train_dataset_shuffle, batch_size=opt.batch_size, drop_last=True)
    end = time.time()

    for _, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = batch.to("cuda")
        data_time.update(time.time() - end)

        bsz = batch.y.shape[0]

        if not opt.classification:
            labels = (batch.y - mu) / std
        else:
            labels = batch.y
        # compute loss
        (
                output_1,
                output_2,
                output_3,
                output_joint_1,
                output_joint_2,
                output_joint_3,
                output_final,
                f1,
                f2,
                f3,
                f1_joint,
                f2_joint,
                f3_joint,
                f1_joint_head,
                f2_joint_head,
                f3_joint_head,
                f1_head,
                f2_head,
                f3_head,
                f_fusion_head,
                f1_recon,
                f2_recon,
                f3_recon,
                f1_raw,
                f2_raw,
                f3_raw,
                loss_vae,
                f_joint_y, 
                output_graph
        ) = model(batch, opt)

        features_cross_1 = torch.cat([f1_joint_head.unsqueeze(1), f1_joint_head.unsqueeze(1)], dim=1)
        features_cross_2 = torch.cat([f2_joint_head.unsqueeze(1), f2_joint_head.unsqueeze(1)], dim=1)
        features_cross_3 = torch.cat([f3_joint_head.unsqueeze(1), f3_joint_head.unsqueeze(1)], dim=1)

        features_cross_4 = torch.cat([f1_head.unsqueeze(1), f1_head.unsqueeze(1)], dim=1)
        features_cross_5 = torch.cat([f2_head.unsqueeze(1), f2_head.unsqueeze(1)], dim=1)
        features_cross_6 = torch.cat([f3_head.unsqueeze(1), f3_head.unsqueeze(1)], dim=1)

        features_cross_7 = torch.cat([f1_joint_head.unsqueeze(1), f2_joint_head.unsqueeze(1)], dim=1)
        features_cross_8 = torch.cat([f1_joint_head.unsqueeze(1), f3_joint_head.unsqueeze(1)], dim=1)
        features_cross_9 = torch.cat([f2_joint_head.unsqueeze(1), f3_joint_head.unsqueeze(1)], dim=1)

        features_cross_10 = torch.cat([f_joint_y.unsqueeze(1), f_joint_y.unsqueeze(1)], dim=1)

        loss_task_tmp = []
        loss_task_tmp_1 = 0
        loss_task_tmp_2 = 0
        loss_task_tmp_3 = 0
        loss_task_tmp_4 = 0
        loss_task_tmp_5 = 0
        loss_task_tmp_6 = 0
        loss_task_tmp_7 = 0
        loss_task_ensemble_tmp = 0
        loss_scl_tmp = 0
        loss_kl_tmp = [] 
        loss_kl_tmp_1 = 0
        loss_kl_tmp_2 = 0
        loss_kl_tmp_3 = 0
        loss_kl_tmp_4 = 0
        loss_kl_tmp_5 = 0
        loss_kl_tmp_6 = 0
        loss_kl_tmp_7 = 0
        total_num = 0

        criterion_diff = DiffLoss()
        criterion_kl = torch.nn.BCELoss(reduction='mean')#nn.MSELoss() #torch.nn.KLDivLoss(reduction='batchmean')        
        criterion_cl = RnCLoss()

        sig_log = torch.nn.LogSigmoid()
        sig = torch.nn.Sigmoid()

        loss_diff = (criterion_diff(f1, f1_joint) + criterion_diff(f2, f2_joint) + criterion_diff(f3, f3_joint)) / 3.0
        loss_recon = (criterion_mse(f1_recon, f1_raw) + criterion_mse(f2_recon, f2_raw) + criterion_mse(f3_recon, f3_raw)) / 3.0

        out = []
        out.append(output_1)
        out.append(output_2)
        out.append(output_3)

        for i in range(labels.shape[1]):
            is_labeled = batch.y[:, i] == batch.y[:, i]

            loss_task_1 = criterion_task(
                output_1[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_2 = criterion_task(
                output_2[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_3 = criterion_task(
                output_3[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_4 = criterion_task(
                output_joint_1[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_5 = criterion_task(
                output_joint_2[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_6 = criterion_task(
                output_joint_3[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_7 = criterion_task(
                    output_graph[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_ensemble = criterion_task(
                output_final[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_scl_1 = criterion_scl(features_cross_1[is_labeled], labels[is_labeled, i])
            loss_scl_2 = criterion_scl(features_cross_2[is_labeled], labels[is_labeled, i])
            loss_scl_3 = criterion_scl(features_cross_3[is_labeled], labels[is_labeled, i])
            
            loss_scl_4 = criterion_scl(features_cross_4[is_labeled], labels[is_labeled, i])
            loss_scl_5 = criterion_scl(features_cross_5[is_labeled], labels[is_labeled, i])
            loss_scl_6 = criterion_scl(features_cross_6[is_labeled], labels[is_labeled, i])

            loss_scl_7 = criterion_scl(features_cross_7[is_labeled], labels[is_labeled, i])
            loss_scl_8 = criterion_scl(features_cross_8[is_labeled], labels[is_labeled, i])
            loss_scl_9 = criterion_scl(features_cross_9[is_labeled], labels[is_labeled, i])

            loss_scl = (loss_scl_7 + loss_scl_8 + loss_scl_9) / 3 

            
            loss_task_tmp_1 = loss_task_tmp_1 + loss_task_1
            loss_task_tmp_2 = loss_task_tmp_2 + loss_task_2
            loss_task_tmp_3 = loss_task_tmp_3 + loss_task_3
            loss_task_tmp_4 = loss_task_tmp_4 + loss_task_4
            loss_task_tmp_5 = loss_task_tmp_5 + loss_task_5
            loss_task_tmp_6 = loss_task_tmp_6 + loss_task_6
            loss_task_tmp_7 = loss_task_tmp_7 + loss_task_7
            loss_task_ensemble_tmp = loss_task_ensemble_tmp + loss_task_ensemble
            
            output_pre = sig(output_final)
            loss_kl_1 = criterion_kl(output_pre[is_labeled, i], sig(output_1[is_labeled, i]))
            loss_kl_2 = criterion_kl(output_pre[is_labeled, i], sig(output_2[is_labeled, i]))
            loss_kl_3 = criterion_kl(output_pre[is_labeled, i], sig(output_3[is_labeled, i]))
            loss_kl_4 = criterion_kl(output_pre[is_labeled, i], sig(output_joint_1[is_labeled, i]))
            loss_kl_5 = criterion_kl(output_pre[is_labeled, i], sig(output_joint_2[is_labeled, i]))
            loss_kl_6 = criterion_kl(output_pre[is_labeled, i], sig(output_joint_3[is_labeled, i]))
            loss_kl_7 = criterion_kl(output_pre[is_labeled, i], sig(output_graph[is_labeled, i]))
            

            loss_kl_tmp_1 = loss_kl_tmp_1 + loss_kl_1
            loss_kl_tmp_2 = loss_kl_tmp_2 + loss_kl_2
            loss_kl_tmp_3 = loss_kl_tmp_3 + loss_kl_3
            loss_kl_tmp_4 = loss_kl_tmp_4 + loss_kl_4
            loss_kl_tmp_5 = loss_kl_tmp_5 + loss_kl_5
            loss_kl_tmp_6 = loss_kl_tmp_6 + loss_kl_6
            loss_kl_tmp_7 = loss_kl_tmp_7 + loss_kl_7
    
            if opt.classification:
                if torch.sum(labels[is_labeled, i], dim=0) > 0:
                    loss_scl_tmp = loss_scl_tmp + loss_scl
                    total_num = total_num + 1
            else:
                loss_scl_tmp = loss_scl_tmp + loss_scl
                total_num = total_num + 1
            
        #if total_num == 0:
        #    continue
        loss_cl = criterion_cl(features_cross_10, labels)
        loss_task = (loss_task_tmp_1 + loss_task_tmp_2 + loss_task_tmp_7 + loss_task_ensemble_tmp) / labels.shape[1]
        
        loss_kl = (loss_kl_tmp_1 + loss_kl_tmp_2 + loss_kl_tmp_7) / labels.shape[1]

        loss_task = loss_task / 4
        loss_kl = loss_kl / 3

        loss =  loss_task + loss_kl + loss_cl #+ loss_kl #+ loss_recon loss_vae

        # update metric
        losses_task.update(loss_task.item(), bsz)
        losses_scl.update(loss_scl.item(), bsz)
        losses_kl.update(loss_kl.item(), bsz)
        losses_recon.update(loss_recon.item(), bsz)
        losses_diff.update(loss_diff.item(), bsz)
        losses.update(loss.item(), bsz)

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step() 
        scheduler.step() 
        '''
        for kk in range(1,4):
            #pdb.set_trace()
            loss = loss_task_tmp[kk-1] + 0.1*loss_kl_tmp[kk-1]
            optimizer[kk].zero_grad()
            if kk<2:
                loss.backward(retain_graph=True)
            elif kk==2:
                loss.backward()
            optimizer[kk].step()
        '''
        '''
        if epoch<30:
             loss =  loss_task  + loss_kl 
        else:
             loss = loss_kl 
        #loss = opt.wscl * loss_scl + loss_task
        # update metric
        losses_task.update(loss_task.item(), bsz)
        #losses_scl.update(loss_scl.item(), bsz)
        losses.update(loss.item(), bsz)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        '''
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses_task.avg, losses_scl.avg, losses_kl.avg, losses_recon.avg, losses_diff.avg, losses.avg


def validation(
    dataset: Set[Data],
    model: torch.nn.Sequential,
    opt: Any,
    mu: int = 0,
    std: int = 0,
    save_feature: int = 0,
    epoch: int =0
):
    """Calculate performance metrics.

    Args:
        dataset (Set[Data]): A dataset.
        model (torch.nn.Sequential): Model.
        opt (Any): Parsed arguments.
        mu (int, optional): Mean value of the train set for the regression task.
            Defaults to 0.
        std (int, optional): Standard deviation of the train set for the regression task.
            Defaults to 0.
        save_feature (int, optional): Whether save the learned features or not.
            Defaults to 0.

    Returns:
        auroc or rmse value.
    """
    model.eval()

    if opt.classification:
        evaluator = Evaluator(name=opt.dataset, num_tasks=opt.num_tasks, eval_metric="rocauc")
    else:
        evaluator = Evaluator(name=opt.dataset, num_tasks=opt.num_tasks, eval_metric="rmse")
    data_loader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False
    )

    with torch.no_grad():
        y_true = []
        y_pred = []
        if save_feature:
            feature_smiles = []
            feature_graph = []
            feature_smiles_sp = []
            feature_graph_sp = []
            feature = []
        for _, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch = batch.to("cuda")
            (
                output_1,
                output_2,
                output_3,
                output_joint_1,
                output_joint_2,
                output_joint_3,
                output_final,
                f1,
                f2,
                f3,
                f1_joint,
                f2_joint,
                f3_joint,
                f1_joint_head,
                f2_joint_head,
                f3_joint_head
            ) = model(batch, opt, "valid")

            if not opt.classification:
                output_4 = (output_final) * std + mu
            if save_feature:
                feature_smiles.append(f2_co.detach().cpu())
                feature_graph.append(f1_co.detach().cpu())
                feature_smiles_sp.append(f2_sp.detach().cpu())
                feature_graph_sp.append(f1_sp.detach().cpu())
                feature.append(h_out.detach().cpu())

            if opt.classification:
                sigmoid = torch.nn.Sigmoid()
                output_1 = sigmoid(output_1)
                output_2 = sigmoid(output_2)
                output_3 = sigmoid(output_3)
                output_4 = sigmoid(output_final)
                output_ensemble = sigmoid((output_1+output_2+output_3)/3)
            output = output_4#output_ensemble#(output_1+output_2+output_3)/3

            y_true.append(batch.y.detach().cpu())
            y_pred.append(output.detach().cpu())

        y_true = torch.cat(y_true, dim=0).squeeze().unsqueeze(1).numpy()
        if opt.num_tasks > 1:
            y_pred = np.concatenate(y_pred)
            input_dict = {"y_true": y_true.squeeze(), "y_pred": y_pred.squeeze()}
        else:
            y_pred = np.expand_dims(np.concatenate(y_pred), 1)
            input_dict = {
                "y_true": np.expand_dims(y_true.squeeze(), 1),
                "y_pred": np.expand_dims(y_pred.squeeze(), 1),
            }

        if opt.classification:
            eval_result = evaluator.eval(input_dict)["rocauc"]
        else:
            eval_result = evaluator.eval(input_dict)["rmse"]

    if save_feature:
        feature_smiles = np.concatenate(feature_smiles)
        feature_graph = np.concatenate(feature_graph)
        feature_smiles_sp = np.concatenate(feature_smiles_sp)
        feature_graph_sp = np.concatenate(feature_graph_sp)
        feature = np.concatenate(feature)

        return (
            eval_result,
            feature_smiles,
            feature_graph,
            y_true,
            y_pred,
            feature_smiles_sp,
            feature_graph_sp,
            feature,
        )
    else:
        return y_pred, eval_result


def main():

    for dataname in [opt.dataset + "_1"]:

        # build data loader
        train_dataset, test_dataset, val_dataset = set_loader(opt, dataname)

        if opt.classification:
            mu, std = 0, 0
        else:
            mu, std = calmean(train_dataset)

        # build model and criterion
        model, criterion_scl, criterion_mse, criterion_task = set_model(opt)

        # build optimizer
        optimizer_teacher = set_optimizer(opt.learning_rate, opt.weight_decay, model)

        model_name = "{}_{}".format(opt.model_name, dataname)

        # save folder
        opt.tb_folder = os.path.join(opt.tb_path, model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        opt.save_folder = os.path.join(opt.model_path, model_name)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)
        # tensorboard
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

        if opt.classification:
            best_acc = 0
        else:
            best_acc = 10000000
        best_model = model
        best_epoch = 0

        #train_dataset_shuffle = train_dataset.shuffle()
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=True)

        num_training_steps =  len(train_loader) * opt.epochs
        num_warmup_steps = int(num_training_steps * 0.1)

        scheduler = optimization.get_linear_schedule_with_warmup(optimizer_teacher[0], num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # training routine
        for epoch in range(opt.epochs):
            torch.cuda.empty_cache()
            #adjust_learning_rate(opt, optimizer_teacher, epoch, opt.learning_rate)
            # train for one epoch
            time1 = time.time()
            loss_task, loss_scl, loss_kl, loss_recon, loss_diff, loss = train(
                train_loader,
                model,
                criterion_scl,
                criterion_mse,
                criterion_task,
                optimizer_teacher,
                scheduler,
                opt,
                mu,
                std,
                epoch
            )
            time2 = time.time()
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

            _, acc = validation(val_dataset, model, opt, mu, std, 0, epoch)

            # tensorboard logger
            logger.log_value("loss_task", loss_task, epoch)
            logger.log_value("loss_scl", loss_scl, epoch)
            logger.log_value("loss_kl", loss_kl, epoch)
            logger.log_value("loss_recon", loss_recon, epoch)
            logger.log_value("loss_diff", loss_diff, epoch)
            logger.log_value("loss", loss, epoch)
            logger.log_value("validation auroc/rmse", acc, epoch)
            logger.log_value("learning rate", optimizer_teacher[0].state_dict()['param_groups'][0]['lr'], epoch)

            if opt.classification:
                if acc > best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    _, test_acc = validation(test_dataset, model, opt, mu, std, 0, epoch)
                    logger.log_value("test auroc", test_acc, epoch)
                    print("test auroc:{}".format(test_acc))
                print("val auroc:{}".format(acc))
                
            else:
                if acc < best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    _, test_acc = validation(test_dataset, model, opt, mu, std, 0, epoch)
                    logger.log_value("test rmse", test_acc, epoch)
                    print("test rmse:{}".format(test_acc))
                    print("val rmse:{}".format(acc))

        train_pred, _ = validation(train_dataset, best_model.cuda(), opt, mu, std, 0, epoch-1)
        y_pred, test_acc = validation(test_dataset, best_model.cuda(), opt, mu, std, 0, epoch-1)
        val_pred, val_acc = validation(val_dataset, best_model.cuda(), opt, mu, std, 0, epoch-1)

        df = pd.DataFrame({'y_pred':np.squeeze(y_pred)})
        df.to_csv('result.csv')

        df = pd.DataFrame({'y_pred':np.squeeze(train_pred)})
        df.to_csv('result_train.csv')

        df = pd.DataFrame({'y_pred':np.squeeze(val_pred)})
        df.to_csv('result_val.csv')

        save_file = os.path.join(opt.save_folder, "result_pre.txt")

        txtFile = open(save_file, "w")
        txtFile.write("validation:" + str(val_acc) + "\n")
        txtFile.write("test:" + str(test_acc) + "\n")
        txtFile.write("best epoch:" + str(best_epoch) + "\n")
        txtFile.close()

        '''
        for name,p in model.named_parameters():
            if "teacher" in name:
                print("stop update",name)
                p.requires_grad = False
        best_acc = 0
        for epoch in range(30, opt.epochs + 1):
            torch.cuda.empty_cache()
            adjust_learning_rate(opt, optimizer_stu, epoch, opt.learning_rate)
            # train for one epoch
            time1 = time.time()
            loss_task, loss_scl, loss = train(
                train_loader,
                model,
                criterion_scl,
                criterion_mse,
                criterion_task,
                optimizer_stu,
                scheduler,
                opt,
                mu,
                std,
                epoch
            )
            time2 = time.time()
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

            acc = validation(val_dataset, model, opt, mu, std, 0, epoch)

            # tensorboard logger
            logger.log_value("loss_task", loss_task, epoch)
            logger.log_value("loss_scl", loss_scl, epoch)
            logger.log_value("loss", loss, epoch)
            logger.log_value("validation auroc/rmse", acc, epoch)
             
            if opt.classification:
                if acc > best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    test_acc = validation(test_dataset, model, opt, mu, std, 0, epoch)
                    logger.log_value("test auroc", test_acc, epoch)
                    print("test auroc:{}".format(test_acc))
                print("val auroc:{}".format(acc))

            else:
                if acc < best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    test_acc = validation(test_dataset, model, opt, mu, std, 0, epoch)
                    logger.log_value("test rmse", test_acc, epoch)
                    print("test rmse:{}".format(test_acc))
                    print("val rmse:{}".format(acc))

        # save the last model
        print("best epoch : {}".format(best_epoch))
        save_file = os.path.join(opt.save_folder, "last_" + str(best_epoch) + ".pth")
        save_model(best_model, optimizer_stu, opt, opt.epochs, save_file)

        test_acc = validation(test_dataset, best_model.cuda(), opt, mu, std, 0, epoch)
        val_acc = validation(val_dataset, best_model.cuda(), opt, mu, std, 0, epoch)
        save_file = os.path.join(opt.save_folder, "result.txt")

        txtFile = open(save_file, "w")
        txtFile.write("validation:" + str(val_acc) + "\n")
        txtFile.write("test:" + str(test_acc) + "\n")
        txtFile.write("best epoch:" + str(best_epoch) + "\n")
        txtFile.close()
        '''

        print("Val Result:{}".format(val_acc))
        print("Test Result:{}".format(test_acc))


if __name__ == "__main__":
    main()
