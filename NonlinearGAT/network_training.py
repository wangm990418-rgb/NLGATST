"""Training routine for NLGATST."""

import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from NonlinearGAT.model import NonlinearGAT_G, NonlinearGAT_P, NonlinearGAT_S, EstimateP
from NonlinearGAT.utils import Noise_Cross_Entropy, load_data, training_performance


def fix_seed(seed):
    """Set all deterministic seeds used by the original implementation."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Training:
    """Fit the graph attention model and store the embedding in AnnData."""

    def __init__(self, num, args, adata):
        self.seed = args.seed
        fix_seed(args.seed)
        self.fastmode = False
        self.log_every = 1
        self.epochs = args.epochs
        self.lr = args.lr
        self.lr_1 = args.lr_1
        self.weight_decay = args.weight_decay
        self.hidden = args.hidden
        self.nb_heads = args.head
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.s_f = args.scale_factor
        self.aggtype = args.aggtype
        self.inner = args.inner_steps
        self.num = num
        self.result = 0
        self.adata = adata
        self.we1 = args.we1
        self.we2 = args.we2
        self.we3 = args.we3

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        graph, edge, features, adj_1 = load_data(self.adata)
        self.graph = graph
        self.features = graph.ndata["feat"]
        self.edge = edge
        self.adj_1 = adj_1
        self.we1 = args.we1
        self.we2 = args.we2
        self.we3 = args.we3

        if args.aggtype == "Generalized-mean":
            self.model = NonlinearGAT_G(
                nfeat=self.features.shape[1],
                nhid=self.hidden,
                dropout=self.dropout,
                alpha=self.alpha,
                nheads=self.nb_heads,
            )
        if args.aggtype == "Polynomial":
            self.model = NonlinearGAT_P(
                nfeat=self.features.shape[1],
                nhid=self.hidden,
                dropout=self.dropout,
                alpha=self.alpha,
                nheads=self.nb_heads,
                s_f=self.s_f,
            )
        if args.aggtype == "Softmax":
            self.model = NonlinearGAT_S(
                nfeat=self.features.shape[1],
                nhid=self.hidden,
                dropout=self.dropout,
                alpha=self.alpha,
                nheads=self.nb_heads,
                s_f=self.s_f,
            )
        self.estimator = EstimateP(self.nb_heads)
        self.BCE_loss = nn.BCEWithLogitsLoss()

        if torch.cuda.is_available():
            device = torch.device("cuda:%d" % 0)
            self.model.cuda()
            self.estimator.cuda()
            self.features = self.features.cuda()
            self.edge = self.edge.cuda()
            self.adj_1 = self.adj_1.cuda()
            self.graph = graph.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_p = optim.SGD(self.estimator.parameters(), momentum=0.9, lr=self.lr_1)

        loss_values = []
        t_total = time.time()
        best_ari = -1
        best_epoch = 0
        best_val = 1e9

        print("Network Fitting...")
        for epoch in range(self.epochs):
            loss, embeddings, loss_adj_1, loss_feature, loss_NCE = self.train_outer(epoch)
            loss_values.append(loss)
            best_val, best_epoch = training_performance(loss_values[-1], best_val, epoch, best_epoch)

            for i in range(self.inner):
                loss, embeddings, loss_adj_1, loss_feature, loss_NCE = self.train_inner(epoch, self.estimator.p)
                loss_values.append(loss)
                best_val, best_epoch = training_performance(loss_values[-1], best_val, epoch, best_epoch)

        total_time = time.time() - t_total
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(total_time))

        np_emb = embeddings.cpu().detach().numpy()
        adata.obsm["NLGATST_embed"] = np_emb

    def train_inner(self, epoch, p):
        """Update encoder weights with the current aggregation parameters."""
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        embeddings, x1 = self.model(self.graph, self.features, self.edge, p)
        reconstructed_adj = torch.sigmoid(torch.matmul(embeddings, embeddings.T))
        loss_adj_1 = self.BCE_loss(reconstructed_adj, self.adj_1)
        loss_NCE = Noise_Cross_Entropy(embeddings, self.adj_1)
        loss_feature = F.l1_loss(x1, self.features)
        loss = self.we3 * loss_feature + self.we1 * loss_adj_1 + self.we2 * loss_NCE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

        if not self.fastmode:
            self.model.eval()
            embeddings, x1 = self.model(self.graph, self.features, self.edge, p)
            s = loss.data.item()

        return loss.data.item(), embeddings, loss_adj_1.item(), loss_feature.item(), loss_NCE.item()

    def train_outer(self, epoch):
        """Update the nonlinear aggregation parameters."""
        t = time.time()
        estimator = self.estimator
        estimator.train()
        self.optimizer_p.zero_grad()

        embeddings, x1 = self.model(self.graph, self.features, self.edge, estimator.p)
        reconstructed_adj = torch.sigmoid(torch.matmul(embeddings, embeddings.T))
        loss_adj_1 = self.BCE_loss(reconstructed_adj, self.adj_1)
        loss_NCE = Noise_Cross_Entropy(embeddings, self.adj_1)
        loss_feature = F.l1_loss(x1, self.features)
        loss = self.we3 * loss_feature + self.we1 * loss_adj_1 + self.we2 * loss_NCE
        loss.backward()
        self.optimizer_p.step()
        torch.cuda.empty_cache()
        estimator.p.data.copy_(estimator.p.data)
        embeddings = self.model(self.graph, self.features, self.edge, estimator.p)

        if not self.fastmode:
            self.model.eval()
            embeddings, x1 = self.model(self.graph, self.features, self.edge, estimator.p)

        if (epoch + 1) % self.log_every == 0:
            print(
                f"Epoch {epoch + 1:03d} | Total Loss: {loss:.4f} | "
                f"Loss_adj1: {loss_adj_1:.4f}  | "
                f"Loss_feat: {loss_feature:.8f} | Loss_NCE: {loss_NCE:.4f}"
            )

        return loss.data.item(), embeddings, loss_adj_1.item(), loss_feature.item(), loss_NCE.item()
