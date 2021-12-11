import torch
import torchvision.transforms as transforms
from dataloaders.domainet_img import *
import pandas as pd

# OPTS FOR DOMAINNET
BATCH_SIZE = 16
TEST_BATCH_SIZE = 100

EPOCHS = 1
STEP = 2
LR = 0.0001
DECAY = 0.000001
MOMENTUM = 0.9
BANDWIDTH = 0.1

# Get text embeds
DOMAINS = ['infograph', 'quickdraw', 'real', 'clipart', 'quickdraw', 'sketch']
domain_id = {'infograph': 1, 'quickdraw':2, 'real':3, 'clipart':4, 'quickdraw':5, 'sketch':6}


# What is this?
CLASSES = 2

NUM_META = 3

DATAROOT = "/shared-network/syadav/domain_images_random"

# What is this? What is to be done in our case?
# Still didnt set up
# REGION_TO_VEC = {'MA': [0, .1], 'NE': [0., 0.], 'South': [1., 1.], 'Pacific': [0., 3.], 'MW': [0., 2.]}

embed_df = pd.read_csv('pred/pred_domain.csv', index=False, header=False)
pred_df = pd.DataFrame(pred_domain)
domain_to_vec = {infograph: <its embedding>, quickdraw: <its embedding>}

# Replace with domain id from dict
def domain_converter(domain_id):
    return domain_id.va


def init_loader(bs, domains=[], shuffle=False, auxiliar=False, size=224, std=[0.229, 0.224, 0.225]):
    data_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std)
    ])

    dataset = DomainNetDataset(DATAROOT, transform=data_transform, domains=domains)

# Helps in sample one domain at a time
    # gives specific domain
    if not auxiliar:
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False, num_workers=4, shuffle=shuffle)

    else:
        # Helpful for auxiliarry sample
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False, num_workers=4,
                                           sampler=PortraitsSampler(dataset, bs))


def compute_edge(x, dt, idx, self_connection=1.):
    edge_w = torch.exp(-torch.pow(torch.norm(x.view(1, -1) - dt, dim=1), 2) / (2. * BANDWIDTH))
    edge_w[idx] = edge_w[idx] * self_connection
    return edge_w / edge_w.sum()

# Just domain embeddigss : meta = domain_name
def get_meta_vector(meta): ## meta =  domain
    return embedding_of_the_passed_domain
