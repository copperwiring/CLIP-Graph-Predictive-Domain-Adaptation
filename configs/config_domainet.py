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
domain_to_id = {'infograph': 1, 'quickdraw': 2, 'real': 3, 'clipart': 4, 'quickdraw': 5, 'sketch': 6}

CLASSES = 345

# Confirm this. Is it 5 (= number of domains)
NUM_META = 3

DATAROOT = "/shared-network/syadav/"

embed_df = pd.read_csv('pred/pred_domain.csv', index_col=False, header=None)

# Map csv file row value to their domain name. TO DO: Save csv first column as domain name
domain_to_pd_idx = {'infograph': 0, 'quickdraw': 1, 'real': 2, 'clipart': 3, 'quickdraw': 4, 'sketch': 5}

# Domain name mapped to Domain embeddings from descriptions
domain_to_vec = {
    'infograph': embed_df[domain_to_pd_idx['infograph']],
     'quickdraw': embed_df[domain_to_pd_idx['quickdraw']],
     'real': embed_df[domain_to_pd_idx['real']],
     'clipart': embed_df[domain_to_pd_idx['clipart']],
     'quickdraw': embed_df[domain_to_pd_idx['quickdraw']],
     'sketch': embed_df[domain_to_pd_idx['sketch']]
    }


# Returns domain for given id
def domain_converter(id):
    domain_val = domain_to_id.keys()[domain_to_id.values().index(id)]
    return domain_val

def init_loader(bs, domains=[], shuffle=False, auxiliar=False, size=224, std=[0.229, 0.224, 0.225]):
    data_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std)
    ])

    dataset = DomainNetDataset(DATAROOT, transform=data_transform, domains=domains)
    import pdb; pdb.set_trace()

    # Helps in sample one domain at a time
    # gives specific domain
    if not auxiliar:
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False, num_workers=4, shuffle=shuffle)

    else:
        # Helpful for auxiliary sample
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False, num_workers=4,
                                           sampler=DomainNetSampler(dataset, bs))


def compute_edge(x, dt, idx, self_connection=1.):
    edge_w = torch.exp(-torch.pow(torch.norm(x.view(1, -1) - dt, dim=1), 2) / (2. * BANDWIDTH))
    edge_w[idx] = edge_w[idx] * self_connection
    return edge_w / edge_w.sum()


# Just domain embeddigs : meta = domain_name
def get_meta_vector(meta):
    embed = domain_to_vec[meta]
    return embed
