import numpy as np
import torch
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import torch.utils.data as data
import pandas as pd
import os

REGIONS_TO_IDX = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}


# Read text labels
def read_split_line(line):
    path, class_id = line.split(' ')
    class_name = path.split('/')[1]
    return path, class_name, int(class_id)


class DomainNetDataset(data.Dataset):
    def __init__(self, data_root, domains, train=True, validation=False, transformer=None):
        self.domains = domains
        self.n_doms = len(domains)
        self.data_root = data_root
        self.train = train
        self.val = validation

        # self.seen = 7
        # self.unseen = 7

        # Is this path of images of each domain?
        self.image_paths = []

        # Labels: aeroplanes, cars etc
        self.labels = []

        self.domain_id = []

        self.loader = default_loader
        if transformer is None:
            self.transformer = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])
        else:
            self.transformer = transformer

        # Read first domain or selected domains
        # Returns domain name and associated index.
        # To Do: Won't index be same as the domain id?
        if isinstance(domains, list):
            for i, d in enumerate(domains):
                self.read_single_domain(d, i)
        else:
            self.read_single_domain(domains, 0)

        # TO DO: Is domain id same as domain index above?
        self.domain_id = torch.LongTensor(self.domain_id)

    # Read data from a single domain
    def read_single_domain(self, domain, id=0):
        if self.val or self.train:
            # Needs to be fixed. What does it do?
            file_names = [domain + '_train.txt']
        else:
            # Note: if we are testing, we use all images of unseen classes contained in the domain,
            # no matter of the split. The images of the unseen classes are NOT present in the training phase.
            file_names = [domain + '_train.txt', domain + '_test.txt']

        for file_name in file_names:
            self.read_single_file(file_name, id)

    # Read all needed elements from file
    def read_single_file(self, filename, id):
        domain_images_list_path = os.path.join(self.data_root, filename)
        with open(domain_images_list_path, 'r') as files_list:
            for line in files_list:
                line = line.strip()
                local_path, class_name, _ = read_split_line(line)
                if class_name in self.valid_classes:
                    self.image_paths.append(os.path.join(self.data_root, local_path))
                    self.labels.append(self.valid_classes.index(class_name))
                    self.domain_id.append(id)
                    self.attributes.append(self.attributes_dict[class_name].unsqueeze(0))

    def get_domains(self):
        return self.domain_id, self.n_doms

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        features = self.loader(self.image_paths[index])
        features = self.transformer(features)
        return features, self.attributes[index], self.domain_id[index], self.labels[index]

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.dataset
        return fmt_str

# Needs to be fixed - what does it do
class DomainNetSampler(torch.utils.data.sampler.Sampler):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source, bs):
        self.data_source = data_source
        self.meta = np.array(self.data_source.get_meta())
        self.dict_meta = {}
        self.indeces = {}
        self.keys = []
        self.bs = bs
        for idx, (year, region) in enumerate(self.meta):
            try:
                self.dict_meta[str(year * 20 + region)].append(idx)
            except:
                self.dict_meta[str(year * 20 + region)] = [idx]
                self.keys.append(str(year * 20 + region))
                self.indeces[str(year * 20 + region)] = 0

        for idx in self.keys:
            shuffle(self.dict_meta[idx])

    def _sampling(self, idx, n):
        if self.indeces[idx] + n >= len(self.dict_meta[idx]):
            self.dict_meta[idx] = self.dict_meta[idx] + self.dict_meta[idx]
        self.indeces[idx] = self.indeces[idx] + n
        return self.dict_meta[idx][self.indeces[idx] - n:self.indeces[idx]]

    def _shuffle(self):
        order = np.random.randint(len(self.keys), size=(len(self.data_source) // (self.bs)))
        sIdx = []
        for i in order:
            sIdx = sIdx + self._sampling(self.keys[i], self.bs)
        return np.array(sIdx)

    def __iter__(self):
        return iter(self._shuffle())

    def __len__(self):
        return len(self.data_source) / self.bs * self.bs

#
# # -- old ---
#     def read_single_domain(self, domain, id):
#         """
#         What does this do?
#         :param domain:
#         :param id:
#         :return:
#         """
#         domain_images_list_path = os.path.join(self.data_root, filename)
#         with open(domain_images_list_path, 'r') as files_list:
#             for line in files_list:
#                 line = line.strip()
#                 local_path, class_name, _ = read_split_line(line)
#                 if class_name in self.valid_classes:
#                     self.image_paths.append(os.path.join(self.data_root, local_path))
#                     self.labels.append(self.valid_classes.index(class_name))
#                     self.domain_id.append(id)
#                     self.attributes.append(self.attributes_dict[class_name].unsqueeze(0))
#
#         if self.train:
#             # To Do
#             # https://github.com/mancinimassimiliano/CuMix/blob/8a1cf3a5071e12ca4eee0fcf2e55edb9fdef48f4/data/domain_dataset.py#L126
#
#         elif self.val:
#             # To Do
#         else:
#             # TO DO
#
#     def get_domains(self):
#         return self.domain_id, self.n_doms
#
#     def get_labels(self):
#         return self.labels
#
#     def __getitem__(self, index):
#         # To change
#
#         # TO DO: What is features below?
#         features = self.loader(self.image_paths[index])
#         features = self.transformer(features)
#
#         # full_embed: numpy array of vector embeddings of domains
#         full_embed = pd.read_csv('pred/pred_domain.csv')
#
#         # each_embed: numpy array of vector embeddings of single domains
#         each_embed = full_embed[self.domain_id]
#
#         return features, self.domain_id[index], full_embed, each_embed[index]
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.dataset
#         return fmt_str
