import numpy as np
import torch
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import torch.utils.data as data
import os


REGIONS_TO_IDX={'clipart': 0,'infograph': 1,'painting': 2,'quickdraw': 3, 'real': 4, 'sketch': 5}


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

        # Do we have labels?
        # self.labels = []

        self.domain_id = []

        self.loader = default_loader
        if transformer is None:
            self.transformer = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])
        else:
            self.transformer = transformer

        # Returns domain name and associated index.
        # To Do: Won't index be same as the domain id?
        if isinstance(domains, list):
            for i, d in enumerate(domains):
                self.read_single_domain(d,i)
        else:
            self.read_single_domain(domains,0)

        # Remove line below?
        # self.labels = torch.LongTensor(self.labels)

        # TO DO: Is domain id same as domain index above?
        self.domain_id = torch.LongTensor(self.domain_id)
        # self.classes = 7

    def read_single_domain(self, domain, id):
        """
        What does this do?
        :param domain:
        :param id:
        :return:
        """
        if self.train:
            # To Do
        elif self.val:
            # To Do
        else:
            # TO DO

        for file_name in file_names:
            self.read_single_file(file_name, id)

    # def read_single_file(self, filename, id):
    #     domain_images_list_path = os.path.join(self.data_root, filename)
    #     with open(domain_images_list_path, 'r') as files_list:
    #         for line in files_list:
    #             line = line.strip()
    #             local_path, _, class_id = read_split_line(line)
    #             self.image_paths.append(os.path.join(self.data_root, 'kfold', local_path))
    #             self.labels.append(class_id-1)
    #             self.domain_id.append(id)
    #             self.attributes.append(0)

    def get_domain_embeddings(self):


    def get_domains(self):
        return self.domain_id, self.n_doms

    # def get_labels(self):
    #     return self.labels

    def __getitem__(self, index):
        features = self.loader(self.image_paths[index])
        features = self.transformer(features)
        return features, self.attributes[index], self.domain_id[index], self.labels[index]

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.dataset
        return fmt_str