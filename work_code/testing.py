import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import random
import pdb
from loss import HLoss

class testing_protocol():
    def __init__(self, device):
        self.device = device

    def test_content(data_loader, C):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                features, labels = data[0].float(), data[1]
                content = C(features).detach()
                m = nn.Softmax(dim=1)
                outputs = m(content)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * (correct / total)
        print("Accuracy of the network on the data: {:2f}".format(100 * (correct / total)))
        return acc

    def check_confidence(self, inputs, labels, Ct, confidence, source_index_dict, source_features):
        x = Ct(inputs).detach()
        m = nn.Softmax(dim=1)
        x = m(x)
        values, indices = torch.max(x, 1)
        #pdb.set_trace()
        #y = (values>confidence)
        pos_source_inputs = torch.tensor([]).to(self.device)
        pos_target_inputs = torch.tensor([]).to(self.device)
        neg_source_inputs, neg_target_inputs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        ind1 = torch.tensor([], dtype = int).to(self.device)
        for i in range(len(values)):
            if values[i]>=confidence and indices[i]!=labels[i]:
                ind1 = torch.cat((ind1, indices[i].unsqueeze(0)))
                c = random.choice(source_index_dict[indices[i].item()])
                pos_source_inputs = torch.cat((pos_source_inputs, torch.tensor([source_features[c]]).float().to(self.device)), 0)
                pos_target_inputs = torch.cat((pos_target_inputs, inputs[i].unsqueeze(0)), 0)
        #pdb.set_trace()
        for i in range(0):
            perm = np.random.permutation(len(ind1))
            ind2 = ind1[perm]
            y = (ind1 == ind2)
            neg_source_inputs, neg_target_inputs = torch.cat((neg_source_inputs, pos_source_inputs[~y]), 0), torch.cat((neg_target_inputs, pos_target_inputs[~y]), 0)
        
        source_inputs, target_inputs = torch.cat((pos_source_inputs, neg_source_inputs), 0), torch.cat((pos_target_inputs, neg_target_inputs), 0)
        Labels = torch.tensor(np.concatenate([np.ones(len(pos_source_inputs), dtype = int), np.zeros(len(neg_source_inputs), dtype = int)])).to(self.device)
        #pdb.set_trace()
        return source_inputs, target_inputs, Labels
    
    def train(self, source_index_dict, source_features, source_labels, target_features, target_loader, Cs, Ss, Rs, Ct, St, Rt, CD, epochs):
        print("Training")
        m = nn.Softmax(dim=1)
        loss_ = nn.CrossEntropyLoss().to(self.device)
        loss_mse = nn.MSELoss().to(self.device)
        loss_ent = HLoss()
        optimizer = optim.SGD([{"params":Ct.parameters(), "lr": 0.001},{"params":St.parameters(), "lr": 0.0},{"params":Rt.parameters(), "lr": 0.0},{"params":CD.parameters(), "lr": 0.0}])
        epoch_loss = list()
        for epoch in range(epochs):  # loop over the dataset multiple times
            batch_loss = list()
            for i, data in enumerate(target_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].float(), data[1]
                source_inputs, target_inputs, Labels = self.check_confidence(inputs, labels, Ct, 0.7, source_index_dict, source_features)
                # zero the parameter gradients
                pdb.set_trace()
                optimizer.zero_grad()
                # forward + backward + optimize
                if(len(source_inputs)==0):
                    continue
                perm = np.random.permutation(len(source_inputs))
                source_content, source_style = Cs(source_inputs).detach(), Ss(source_inputs).detach()
                target_content, target_style = Ct(target_inputs), St(target_inputs)
                X1, X2 = source_inputs,  Rt(target_content, source_style) 
                # Labels = torch.tensor(np.ones(len(source_inputs)), dtype = int).to(self.device)
                
                logits = CD(X1, X2)
                probs = m(logits)
                val, ind = torch.max(probs, 1)
                #pdb.set_trace()
                loss = loss_(logits, Labels)
                loss.backward()
                optimizer.step()
                # print statistics
                batch_loss.append(loss.item())

            if(epoch%10==0):
                accuracy_t = self.test_content(target_loader, Ct)
                logits_t = Ct(torch.tensor(target_features).float().to(self.device)).detach()
                entropy = loss_ent(logits_t).item()
                print("epoch {} : Classification Loss {:2f} Entropy {:2f} Accuracy {:2f}".format(epoch, np.mean(batch_loss), entropy, accuracy_t))
        torch.save(Ct.state_dict(), "../modules/Ct_module")
        print('Finished Training')
