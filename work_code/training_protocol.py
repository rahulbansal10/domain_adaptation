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

class training_protocol():
    def __init__(self, device):
        self.device = device
    
    def train_class_differentiator_module(self, data_loader, CD, epochs):
        print("Training class differentiator module")
        loss_ = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(CD.parameters(), lr = 1e-3)
        epoch_loss = list()
        for epoch in range(epochs):
            batch_loss = list()
            total, correct = 0, 0
            for i, data in enumerate(data_loader, 0):
                X1, X2, Labels = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([], dtype = int).to(self.device)
                for j in range(10):
                    perm = np.random.permutation(len(data[0]))
                    inputs1, labels1 = data[0].float(), data[1]
                    inputs2, labels2 = inputs1[perm], labels1[perm]
                    y = (labels1==labels2)
                    x1_pos, x2_pos = inputs1[y], inputs2[y]
                    x1_neg, x2_neg = inputs1[~y], inputs2[~y]
                    id_neg = np.random.choice(len(x1_neg), len(x1_pos))
                    x1 = torch.cat((x1_pos, x1_neg[id_neg]), 0)
                    x2 = torch.cat((x2_pos, x2_neg[id_neg]), 0)
                    labels = torch.tensor(np.concatenate([np.ones(len(x1_pos), dtype = int), np.zeros(len(x1_neg[id_neg]), dtype = int)])).to(self.device)
                    X1, X2, Labels = torch.cat((X1, x1), 0), torch.cat((X2, x2), 0), torch.cat((Labels, labels))
                logits = CD(X1, X2)
                optimizer.zero_grad()
                loss = loss_(logits, Labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                m = nn.Softmax(dim=1)
                outputs = m(logits)
                _, predicted = torch.max(outputs.data, 1)
                total += Labels.size(0)
                correct += (predicted == Labels).sum().item()

            if(epoch%5==0):
                print("epoch {} : Classification Loss {:2f} Accuracy {:2f}".format(epoch, np.mean(batch_loss), 100*(correct / total)))
        torch.save(CD.state_dict(), "../modules/CD_module")
        print('Finished Training')

    def train_domain_identifier_module(self, data_loader, DI, epochs):
        print("Training domain identifier module")
        loss_ = nn.CrossEntropyLoss()
        optimizer = optim.SGD(DI.parameters(), lr = 0.003)
        epoch_loss = list()
        for epoch in range(epochs):
            batch_loss = list()
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data[0].float(), data[1]
                logits = DI(inputs)
                optimizer.zero_grad()
                loss = loss_(logits, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                batch_loss.append(loss.item())

            if(epoch%5==0):
                print("epoch {} : Classification Loss {:2f}".format(epoch, np.mean(batch_loss)))
        print('Finished Training')
                    
    def train_content_module(self, data_loader, C, content_clf, epochs):
        print("Training Content Separator")
        loss_ = nn.CrossEntropyLoss().to(self.device)
        params = list(C.parameters()) + list(content_clf.parameters())
        optimizer = optim.SGD(params, lr = 0.001)
        epoch_loss = list()
        for epoch in range(epochs):  # loop over the dataset multiple times
            batch_loss = list()
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].float(), data[1]
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                content = C(inputs)
                logits = content_clf(content)
                loss = loss_(logits, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                batch_loss.append(loss.item())

            if(epoch%5==0):
                print("epoch {} : Classification Loss {:2f}".format(epoch, np.mean(batch_loss)))
                #self.test_content(data_loader, C, content_clf)
        torch.save(C.state_dict(), "../modules/Cs_module")
        torch.save(content_clf.state_dict(), "../modules/content_clf_module")
        print('Finished Training')
    
    def train_style_module(self, data_loader, C, S, R, adv_clf, epochs):
        print("Training Style Separator")
        loss_mse = nn.MSELoss().to(self.device)
        loss_cent = nn.CrossEntropyLoss().to(self.device)
        loss_ent = HLoss().to(self.device)
        params = list(S.parameters()) + list(R.parameters())
        optimizer_enc = optim.Adam(params)
        optimizer_adv = optim.SGD(adv_clf.parameters(), lr = 0.001)
          
        epoch_adv_loss, epoch_rec_loss = list(), list()
        for epoch in range(epochs):  #loop over the dataset multiple times
            batch_adv_loss, batch_rec_loss, batch_ent_loss = list(), list(), list()
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].float(), data[1]
                # forward + backward + optimize
                optimizer_enc.zero_grad()
                optimizer_adv.zero_grad()
                content = C(inputs).detach()
                style = S(inputs)
                inputs_hat = R(content, style)
                logits = adv_clf(style)
                loss1 = loss_mse(inputs, inputs_hat)
                loss2 = -loss_cent(logits, labels)
                if(i%3==0):
                    loss = loss1 + loss2
                    loss.backward()
                    optimizer_enc.step()
                    batch_rec_loss.append(loss1.item())
                    batch_adv_loss.append(loss2.item())
                else:
                    loss = -(loss2)
                    loss.backward()
                    optimizer_adv.step()
                    batch_adv_loss.append(loss2.item())
                
        
            if(epoch%5==0):
                print("epoch {} : Adversarial Loss {:2f} Reconstruction Loss {:2f}".format(epoch, np.mean(batch_adv_loss),np.mean(batch_rec_loss)))   
        torch.save(S.state_dict(), "../modules/Ss_module")
        torch.save(R.state_dict(), "../modules/Rs_module")
        torch.save(adv_clf.state_dict(), "../modules/adv_clf")
        print('Finished Training')
    
    def test_content(self, data_loader, C, content_clf):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                features, labels = data[0].float(), data[1]
                content = C(features).detach()
                logits = content_clf(content)
                m = nn.Softmax(dim=1)
                outputs = m(logits)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy of the network on the data: {:2f}".format(100 * (correct / total)))
    
    def test_style(self, data_loader, S, adv_clf):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                features, labels = data[0].float(), data[1]
                style = S(features)
                logits = adv_clf(style)
                m = nn.Softmax(dim=1)
                outputs = m(logits)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy of the network on the data: {:2f}".format(100 * (correct / total)))
    
    def test_class_differentiator_module(self, data_loader, CD, epochs):
        print("Testing class differentiator module")
        loss_ = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(CD.parameters(), lr = 1e-3)
        epoch_loss = list()
        for epoch in range(epochs):
            batch_loss = list()
            total, correct = 0, 0
            with torch.no_grad():
                for i, data in enumerate(data_loader, 0):
                    X1, X2, Labels = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([], dtype = int).to(self.device)
                    for j in range(10):
                        perm = np.random.permutation(len(data[0]))
                        inputs1, labels1 = data[0].float(), data[1]
                        inputs2, labels2 = inputs1[perm], labels1[perm]
                        y = (labels1==labels2)
                        x1_pos, x2_pos = inputs1[y], inputs2[y]
                        x1_neg, x2_neg = inputs1[~y], inputs2[~y]
                        id_neg = np.random.choice(len(x1_neg), len(x1_pos))
                        x1 = torch.cat((x1_pos, x1_neg[id_neg]), 0)
                        x2 = torch.cat((x2_pos, x2_neg[id_neg]), 0)
                        labels = torch.tensor(np.concatenate([np.ones(len(x1_pos), dtype = int), np.zeros(len(x1_neg[id_neg]), dtype = int)])).to(self.device)
                        X1, X2, Labels = torch.cat((X1, x1), 0), torch.cat((X2, x2), 0), torch.cat((Labels, labels))
                    logits = CD(X1, X2)
                    m = nn.Softmax(dim=1)
                    outputs = m(logits)
                    _, predicted = torch.max(outputs.data, 1)
                    total += Labels.size(0)
                    correct += (predicted == Labels).sum().item()

                print("epoch {} : Accuracy {:2f}".format(epoch, 100*(correct / total)))    
    
    
    def check_confidence(self, inputs, Ct,content_clf, confidence, source_index_dict, source_features):
        x = Ct(inputs).detach()
        x = content_clf(x).detach()
        m = nn.Softmax(dim=1)
        x = m(x)
        values, indices = torch.max(x, 1)
        source_inputs = torch.tensor([]).to(self.device)
        target_inputs = torch.tensor([]).to(self.device)
        for i in range(len(values)):
            if values[i]>=confidence:
                c = random.choice(source_index_dict[indices[i].item()])
                source_inputs = torch.cat((source_inputs, torch.tensor([source_features[c]]).float().to(self.device)), 0)
                target_inputs = torch.cat((target_inputs, inputs[i].unsqueeze(0)), 0)
        return source_inputs, target_inputs
    
    def train(self, source_index_dict, source_features, source_labels, target_loader, Cs, content_clf, Ss, Rs, Ct, St, Rt, CD, epochs):
        print("Training")
        m = nn.Softmax(dim=1)
        loss_ = nn.CrossEntropyLoss()
        loss_mse = nn.MSELoss()
        optimizer = optim.SGD([{"params":Ct.parameters(), "lr": 0.0001},{"params":St.parameters(), "lr": 0.0},{"params":Rt.parameters(), "lr": 0.0},{"params":CD.parameters(), "lr": 0.0}])
        epoch_loss = list()
        for epoch in range(epochs):  # loop over the dataset multiple times
            batch_loss = list()
            for i, data in enumerate(target_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, _ = data[0].float(), data[1]
                source_inputs, target_inputs = self.check_confidence(inputs, Ct,content_clf, 0.5, source_index_dict, source_features)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                if(len(source_inputs)==0):
                    continue
                perm = np.random.permutation(len(source_inputs))
                source_content, source_style = Cs(source_inputs).detach(), Ss(source_inputs).detach()
                target_content, target_style = Ct(target_inputs), St(target_inputs)
                X1, X2 = source_inputs,  Rt(target_content, source_style) 
                Labels = torch.tensor(np.ones(len(source_inputs)), dtype = int).to(self.device)
                
                logits = CD(X1, X2)
                pdb.set_trace()
                loss = loss_(logits, Labels)
                loss.backward()
                optimizer.step()
                # print statistics
                batch_loss.append(loss.item())

            if(epoch%5==0):
                print("epoch {} : Classification Loss {:2f}".format(epoch, np.mean(batch_loss)))
        torch.save(Ct.state_dict(), "../modules/Ct_module")
        print('Finished Training')