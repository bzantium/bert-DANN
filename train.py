"""Adversarial adaptation to train target encoder."""

import torch
from utils import make_cuda
import torch.nn as nn
import param
import torch.optim as optim
from utils import save_model


def train(args, encoder, cls_classifier, dom_classifier,
          src_data_loader, src_data_loader_eval,
          tgt_data_loader, tgt_data_loader_all):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    cls_classifier.train()
    dom_classifier.train()

    # setup criterion and optimizer
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(cls_classifier.parameters()) +
                           list(dom_classifier.parameters()),
                           lr=param.c_learning_rate)

    ####################
    # 2. train network #
    ####################

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((src_reviews, src_mask, src_labels), (tgt_reviews, tgt_mask, _)) in data_zip:
            src_reviews = make_cuda(src_reviews)
            src_mask = make_cuda(src_mask)
            src_labels = make_cuda(src_labels)
            tgt_reviews = make_cuda(tgt_reviews)
            tgt_mask = make_cuda(tgt_mask)

            # extract and concat features
            src_feat = encoder(src_reviews, src_mask)
            tgt_feat = encoder(tgt_reviews, tgt_mask)
            feat_concat = torch.cat((src_feat, tgt_feat), 0)
            src_preds = cls_classifier(src_feat)
            dom_preds = dom_classifier(feat_concat, alpha=args.alpha)

            # prepare real and fake label
            optimizer.zero_grad()
            label_src = make_cuda(torch.ones(src_feat.size(0)))
            label_tgt = make_cuda(torch.zeros(tgt_feat.size(0)))
            label_concat = torch.cat((label_src, label_tgt), 0).long()
            loss_cls = CELoss(src_preds, src_labels)
            loss_dom = CELoss(dom_preds, label_concat)
            loss = loss_cls + args.beta * loss_dom

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f dom_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len(src_data_loader),
                         loss_cls.item(),
                         loss_dom.item()))

        evaluate(encoder, cls_classifier, src_data_loader)
        evaluate(encoder, cls_classifier, src_data_loader_eval)
        evaluate(encoder, cls_classifier, tgt_data_loader_all)

    save_model(encoder, param.encoder_path)
    save_model(cls_classifier, param.cls_classifier_path)
    save_model(dom_classifier, param.dom_classifier_path)

    return encoder, cls_classifier, dom_classifier


def evaluate(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, mask, labels) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        labels = make_cuda(labels)

        with torch.no_grad():
            feat = encoder(reviews, mask)
            preds = classifier(feat)
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    return acc
