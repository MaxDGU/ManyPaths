import torch
import numpy as np
from utils import calculate_accuracy
from torch.optim import AdamW


def evaluate(meta, dataset, criterion, device, adaptation_steps, return_results=False):
    meta.train()
    meta_loss, meta_acc_list, results = 0.0, [], []
    for task_data in dataset.tasks:
        if len(task_data) == 9:
            X_s_processed, X_s_original, y_s_tensor, \
            X_q_processed, X_q_original, y_q_tensor, \
            num_support_actual, concept_literals, concept_depth = task_data
            
            meta_info = {
                "type": "pcfg_concept",
                "num_support_actual": num_support_actual,
                "concept_literals": concept_literals,
                "concept_depth": concept_depth
            }
        elif len(task_data) == 7:
            X_s_processed, X_s_original, y_s_tensor, \
            X_q_processed, X_q_original, y_q_tensor, old_meta_info = task_data
            meta_info = {"type": "other", "value": old_meta_info }
        else:
            raise ValueError(f"Unexpected task data length: {len(task_data)}")

        X_s, y_s, X_q, y_q = (
            X_s_processed.to(device),
            y_s_tensor.to(device),
            X_q_processed.to(device),
            y_q_tensor.to(device),
        )
        preds, losses, accs = [], [], []
        for steps in adaptation_steps:
            learner = meta.clone()
            # Adaptation on the support set
            for _ in range(steps):
                support_pred = learner(X_s)
                support_loss = criterion(support_pred, y_s)
                learner.adapt(support_loss)

            # Evaluate on the query set (post-adaptation)
            with torch.no_grad():
                pred = learner(X_q)
                preds.append(pred)
                losses.append(criterion(pred, y_q).item())
                accs.append(calculate_accuracy(pred, y_q))

        results.append(
            {
                "meta_info": meta_info,
                "X_s_orig": X_s_original,
                "y_s": y_s,
                "X_q_orig": X_q_original,
                "y_q": y_q,
                "predictions": preds,
                "losses": losses,
                "accuracies": accs,
            }
        )
        if accs:
            meta_acc_list.append(accs[-1])
        if losses:
            meta_loss += losses[-1]

    if meta_acc_list:
        meta_acc = np.mean(meta_acc_list)
    else:
        meta_acc = 0.0
    
    if len(dataset.tasks) > 0:
        meta_loss /= len(dataset.tasks)
    else:
        meta_loss = 0.0

    if return_results:
        return meta_loss, results
    else:
        return meta_loss, meta_acc

def baseline_evaluate(model, dataset, criterion, device, adaptation_steps, return_results=False):
    model.train()
    state_dict = model.state_dict()
    meta_loss, meta_acc_list, results = 0.0, [], []
    for task_data in dataset.tasks:
        if len(task_data) == 9:
            X_s_processed, X_s_original, y_s_tensor, \
            X_q_processed, X_q_original, y_q_tensor, \
            num_support_actual, concept_literals, concept_depth = task_data
            meta_info = {
                "type": "pcfg_concept",
                "num_support_actual": num_support_actual,
                "concept_literals": concept_literals,
                "concept_depth": concept_depth
            }
        elif len(task_data) == 7:
            X_s_processed, X_s_original, y_s_tensor, \
            X_q_processed, X_q_original, y_q_tensor, old_meta_info = task_data
            meta_info = {"type": "other", "value": old_meta_info }
        else:
            raise ValueError(f"Unexpected task data length: {len(task_data)}")

        X_s, y_s, X_q, y_q = (
            X_s_processed.to(device),
            y_s_tensor.to(device),
            X_q_processed.to(device),
            y_q_tensor.to(device),
        )
        model.load_state_dict(state_dict)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        preds, losses, accs = [], [], []
        for i in range(max(adaptation_steps)):
            optimizer.zero_grad()
            support_pred = model(X_s)
            support_loss = criterion(support_pred, y_s)
            support_loss.backward()
            optimizer.step()

            if (i + 1) in adaptation_steps:
                with torch.no_grad():
                    query_pred = model(X_q)
                    preds.append(query_pred)
                    losses.append(criterion(query_pred, y_q).item())
                    accs.append(calculate_accuracy(query_pred, y_q))

        results.append(
            {
                "meta_info": meta_info,
                "X_s_orig": X_s_original,
                "y_s": y_s,
                "X_q_orig": X_q_original,
                "y_q": y_q,
                "predictions": preds,
                "losses": losses,
                "accuracies": accs,
            }
        )
        if accs:
            meta_acc_list.append(accs[-1])
        if losses:
            meta_loss += losses[-1]

    if meta_acc_list:
        meta_acc = np.mean(meta_acc_list)
    else:
        meta_acc = 0.0
        
    if len(dataset.tasks) > 0:
        meta_loss /= len(dataset.tasks)
    else:
        meta_loss = 0.0
        
    if return_results:
        return meta_loss, results
    else:
        return meta_loss, meta_acc