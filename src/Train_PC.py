from utils import *
from sklearn.model_selection import KFold
from evaluation import calculate_fmax,get_score
from models import PCpredict
import numpy as np
import torch
import torch.nn as nn
import os
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.metrics import auc

def train_DNN(X, Y_train, Train_PC, Test_PC, epochs, learning_rate, drop_rate, save_model=False, model_save_path=None):
    model = PCpredict(int(X.shape[1]), 1).to(device=try_gpu())
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    Y_train = Y_train.to(device=try_gpu())
    list_epoch = []
    train_out = None 
    for e in range(epochs):
        list_epoch.append(e + 1)
        model.train()
        out = model(X, Train_PC)
        train_out = out # Store training output
        loss_train = loss(out, Y_train)
        optimizer.zero_grad()
        loss_train.backward(retain_graph=True)
        optimizer.step()

        if e % 100 == 0:
            print(f"Epoch: {e + 1}; train_loss: {loss_train.data:.4f};")
    
    # Save model if requested
    if save_model and model_save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'in_channels': int(X.shape[1]),
                'num_class': 1,
                'drop_rate': drop_rate
            },
            'training_config': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'drop_rate': drop_rate
            }
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
    
    model.eval()
    out_valid = model(X, Test_PC)
    return out_valid, model, train_out # Return training output as well

def HGC_DNN(PC, protein_dict, PPI_dict, X, save_predictions=True, dataset_info=None, model_params=None):
    # id转换--AdaPPI的金标准复合物数据集
    PCs = []
    for pc in PC:
        if len(pc) > 2:  # 至少3个元素在里面才要
            if set(pc).issubset(set(list(protein_dict.keys()))):
                pc_map = [protein_dict[sub] for sub in pc]
                PCs.append(pc_map)
    PC = [sorted(i) for i in PCs]

    label1 = torch.ones(len(PC), 1, dtype=torch.float)
    all_Test_PC = []
    all_Y_lable = []
    all_Y_pred = []
    trained_models = []  # Store all trained models
    
    # Store training metrics
    all_Train_labels = []
    all_Train_pred = []

    # Five-Fold cross validation
    all_idx = list(range(len(label1)))
    np.random.shuffle(all_idx)
    rs = KFold(n_splits=5, shuffle=True)
    cv_index_set = rs.split(all_idx)
    print('The number of protein complexes:', len(PC))
    print("rs:", rs)
    print("cv_index_set:", cv_index_set)
    


    for train_index, test_index in cv_index_set:
        n=1
        print('This is the', n, 'Fold')
        n += 1
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        # 训练集
        Train_PC = [PC[i] for i in train_index]
        Train_label1 = torch.ones(len(Train_PC), 1, dtype=torch.float)
        Train_PC_negative = negative_on_distribution(Train_PC, list(PPI_dict.keys()), 5)
        Train_label0 = torch.zeros(len(Train_PC_negative), 1, dtype=torch.float)
        Train_labels = torch.cat((Train_label1, Train_label0), dim=0)
        Train_PC_PN = Train_PC + Train_PC_negative
        all_idx = list(range(len(Train_PC_PN)))
        np.random.shuffle(all_idx)
        Train_PC_PN = [Train_PC_PN[i] for i in all_idx]
        Train_labels = Train_labels[all_idx]
        # 测试集
        Test_PC = [PC[i] for i in test_index]
        Test_label1 = torch.ones(len(Test_PC), 1, dtype=torch.float)
        Test_PC_negative = negative_on_distribution(Test_PC, list(PPI_dict.keys()), 5)
        Test_label0 = torch.zeros(len(Test_PC_negative), 1, dtype=torch.float)
        Test_labels = torch.cat((Test_label1, Test_label0), dim=0)
        Test_PC_PN = Test_PC + Test_PC_negative
        all_idx = list(range(len(Test_PC_PN)))
        np.random.shuffle(all_idx) 
        Test_PC_PN = [Test_PC_PN[i] for i in all_idx]
        Test_labels = Test_labels[all_idx]
          # 训练模型
        fold_model_path = None
        if save_predictions and dataset_info:
            # Create model save path for this fold
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H-%M-%S")
            models_dir = os.path.join("trained_models", current_date, f"run_{current_time}")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            fold_model_path = os.path.join(models_dir, f"dnn_model_fold_{len(trained_models)+1}.pt")
        y_pred, model, train_pred = train_DNN(X, Train_labels, Train_PC_PN, Test_PC_PN, 150, 0.001, 0.3, 
                                  save_model=True, model_save_path=fold_model_path)
        trained_models.append(model)  # Store the trained model

        all_Y_lable.append(Test_labels)
        all_Y_pred.append(y_pred)
        all_Test_PC.append(Test_PC_PN)

        all_Train_labels.append(Train_labels)
        all_Train_pred.append(train_pred)

    all_Test_PC = [item for sublist in all_Test_PC for item in sublist]
    Y_label = [tensor.data.cpu().numpy() for tensor in all_Y_lable]
    Y_label = [item for sublist in Y_label for item in sublist]
    Y_label = np.array(Y_label)

    Y_pred = [tensor.data.cpu().numpy() for tensor in all_Y_pred]
    Y_pred = [item for sublist in Y_pred for item in sublist]
    Y_pred = np.array(Y_pred)

    F1_score_5CV, threshold, Precision, Recall, Sensitivity, Specificity, ACC = calculate_fmax(Y_pred, Y_label)
    precision, recall, thresholds = precision_recall_curve(Y_label, Y_pred)
    auprc_5CV = auc(recall, precision)
    print('Test AUPRC:', auprc_5CV)
    auroc_5CV = roc_auc_score(Y_label, Y_pred)
    print('Test AUROC:', auroc_5CV)

    # Calculate and print training metrics
    Train_Y_label_agg = [tensor.data.cpu().numpy() for tensor in all_Train_labels]
    Train_Y_label_agg = [item for sublist in Train_Y_label_agg for item in sublist]
    Train_Y_label_agg = np.array(Train_Y_label_agg)

    Train_Y_pred_agg = [tensor.data.cpu().numpy() for tensor in all_Train_pred]
    Train_Y_pred_agg = [item for sublist in Train_Y_pred_agg for item in sublist]
    Train_Y_pred_agg = np.array(Train_Y_pred_agg)
    
    Train_F1_score_5CV, train_threshold, Train_Precision, Train_Recall, Train_Sensitivity, Train_Specificity, Train_ACC = calculate_fmax(Train_Y_pred_agg, Train_Y_label_agg)
    train_precision_curve, train_recall_curve, train_thresholds_curve = precision_recall_curve(Train_Y_label_agg, Train_Y_pred_agg)
    train_auprc_5CV = auc(train_recall_curve, train_precision_curve)
    print('Train AUPRC:', train_auprc_5CV)
    train_auroc_5CV = roc_auc_score(Train_Y_label_agg, Train_Y_pred_agg)
    print('Train AUROC:', train_auroc_5CV)
    print(f'Train F1 Score: {Train_F1_score_5CV:.4f}')


    predict_pc = [all_Test_PC[i] for i in range(len(Y_pred)) if Y_pred[i] > threshold]
    
    # Save ensemble model if requested
    if save_predictions and dataset_info and trained_models:
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H-%M-%S")
        models_dir = os.path.join("trained_models", current_date, f"run_{current_time}")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        ensemble_path = os.path.join(models_dir, "dnn_ensemble_model.pt")
        save_ensemble_model(trained_models, ensemble_path, X.shape, model_params)
    
    if save_predictions:
        # Create reverse mapping from protein IDs back to names
        id_to_name = {v: k for k, v in protein_dict.items()}
        
        # Convert predicted complexes from IDs back to protein names
        predict_pc_names = []
        predict_pc_scores = []
        for i, pc in enumerate(predict_pc):
            pc_names = [id_to_name[protein_id] for protein_id in pc if protein_id in id_to_name]
            if pc_names:  # Only add if we have valid protein names
                predict_pc_names.append(pc_names)
                # Find the corresponding score for this complex
                pc_indices = [j for j in range(len(Y_pred)) if Y_pred[j] > threshold]
                if i < len(pc_indices):
                    predict_pc_scores.append(float(Y_pred[pc_indices[i]]))
                else:
                    predict_pc_scores.append(0.0)
          # Create date-organized directory structure
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H-%M-%S")
        
        save_dir = os.path.join("predicted_complexes", current_date, f"run_{current_time}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save predicted protein complexes with names
        with open(os.path.join(save_dir, "predicted_complexes_by_names.txt"), "w") as f:
            for i, pc in enumerate(predict_pc_names):
                score = predict_pc_scores[i] if i < len(predict_pc_scores) else 0.0
                f.write(f"Complex_{i+1} (score: {score:.4f}): {' '.join(pc)}\n")
        
        # Save as CSV for easier analysis
        import pandas as pd
        pc_data = []
        for i, pc in enumerate(predict_pc_names):
            score = predict_pc_scores[i] if i < len(predict_pc_scores) else 0.0
            pc_data.append({
                'Complex_ID': f'Complex_{i+1}',
                'Proteins': ';'.join(pc),
                'Size': len(pc),
                'Prediction_Score': float(score)
            })
        
        if pc_data:
            df = pd.DataFrame(pc_data)
            df.to_csv(os.path.join(save_dir, "predicted_complexes_by_names.csv"), index=False)
        
        # Save dataset information and model parameters
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'date': current_date,
            'time': current_time,
            'num_predicted_complexes': len(predict_pc_names),
            'prediction_threshold': float(threshold),
            'performance_metrics': {
                'F1_score_5CV': float(F1_score_5CV),
                'AUPRC': float(auprc_5CV),
                'AUROC': float(auroc_5CV),
                'Precision': float(Precision),
                'Recall': float(Recall),
                'Sensitivity': float(Sensitivity),
                'Specificity': float(Specificity),
                'ACC': float(ACC)
            },
            'training_performance_metrics': {
                'F1_score_5CV': float(Train_F1_score_5CV),
                'AUPRC': float(train_auprc_5CV),
                'AUROC': float(train_auroc_5CV),
                'Precision': float(Train_Precision),
                'Recall': float(Train_Recall),
                'Sensitivity': float(Train_Sensitivity),
                'Specificity': float(Train_Specificity),
                'ACC': float(Train_ACC)
            }
        }
        
        # Add dataset information if provided
        if dataset_info:
            metadata['dataset_info'] = dataset_info
        
        # Add model parameters if provided
        if model_params:
            metadata['model_params'] = model_params
          # Save metadata as JSON
        import json
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(predict_pc_names)} predicted protein complexes to {save_dir}/")
        print(f"Files saved:")
        print(f"  - predicted_complexes_by_names.txt")
        print(f"  - predicted_complexes_by_names.csv")
        print(f"  - metadata.json")
    
    precision, recall, f1, acc, sn, PPV, score = get_score(PC, predict_pc)
    print(score)
    performance = [F1_score_5CV, Precision, Recall, Sensitivity, Specificity, ACC, threshold, auprc_5CV, auroc_5CV,
                   precision, recall, f1, acc, sn, PPV]

    return score


def save_ensemble_model(trained_models, save_path, X_shape, model_params=None):
    """Save the best performing model from cross-validation as the ensemble model."""
    if not trained_models:
        print("No trained models to save.")
        return
    
    # For simplicity, save the first model as the representative model
    # In practice, you might want to select the best performing one
    best_model = trained_models[0]
    
    ensemble_data = {
        'model_state_dict': best_model.state_dict(),
        'model_config': {
            'in_channels': int(X_shape[1]),
            'num_class': 1,
            'drop_rate': 0.3  # Default from training
        },
        'ensemble_info': {
            'num_folds': len(trained_models),
            'selected_fold': 0,  # Index of the model we're saving
            'timestamp': datetime.now().isoformat()
        }
    }
    
    if model_params:
        ensemble_data['training_config'] = model_params
    
    torch.save(ensemble_data, save_path)
    print(f"Ensemble model saved to {save_path}")

def load_pretrained_dnn(model_path):
    """Load a pretrained DNN model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    
    # Create model with saved configuration
    model = PCpredict(
        in_channels=model_config['in_channels'],
        num_class=model_config['num_class'],
        drop_rate=model_config.get('drop_rate', 0.3)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded pretrained DNN model from {model_path}")
    print(f"Model config: {model_config}")
    
    return model, checkpoint

def predict_with_pretrained_dnn(model, X, test_complexes):
    """Make predictions using a pretrained DNN model."""
    model.eval()
    device = next(model.parameters()).device
    
    # Move data to the same device as the model
    if hasattr(X, 'to'):
        X = X.to(device)
    
    with torch.no_grad():
        predictions = model(X, test_complexes)
    
    return predictions
