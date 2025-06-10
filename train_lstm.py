# train_lstm.py
import numpy as np
import matplotlib.pyplot as plt
import torch # usei torch pq tensorflow não é compativel com python 3.12.x
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

X_train = np.load('pipeline/data/X_train_lstm.npy') 
y_train = np.load('pipeline/data/y_train_lstm.npy')
X_val   = np.load('pipeline/data/X_val_lstm.npy')
y_val   = np.load('pipeline/data/y_val_lstm.npy')
X_test  = np.load('pipeline/data/X_test_lstm.npy')
y_test  = np.load('pipeline/data/y_test_lstm.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val_t   = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

batch_size = 32 # recomendo manter 32 mas para melhores testes em GPU use 64 ou 128

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# Instanciar Attention 
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output, return_weights=False):
        attn_weights = torch.softmax(self.attn(lstm_output).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        if return_weights:
            return context, attn_weights
        return context


# LSTM + Attention Model
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # captura dos pesos de atenção do lstm com 2 layers
        context, attn_weights = self.attention(out, return_weights=True)
        x = self.dropout(context)
        x = self.fc1(self.relu(x))
        x = self.fc2(x)
        return x, attn_weights


# Instanciar modelo
model = LSTMWithAttention(input_size=X_train.shape[2]).to(device) 

criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
writer = SummaryWriter(log_dir='runs/lstm_attention')


# avaliação do model
def eval_metrics(model, loader): 
    model.eval()
    running_loss = 0.0 
    preds_transformed = []
    trues_transformed = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            output, _ = model(X_batch)
            loss_transformed = criterion(output, y_batch) 
            running_loss += loss_transformed.item() * X_batch.size(0)
            preds_transformed.append(output.cpu().numpy())
            trues_transformed.append(y_batch.cpu().numpy())

    preds_transformed = np.vstack(preds_transformed)
    trues_transformed = np.vstack(trues_transformed)

    preds_original_scale = np.expm1(preds_transformed)
    trues_original_scale = np.expm1(trues_transformed)

    # Garante que não há valores negativos 
    preds_original_scale[preds_original_scale < 0] = 0 
    trues_original_scale[trues_original_scale < 0] = 0 # não deveria acontecer mas há uma prevenção 


    mse_original_scale = mean_squared_error(trues_original_scale, preds_original_scale)
    rmse_original_scale = np.sqrt(mse_original_scale)
    mae_original_scale = mean_absolute_error(trues_original_scale, preds_original_scale)
    r2_original_scale = r2_score(trues_original_scale, preds_original_scale)

    return mse_original_scale, rmse_original_scale, mae_original_scale, r2_original_scale


# Loop de treino
best_val_loss = float('inf')
patience = 20
counter = 0
num_epochs = 100

train_losses = [] 
val_mses = [] 

for epoch in range(1, num_epochs+1):
    # ======== Treino ========
    model.train()
    running_loss_transformed_scale = 0.0 

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output, _ = model(X_batch)
        loss = criterion(output, y_batch) #Loss calculada em escala transf.
        loss.backward()
        optimizer.step()
        running_loss_transformed_scale += loss.item() * X_batch.size(0)

    train_loss = running_loss_transformed_scale / len(train_loader.dataset)

    # Validação
    val_mse, val_rmse, val_mae, val_r2 = eval_metrics(model, val_loader) 

    # Logs no TensorBoard
    writer.add_scalar('Loss/Train_Transformed_Scale', train_loss, epoch) 
    writer.add_scalar('Loss/Val_Original_Scale_MSE', val_mse, epoch) 
    writer.add_scalar('Metrics/Val_RMSE_Original_Scale', val_rmse, epoch)
    writer.add_scalar('Metrics/Val_MAE_Original_Scale', val_mae, epoch)
    writer.add_scalar('Metrics/Val_R2_Original_Scale', val_r2, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

    train_losses.append(train_loss) 
    val_mses.append(val_mse)
    # marcação por epochs e metricas
    print(f'Epoch {epoch:02d} | Train Loss (Transformed): {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f}')

    # Early Stopping 
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        counter = 0
        torch.save(model.state_dict(), 'pipeline/models/lstm_attention.pt')
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping ativado')
            break

    # Scheduler step
    scheduler.step(val_mse)

writer.close()

# Avaliação em teste
model.load_state_dict(torch.load('pipeline/models/lstm_attention.pt'))
test_mse, test_rmse, test_mae, test_r2 = eval_metrics(model, test_loader) 

print(f'\n[Test LSTM+Attention]')
print(f'MSE:   {test_mse:.4f}')
print(f'RMSE: {test_rmse:.4f}')
print(f'MAE:   {test_mae:.4f}')
print(f'R2:    {test_r2:.4f}')


# Plot das Losses
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Train Loss (Transformed Scale)')
plt.plot(val_mses, label='Val MSE (Original Scale)') 
plt.title('Evolução das Losses/MSE (LSTM + Attention)')
plt.xlabel('Epoch')
plt.ylabel('Loss/MSE')
plt.legend()
plt.grid()
plt.savefig(f'pipeline/plots/lstm/loss_plot{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
plt.show()


# Plot: Pred vs Real
model.eval()
test_preds_list = []
test_trues_list = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output, _ = model(X_batch)
        test_preds_list.append(output.cpu().numpy())
        test_trues_list.append(y_batch.cpu().numpy())
test_preds_transformed = np.vstack(test_preds_list)
test_trues_transformed = np.vstack(test_trues_list)
test_preds_original_scale = np.expm1(test_preds_transformed)
test_trues_original_scale = np.expm1(test_trues_transformed)

# garantia de que nenhuma previsão seja negativa
test_preds_original_scale[test_preds_original_scale < 0] = 0
test_trues_original_scale[test_trues_original_scale < 0] = 0

plt.figure(figsize=(12, 7))
plt.plot(test_trues_original_scale, label='Valores Reais (Test)', color='blue', alpha=0.7)
plt.plot(test_preds_original_scale, label='Previsões (Test)', color='red', linestyle='--', alpha=0.7)
plt.title('Previsões do Modelo LSTM vs. Valores Reais de Ocorrências (Conjunto de Teste)')
plt.xlabel('Ponto Temporal (Dias)')
plt.ylabel('Número de Ocorrências')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pipeline/plots/lstm/predictions_vs_real_test.png')
plt.show()

# Plot dos pesos de attention
model.eval()
sample_X_batch, sample_y_batch = next(iter(test_loader))
sample_output, sample_attn_weights = model(sample_X_batch) 
sample_output_np = np.expm1(sample_output.detach().cpu().numpy())
sample_y_batch_np = np.expm1(sample_y_batch.cpu().numpy()) 
sample_attn_weights_np = sample_attn_weights.detach().cpu().numpy() 
num_samples_to_plot = 2 
window_size = X_train.shape[1] 

plt.figure(figsize=(15, 6 * num_samples_to_plot))
for i in range(min(num_samples_to_plot, sample_X_batch.shape[0])):
    plt.subplot(num_samples_to_plot, 1, i + 1)
    plt.bar(range(window_size), sample_attn_weights_np[i, :], alpha=0.7)
    plt.title(f'Pesos de Atenção para Amostra {i+1} '
              f'(Real: {sample_y_batch_np[i, 0]:.1f}, Previsto: {sample_output_np[i, 0]:.1f})')
    plt.xlabel('Passo Temporal na Janela')
    plt.ylabel('Peso da Atenção')
    plt.xticks(range(window_size))
    plt.grid(True)

plt.tight_layout()
plt.savefig('pipeline/plots/lstm/attention_weights_plot.png')
plt.show()

print("Gráficos salvos em pipeline/plots/lstm/_.png")
print("Modelo salvo em pipeline/models/lstm_attention.pt")
print("Logs TensorBoard em runs/, use 'tensorboard --logdir=runs' no terminal para logs do LSTM com ATTN")
print("Alan Turing deu bença!! 🙏🙏")