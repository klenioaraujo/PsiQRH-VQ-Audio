# ===== VQ ÁUDIO + ΨQRH BENCHMARK COMPLETO (LOCAL) =====

import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from scipy.signal import sawtooth  # ✅ Correção crítica: sawtooth vem do scipy.signal

print("=== INICIANDO VQ AUDIO + PSI-QRH BENCHMARK COMPLETO ===")

# ==================== 1. GERAÇÃO DE ÁUDIO SINTÉTICO REALISTA ====================
print("\n1. Gerando dataset de audio sintetico realista...")

def generate_realistic_audio(duration=2.0, sr=22050):
    """Gera audio sintetico realista com multiplos instrumentos"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Violino (onda serrada com vibrato)
    violin_freq = 440 + 5 * np.sin(2 * np.pi * 6 * t)  # vibrato
    violin = 0.4 * sawtooth(2 * np.pi * violin_freq * t)  # ✅ sawtooth do scipy
    
    # Piano (multiplos harmonicos)
    piano = 0.3 * np.sin(2*np.pi*220*t) + 0.2 * np.sin(2*np.pi*440*t) + 0.1 * np.sin(2*np.pi*880*t)
    piano *= np.exp(-0.5 * t)  # decay
    
    # Bateria (transientes)
    drum = np.zeros_like(t)
    drum_peaks = [0.1, 0.5, 1.2, 1.8]  # posicoes em segundos
    for peak in drum_peaks:
        idx = int(peak * sr)
        if idx < len(drum):
            drum[idx:idx+100] = 0.8 * np.exp(-50 * (np.arange(100) / 100.0))
    
    # Voz (formantes)
    voice = 0.3 * np.sin(2*np.pi*200*t) + 0.2 * np.sin(2*np.pi*800*t) + 0.1 * np.sin(2*np.pi*1200*t)
    voice *= (0.5 + 0.5 * np.sin(2*np.pi*3*t))  # modulacao
    
    # Combina os sons
    combined = violin + piano + drum + voice
    if np.max(np.abs(combined)) > 0:
        combined = 0.3 * combined / np.max(np.abs(combined))  # normaliza
    else:
        combined = np.zeros_like(combined)
    
    # Adiciona ruido realista
    noise = 0.01 * np.random.normal(0, 1, len(combined))
    
    return combined + noise

# Gera dataset balanceado
audio_classes = ['violin', 'piano', 'drum', 'voice', 'mixed']
samples_per_class = 50
audio_dataset = {}

sr = 22050
duration = 2.0
n_samples = int(sr * duration)

for class_name in audio_classes:
    class_audios = []
    for i in range(samples_per_class):
        if class_name == 'violin':
            t = np.linspace(0, duration, n_samples)
            audio = 0.6 * sawtooth(2 * np.pi * (440 + i) * t)  # ✅ corrigido
            audio += 0.1 * np.random.normal(0, 1, len(audio))
        elif class_name == 'piano':
            audio = generate_realistic_audio()
        elif class_name == 'drum':
            audio = np.random.normal(0, 0.1, n_samples)
            for _ in range(10):
                idx = np.random.randint(50, n_samples - 100)
                audio[idx:idx+100] += 0.5 * np.exp(-50 * np.arange(100) / 100.0)
        elif class_name == 'voice':
            t = np.linspace(0, duration, n_samples)
            audio = 0.4 * np.sin(2*np.pi*200*t) + 0.3 * np.sin(2*np.pi*800*t)
            audio *= (0.7 + 0.3 * np.sin(2*np.pi*2*t))
        else:  # mixed
            audio = generate_realistic_audio()
        
        # Normalização segura
        if np.max(np.abs(audio)) > 0:
            audio = 0.9 * audio / np.max(np.abs(audio))
        else:
            audio = np.zeros_like(audio)
        
        # Garante tamanho fixo
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        else:
            audio = audio[:n_samples]
        
        class_audios.append(audio)
    
    audio_dataset[class_name] = class_audios

print(f"Dataset gerado: {len(audio_classes)} classes, {samples_per_class} amostras cada")

# ==================== 2. TREINAMENTO DO CODEBOOK VQ ====================
print("\n2. Treinando codebook VQ com audio realista...")

all_frames = []
for class_name, audios in audio_dataset.items():
    for audio in audios:
        # Garante que audio tenha pelo menos 512 amostras
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)))
        frames = librosa.util.frame(audio, frame_length=512, hop_length=256).T
        if len(frames) > 0:
            # Subamostragem para acelerar treino
            step = max(1, len(frames) // 200)
            selected_frames = frames[::step]
            all_frames.extend(selected_frames)

all_frames = np.array(all_frames)
print(f"Total de frames para treino: {all_frames.shape}")

# Treina codebook com K-means
kmeans = MiniBatchKMeans(n_clusters=256, batch_size=1024, random_state=42, n_init=3, max_iter=100)
kmeans.fit(all_frames)
codebook = kmeans.cluster_centers_

print(f"Codebook treinado: {codebook.shape} -> 256 atomos acusticos")

# ==================== 3. ENCODING VQ DOS AUDIOS ====================
def vq_encode_audio(audio, kmeans_model):
    """Converte audio para sequencia de indices VQ"""
    if len(audio) < 512:
        audio = np.pad(audio, (0, 512 - len(audio)))
    frames = librosa.util.frame(audio, frame_length=512, hop_length=256).T
    if len(frames) == 0:
        return []
    indices = kmeans_model.predict(frames)
    return indices.tolist()

# Codifica todo o dataset
encoded_dataset = {}
for class_name, audios in audio_dataset.items():
    encoded_audios = []
    for audio in audios:
        indices = vq_encode_audio(audio, kmeans)
        # Garante sequência mínima
        if len(indices) == 0:
            indices = [0] * 10
        encoded_audios.append(indices[:500])
    encoded_dataset[class_name] = encoded_audios

print("Dataset codificado com VQ")

# ==================== 4. IMPLEMENTAÇÃO PSI-QRH PARA AUDIO ====================
class SpectralQRHAudio(nn.Module):
    """PSI-QRH Layer otimizada para sequencias acusticas VQ"""
    
    def __init__(self, embed_dim=64, alpha=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha
        
        self.theta_left = nn.Parameter(torch.tensor(0.15))
        self.omega_left = nn.Parameter(torch.tensor(0.08))
        self.phi_left = nn.Parameter(torch.tensor(0.04))
        self.theta_right = nn.Parameter(torch.tensor(0.12))
        self.omega_right = nn.Parameter(torch.tensor(0.06))
        self.phi_right = nn.Parameter(torch.tensor(0.03))
    
    def spectral_filter(self, k):
        """Filtro espectral adaptado para caracteristicas acusticas"""
        epsilon = 1e-10
        k_mag = torch.abs(k) + epsilon
        phase = torch.atan(torch.log(k_mag + 1.0))
        return torch.exp(1j * self.alpha * phase)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        if embed_dim % 4 != 0:
            new_dim = ((embed_dim + 3) // 4) * 4
            x = nn.functional.pad(x, (0, new_dim - embed_dim))
            embed_dim = new_dim
        
        x_quat = x.view(batch_size, seq_len, embed_dim // 4, 4)
        
        x_fft = torch.fft.fft(x_quat, dim=2)
        
        freqs = torch.fft.fftfreq(embed_dim // 4)
        k = 2 * np.pi * freqs.view(1, 1, -1, 1).to(x.device)
        filter_response = self.spectral_filter(k)
        
        x_filtered_fft = x_fft * filter_response
        x_filtered = torch.fft.ifft(x_filtered_fft, dim=2).real
        
        return x_filtered.reshape(batch_size, seq_len, embed_dim)

class PsiQRHAudioClassifier(nn.Module):
    """Classificador PSI-QRH para audio VQ"""
    
    def __init__(self, vocab_size=256, embed_dim=64, num_classes=5, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.qrh_layers = nn.Sequential(*[
            SpectralQRHAudio(embed_dim) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.qrh_layers(x)
        x = x.permute(0, 2, 1)
        return self.classifier(x)

# ==================== 5. DATASET E TREINAMENTO ====================
class VQAudioDataset(Dataset):
    def __init__(self, encoded_data, class_to_idx):
        self.data = []
        self.labels = []
        
        for class_name, sequences in encoded_data.items():
            for seq in sequences:
                if len(seq) > 5:
                    self.data.append(seq)
                    self.labels.append(class_to_idx[class_name])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Padding ou truncamento para 500
        if len(sequence) < 500:
            sequence = sequence + [0] * (500 - len(sequence))
        else:
            sequence = sequence[:500]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Prepara dados para treino
class_to_idx = {name: idx for idx, name in enumerate(audio_classes)}
dataset = VQAudioDataset(encoded_dataset, class_to_idx)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Garante que train_size > 0
if train_size == 0:
    train_size = 1
    test_size = len(dataset) - 1
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=min(16, train_size), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=min(32, test_size), shuffle=False)

print(f"Dataset preparado: {len(dataset)} amostras ({train_size} treino, {test_size} teste)")

# ==================== 6. TREINAMENTO DO MODELO ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = PsiQRHAudioClassifier(
    vocab_size=256,
    embed_dim=64,
    num_classes=len(audio_classes),
    num_layers=2
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

print(f"Modelo PSI-QRH inicializado: {sum(p.numel() for p in model.parameters()):,} parametros")

# Funcoes de treino
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted')

print("\n3. Iniciando treinamento PSI-QRH...")
training_history = []

# Treina por até 10 épocas (ou menos se dataset pequeno)
num_epochs = min(10, max(3, len(train_loader)))

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_acc, val_f1 = evaluate(model, test_loader)
    
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_f1': val_f1
    })
    
    print(f"Epoca {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# ==================== 7. RESULTADOS E ANÁLISE ====================
final_val_acc = training_history[-1]['val_accuracy'] if training_history else 0.0
final_val_f1 = training_history[-1]['val_f1'] if training_history else 0.0

print(f"\nRESULTADOS FINAIS:")
print(f"  Acuracia Validacao: {final_val_acc:.4f}")
print(f"  F1-Score Validacao: {final_val_f1:.4f}")

# Salva resultados
qrh_vq_results = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "device": str(device),
    "framework": "PyTorch + PSI-QRH + VQ-Audio",
    "vq_codebook_size": 256,
    "audio_classes": audio_classes,
    "dataset_size": len(dataset),
    "model_parameters": sum(p.numel() for p in model.parameters()),
    "final_metrics": {
        "validation_accuracy": final_val_acc,
        "validation_f1": final_val_f1
    },
    "training_history": training_history,
    "architecture": {
        "embed_dim": 64,
        "qrh_layers": 2,
        "vocab_size": 256
    }
}

with open("./qrh_vq_audio_benchmark.json", "w") as f:
    json.dump(qrh_vq_results, f, indent=2)

print("Benchmark salvo em: ./qrh_vq_audio_benchmark.json")

# ==================== 8. VISUALIZAÇÃO DOS RESULTADOS ====================
plt.figure(figsize=(15, 5))

# Gráfico 1: Loss e Acuracia
if training_history:
    plt.subplot(1, 3, 1)
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    train_accs = [h['train_accuracy'] for h in training_history]
    val_accs = [h['val_accuracy'] for h in training_history]

    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, train_accs, 'g--', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    plt.xlabel('Época')
    plt.ylabel('Métrica')
    plt.title('Treinamento PSI-QRH + VQ Audio')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Gráfico 2: Codebook VQ
plt.subplot(1, 3, 2)
for i in range(min(16, len(codebook))):
    plt.plot(codebook[i][:100], alpha=0.7, linewidth=0.8)
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
plt.title('16 Primeiros Átomos do Codebook VQ')
plt.grid(True, alpha=0.3)

# Gráfico 3: Exemplo de áudio codificado
plt.subplot(1, 3, 3)
example_audio = audio_dataset['mixed'][0]
t_plot = np.arange(len(example_audio[:1000])) / 22050
plt.plot(t_plot, example_audio[:1000], 'b-', alpha=0.7, linewidth=0.9)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Exemplo: Áudio Misto (0–0.045s)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./qrh_vq_audio_results.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\n✅ BENCHMARK COMPLETO!")
print(f"  Codebook VQ: 256 átomos acústicos")
print(f"  Modelo PSI-QRH: {sum(p.numel() for p in model.parameters()):,} parâmetros")
print(f"  Acurácia Final: {final_val_acc:.1%}")
print(f"  Arquitetura: Áudio → VQ → PSI-QRH → Classificação")
print(f"  Resultados salvos:")
print(f"    • JSON: ./qrh_vq_audio_benchmark.json")
print(f"    • PNG:  ./qrh_vq_audio_results.png")