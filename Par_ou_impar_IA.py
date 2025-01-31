import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from sklearn.model_selection import KFold

ARQUIVO = "dados_para_treinamento.json"
CAMINHO_ARQUIVO = os.path.join(os.path.dirname(os.path.abspath(__file__)), ARQUIVO)
print(f"Arquivo sendo salvo em: {CAMINHO_ARQUIVO}")  # Mostra o caminho completo do arquivo

# Função para carregar os dados
def carregar_dados(CAMINHO_ARQUIVO):
    try:
        with open(CAMINHO_ARQUIVO, "r") as file:
            dados = json.load(file)
            X_Treino, Y_Treino = dados["X_Treino"], dados["Y_Treino"]
            #print(X_Treino, Y_Treino)
            return X_Treino, Y_Treino
    except FileNotFoundError:
        print("erro ao carregar dados")
        # Dados iniciais se o arquivo não existir
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 20], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]

# Função para salvar os dados
def salvar_dados(CAMINHO_ARQUIVO, X_Treino, Y_Treino):
    with open(CAMINHO_ARQUIVO, "w") as file:
        json.dump({
            "X_Treino": X_Treino,
            "Y_Treino": Y_Treino
        }, file, indent=4)

# Definição do modelo simples com ReLU
class Classificador(nn.Module): 
    def __init__(self):
        super(Classificador, self).__init__()
        self.fc = nn.Linear(1, 8)  # Primeira camada totalmente conectada
        self.relu = nn.ReLU()  # Função de ativação ReLU
        self.output = nn.Linear(8, 1)  # Camada de saída

    def forward(self, x):
        x = self.fc(x)  # Passando pela camada densa
        x = self.relu(x)  # Aplicando a ReLU
        x = self.output(x)  # Camada de saída
        return torch.sigmoid(x)  # Sigmoid na saída para classificação binária

# Função para classificar par ou ímpar
def classificar_par_impar(model, numero, ultimo_numero):
    with torch.no_grad():  # Desabilitando o cálculo do gradiente
        numero_normalizado = numero / ultimo_numero  # Normalizando o número
        resultado = model(torch.tensor([[numero_normalizado]], dtype=torch.float32))
        if resultado.item() < 0.5:
            return "Par"
        else:
            return "Ímpar"

# Função para mover um número para o treinamento
def mover_para_treinamento(X_Treino, Y_Treino, numero, resultado):
    # Adiciona o número ao conjunto de treinamento
    X_Treino.append(numero)
    Y_Treino.append(0 if resultado == "Par" else 1)

    # Ordenar os dados (opcional, se precisar manter organização crescente)
    dados_ordenados = sorted(zip(X_Treino, Y_Treino))
    X_Treino, Y_Treino = zip(*dados_ordenados)

    # Converter novamente para listas
    return list(X_Treino), list(Y_Treino)

# Inicialização do modelo, critério e otimizador
model = Classificador()
criterion = nn.BCELoss()  # Função de perda para classificação binária
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Carregar dados
X_Treino, Y_Treino = carregar_dados(CAMINHO_ARQUIVO)

# Normalizando os dados
ultimo_numero = X_Treino[-1]  # Último número da lista
X_Treino_tensor = torch.tensor(X_Treino, dtype=torch.float32).unsqueeze(1)
Y_Treino_tensor = torch.tensor(Y_Treino, dtype=torch.float32).unsqueeze(1)
X_Treino_tensor = X_Treino_tensor / ultimo_numero

# Converta os tensores para arrays numpy
X_Treino_numpy = X_Treino_tensor.numpy()
Y_Treino_numpy = Y_Treino_tensor.numpy()

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(X_Treino_numpy)):
    print(f"Treinamento no Fold {fold+1}")
    
    # Dividir os dados em treinamento
    X_Treino_fold, X_Val_fold = X_Treino_numpy[train_index], X_Treino_numpy[val_index]
    Y_Treino_fold, Y_Val_fold = Y_Treino_numpy[train_index], Y_Treino_numpy[val_index]
    
    # Inicializar o modelo e o otimizador novamente para cada fold
    model = Classificador()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Treinamento do modelo
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(torch.tensor(X_Treino_fold, dtype=torch.float32).unsqueeze(1))
        loss = criterion(output, torch.tensor(Y_Treino_fold, dtype=torch.float32).unsqueeze(1))
        loss.backward()
        optimizer.step()

        # Validação
        model.eval()  # Muda o modelo para o modo de avaliação
        with torch.no_grad():
            val_output = model(torch.tensor(X_Val_fold, dtype=torch.float32).unsqueeze(1))
            val_loss = criterion(val_output, torch.tensor(Y_Val_fold, dtype=torch.float32).unsqueeze(1))

        if (epoch + 1) % 100 == 0:
            print(f'Época [{epoch+1}/{epochs}], Perda no Fold {fold+1}: {loss.item():.4f}, Validação: {val_loss.item():.4f}')

# Entrada de número para classificar
valor = int(input("Digite um número: "))
resultado = classificar_par_impar(model, valor, ultimo_numero)
print(f"O número {valor} é: {resultado}")

# Validação e salvamento dos dados
Validação = input("A resposta está correta?(Y/N): ").lower()

if Validação == "y":
    # Mover para o treinamento
    X_Treino, Y_Treino = mover_para_treinamento(X_Treino, Y_Treino, valor, resultado)
    print(f"O número {valor} foi movido para o treinamento.")
else:
    # Inverter o valor (se é par vira ímpar e vice-versa)
    resultado_invertido = "Par" if resultado == "Ímpar" else "Ímpar"
    print(f"A resposta estava errada. O número {valor} será movido para o treinamento com o valor invertido ({resultado_invertido}).")
    
    # Adicionar o número invertido ao treinamento
    X_Treino.append(valor)
    Y_Treino.append(0 if resultado_invertido == "Par" else 1)


# Garantir que Y_Treino seja reordenado de acordo com a ordem de X_Treino
dados_combinados = list(zip(X_Treino, Y_Treino))
dados_combinados.sort()  # Ordena pelos valores de X_Treino

# Separar novamente X_Treino e Y_Treino
X_Treino, Y_Treino = zip(*dados_combinados)

# Converter de volta para listas, se necessário
X_Treino = list(X_Treino)
Y_Treino = list(Y_Treino)

# Salvar os dados de treinamento no arquivo
salvar_dados(CAMINHO_ARQUIVO, X_Treino, Y_Treino)
if salvar_dados:
    print("Dados salvos com sucesso!")
#pqp#