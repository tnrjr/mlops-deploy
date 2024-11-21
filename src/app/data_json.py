import pandas as pd

# Ler o arquivo CSV
df = pd.read_csv('C:\\Users\\tary.nascimento\\OneDrive\\Área de Trabalho\\mlops-deploy\\data\\data_set.csv')


# Converter para JSON
json_data = df.to_json(orient='records', lines=False)

# Salvar o JSON em um arquivo
with open('dados.json', 'w') as f:
    f.write(json_data)

print("Arquivo JSON criado com sucesso!")
