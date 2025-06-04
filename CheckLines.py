import os
import json

# Lista de event_ids a serem verificados
event_ids_to_check = [
    
    "b4723d7b-d454-4230-a728-9a546ea3134f",
    "ccf01bda-afc9-4299-9d52-62866597d7cc",
    #"59e338b1-4f1c-4726-a372-f3ef14b9a93e",
    # Adicione outros event_ids conforme necessário
]

# Diretório onde os arquivos part_X.json estão localizados
output_dir = 'split_batches_with_sequences'

# Função para verificar as ocorrências dos event_ids nos arquivos part_X.json
def check_event_ids_in_file(file_path, event_ids_to_check):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        event_id_counts = {event_id: 0 for event_id in event_ids_to_check}
        
        for key, value in data.items():
            group = value.get('group', [])
            for event in group:
                event_id = event.get('event_id')
                if event_id in event_id_counts:
                    event_id_counts[event_id] += 1
        
        # Verifica se todos os event_ids aparecem pelo menos 2 vezes
        return all(count >= 2 for count in event_id_counts.values())

# Verificar todos os arquivos part_X.json
files_with_matching_event_ids = []
for file_name in os.listdir(output_dir):
    if file_name.startswith('part_') and file_name.endswith('.json'):
        file_path = os.path.join(output_dir, file_name)
        if check_event_ids_in_file(file_path, event_ids_to_check):
            files_with_matching_event_ids.append(file_name)

# Exibir arquivos que contêm pelo menos 2 vezes cada event_id
print("Arquivos que atendem a condição de ter pelo menos 2 ocorrências de cada event_id:")
for file in files_with_matching_event_ids:
    print(file)
