import os
import json

# Lista de event_ids a serem verificados
event_ids_to_check = [
    "ff9f4bb3-6876-46ef-a858-d8f84c7b0414",
    # "4e3f21df-484d-48ce-950e-01319588822a",
    # "5e81a0ea-28f1-4999-b01f-0d613d27f59d",
    # "a5cf75d8-27fa-429e-9d26-fc1f04436ce8",
    # "8c604fdd-53c9-4c2a-9433-fb14e1ed6362",
    # "07452e03-006a-40d5-b998-522408afbc33",
    # "c5932d55-fc67-4100-901b-96a1582567d7",
    # "7e6153b1-3b2d-45e4-acf4-4a54c936ba3b",
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
