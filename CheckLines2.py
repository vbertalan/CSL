import os
import json

# Lista de event_ids a serem verificados
event_ids_to_check = [
    "ff9f4bb3-6876-46ef-a858-d8f84c7b0414",
    "4e3f21df-484d-48ce-950e-01319588822a",
    "5e81a0ea-28f1-4999-b01f-0d613d27f59d",
    "a5cf75d8-27fa-429e-9d26-f01c1f04436ce8",
    "8c604fdd-53c9-4c2a-9433-fb14e1ed6362",
    "07452e03-006a-40d5-b998-522408afbc33",
    "c5932d55-fc67-4100-901b-96a1582567d7",
    "7e6153b1-3b2d-45e4-acf4-4a54c936ba3b",
    # Adic1ione outros event_ids conforme necessário
]
# Diretório onde os arquivos part_X.json estão localizados
output_dir = 'split_batches_with_sequences'

# Função para verificar as ocorrências dos event_ids nos arquivos part_X.json
def check_event_ids_in_file(file_path, event_ids_to_check):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        event_id_counts = {event_id: 0 for event_id in event_ids_to_check}
        found_event_ids = set()

        for key, value in data.items():
            group = value.get('group', [])
            for event in group:
                event_id = event.get('event_id')
                if event_id in event_ids_to_check:
                    event_id_counts[event_id] += 1
                    found_event_ids.add(event_id)

        # Retorna True se algum event_id foi encontrado e as contagens dos event_ids encontrados
        return found_event_ids, event_id_counts

# Verificar todos os arquivos part_X.json
files_with_matching_event_ids = []
for file_name in os.listdir(output_dir):
    if file_name.startswith('part_') and file_name.endswith('.json'):
        file_path = os.path.join(output_dir, file_name)
        found_event_ids, event_id_counts = check_event_ids_in_file(file_path, event_ids_to_check)

        if found_event_ids:
            files_with_matching_event_ids.append((file_name, event_id_counts))

# Exibir arquivos que contêm pelo menos um dos event_ids e as contagens de ocorrências
print("Arquivos que contêm pelo menos um dos event_ids:")
for file_name, event_id_counts in files_with_matching_event_ids:
    print(f"\nArquivo: {file_name}")
    for event_id, count in event_id_counts.items():
        if count > 0:
            print(f"  Event ID: {event_id}, Ocorrências: {count}")
