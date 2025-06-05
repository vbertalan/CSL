import os
import json
from collections import Counter

# Diretório onde os arquivos part_X.json estão localizados
output_dir = 'split_batches_with_sequences'

# Função para contar as ocorrências de event_ids nos arquivos part_X.json
def count_event_ids_in_file(file_path, event_id_counts, event_data):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for key, value in data.items():
            group = value.get('group', [])
            for event in group:
                event_id = event.get('event_id')
                if event_id:
                    event_id_counts[event_id] += 1

                    # Armazena os atributos "raw" e "template" para cada event_id
                    if event_id not in event_data:
                        event_data[event_id] = {
                            "raw": event.get('raw'),
                            "template": event.get('template')
                        }

# Contador global para armazenar frequências e dados dos events
event_id_counts = Counter()
event_data = {}

# Verificar todos os arquivos part_X.json e contar as ocorrências de event_ids
for file_name in os.listdir(output_dir):
    if file_name.startswith('part_') and file_name.endswith('.json'):
        file_path = os.path.join(output_dir, file_name)
        count_event_ids_in_file(file_path, event_id_counts, event_data)

# Encontrar os 20 event_ids mais frequentes
most_common_event_ids = event_id_counts.most_common(20)

# Criar o arquivo para salvar as informações
output_file = 'top_20_event_ids.json'

with open(output_file, 'w', encoding='utf-8') as f_out:
    result = []
    for event_id, frequency in most_common_event_ids:
        result.append({
            "event_id": event_id,
            "frequencia": frequency,
            "raw": event_data[event_id]["raw"],
            "template": event_data[event_id]["template"]
        })

    # Salva os dados no arquivo externo
    json.dump(result, f_out, indent=2, ensure_ascii=False)

print(f"Os 20 event_ids mais frequentes foram salvos em '{output_file}'")
