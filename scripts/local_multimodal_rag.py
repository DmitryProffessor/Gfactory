import os
import torch
import gc
import transformers
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from byaldi import RAGMultiModalModel

# 1. НАСТРОЙКИ ОФФЛАЙН
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# ФИКС СОВМЕСТИМОСТИ
transformers.PreTrainedModel.all_tied_weights_keys = {}

def run_local_solution():
    MOONDREAM_ID = "vikhyatk/moondream2"
    index_name = "metallurgy_demo"
    image_dir = "data/sample_images"
    # Английский запрос для точности
    query = "Describe the mechanism of anchoring apolar collectors on the mineral surface shown in this diagram."

    print("\n--- 1. ЭТАП: ПОИСК СХЕМЫ (GPU) ---")
    try:
        search_model = RAGMultiModalModel.from_index(index_name, device="cuda")
        results = search_model.search(query, k=1)
        
        if not results:
            print("❌ Ничего не найдено.")
            return

        res = results[0]
        doc_id = res.doc_id
        if hasattr(search_model.model, 'doc_ids_to_file_names'):
            image_filename = search_model.model.doc_ids_to_file_names.get(doc_id, f"page_{doc_id}.png")
        else:
            image_filename = str(doc_id)

        image_filename = os.path.basename(image_filename)
        image_path = os.path.join(image_dir, image_filename)
        print(f"✅ Найдена страница: {image_path}")

        # ПОЛНАЯ ОЧИСТКА GPU
        del search_model
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return

    print("\n--- 2. ЭТАП: АНАЛИЗ ИЗОБРАЖЕНИЯ (CPU - для стабильности) ---")
    try:
        print("Загружаю Vision-модель на процессор (подождите 15-20 сек)...")
        
        # Загружаем на CPU в float32 (самый надежный режим)
        model = AutoModelForCausalLM.from_pretrained(
            MOONDREAM_ID, 
            trust_remote_code=True, 
            torch_dtype=torch.float32, 
            device_map="cpu" 
        )
        tokenizer = AutoTokenizer.from_pretrained(MOONDREAM_ID)

        image = Image.open(image_path)
        
        print("🤖 Нейросеть анализирует схему на CPU... Это может занять до минуты.")
        
        with torch.no_grad():
            # Кодируем
            image_embeds = model.encode_image(image)
            # Генерируем
            answer = model.answer_question(image_embeds, query, tokenizer=tokenizer)

        print("\n" + "="*60)
        print(f"ЗАПРОС: {query}")
        print(f"ОТВЕТ ЛОКАЛЬНОЙ МОДЕЛИ (CPU-стабильный):")
        print("-" * 30)
        print(answer)
        print("="*60)

    except Exception as e:
        print(f"\n❌ Ошибка генерации: {e}")

if __name__ == "__main__":
    run_local_solution()
