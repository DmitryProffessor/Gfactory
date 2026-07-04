import os
import torch
from byaldi import RAGMultiModalModel
from PIL import Image

# 1. Настройки
index_name = "metallurgy_demo"
image_dir = "data/sample_images"
query = "механизм закрепления аполярных собирателей на поверхности минералов"

def run_demo():
    print(f"--- 🔍 МУЛЬТИМОДАЛЬНЫЙ ПОИСК (ColPali) ---")
    
    try:
        # Загружаем индекс (это очень быстро)
        model = RAGMultiModalModel.from_index(index_name, device="cuda")
        
        print(f"Запрос: {query}")
        results = model.search(query, k=1)
        
        if results:
            res = results[0]
            # Получаем имя файла
            doc_id = res.doc_id
            if hasattr(model.model, 'doc_ids_to_file_names'):
                image_filename = model.model.doc_ids_to_file_names.get(doc_id, f"page_{doc_id}.png")
            else:
                image_filename = str(doc_id)
            
            image_path = os.path.join(image_dir, os.path.basename(image_filename))
            
            print(f"\n✅ УСПЕХ! Система нашла наиболее релевантную страницу.")
            print(f"📄 Файл: {image_filename}")
            print(f"🔥 Уверенность (Score): {res.score:.2f}")
            
            # АВТОМАТИЧЕСКИ ОТКРЫВАЕМ КАРТИНКУ (фишка для демо)
            if os.path.exists(image_path):
                print("Открываю изображение для демонстрации...")
                img = Image.open(image_path)
                img.show() 
            
            print("\nБизнес-ценность: Система нашла сложную технологическую схему без OCR,")
            print("используя только визуальные признаки, что критично для технических регламентов.")

    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    run_demo()
