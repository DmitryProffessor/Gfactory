import json
from byaldi import RAGMultiModalModel

def run_test():
    # Загружаем созданный индекс
    model = RAGMultiModalModel.from_index(".byaldi/multimodal_metallurgy_index")
    
    # Наша тестовая выборка (Ground Truth)
    # В реальности добавь сюда 5-10 вопросов
    test_cases = [
        {
            "question": "Какова схема извлечения золота из упорных руд?",
            "expected_page": 12, # Пример номера страницы
            "doc_name": "geokniga_lodeyshchikov..."
        }
    ]

    results = []
    for case in test_cases:
        # Ищем ответ (топ-3 результата)
        results_from_rag = model.search(case["question"], k=3)
        
        # Проверяем, попала ли нужная страница в топ-3 (Метрика Recall@3)
        found_pages = [res.page_num for res in results_from_rag]
        is_correct = case["expected_page"] in found_pages
        
        results.append({
            "question": case["question"],
            "found_pages": found_pages,
            "correct": is_correct
        })

    # Считаем точность
    accuracy = sum([1 for r in results if r["correct"]]) / len(results)
    print(f"Точность поиска (Recall@3): {accuracy * 100}%")

if __name__ == "__main__":
    run_test()
