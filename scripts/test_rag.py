from byaldi import RAGMultiModalModel
import json

def run_evaluation():
    # Загружаем наш индекс
    model = RAGMultiModalModel.from_index(".byaldi/metallurgy_index")
    
    # Загружаем вопросы
    with open("tests/ground_truth.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"{'Вопрос':<50} | {'Найдена стр.':<12} | {'Статус':<10}")
    print("-" * 80)

    correct_count = 0
    for case in test_cases:
        # Ищем (k=3 означает, что мы смотрим топ-3 результата)
        results = model.search(case["question"], k=3)
        
        # Проверяем, есть ли ожидаемая страница в результатах
        found_pages = [res.page_num for res in results]
        is_correct = case["expected_page"] in found_pages
        
        if is_correct:
            correct_count += 1
            status = "✅ УСПЕХ"
        else:
            status = "❌ НЕУДАЧА"
            
        print(f"{case['question'][:47]+'...':<50} | {str(found_pages):<12} | {status}")

    accuracy = (correct_count / len(test_cases)) * 100
    print(f"\nИтоговая точность (Recall@3): {accuracy}%")

if __name__ == "__main__":
    run_evaluation()
