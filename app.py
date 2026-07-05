import os
import torch
import gc
import gradio as gr
import fitz  # PyMuPDF
from PIL import Image
from byaldi import RAGMultiModalModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
from concurrent.futures import ThreadPoolExecutor

transformers.PreTrainedModel.all_tied_weights_keys = {}

# --- CONFIG ---
INDEX_ROOT = ".byaldi"
INDEX_NAME = "ui_demo_index"
MODEL_ID = "vidore/colpali-v1.2"
MOONDREAM_ID = "vikhyatk/moondream2"
TEMP_IMAGE_DIR = "data/temp_pdf_pages" 

RAG = None
V_MODEL = None
V_TOK = None

def get_vision_model():
    global V_MODEL, V_TOK
    if V_MODEL is None:
        print("📥 Загрузка Moondream2 в RAM (CPU mode)...")
        V_MODEL = AutoModelForCausalLM.from_pretrained(
            MOONDREAM_ID, 
            trust_remote_code=True, 
            dtype=torch.float32,
            device_map="cpu" 
        )
        V_TOK = AutoTokenizer.from_pretrained(MOONDREAM_ID)
        print("✅ Moondream2 готова.")
    return V_MODEL, V_TOK

def save_page(args):
    """Вспомогательная функция для параллельной нарезки PDF"""
    pdf_path, page_num, zoom, output_path = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    pix.save(output_path)
    doc.close()

def pdf_to_images(pdf_path, progress, fast_mode=True):
    if os.path.exists(TEMP_IMAGE_DIR):
        shutil.rmtree(TEMP_IMAGE_DIR)
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    zoom = 1.2 if fast_mode else 2.0
    
    # ОПТИМИЗАЦИЯ: Параллельная нарезка страниц (используем все ядра CPU)
    tasks = []
    for i in range(num_pages):
        img_path = os.path.join(TEMP_IMAGE_DIR, f"{i}.png")
        tasks.append((pdf_path, i, zoom, img_path))
    
    progress(0.1, desc=f"Параллельная нарезка {num_pages} страниц...")
    with ThreadPoolExecutor() as executor:
        list(executor.map(save_page, tasks))
    
    doc.close()
    return TEMP_IMAGE_DIR

def index_document(pdf_file, fast_mode, progress=gr.Progress()):
    global RAG
    if pdf_file is None: return "⚠️ Файл не выбран!"
    
    try:
        # 1. Быстрая конвертация
        img_folder = pdf_to_images(pdf_file.name, progress, fast_mode)
        
        # 2. Загрузка модели ColPali
        progress(0.4, desc="Подготовка GPU...")
        if RAG is None:
            # ОПТИМИЗАЦИЯ: Если памяти впритык, можно добавить load_in_4bit=True (если byaldi поддерживает)
            RAG = RAGMultiModalModel.from_pretrained(MODEL_ID, device="cuda")
        
        # 3. Индексация
        progress(0.6, desc="Индексация (ColPali)...")
        RAG.index(
            input_path=img_folder,
            index_name=INDEX_NAME,
            store_collection_with_index=True,
            overwrite=True
        )
        
        torch.cuda.empty_cache()
        gc.collect()
        return f"✅ Проиндексировано страниц: {len(os.listdir(img_folder))}"
    except Exception as e:
        return f"❌ Ошибка индексации: {str(e)}"

def run_search(query, progress=gr.Progress()):
    global RAG
    if not query: return None, "Введите запрос."
    try:
        if RAG is None:
            print("🔍 Загрузка индекса ColPali на GPU...")
            RAG = RAGMultiModalModel.from_index(INDEX_NAME, device="cuda")
        
        progress(0.3, desc="Поиск в визуальном индексе...")
        print(f"🔎 Поиск по запросу: {query}")
        results = RAG.search(query, k=1)
        
        if not results: return None, "Ничего не найдено."
        
        res = results[0]
        doc_id = res.doc_id
        img_path = os.path.join(TEMP_IMAGE_DIR, f"{doc_id}.png")
        
        if not os.path.exists(img_path):
            return None, f"Ошибка: картинка {doc_id} не найдена."
            
        raw_img = Image.open(img_path)
        
        # Уменьшаем картинку для Moondream
        # Это сэкономит время на CPU без потери смысла
        progress(0.6, desc="Оптимизация изображения...")
        print("⚙️ Масштабирование картинки для Vision LLM...")
        analysis_img = raw_img.copy()
        analysis_img.thumbnail((768, 768)) # Moondream лучше всего работает с такими размерами
        
        progress(0.7, desc="Moondream2: Анализ (ждем CPU)...")
        print("🤖 Moondream2 начинает кодирование (это может занять 20-40 сек)...")
        v_model, v_tok = get_vision_model()
        
        with torch.no_grad():
            image_embeds = v_model.encode_image(analysis_img)
            print("📡 Генерация текстового ответа...")
            answer = v_model.answer_question(image_embeds, query, tokenizer=v_tok)
        
        print(f"🎉 Готово! Score: {res.score:.2f}")
        return raw_img, f"### Результат\n**Score:** {res.score:.2f} | **Page:** {doc_id}\n\n**🤖 ИИ:** {answer}"
        
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        return None, f"❌ Ошибка поиска: {str(e)}"

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="Gfactory RAG") as demo:
    gr.Markdown("# 🏭 Gfactory Hybrid RAG")
    
    with gr.Tabs():
        with gr.Tab("📁 Загрузка и Индексация"):
            pdf_input = gr.File(label="Загрузите PDF")
            fast_chk = gr.Checkbox(label="Fast Mode (Ускорить индексацию)", value=True)
            btn_idx = gr.Button("🚀 Индексировать", variant="primary")
            status = gr.Textbox(label="Статус")
            
        with gr.Tab("🔍 Поиск"):
            with gr.Row():
                with gr.Column(scale=1):
                    q_input = gr.Textbox(label="Ваш запрос")
                    btn_search = gr.Button("Найти", variant="primary")
                    out_text = gr.Markdown()
                with gr.Column(scale=2):
                    out_img = gr.Image(label="Найденный фрагмент", type="pil")

    btn_idx.click(index_document, inputs=[pdf_input, fast_chk], outputs=[status])
    btn_search.click(run_search, inputs=[q_input], outputs=[out_img, out_text])

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    demo.launch(server_name="0.0.0.0")