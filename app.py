"""
Веб-интерфейс для парсера научных PDF-документов на базе Streamlit.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

from parser.pipeline import process_single_file

# Настройка страницы
st.set_page_config(page_title="PDF to JSON Parser", page_icon="📄", layout="wide")

st.title("📄 Парсер научных статей (PDF ➡️ JSON)")
st.markdown(
    "Загрузи PDF-документ, "
    "и система разберет его на метаданные, секции, формулы и картинки."
)

# Боковая панель с настройками
st.sidebar.header("Настройки парсера")
use_offline = st.sidebar.checkbox("Offline режим (без сети)", value=False)
use_crossref = st.sidebar.checkbox(
    "Использовать CrossRef", value=True, disabled=use_offline
)
use_llm = st.sidebar.checkbox("Использовать локальную LLM (Ollama)", value=True)
extract_images = st.sidebar.checkbox("Извлекать фигуры и таблицы (YOLO)", value=True)

uploaded_file = st.file_uploader("Загрузите научную статью (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.info(f"Файл **{uploaded_file.name}** загружен. Нажмите кнопку ниже для старта.")

    if st.button("🚀 Запустить парсинг", type="primary"):
        # Создаем временную директорию для безопасной работы
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Сохраняем загруженный файл на диск
            pdf_path = tmp_path / uploaded_file.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            output_dir = tmp_path / "results"
            output_dir.mkdir(exist_ok=True)

            with st.spinner(
                "🧠 Нейронки и эвристики переваривают документ. Подождите..."
            ):
                try:
                    # Запускаем ваш боевой конвейер
                    process_single_file(
                        pdf_path=pdf_path,
                        output_dir=output_dir,
                        overwrite=True,
                        offline=use_offline,
                        use_crossref=use_crossref,
                        use_llm=use_llm,
                        extract_images=extract_images,
                    )

                    json_filename = f"{pdf_path.stem}.json"
                    json_path = output_dir / json_filename

                    if json_path.exists():
                        st.success("✅ Парсинг успешно завершен!")

                        # Читаем результат
                        with open(json_path, "r", encoding="utf-8") as f:
                            parsed_data = json.load(f)

                        # Делим экран пополам для красоты
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("📦 Итоговый JSON")
                            # Красивый вывод JSON в интерфейс
                            st.json(parsed_data, expanded=False)

                            # Кнопка для скачивания файла
                            with open(json_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    label="💾 Скачать JSON",
                                    data=f,
                                    file_name=json_filename,
                                    mime="application/json",
                                )

                        with col2:
                            st.subheader("🖼 Извлеченные изображения")
                            images_dir = output_dir / "images" / pdf_path.stem
                            if images_dir.exists() and any(images_dir.iterdir()):
                                for img_file in images_dir.glob("*.png"):
                                    st.image(
                                        str(img_file),
                                        caption=img_file.name,
                                        use_container_width=True,
                                    )
                            else:
                                st.warning(
                                    "Картинки и таблицы не найдены "
                                    "(или извлечение отключено)."
                                )
                    else:
                        st.error("Что-то пошло не так. JSON файл не был создан.")

                except Exception as e:
                    st.error(f"❌ Критическая ошибка во время парсинга: {e}")
