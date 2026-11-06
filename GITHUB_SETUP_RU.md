# Как загрузить этот проект на GitHub

1. Открой https://github.com/new и войди в свой GitHub.
2. Repository name: `FoodVision-Assignment2`.
3. Visibility: `Public`, если преподаватель должен открывать ссылку без приглашения.
4. Не ставь галочку «Add a README file», потому что наш README уже готов.
5. Нажми «Create repository».
6. Распакуй `FoodVision_GitHub_Ready.zip` на компьютере.
7. В новом репозитории выбери «uploading an existing file»
   (или «Add file» → «Upload files»).
8. Перетащи **содержимое** распакованной папки — все файлы и папку `artifacts`.
   Не загружай просто ZIP-файл: GitHub не превратит его автоматически в исходный код.
9. В поле сообщения коммита введи `Add FoodVision Assignment 2 project`.
10. Нажми «Commit changes» (или «Propose changes», если такая надпись).
11. Убедись, что на главной странице видны `README.md`, `app.py`, `train.py`,
    `requirements.txt` и папка `artifacts/`.
12. Скопируй ссылку вида `https://github.com/USERNAME/FoodVision-Assignment2`
    для сдачи в Moodle.

**Важно:** не загружай оригинальные фотографии и файл `food_dataset.zip`
в публичный репозиторий без проверки лицензий.
Сайт из такого репозитория сможет классифицировать фотографии
после установки зависимостей, но показ соседних фотографий
потребует локального датасета с исходными названиями файлов.

Созданный в Google Colab адрес trycloudflare.com временный:
он перестанет открываться после отключения сессии или туннеля.
В README есть инструкция локального запуска, как требует задание.
