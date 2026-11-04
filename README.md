Wheelczur - Śledzenie kończyn psa w nagraniach wideo

Projekt Wheelczur to uruchamiana w terminalu aplikacja do śledzenia wybranej kończyny psa na nagraniu wideo, co wspomaga proces stwierdzania chorób narządu ruchu. Program w kolejnych klatkach wykorzystuje model YOLOX-Tiny do wyznaczania ramki otaczającej psa, a następnie model RTMPose-m AP-10K do estymacji punktów anatomicznych w wyciętym obszarze.   Wymagania i środowisko
Aplikacja działa na systemach operacyjnych Linux oraz Windows.
Wymagane środowisko to Python w wersji od 3.10 do 3.13.

## Szybki setup

Rozpakuj całe archiwum i otwórz terminal w folderze projektu. Skrypt korzysta
wyłącznie z biblioteki standardowej Pythona; do pobrania zależności potrzebny jest
internet.

Domyślna instalacja dla procesora (CPU):

```bash
python setup_env.py
```

Opcjonalna instalacja dla karty NVIDIA (CUDA):

```bash
python setup_env.py --device cuda
```

W Linuxie zamiast `python` może być potrzebne `python3`, a w Windows można użyć
`py -3.13`. Jeśli system nie ma modułu `venv`/`ensurepip`, doinstaluj go dla używanej
wersji Pythona (np. pakiet `python3-venv` w Debianie/Ubuntu).

Skrypt tworzy `.venv`, instaluje projekt w trybie edytowalnym, sprawdza zależności
i import aplikacji, a następnie wyświetla gotowe komendy uruchomienia. Ponowne
wywołanie korzysta z tego samego środowiska. Aktywowanie `.venv` nie jest konieczne.
Ścieżka do skryptu może być bezwzględna, więc setup działa też z innego katalogu.

Dodatkowe opcje:

- `--dev` — instaluje także narzędzia do testów, np. `python setup_env.py --dev`.
- `--venv .venv-cpu` — wybiera inny katalog środowiska; ścieżki względne odnoszą się do folderu projektu.
- `--help` — wyświetla pomoc.

Instalacja CUDA wymaga karty NVIDIA i odpowiedniego sterownika, a w Windows także
Visual C++ x64 Runtime. Skrypt podaje komendę `--check-cuda --device cuda:0`,
która sprawdza rzeczywiste obliczenia modeli na GPU. CPU jest domyślne także
przy uruchamianiu aplikacji; użycie GPU wymaga jawnego `--device cuda:0`.

Obliczenia wykonywane są domyślnie na procesorze (CPU). Użytkownik może wybrać kartę graficzną NVIDIA z obsługą CUDA przez `--device cuda:0`.   Sposób użycia i parametry
Program przetwarza wideo klatka po klatce i wymaga podania ścieżki do filmu oraz wyboru śledzonej kończyny (np. left-hind dla lewej tylnej lub all dla 12 punktów wszystkich kończyn).

Użytkownik może przekazać dodatkowe parametry konfigurujące działanie programu:
--fps: Ręczne wymuszenie liczby klatek na sekundę przy błędnych metadanych wideo.
--confidence i --detector-confidence: Określenie własnych progów pewności dla punktów oraz detekcji psa.
--bbox: Wskazanie stałego obszaru obejmującego całego psa, zastępujące detekcję YOLOX. 
--output-dir i --overwrite: Zmiana folderu docelowego i zezwolenie na nadpisanie istniejących plików wynikowych.

Wyniki działania
Wyniki zapisywane są domyślnie w nowym katalogu <nazwa_filmu>_tracking. Zakończenie analizy generuje trzy pliki:
Plik CSV: Zawiera indeks klatki, czas, współrzędne pikseli (x, y), miarę pewności (score) oraz status obserwacji (np. detected, no_dog).
Plik wideo: Kopia oryginalnego nagrania (w formacie MP4 lub AVI) z naniesionymi połączonymi znacznikami dla prawidłowo wykrytych punktów.
Plik JSON: Zapis konfiguracji uruchomienia, wykorzystanego urządzenia, sum kontrolnych modeli oraz statystyk przebiegu.


Projekt stworzony w Katedrze Informatyki na Wydziale Nauk Inżynieryjnych Akademii Nauk Stosowanych w Nowym Sączu pod nadzorem dr inż. Przemysława Pardela. 
Zespół autorski tworzą: Dawid Plewa (Lider),Patrycja Obrzut, Kamil Obrzut, Jakub Pyrdoł oraz Tymoteusz Zaczyk.   
