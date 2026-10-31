Wheelczur - Śledzenie kończyn psa w nagraniach wideo

Projekt Wheelczur to uruchamiana w terminalu aplikacja do śledzenia wybranej kończyny psa na nagraniu wideo, co wspomaga proces stwierdzania chorób narządu ruchu. Program w kolejnych klatkach wykorzystuje model YOLOX-Tiny do wyznaczania ramki otaczającej psa, a następnie model RTMPose-m AP-10K do estymacji punktów anatomicznych w wyciętym obszarze.   Wymagania i środowisko
Aplikacja działa na systemach operacyjnych Linux oraz Windows.
Wymagane środowisko to Python w wersji od 3.10 do 3.13.

Obliczenia wykonywane są domyślnie na karcie graficznej NVIDIA z obsługą CUDA, ale użytkownik może jawnie wymusić działanie na procesorze (CPU).   Sposób użycia i parametry
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
