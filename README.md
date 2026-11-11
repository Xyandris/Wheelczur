# Wheelczur

Program do śledzenia kończyn psa na nagraniach wideo. Wykorzystuje modele YOLOX-Tiny i RTMPose, a wyniki zapisuje jako dane CSV oraz film z zaznaczonymi punktami.

## 1. Instalacja Pythona

Program działa na Windowsie i Linuxie, wymaga Pythona **3.10–3.13** oraz internetu podczas instalacji.

### Windows

1. Na [python.org](https://www.python.org/downloads/windows/) znajdź **Python 3.13.x** i pobierz **Windows installer (64-bit)**.
2. Uruchom instalator, zaznacz **Add python.exe to PATH** i kliknij **Install Now**. Narzędzie `pip` zainstaluje się razem z Pythonem.
3. Otwórz nowe okno PowerShell i sprawdź instalację:

```powershell
python --version
python -m pip --version
```

Pierwsze polecenie powinno wyświetlić wersję `3.13.x`.

### Linux (Ubuntu 24.04 LTS / Debian 13)

Otwórz terminal i zainstaluj Pythona, `pip` oraz obsługę środowisk wirtualnych:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

Wersja musi mieścić się w zakresie **3.10–3.13**. W dalszych poleceniach na Linuxie używaj **`python3` zamiast `python`**. Na innych dystrybucjach zainstaluj zgodną wersję Pythona i moduł `venv` przez menedżer pakietów systemu.

### Brak polecenia `pip` lub modułu `pip`

Jeśli terminal zgłasza, że `pip` nie jest rozpoznawany, najpierw sprawdź `python -m pip --version` (na Linuxie: `python3 -m pip --version`). Jeśli to działa, używaj `python -m pip` zamiast samego `pip`.

**Windows — komunikat `No module named pip`:** zamknij menu launchera opcją **0**, a następnie wykonaj w PowerShell:

```powershell
python -m ensurepip --upgrade
python -m pip --version
```

Jeśli polecenie `python` nie działa, ale działa `py`, zastąp `python` przez `py` we wszystkich poleceniach. Jeśli brakuje również modułu `ensurepip`, uruchom ponownie instalator tej samej wersji Pythona 3.10–3.13 z [python.org](https://www.python.org/downloads/windows/), wybierz **Modify** i zaznacz składnik **pip**. Po instalacji otwórz nowe okno terminala i ponownie sprawdź wersję `pip`.

**Linux (Ubuntu / Debian) — brak `pip` lub `ensurepip`:** zainstaluj wymagane pakiety przez menedżer systemowy:

```bash
sudo apt update
sudo apt install python3-pip python3-venv
python3 -m pip --version
```

Jeśli korzystasz z innej wersji interpretera, np. `python3.12`, doinstaluj odpowiadający jej pakiet `python3.12-venv`.

**Brak `pip` w istniejącym środowisku Wheelczura:** jeśli błąd pochodzi z `.venv-cpu`, przywróć `pip` bezpośrednio w tym środowisku:

Windows (PowerShell):

```powershell
.\.venv-cpu\Scripts\python.exe -m ensurepip --upgrade
.\.venv-cpu\Scripts\python.exe -m pip --version
```

Linux:

```bash
./.venv-cpu/bin/python -m ensurepip --upgrade
./.venv-cpu/bin/python -m pip --version
```

Dla GPU zamień `.venv-cpu` na `.venv-cuda`. Jeśli brakuje `ensurepip`, najpierw napraw instalację Pythona zgodnie z instrukcją dla swojego systemu powyżej. Następnie ponów `python launcher.py --setup cpu` (dla GPU: `--setup cuda`; na Linuxie użyj `python3`).

Opis mechanizmu przywracania `pip`: [dokumentacja Pythona — ensurepip](https://docs.python.org/3.13/library/ensurepip.html).

## 2. Instalacja programu

Na GitHubie wybierz **Code → Download ZIP** i rozpakuj archiwum. Otwórz terminal w folderze zawierającym `launcher.py` i wykonaj kolejno:

```bash
python launcher.py --setup cpu
python launcher.py --models
```

Pierwsze polecenie tworzy środowisko `.venv-cpu` i instaluje wszystkie wymagane pakiety, w tym `numpy` i `opencv-python-headless`. Drugie pobiera i sprawdza modele. Nie trzeba ręcznie aktywować środowiska.

### Błąd certyfikatu przy pobieraniu modeli (Windows)

Jeśli przy pobieraniu modeli pojawi się `CERTIFICATE_VERIFY_FAILED` lub `unable to get local issuer certificate`, spróbuj poniższego rozwiązania. Wyjdź z menu launchera opcją **0**, jeśli jest otwarte. Następnie w terminalu, w folderze projektu, wykonaj kolejno:

```bat
python -m pip install --upgrade truststore
python -c "import truststore; truststore.inject_into_ssl(); import download_models; raise SystemExit(download_models.main())"
```

**Wykonaj obie komendy** — druga uruchamia pobieranie z użyciem certyfikatów systemowych. Samo zainstalowanie `truststore` nie zmienia działania opcji 7 w launcherze.

Po komunikacie **Oba modele są gotowe** uruchom ponownie `python launcher.py` i kontynuuj pracę z menu.

## 3. Użycie

```bash
python launcher.py --run cpu
```

1. Podaj ścieżkę do filmu, np. `test-vid/doggi-walk-01.mp4`.
2. Wybierz jedną z czterech kończyn: `left-fore`, `right-fore`, `left-hind` lub `right-hind`. Lewa i prawa strona odnoszą się do stron psa.
3. Poczekaj na zakończenie analizy.

Wyniki znajdziesz obok filmu, w folderze `<nazwa_filmu_bez_rozszerzenia>_tracking/`:

- **CSV** — współrzędne punktów, czas i pewność wykrycia.
- **MP4** — film z zaznaczonymi punktami i połączeniami.
- **JSON** — ustawienia i podsumowanie analizy.

### Obsługa przez menu tekstowe

Po zainstalowaniu Pythona możesz wykonać wszystkie kroki z menu tekstowego. W folderze projektu wystarczy uruchomić:

```bash
python launcher.py
```

Menu umożliwia instalację pakietów, pobranie modeli, analizę na CPU/GPU oraz sprawdzenie środowiska. Wpisz numer wybranej opcji i zatwierdź klawiszem **Enter**. Przy pierwszym użyciu na CPU wybierz kolejno **1 → 7 → 3**.

## Opcjonalnie: karta NVIDIA (CUDA)

Po pobraniu modeli zamiast poleceń CPU użyj:

```bash
python launcher.py --setup cuda
python launcher.py --run cuda
```

Instalator automatycznie doda `onnxruntime-gpu` z bibliotekami CUDA/cuDNN. Wymagana jest karta NVIDIA z odpowiednim sterownikiem, a w Windows także Visual C++ x64 Runtime.

### Windows: błąd ładowania biblioteki cuDNN

Jeśli karta NVIDIA jest wykrywana, ale test modeli kończy się błędem `Could not locate cudnn_engines_tensor_ir64_9.dll` lub `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`, problem dotyczy ładowania podbibliotek cuDNN. Aktualna wersja Wheelczura udostępnia katalogi bibliotek NVIDIA z wybranego środowiska i ładuje dodatkowe biblioteki przed uruchomieniem modelu. Nie trzeba ustawiać systemowej zmiennej `PATH`.

Zamknij launcher opcją **0**, rozpakuj aktualne pliki projektu do swojego folderu Wheelczura, a następnie sprawdź oba modele w PowerShell:

```powershell
.\.venv-cuda\Scripts\python.exe track_limb.py --check-cuda --device cuda:0
```

Jeśli test się powiedzie, uruchom analizę:

```powershell
.\.venv-cuda\Scripts\python.exe launcher.py --run cuda
```

Jeśli błąd nadal wskazuje brak tego pliku, sprawdź, czy pakiet go zawiera:

```powershell
Test-Path .\.venv-cuda\Lib\site-packages\nvidia\cudnn\bin\cudnn_engines_tensor_ir64_9.dll
```

Dla cuDNN **9.28.1.4** z opisanego komunikatu brakujący lub uszkodzony pakiet można przywrócić poleceniem:

```powershell
.\.venv-cuda\Scripts\python.exe -m pip install --force-reinstall --no-deps "nvidia-cudnn-cu12==9.28.1.4"
```

Jeśli masz inną wersję, sprawdź ją poleceniem `.\.venv-cuda\Scripts\python.exe -m pip show nvidia-cudnn-cu12` i użyj jej numeru zamiast `9.28.1.4`. Po naprawie ponów test `--check-cuda`. Nie kopiuj pojedynczych bibliotek DLL z innej wersji cuDNN.

Dokumentacja: [ONNX Runtime — ładowanie bibliotek CUDA/cuDNN](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls) oraz [NVIDIA — instalacja cuDNN w Windows](https://docs.nvidia.com/cudnn/installation/backend/windows/).
