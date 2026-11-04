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
2. Wybierz kończynę z listy lub `all`, aby śledzić wszystkie. Lewa i prawa strona odnoszą się do stron psa.
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
